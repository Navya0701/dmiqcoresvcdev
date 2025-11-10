"""
Enhanced RAG QA system with conversation history, citation extraction, and cost tracking.
Used by the web UI and batch processing modes.
"""
import os
from dotenv import load_dotenv
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import heapq
import joblib

from agents.embedding_agent import EmbeddingAgent
from utils.vector_store import FaissStore
from utils.pdf_reader import read_pdf, read_text_file
from utils.text_chunker import chunk_text as chunk_text_fn

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_community.callbacks import get_openai_callback
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available. Install with: pip install langchain langchain-openai langchain-community")


SYSTEM_PROMPT = (
    "You are an expert assistant guiding a physician in providing guideline based recommendations for patient care. "
    "Take the most recent guideline data as primary source and use any other guidelines papers that were published within 2 years of the primary source for comparison. "
    "If there is no relevant guideline data in the last 2 years, use older data, but explicitly mention that there are no recent guidelines on the topic. "
    "Use the following pieces of retrieved context to answer the question in the following format: "
    "\n\n (Recommendation:)"
    "\n    * [Provide a clear and concise recommendation along with context like what data and guidelines the recommendation is taken from]"
    "\n\n (Rationale and Supportive Arguments:)"
    "\n    * [Provide a detailed explanation of the recommendation along with the rationale for the recommendation with context such as what are the main pathophysiological considerations for the question in context, data behind management strategy and guidelines the recommendation is taken from]"
    "\n    * [Detail the Reasoning behind the recommendation, rationale for the recommendation with pathophysiological context from the guidelines as well as the references contained within the guidelines and detail the risk of harm without the recommended management strategy]" 
    "\n    * [List of factors and links supporting the recommendation] "
    "\n\n (Important Considerations:)"
    "\n    * [List of key factors to consider along with links to the references]"
    "\n    * [List any key factors that might make this recommendation unsuitable for a particular patient]"
    "\n    * [List any key risks, complications or harm that could occur with the recommendation]" 
    "\n    * [List any alternative management strategies]"  
    "\n\n (Relevant guidelines:)"
    "\n    * [Mention the main guidelines used to formulate the above recommendation. If multiple guidelines were used to synthesis the above recommendations, mention all with title of guidelines and year of publication and links for references]"
    "\n    * [If there is difference in opinion between different guidelines, Summarize and highlight the differences in guidelines other than the main guideline used for the recommendation]" 
    "\n    * [Summarize and highlight if there is a different recommendation from different regional or older guidelines]"       
    "\n\n (Areas of uncertainty and Controversies in management)"
    "\n   * [List any opposing schools of thought, and any differences in recommendation in other recent guidelines within 4 years of the primary guideline. Explain any controversies on the topic]"
    "\n (Primary source of data:)"
    "\n   * [list the Titles of the guidelines used for this recommendation]"
    "\n Year of publication:"
    "\n   * [show the year of guideline used for recommendation along with society of the guideline]"
    "\n If there is not enough guideline supported data to make a recommendation, please explain that."
    "\n If you don't know the answer, say that you don't know. "
    "\n\nContext:\n{context}"
)


class RAGQASystem:
    """Enhanced RAG QA system with conversation history and cost tracking."""
    
    def __init__(self, stores_base: str, model: str = "gpt-4o", chunk_chars: int = 1500, overlap: int = 200):
        self.stores_base = stores_base
        self.model = model
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self.conversation_history: List[Dict[str, Any]] = []
        self.total_cost = 0.0
        self.total_tokens = 0
        
        # Load environment variables from a local .env if present
        load_dotenv()

        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required. Install with: pip install langchain langchain-openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Set it in your environment or in a .env file at project root.")

        self.llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
        self.stores = self._find_stores()
        
    def _find_stores(self) -> List[Path]:
        """Find all FAISS stores in the base directory."""
        base_path = Path(self.stores_base)
        stores: List[Path] = []
        if not base_path.exists():
            return stores
        
        for child in base_path.iterdir():
            if child.is_dir() and (child / "index.faiss").exists() and (child / "metadata.json").exists():
                stores.append(child)
        
        if (base_path / "index.faiss").exists() and (base_path / "metadata.json").exists():
            stores.append(base_path)
        
        return stores
    
    def retrieve_contexts(self, query: str, top_k: int = 10, per_shard_k: int = 10) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        """Retrieve contexts and extract citations."""
        results_heap: List[tuple] = []
        counter = 0
        
        for store_path in self.stores:
            store = FaissStore(dim=384, store_path=str(store_path))
            embedder = EmbeddingAgent()
            
            vect = store_path / "tfidf_vectorizer.joblib"
            if vect.exists():
                try:
                    embedder.vectorizer = joblib.load(vect)
                    embedder.use_tfidf = True
                except Exception:
                    pass
            
            q_emb = embedder.embed_texts([query])[0]
            shard_results = store.search(q_emb, top_k=per_shard_k)
            
            for r in shard_results:
                md = r.get("metadata", {})
                md = dict(md)
                md["store"] = str(store_path)
                md["store_name"] = store_path.name
                item = {"score": r.get("score", 0.0), "metadata": md}
                heapq.heappush(results_heap, (-item["score"], counter, item))
                counter += 1
        
        # Get top_k results
        results: List[Dict[str, Any]] = []
        while results_heap and len(results) < top_k:
            _, _, item = heapq.heappop(results_heap)
            results.append(item)
        
        # Extract citations
        citations = []
        seen_sources = set()
        for r in results:
            md = r.get("metadata", {})
            source = md.get("source", "Unknown")
            if source != "Unknown" and source not in seen_sources:
                seen_sources.add(source)
                citations.append({
                    "source": source,
                    "filename": Path(source).name,
                    "store": md.get("store_name", "Unknown"),
                    "score": r.get("score", 0.0)
                })
        
        return results, citations
    
    def build_context_string(self, results: List[Dict[str, Any]]) -> str:
        """Build context string with actual chunk text."""
        context_parts = []
        for i, r in enumerate(results, 1):
            md = r.get("metadata", {})
            source = md.get("source", "Unknown")
            chunk_idx = md.get("chunk_index", -1)
            
            chunk_text_value = ""
            try:
                if source and source != "Unknown":
                    p = Path(source)
                    if p.exists():
                        if p.suffix.lower() == ".pdf":
                            full_text = read_pdf(str(p))
                        else:
                            full_text = read_text_file(str(p))
                        
                        chunks = chunk_text_fn(full_text, chunk_chars=self.chunk_chars, overlap_chars=self.overlap)
                        if 0 <= chunk_idx < len(chunks):
                            chunk_text_value = chunks[chunk_idx]
            except Exception as e:
                print(f"Warning: could not read chunk from {source}: {e}")
            
            if chunk_text_value:
                context_parts.append(f"[Source: {Path(source).name}, Chunk {chunk_idx}]\n{chunk_text_value}")
            else:
                context_parts.append(f"[Source: {Path(source).name}, Chunk {chunk_idx}] (text unavailable)")
        
        return "\n\n---\n\n".join(context_parts)
    
    def generate_followup_questions(self, question: str, answer: str, context: str) -> List[str]:
        """Generate followup questions."""
        followup_template = """
Based on the original question: {original_question}
And the answer: {answer}
With context: {context}

Generate 3 most relevant followup questions that would help explore this topic further.
Format as a simple numbered list:
1. [first question]
2. [second question]
3. [third question]
"""
        
        try:
            followup_prompt = PromptTemplate(
                input_variables=["original_question", "answer", "context"],
                template=followup_template
            )
            
            # Use LCEL (LangChain Expression Language) - new syntax in LangChain 1.0+
            followup_chain = followup_prompt | self.llm
            
            with get_openai_callback() as cb:
                response = followup_chain.invoke({
                    "original_question": question,
                    "answer": answer,
                    "context": context[:2000]  # Truncate context for followup generation
                })
                
                self.total_cost += cb.total_cost
                self.total_tokens += cb.total_tokens
            
            # Extract text content from response
            text = response.content if hasattr(response, 'content') else str(response)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            return lines[:3]
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error generating followup questions: {e}")
            print(f"Full traceback:\n{error_details}")
            return []
    
    def query(self, question: str, top_k: int = 10, per_shard_k: int = 10, include_history: bool = False) -> Dict[str, Any]:
        """Query the RAG system with cost tracking."""
        
        # Retrieve contexts
        results, citations = self.retrieve_contexts(question, top_k=top_k, per_shard_k=per_shard_k)
        
        if not results:
            response = {
                "question": question,
                "answer": "No relevant context found in the knowledge base.",
                "citations": [],
                "followup_questions": [],
                "cost": 0.0,
                "tokens": 0,
                "timestamp": datetime.now().isoformat()
            }
            self.conversation_history.append(response)
            return response
        
        # Build context
        context_str = self.build_context_string(results)
        
        # Add conversation history if requested
        history_context = ""
        if include_history and self.conversation_history:
            history_parts = []
            for i, entry in enumerate(self.conversation_history[-3:], 1):  # Last 3 exchanges
                history_parts.append(f"Previous Q{i}: {entry['question']}\nPrevious A{i}: {entry['answer'][:500]}...")
            history_context = "\n\n".join(history_parts) + "\n\n---\n\n"
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ])
        
        # Invoke LLM with cost tracking
        try:
            chain = prompt | self.llm
            
            with get_openai_callback() as cb:
                response_obj = chain.invoke({
                    "context": history_context + context_str,
                    "input": question
                })
                
                answer = response_obj.content
                cost = cb.total_cost
                tokens = cb.total_tokens
                
                self.total_cost += cost
                self.total_tokens += tokens
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error generating answer: {e}")
            print(f"Full traceback:\n{error_details}")
            answer = f"An error occurred while generating the answer: {e}"
            cost = 0.0
            tokens = 0
        
        # Generate followup questions
        followups = self.generate_followup_questions(question, answer, context_str)
        
        response = {
            "question": question,
            "answer": answer,
            "citations": citations,
            "followup_questions": followups,
            "cost": cost,
            "tokens": tokens,
            "timestamp": datetime.now().isoformat(),
            "num_contexts": len(results)
        }
        
        self.conversation_history.append(response)
        return response
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_total_cost(self) -> float:
        """Get total API cost."""
        return self.total_cost
    
    def get_total_tokens(self) -> int:
        """Get total tokens used."""
        return self.total_tokens
    
    def export_history(self, filepath: str):
        """Export conversation history to JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "conversation_history": self.conversation_history,
                "total_cost": self.total_cost,
                "total_tokens": self.total_tokens,
                "model": self.model,
                "stores_base": self.stores_base
            }, f, indent=2, ensure_ascii=False)
