from typing import List
import numpy as np
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False
    torch = None
    AutoTokenizer = None
    AutoModel = None
    # fallbacks
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingAgent:
    """Create embeddings using a HuggingFace transformer model.
    Produces normalized float32 vectors suitable for FAISS IndexFlatIP.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self.use_tfidf = False
        self.model_name = model_name
        if _HAS_TRANSFORMERS:
            try:
                self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(self.device)
                self.model.eval()
            except Exception:
                # fallback to TF-IDF if model loading fails at runtime
                self.use_tfidf = True
                self.vectorizer = TfidfVectorizer()
        else:
            # No transformers/torch available: use TF-IDF fallback
            self.use_tfidf = True
            self.vectorizer = TfidfVectorizer()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_texts(self, texts: List[str], batch_size: int = 16) -> List[np.ndarray]:
        embeddings: List[np.ndarray] = []
        if self.use_tfidf:
            # If the vectorizer already has a fixed vocabulary (e.g. loaded from disk),
            # use transform to keep dimensions stable. Otherwise fit on the provided
            # texts (fallback behavior).
            if hasattr(self.vectorizer, "vocabulary_") and self.vectorizer.vocabulary_:
                X = self.vectorizer.transform(texts).toarray()
            else:
                X = self.vectorizer.fit_transform(texts).toarray()
            # normalize
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            X = X / norms
            for row in X:
                embeddings.append(row.astype("float32"))
            return embeddings

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc)
                pooled = self._mean_pooling(out, enc["attention_mask"]).cpu().numpy()
                # normalize
                norms = np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12
                pooled = pooled / norms
                for row in pooled:
                    embeddings.append(row.astype("float32"))
        return embeddings
