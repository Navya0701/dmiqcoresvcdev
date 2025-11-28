"""
DMIQ Core Service - Flask API Application
This is the service that the front end app calls
"""
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import os
import logging
from flask_cors import CORS
from src.rag_qa_enhanced import RAGQASystem
from services.firestore_service import save_chat


# Load environment variables
load_dotenv()

# Initialize Flask
app = Flask(__name__)
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

# CORS Configuration (allow all for now)
CORS(app, origins="*", supports_credentials=True)

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Vector Store Path
vecstore_path = os.getenv('VECSTORE_PATH', '/mnt/medical_data')
logger.info(f"Vector store path: {vecstore_path}")

# Initialize RAG System ONCE
RAG_SYSTEM = None
try:
    logger.info("🔄 Initializing RAG system...")
    if not os.path.exists(vecstore_path):
        logger.warning(f"⚠️ Vector store path does not exist: {vecstore_path}")
    else:
        RAG_SYSTEM = RAGQASystem(
            stores_base=vecstore_path,
            model="gpt-4o"
        )
        logger.info("🚀 RAG system initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize RAG: {str(e)}")

# Default route
@app.route('/')
def index():
    return jsonify({"service": "DMIQ Core Service", "status": "running"}), 200

# Test Question → Query AI + Save Chat
@app.route('/api/v1/testq', methods=['GET'])
def test_question():
    try:
        question = request.args.get('question')
        if not question:
            return jsonify({"error": "Missing question parameter"}), 400

        logger.info(f"📩 Received question: {question}")

        # Query RAG
        result = RAG_SYSTEM.query(
            question=question,
            top_k=10,
            per_shard_k=10,
            include_history=False,
        )
        logger.info("🤖 RAG Query completed")

        # Extract answer for logging
        answer = result.get("answer", "No answer returned")

        # Save to Firestore
        try:
            save_chat(
                user_id="anonymous",
                question=question,
                answer=answer
            )
            logger.info("🔥 Chat saved to Firestore")
        except Exception as firestore_error:
            logger.error(f"❌ Firestore write failed: {firestore_error}")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"❌ Error handling request: {e}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

# 404 Handler
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found"}), 404

# Run server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
