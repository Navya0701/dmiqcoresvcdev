"""
DMIQ Core Service - Flask API Application
"""
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import os
import logging
from flask_cors import CORS
from src.rag_qa_enhanced import RAGQASystem
from services.firestore_service import create_thread, save_message, get_threads, get_messages

# Load ENV variables
load_dotenv()

app = Flask(__name__)
app.config['DEBUG'] = os.getenv("FLASK_DEBUG", "False").lower() == "true"

# Enable CORS for frontend
CORS(app, origins="*", supports_credentials=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vector DB path
vecstore_path = os.getenv("VECSTORE_PATH", "/mnt/medical_data")

# Initialize RAG System
RAG_SYSTEM = None
try:
    if os.path.exists(vecstore_path):
        RAG_SYSTEM = RAGQASystem(stores_base=vecstore_path, model="gpt-4o")
        logger.info("🔥 RAG Initialized Successfully!")
    else:
        logger.warning(f"⚠ Vector DB not found: {vecstore_path}")
except Exception as e:
    logger.error(f"❌ RAG Startup Failed: {e}")


@app.route('/')
def home():
    return jsonify({"status": "DeepMedIQ Backend Running"}), 200


@app.get('/api/query')
def ai_query():
    question = request.args.get("question")

    if not question:
        return jsonify({"error": "Missing question"}), 400

    logger.info(f"📩 Query Received: {question}")
    try:
        result = RAG_SYSTEM.query(
            question=question,
            top_k=10,
            include_history=False
        )
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"❌ AI Query Failed: {e}")
        return jsonify({"error": "AI Query Failed", "details": str(e)}), 500


@app.post("/threads")
def api_create_thread():
    data = request.get_json()
    user_id = data.get("userId")

    if not user_id:
        return jsonify({"error": "userId required"}), 400

    thread_id = create_thread(user_id)
    logger.info(f"🧵 Thread Created → {thread_id}")

    return jsonify({"threadId": thread_id}), 200


@app.get("/threads")
def api_get_threads():
    user_id = request.args.get("userId")

    if not user_id:
        return jsonify({"error": "userId required"}), 400

    return jsonify(get_threads(user_id)), 200




@app.get("/threads/<thread_id>/messages")
def api_get_messages(thread_id):
    user_id = request.args.get("userId")

    if not user_id:
        return jsonify({"error": "userId required"}), 400

    return jsonify(get_messages(user_id, thread_id)), 200


@app.post("/threads/<thread_id>/messages")
def api_add_message(thread_id):
    data = request.get_json()
    user_id = data.get("userId")
    role = data.get("role")
    content = data.get("content")

    if not user_id or not role or not content:
        return jsonify({"error": "Missing fields"}), 400

    save_message(user_id, thread_id, role, content)
    logger.info(f"💬 Message Saved Under Thread {thread_id}")
    return jsonify({"status": "saved"}), 200


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Route Not Found"}), 404


