"""
DMIQ Core Service - Flask API Application
This is the service that the front end app calls
"""
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import os
import logging
import traceback
import psutil
from src.rag_qa_enhanced import RAGQASystem

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Vector store path
vecstore_path = os.getenv('VECSTORE_PATH', '/mnt/medical_data')
logger.info(f"Vector store path: {vecstore_path}")

# Initialize Flask app
app = Flask(__name__)
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

# ---------------------------------------------------------------
# ⭐ FIX: Initialize RAG system ONCE at container startup
# ---------------------------------------------------------------
RAG_SYSTEM = None

try:
    logger.info("🔄 Initializing RAG system ONCE at startup...")

    if not os.path.exists(vecstore_path):
        logger.error(f"❌ Vector store path does not exist: {vecstore_path}")
    else:
        RAG_SYSTEM = RAGQASystem(
            stores_base=vecstore_path,
            model="gpt-4o"
        )
        logger.info("✅ RAG system initialized successfully.")

except Exception as e:
    logger.error(f"❌ Failed to initialize RAG system during startup: {str(e)}")


# ---------------------------------------------------------------
# Basic routes
# ---------------------------------------------------------------
@app.route('/')
def index():
    return jsonify({
        'service': 'DMIQ Core Service',
        'version': '1.0.0',
        'status': 'running'
    })


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200


@app.route('/api/v1/status')
def status():
    return jsonify({'api_version': 'v1', 'status': 'operational'}), 200


# ---------------------------------------------------------------
# Diagnostics endpoint
# ---------------------------------------------------------------
@app.route('/api/v1/diagnostics')
def diagnostics():
    try:
        info = {
            'vecstore_path': vecstore_path,
            'vecstore_exists': os.path.exists(vecstore_path),
            'rag_initialized': RAG_SYSTEM is not None,
            'python_version': os.sys.version,
            'cwd': os.getcwd()
        }

        if os.path.exists(vecstore_path):
            contents = os.listdir(vecstore_path)
            info['vecstore_contents'] = contents[:20]
            info['vecstore_count'] = len(contents)

        return jsonify(info), 200

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ---------------------------------------------------------------
# MAIN QUERY ROUTE (working version)
# ---------------------------------------------------------------
@app.route('/api/v1/testq', methods=['GET', 'POST'])
def test_question():

    try:
        # Get question
        if request.method == 'GET':
            question = request.args.get('question')
        else:
            body = request.get_json(silent=True)
            question = body.get('question') if body else None

        if not question:
            return jsonify({
                "error": "Missing question parameter",
                "example": "/api/v1/testq?question=What is diabetes"
            }), 400

        logger.info(f"📩 Received question: {question}")

        # Ensure RAG system is loaded
        if RAG_SYSTEM is None:
            logger.error("❌ RAG not initialized")
            return jsonify({
                "error": "RAG system not initialized",
                "message": "Check vector store mount or startup logs"
            }), 500

        # Execute query
        result = RAG_SYSTEM.query(
            question=question,
            top_k=10,
            per_shard_k=10,
            include_history=False
        )

        logger.info(f"✅ Query successful. Cost: ${result.get('cost', 0):.4f}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(str(e))
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "trace": traceback.format_exc() if app.config['DEBUG'] else None
        }), 500


# ---------------------------------------------------------------
# Additional routes
# ---------------------------------------------------------------
@app.route('/api/v1/data', methods=['GET'])
def get_data():
    return jsonify({"data": [{"id": 1}, {"id": 2}]})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found"}), 404


# ---------------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
