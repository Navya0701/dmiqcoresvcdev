"""
DMIQ Core Service - Flask API Application
This is the service that the front end app calls
"""
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import os
import logging
import traceback
from src.rag_qa_enhanced import RAGQASystem

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to vector store
vecstore_path = os.getenv('VECSTORE_PATH', '/home/filesharemount')
logger.info(f"Vector store path: {vecstore_path}")

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
app.config['HOST'] = os.getenv('FLASK_HOST', '0.0.0.0')
app.config['PORT'] = int(os.getenv('FLASK_PORT', 5000))


@app.route('/')
def index():
    """Root endpoint - API information"""
    return jsonify({
        'service': 'DMIQ Core Service',
        'version': '1.0.0',
        'status': 'running'
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'dmiqcoresvc'
    }), 200


@app.route('/api/v1/status')
def api_status():
    """API status endpoint"""
    return jsonify({
        'api_version': 'v1',
        'status': 'operational'
    }), 200


@app.route('/api/v1/diagnostics')
def diagnostics():
    """Diagnostics endpoint - check environment and file system"""
    try:
        diagnostics_info = {
            'vecstore_path': vecstore_path,
            'vecstore_exists': os.path.exists(vecstore_path),
            'environment': {
                'OPENAI_API_KEY_SET': bool(os.getenv('OPENAI_API_KEY')),
                'VECSTORE_PATH': os.getenv('VECSTORE_PATH', 'not set'),
                'FLASK_DEBUG': os.getenv('FLASK_DEBUG', 'not set'),
            },
            'current_directory': os.getcwd(),
            'python_version': os.sys.version,
        }
        
        # Check if vecstore path exists and list contents
        if os.path.exists(vecstore_path):
            try:
                contents = os.listdir(vecstore_path)
                diagnostics_info['vecstore_contents'] = contents[:20]  # First 20 items
                diagnostics_info['vecstore_item_count'] = len(contents)
            except Exception as e:
                diagnostics_info['vecstore_error'] = str(e)
        
        # Check home directory structure
        home_path = '/home'
        if os.path.exists(home_path):
            diagnostics_info['home_contents'] = os.listdir(home_path)
        
        return jsonify(diagnostics_info), 200
        
    except Exception as e:
        logger.error(f"Diagnostics error: {str(e)}")
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/v1/testq')
def test_question():
    """Test queue endpoint"""
    try:
        logger.info("Starting test question query...")
        logger.info(f"Vector store path: {vecstore_path}")
        
        # Check if vector store path exists
        if not os.path.exists(vecstore_path):
            logger.error(f"Vector store path does not exist: {vecstore_path}")
            return jsonify({
                'error': 'Vector store not found',
                'path': vecstore_path,
                'message': 'Please ensure the Azure File Share is mounted correctly'
            }), 500
        
        # Initialize RAG system
        logger.info("Initializing RAG system...")
        system = RAGQASystem(
            stores_base=vecstore_path,
            model="gpt-4o"
        )
        
        # Query
        logger.info("Executing query...")
        result = system.query(
            question="What are the indications for Barrett's esophagus screening?",
            top_k=10,
            per_shard_k=10,
            include_history=False
        )
        
        # Log results
        logger.info(f"Query successful. Cost: ${result['cost']:.4f}, Citations: {len(result['citations'])}")
        
        # Return the full result as JSON
        return jsonify(result), 200
        
    except Exception as e:
        # Log the full error with traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in test_question: {str(e)}")
        logger.error(f"Full traceback:\n{error_details}")
        
        # Return detailed error (only in debug mode for security)
        if app.config['DEBUG']:
            return jsonify({
                'error': str(e),
                'traceback': error_details,
                'vecstore_path': vecstore_path
            }), 500
        else:
            return jsonify({
                'error': 'Internal server error',
                'message': str(e),
                'type': type(e).__name__
            }), 500

@app.route('/api/v1/data', methods=['GET', 'POST'])
def handle_data():
    """Example data endpoint that handles GET and POST requests"""
    if request.method == 'GET':
        # Return sample data
        return jsonify({
            'data': [
                {'id': 1, 'name': 'Sample 1'},
                {'id': 2, 'name': 'Sample 2'}
            ]
        }), 200
    
    elif request.method == 'POST':
        # Handle data submission
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Echo back the received data with a success message
        return jsonify({
            'message': 'Data received successfully',
            'received_data': data
        }), 201


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    logger.warning(f"404 error: {request.url}")
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"500 error: {str(error)}")
    logger.error(traceback.format_exc())
    return jsonify({
        'error': 'Internal server error',
        'message': str(error) if app.config['DEBUG'] else 'An error occurred'
    }), 500


if __name__ == '__main__':
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )
