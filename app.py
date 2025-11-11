"""
DMIQ Core Service - Flask API Application
This is the service that the front end app calls
"""
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import os
from src.rag_qa_enhanced import RAGQASystem

# Load environment variables
load_dotenv()

# Path to vector store
vecstore_path = '/home/filesharemount'

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


@app.route('/api/v1/testq')
def test_question():
    """Test queue endpoint"""

    # Initialize
    system = RAGQASystem(
        stores_base=r"vecstore_path",
        model="gpt-4o"
    )

    # Query
    result = system.query(
        question="What are the indications for Barrett's esophagus screening?",
        top_k=10,
        per_shard_k=10,
        include_history=False
    )

    # Access results (for logging)
    print(result['answer'])
    print(f"Cost: ${result['cost']:.4f}")
    print(f"Citations: {len(result['citations'])}")
    
    # Return the full result as JSON
    return jsonify(result), 200

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
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )
