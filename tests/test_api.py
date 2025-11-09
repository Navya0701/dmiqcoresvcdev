"""
Unit tests for DMIQ Core Service API endpoints
"""
import unittest
import json
from app import app


class TestAPIEndpoints(unittest.TestCase):
    """Test cases for API endpoints"""

    def setUp(self):
        """Set up test client"""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_index(self):
        """Test root endpoint"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['service'], 'DMIQ Core Service')
        self.assertEqual(data['status'], 'running')

    def test_health(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['service'], 'dmiqcoresvc')

    def test_api_status(self):
        """Test API status endpoint"""
        response = self.client.get('/api/v1/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['api_version'], 'v1')
        self.assertEqual(data['status'], 'operational')

    def test_get_data(self):
        """Test GET request to data endpoint"""
        response = self.client.get('/api/v1/data')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertEqual(len(data['data']), 2)

    def test_post_data(self):
        """Test POST request to data endpoint"""
        test_data = {'name': 'Test Item', 'value': 123}
        response = self.client.post(
            '/api/v1/data',
            data=json.dumps(test_data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        self.assertEqual(data['message'], 'Data received successfully')
        self.assertEqual(data['received_data'], test_data)

    def test_post_data_no_json(self):
        """Test POST request to data endpoint without JSON data"""
        response = self.client.post('/api/v1/data')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'No data provided')

    def test_not_found(self):
        """Test 404 error handler"""
        response = self.client.get('/nonexistent')
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Resource not found')


if __name__ == '__main__':
    unittest.main()
