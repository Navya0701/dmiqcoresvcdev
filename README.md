# DMIQ Core Service

This is the service that the front end app calls. It's a Python Flask application that hosts RESTful APIs.

## Features

- RESTful API endpoints
- Health check endpoint
- JSON response format
- Error handling
- Environment-based configuration
- Unit tests

## Requirements

- Python 3.7+
- Flask 3.0.0
- python-dotenv 1.0.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dowlurukk/dmiqcoresvc.git
cd dmiqcoresvc
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Create a `.env` file for custom configuration:
```bash
cp .env.example .env
```

## Configuration

The application can be configured using environment variables. Create a `.env` file in the root directory:

```
FLASK_DEBUG=False
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
```

## Running the Application

Start the Flask server:

```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Root
- **GET** `/` - Service information
  ```json
  {
    "service": "DMIQ Core Service",
    "version": "1.0.0",
    "status": "running"
  }
  ```

### Health Check
- **GET** `/health` - Health status
  ```json
  {
    "status": "healthy",
    "service": "dmiqcoresvc"
  }
  ```

### API Status
- **GET** `/api/v1/status` - API operational status
  ```json
  {
    "api_version": "v1",
    "status": "operational"
  }
  ```

### Data Endpoint
- **GET** `/api/v1/data` - Get sample data
  ```json
  {
    "data": [
      {"id": 1, "name": "Sample 1"},
      {"id": 2, "name": "Sample 2"}
    ]
  }
  ```

- **POST** `/api/v1/data` - Submit data
  - Request body (JSON):
    ```json
    {
      "name": "Example",
      "value": 123
    }
    ```
  - Response:
    ```json
    {
      "message": "Data received successfully",
      "received_data": {...}
    }
    ```

## Testing

Run the unit tests:

```bash
python -m unittest discover tests
```

Or run a specific test file:

```bash
python -m unittest tests.test_api
```

## Development

To run the application in development mode with debug enabled:

1. Set `FLASK_DEBUG=True` in your `.env` file
2. Run: `python app.py`

## License

This project is part of the DMIQ application suite. 
