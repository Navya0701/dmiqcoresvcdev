# Use the official lightweight Python image
FROM python:3.11-slim

# Working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Set dynamic port for Cloud Run
ENV PORT=8080

# Use Gunicorn for production server
CMD exec gunicorn --bind "0.0.0.0:${PORT}" app:app
