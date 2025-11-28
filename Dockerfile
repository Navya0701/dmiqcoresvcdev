# Use the official lightweight Python image.
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (so Docker caching works)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port 8080 for Cloud Run
ENV PORT=8080
EXPOSE 8080

# Run the Flask app
CMD ["python", "app.py"]