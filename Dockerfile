# Use Python base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install pip requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Cloud Run expects 8080)
EXPOSE 8080

# Start the app with gunicorn
CMD ["gunicorn", "-b", ":8080", "app:app"]
