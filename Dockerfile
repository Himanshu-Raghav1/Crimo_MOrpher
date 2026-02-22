# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy download_models script and run it to bake models into the image
COPY download_models.py .
RUN python download_models.py

# Copy all project files
COPY . .

# Create output directory
RUN mkdir -p output

# Expose Flask port (Render uses $PORT, but we expose 5000 for local testing)
EXPOSE 5000

# Use Gunicorn for production
# -w 1: Single worker to save RAM on free tiers
# --bind: Use the PORT environment variable
CMD gunicorn -w 1 --bind 0.0.0.0:${PORT:-5000} app:app
