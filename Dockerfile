# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/tmp \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR $HOME/app

# Switch to root to install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
USER user

# Copy requirements first (for Docker layer caching)
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy download_models script and run it
COPY --chown=user download_models.py .
RUN python download_models.py

# Copy all project files
COPY --chown=user . .

# Create output directory
RUN mkdir -p output && chown user:user output

# Hugging Face Spaces uses Port 7860
EXPOSE 7860

# Use Gunicorn for production
CMD gunicorn -w 1 --bind 0.0.0.0:7860 app:app
