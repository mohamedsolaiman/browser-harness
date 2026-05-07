FROM python:3.11-slim

# Install system dependencies (curl for health check + ffmpeg for video)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directories
RUN mkdir -p /tmp/content-studio/videos /tmp/content-studio/plans /tmp/content-studio/audio

# Expose Gradio port
EXPOSE 7860

# Health check (curl is now installed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the app
CMD ["python", "app.py"]
