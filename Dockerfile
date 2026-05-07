FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    ffmpeg \
    fonts-liberation \
    fonts-noto-color-emoji \
    fonts-noto-cjk \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    libxshmfence1 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install Chromium
RUN apt-get update && apt-get install -y --no-install-recommends chromium && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Create output directories
RUN mkdir -p /app/output/videos /app/output/plans /app/output/audio /app/output/frames

# Set environment variables
ENV DISPLAY=:99
ENV PYTHONUNBUFFERED=1
ENV BH_VIDEO_DIR=/app/output/videos
ENV BH_PLAN_DIR=/app/output/plans
ENV CHROME_PATH=/usr/bin/chromium
ENV CHROMIUM_FLAGS="--no-sandbox --disable-gpu --disable-dev-shm-usage --headless=new"

# Expose the web UI port
EXPOSE 7860

# Start Xvfb and the app
CMD Xvfb :99 -screen 0 1920x1080x24 & \
    sleep 2 && \
    python app.py
