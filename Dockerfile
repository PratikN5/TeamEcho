# Use a full image (not slim) because of heavy ML dependencies
FROM python:3.11-bullseye

WORKDIR /app

# Environment configs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install required system dependencies for OCR, CV, PDF, and AI
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libpoppler-cpp-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    ghostscript \
    openjdk-11-jre-headless \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy only requirements first (for caching)
COPY requirements.txt .

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default command
CMD ["python", "-m", "app.scripts.run_ingestion_agent"]
