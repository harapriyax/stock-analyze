# ==============================================================================
# AlphaPulse v2.0 - Production Dockerfile
# ==============================================================================

# Lightweight official Python runtime
FROM python:3.11-slim

# Prevent Python from writing .pyc bytecode and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install minimal OS dependencies for healthcheck & build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU first to avoid heavy GPU driver downloads
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application source code
COPY . .

# Expose default Streamlit port
EXPOSE 8501

# Streamlit container health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit server
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
