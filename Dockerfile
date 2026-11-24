# Production Dockerfile for Korba PM2.5 & PM10 Forecasting System
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and pre-install lightweight CPU-only PyTorch
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Ensure runtime directories exist
RUN mkdir -p /tmp/matplotlib_cache outputs/forecasts outputs/reports outputs/alerts data/raw/user_uploads

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib_cache \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Expose default port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit with dynamic $PORT binding for Render, Railway, Cloud Run
CMD sh -c "streamlit run app/dashboard.py --server.port=${PORT:-8501} --server.address=0.0.0.0"
