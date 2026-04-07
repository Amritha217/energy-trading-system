# Dockerfile
# ==========
# Multi-purpose container for the Energy Trading AI project.
# Supports running the Streamlit dashboard, FastAPI server, or full ML pipeline.
#
# Build:  docker build -t energy-trading-ai .
# Run dashboard:  docker run -p 8501:8501 -v $(pwd)/data:/app/data energy-trading-ai \
#                   streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
# Run API:        docker run -p 8000:8000 -v $(pwd)/data:/app/data energy-trading-ai \
#                   uvicorn api.main:app --host 0.0.0.0 --port 8000
# Run pipeline:   docker run -v $(pwd)/data:/app/data energy-trading-ai python main.py

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed by Prophet (Stan) and LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project source
COPY . .

# Create the data/models directory so the pipeline can write model files
RUN mkdir -p data/models

# Expose ports for dashboard (8501) and API (8000)
EXPOSE 8501 8000

# Default command: run the full ML pipeline
CMD ["python", "main.py"]