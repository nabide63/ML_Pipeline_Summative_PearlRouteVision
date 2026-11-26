# Docker image: the FastAPI service (port 8000) and the Streamlit UI (port 8501) in one container.
#   docker build -t pearlroute-vision .
#   docker run -p 8000:8000 -p 8501:8501 --env-file .env pearlroute-vision
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    # config.py finds configs/, data/ and models/ from here (the package is installed, not run from src/)
    PEARLROUTE_ROOT=/app \
    PEARLROUTE_API_URL=http://127.0.0.1:8000

WORKDIR /app

# Dependencies first, so code changes don't reinstall TensorFlow
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-deps .

COPY configs/ configs/
COPY scripts/ scripts/
COPY app/ app/
COPY models/ models/
# Small data files the app and API read (the images themselves are mounted, see docker-compose.yml)
COPY data/counts.json data/metadata.csv data/stats.json data/exclude.txt data/train_exclude.txt data/retrain_sample.zip data/
COPY start.sh .

RUN sed -i 's/\r$//' start.sh && chmod +x start.sh \
    && useradd --create-home --uid 1000 pearl \
    && mkdir -p data/processed data/demo data/cache data/retrain_runs \
    && chown -R pearl /app/models /app/data
USER pearl

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["./start.sh"]
