# ── SmartDoc-Insight Dockerfile ───────────────────────────────────────────────
FROM python:3.11-slim

# System dependencies (libgl1/libglib for opencv, curl for the healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies (cached layer). Set INSTALL_OCR=1 to add PaddleOCR.
ARG INSTALL_OCR=0
COPY requirements.txt requirements-ocr.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    if [ "$INSTALL_OCR" = "1" ]; then pip install --no-cache-dir -r requirements-ocr.txt; fi

# App code
COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY test_data/ ./test_data/
COPY .env.example .env

# Create data directories
RUN mkdir -p /app/data/chroma_db /app/data/uploads /app/data/cache

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.maxUploadSize=50", \
     "--browser.gatherUsageStats=false"]
