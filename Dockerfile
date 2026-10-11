# ============================================================
# Surface Crack Detection — API FastAPI
# Image CPU pour inférence avec TensorFlow
# ============================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Dépendance système requise par certaines bibliothèques ML.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python.
COPY requirements-docker.txt /app/

RUN python -m pip install --no-cache-dir \
    -r requirements-docker.txt

# Copier uniquement le code applicatif.
COPY src/ /app/src/

# Le modèle sera monté via Docker Compose.
RUN mkdir -p /app/artifacts \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Vérification HTTP sans installer curl.
HEALTHCHECK --interval=30s --timeout=10s \
    --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8)" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
