# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

# System dependencies + non-root runtime user
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10001 ncmgroup \
    && useradd -u 10001 -g ncmgroup -s /usr/sbin/nologin -m ncmuser

WORKDIR /app

# Pin cache environment variables for strict offline enforcement
ENV NCM_HOME=/app \
    NCM_ROOT=/app/ncm \
    NCM_DEVICE=cpu \
    HF_HOME=/app/hf-home \
    HF_HUB_CACHE=/app/hf-home/hub \
    TRANSFORMERS_CACHE=/app/hf-home/hub \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=8 \
    PYTHONUNBUFFERED=1

# Install PyTorch CPU with MKL/oneDNN optimization
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt pyproject.toml /app/
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir requests

# Copy application source, SHA256SUMS, and vendored upstream code
COPY SHA256SUMS /app/SHA256SUMS
COPY ncm_bench/ /app/ncm_bench/
COPY service/ /app/service/
COPY ncm/ /app/ncm/
COPY artifacts/ /app/artifacts/
COPY hf-home/ /app/hf-home/

RUN chown -R ncmuser:ncmgroup /app

USER ncmuser

EXPOSE 8031

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8031/health || exit 1

ENTRYPOINT ["python", "service/summarize_server.py"]
CMD ["--host", "0.0.0.0", "--port", "8031", "--quantize", "int8"]
