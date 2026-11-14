# syntax=docker/dockerfile:1
ARG TARGETPLATFORM=linux/amd64
FROM --platform=$TARGETPLATFORM python:3.12-slim-bookworm

# Runtime dependencies + non-root user
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10001 ncmgroup \
    && useradd -u 10001 -g ncmgroup -s /usr/sbin/nologin -m ncmuser

WORKDIR /app

# Pin offline cache environment variables
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

# Clone and patch upstream NCM at the exact verified SHA
COPY patches/device-support.patch /app/patches/device-support.patch
RUN git clone https://github.com/HazyResearch/ncm.git /app/ncm \
    && cd /app/ncm \
    && git checkout c70274273f489abc73832b2b98584a8483b6b203 \
    && git apply /app/patches/device-support.patch \
    && rm -rf /app/ncm/.git

# Install PyTorch CPU (MKL/oneDNN) + dependencies + local package
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt pyproject.toml /app/
RUN pip install --no-cache-dir -r requirements.txt safetensors pypdf anyio

# Copy source and checksums (mount ./artifacts and ./hf-home read-only at runtime)
COPY SHA256SUMS /app/SHA256SUMS
COPY ncm_bench/ /app/ncm_bench/
COPY ncm_train/ /app/ncm_train/
COPY service/ /app/service/
COPY scripts/ /app/scripts/

RUN pip install --no-cache-dir --no-deps -e . \
    && mkdir -p /app/artifacts /app/hf-home \
    && chown -R ncmuser:ncmgroup /app

USER ncmuser

EXPOSE 8031

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8031/ready || exit 1

ENTRYPOINT ["sh", "-c", "ncm-bench verify --artifacts-only && exec python service/summarize_server.py \"$@\"", "--"]
CMD ["--host", "0.0.0.0", "--port", "8031", "--quantize", "int8"]
