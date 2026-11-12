#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
echo "========================================================"
echo "Starting NCM Sovereign Summarizer & Inspector Service"
echo "========================================================"

export NCM_HOME="$HERE"
export NCM_ROOT="$HERE/ncm"
export NCM_DEVICE=${NCM_DEVICE:-cpu}
export HF_HOME="$HERE/hf-home"
export HF_HUB_CACHE="$HERE/hf-home/hub"
export TRANSFORMERS_CACHE="$HERE/hf-home/hub"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1

PY="python"
if [ -x "$HERE/venv/bin/python" ]; then
    PY="$HERE/venv/bin/python"
elif [ -x "$HERE/.venv-cpu/bin/python" ]; then
    PY="$HERE/.venv-cpu/bin/python"
fi

echo "Using Python: $PY"
exec $PY service/summarize_server.py --port 8031 --host 127.0.0.1 --quantize int8
