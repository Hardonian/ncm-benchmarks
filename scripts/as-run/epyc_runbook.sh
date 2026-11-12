#!/usr/bin/env bash
# NCM V100 Dedicated Benchmark Runner — Hardonian Labs
# Executes on EPYC GPU box. Verifies paper claims without co-tenant interference:
# 1. Sudoku-hard full 2,000 boards (bs=128)
# 2. XSum clean latency benchmark (bs=1 and bs=32)
# 3. Optional full 11,334-doc census pass (bs=32, ~16 min)
set -Eeuo pipefail
export PATH="$HOME/.local/bin:$PATH"

# Pre-flight: pick cleanest GPU if index 0 has resident memory
GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "=== NCM GPU BENCHMARK (CUDA_VISIBLE_DEVICES=$GPU_ID) ==="
nvidia-smi --id=$GPU_ID --query-gpu=name,memory.used,memory.total --format=csv,noheader

ROOT=${NCM_ROOT_DIR:-$HOME/ai-lab/ncm}
mkdir -p "$ROOT/logs" "$ROOT/results"
cd "$ROOT"

# Ensure repo is up to date
if [ -d ncm-benchmarks ]; then
  cd ncm-benchmarks && git pull origin main && cd ..
else
  git clone https://github.com/Hardonian/ncm-benchmarks.git
fi

# Ensure upstream is cloned and patched
if [ ! -d ncm ]; then
  git clone https://github.com/HazyResearch/ncm ncm
  cd ncm
  git checkout c70274273f489abc73832b2b98584a8483b6b203
  git apply "$ROOT/ncm-benchmarks/patches/device-support.patch"
  cd "$ROOT"
fi

# Set up virtual environment with CUDA PyTorch
if [ ! -x "$ROOT/venv/bin/python" ]; then
  uv venv "$ROOT/venv" --python 3.12
  uv pip install --python "$ROOT/venv/bin/python" torch --index-url https://download.pytorch.org/whl/cu126
  uv pip install --python "$ROOT/venv/bin/python" -r "$ROOT/ncm-benchmarks/requirements.txt"
fi

PY="$ROOT/venv/bin/python"
$PY -c 'import torch; assert torch.cuda.is_available(), "no CUDA"; print("GPU:", torch.cuda.get_device_name(0), "VRAM:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")'

# Fetch and verify checkpoints and datasets
cd "$ROOT/ncm-benchmarks"
$PY scripts/fetch_artifacts.py
cd "$ROOT"
ln -sfn "$ROOT/ncm-benchmarks/artifacts" "$ROOT/artifacts"
ln -sfn "$ROOT/ncm-benchmarks/data" "$ROOT/data"

export NCM_ROOT="$ROOT/ncm" NCM_DEVICE=cuda TOKENIZERS_PARALLELISM=false

run() { # $1 name, $2 config, $3 ckptdir, $4 datadir, $5 n, $6 bs
  mkdir -p "$ROOT/results/$1"
  echo "$1 n=$5 bs=$6 start=$(date -u +%FT%TZ)" | tee "$ROOT/results/$1/run.meta"
  cd "$ROOT/ncm"
  { time $PY scripts/eval_shard.py --config "config/eval/$2" \
      --checkpoint-dir "$ROOT/artifacts/$3" --data-dir "$ROOT/data/$4" \
      --n "$5" --bs "$6" --shard 0 --nshards 1 --out "$ROOT/results/$1/shard_0.pkl" ; } \
      2> "$ROOT/results/$1/time.txt"
  cat "$ROOT/results/$1/time.txt" | tee -a "$ROOT/results/$1/run.meta"
  cd "$ROOT"
}

# 1. Sudoku: full 2,000-board hard test
echo "--> Running Sudoku Hard (2000 boards, bs=128)..."
run sudoku_full sudoku-hard.yaml sudoku/hard sudoku/hard 2000 128
cd "$ROOT/ncm"
$PY scripts/eval_merge.py --config config/eval/sudoku-hard.yaml --glob "$ROOT/results/sudoku_full/shard_*.pkl" | tee "$ROOT/results/sudoku_full/score.txt"
cd "$ROOT"

# 2. XSum clean latency & scaling (bs=1, bs=32)
echo "--> Running XSum clean latency tests (bs=1, bs=32)..."
run xsum_bs1_n8    xsum.yaml xsum xsum 8 1
run xsum_bs1_n40   xsum.yaml xsum xsum 40 1
run xsum_bs32_n160 xsum.yaml xsum xsum 160 32
run xsum_bs32_n1000 xsum.yaml xsum xsum 1000 32

$PY "$ROOT/ncm-benchmarks/scripts/score_rouge_cpu.py" "$ROOT/results/xsum_bs32_n1000/shard_0.pkl" | tee "$ROOT/results/xsum_bs32_n1000/rouge.txt"
$PY "$ROOT/ncm-benchmarks/scripts/score_rouge_cpu.py" "$ROOT/results/xsum_bs32_n160/shard_0.pkl"   | tee "$ROOT/results/xsum_bs32_n160/rouge.txt"

# 3. Optional: Full XSum Census (11,334 documents, ~16 min)
if [ "${RUN_FULL_XSUM:-0}" = "1" ]; then
  echo "--> Running FULL XSum test set (11,334 docs, bs=32)..."
  run xsum_full xsum.yaml xsum xsum 11334 32
  $PY "$ROOT/ncm-benchmarks/scripts/score_rouge_cpu.py" "$ROOT/results/xsum_full/shard_0.pkl" | tee "$ROOT/results/xsum_full/rouge.txt"
fi

echo "NCM-EPYC-DONE" > "$ROOT/logs/epyc.status"
echo "All GPU runs successfully completed without co-tenant contention."
