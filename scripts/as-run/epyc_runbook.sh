#!/usr/bin/env bash
# NCM V100 benchmark runner — execute ON EPYC after Tailscale auth clears.
# Verifies paper claims on our GPU lane: sudoku-hard full 2000 boards + XSum n=1000 + bs=1 latency.
set -Eeuo pipefail
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
ROOT=${NCM_ROOT_DIR:-$HOME/ai-lab/ncm}
mkdir -p "$ROOT/logs"
cd "$ROOT"

[ -d ncm ] || { echo "FATAL: patched repo not synced yet"; exit 1; }
cd ncm

# venv with CUDA torch (cu126 to match lane fleet convention)
if [ ! -x "$ROOT/venv/bin/python" ]; then
  uv venv "$ROOT/venv" --python 3.12
  uv pip install --python "$ROOT/venv/bin/python" torch --index-url https://download.pytorch.org/whl/cu126
  uv pip install --python "$ROOT/venv/bin/python" numpy omegaconf hydra-core 'transformers>=4.51' 'huggingface_hub[cli]' regex rouge_score sacrebleu
fi
PY="$ROOT/venv/bin/python"
$PY -c 'import torch; assert torch.cuda.is_available(), "no CUDA"; print("GPU:", torch.cuda.get_device_name(0), "bf16:", torch.cuda.get_device_capability())'

# artifacts (explicit filenames; hf v2 quirk)
cd "$ROOT"
mkdir -p artifacts data
export HF_HOME="$ROOT/hf-home" HF_HUB_CACHE="$ROOT/hf-home/hub" TRANSFORMERS_CACHE="$ROOT/hf-home/hub" HF_HUB_DISABLE_XET=1
[ -f artifacts/xsum/generator.pt ] || "$ROOT/venv/bin/hf" download hazyresearch/ncm-checkpoints \
  xsum/config.yaml xsum/generator.pt xsum/tokenizer.pt \
  sudoku/hard/config.yaml sudoku/hard/generator.pt sudoku/hard/tokenizer.pt --local-dir artifacts > logs/hf-ckpts.log 2>&1
[ -f data/xsum/passages.pickle ] || "$ROOT/venv/bin/hf" download hazyresearch/ncm-datasets \
  xsum/passages.pickle xsum/reference_summaries.pickle sudoku/hard/puzzles.pickle sudoku/hard/solutions.pickle \
  --repo-type dataset --local-dir data > logs/hf-data.log 2>&1

export NCM_ROOT="$ROOT/ncm" NCM_DEVICE=cuda TOKENIZERS_PARALLELISM=false

run() { cd "$ROOT/ncm"  # $1 name, $2 config, $3 ckptdir, $4 datadir, $5 n, $6 bs
  mkdir -p "$ROOT/results/$1"
  echo "$1 n=$5 bs=$6 start=$(date -u +%FT%TZ)" | tee "$ROOT/results/$1/run.meta"
  { time $PY scripts/eval_shard.py --config "config/eval/$2" \
      --checkpoint-dir "$ROOT/artifacts/$3" --data-dir "$ROOT/data/$4" \
      --n "$5" --bs "$6" --shard 0 --nshards 1 --out "$ROOT/results/$1/shard_0.pkl" ; } \
      2> "$ROOT/results/$1/time.txt"
  cat "$ROOT/results/$1/time.txt" | tee -a "$ROOT/results/$1/run.meta"
}

# Sudoku: full 2000-board hard test (their exact metric path is GPU-safe)
run sudoku_full  sudoku-hard.yaml sudoku/hard sudoku/hard 2000 128
$PY scripts/eval_merge.py --config config/eval/sudoku-hard.yaml --glob "$ROOT/results/sudoku_full/shard_*.pkl" | tee "$ROOT/results/sudoku_full/score.txt"

# XSum latency (bs=1) + throughput/quality set (bs=32 n=1000)
run xsum_bs1_n8   xsum.yaml xsum xsum 8 1
run xsum_bs1_n40  xsum.yaml xsum xsum 40 1
run xsum_bs32_n160 xsum.yaml xsum xsum 160 32
run xsum_bs32_n1000 xsum.yaml xsum xsum 1000 32
$PY "$ROOT/score_rouge_cpu.py" "$ROOT/results/xsum_bs32_n1000/shard_0.pkl" | tee "$ROOT/results/xsum_bs32_n1000/rouge.txt"
$PY "$ROOT/score_rouge_cpu.py" "$ROOT/results/xsum_bs32_n160/shard_0.pkl"   | tee "$ROOT/results/xsum_bs32_n160/rouge.txt"

echo "NCM-EPYC-DONE" > "$ROOT/logs/epyc.status"
