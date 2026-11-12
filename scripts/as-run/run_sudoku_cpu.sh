#!/usr/bin/env bash
# Sudoku-hard CPU reproduction: NCM claims 86.1% solved on the 2000-board hard test (12 NFE, 25M model).
# We run n=200 boards on HX370 CPU and score with their own exact() metric path (pure CPU).
set -Eeuo pipefail
cd /home/scott/ai-lab/ncm/ncm
export NCM_ROOT=/home/scott/ai-lab/ncm/ncm
export NCM_DEVICE=cpu
export TOKENIZERS_PARALLELISM=false
PY=/home/scott/ai-lab/ncm/venv/bin/python
OUT=/home/scott/ai-lab/ncm/results/sudoku_cpu
mkdir -p "$OUT"
N=${1:-200}
BS=${2:-64}
echo "sudoku-hard cpu n=$N bs=$BS start=$(date -u +%FT%TZ)" | tee "$OUT/run.meta"
{ time $PY scripts/eval_shard.py --config config/eval/sudoku-hard.yaml \
    --checkpoint-dir /home/scott/ai-lab/ncm/artifacts/sudoku/hard \
    --data-dir /home/scott/ai-lab/ncm/data/sudoku/hard \
    --n "$N" --bs "$BS" --shard 0 --nshards 1 --out "$OUT/shard_0.pkl" ; } 2> "$OUT/time.txt"
cat "$OUT/time.txt" | tee -a "$OUT/run.meta"
$PY scripts/eval_merge.py --config config/eval/sudoku-hard.yaml --glob "$OUT/shard_*.pkl" | tee "$OUT/score.txt"
cp "$OUT/shard_0.pkl" "$OUT/shard_0.pkl.keep" 2>/dev/null || true
echo DONE > /home/scott/ai-lab/ncm/logs/sudoku.status
