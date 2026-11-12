#!/usr/bin/env bash
# XSum CPU spike: measure per-doc latency (bs=1) and throughput (bs=8), then ROUGE via CPU scorer.
# Timing trick: two sizes per batch config so model-load time cancels: (T_96 - T_32)/64 = marginal doc cost.
set -Eeuo pipefail
cd /home/scott/ai-lab/ncm/ncm
export HF_HOME=/home/scott/ai-lab/ncm/hf-home HF_HUB_CACHE=/home/scott/ai-lab/ncm/hf-home/hub TRANSFORMERS_CACHE=/home/scott/ai-lab/ncm/hf-home/hub HF_HUB_DISABLE_XET=1
export NCM_ROOT=/home/scott/ai-lab/ncm/ncm
export NCM_DEVICE=cpu
export TOKENIZERS_PARALLELISM=false
PY=/home/scott/ai-lab/ncm/venv/bin/python
OUT=/home/scott/ai-lab/ncm/results/xsum_cpu
mkdir -p "$OUT"
CK=/home/scott/ai-lab/ncm/artifacts/xsum
DD=/home/scott/ai-lab/ncm/data/xsum
echo "xsum cpu start=$(date -u +%FT%TZ)" | tee "$OUT/run.meta"

# marginal single-doc latency (bs=1): n=4 vs n=16
for N in 4 16; do
  { time $PY scripts/eval_shard.py --config config/eval/xsum.yaml \
      --checkpoint-dir "$CK" --data-dir "$DD" \
      --n "$N" --bs 1 --shard 0 --nshards 1 --out "$OUT/bs1_n$N.pkl" ; } 2> "$OUT/time_bs1_n$N.txt"
  echo "bs1 n=$N $(grep real "$OUT/time_bs1_n$N.txt")" | tee -a "$OUT/run.meta"
done

# throughput + ROUGE set (bs=8): n=32 vs n=96
for N in 32 96; do
  { time $PY scripts/eval_shard.py --config config/eval/xsum.yaml \
      --checkpoint-dir "$CK" --data-dir "$DD" \
      --n "$N" --bs 8 --shard 0 --nshards 1 --out "$OUT/bs8_n$N.pkl" ; } 2> "$OUT/time_bs8_n$N.txt"
  echo "bs8 n=$N $(grep real "$OUT/time_bs8_n$N.txt")" | tee -a "$OUT/run.meta"
done

$PY /home/scott/ai-lab/ncm/score_rouge_cpu.py "$OUT/bs8_n96.pkl" | tee "$OUT/rouge.txt"
echo DONE > /home/scott/ai-lab/ncm/logs/xsum.status
