#!/usr/bin/env bash
# NCM benchmark suite — portable wrapper.
#   ./run_all.sh          full suite (~25-30 min on a 16-core CPU box)
#   ./run_all.sh --quick  smoke (sudoku n=8, xsum n=8)
# Verified as-run counterparts with absolute paths: scripts/as-run/.
set -Eeuo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export NCM_HOME=${NCM_HOME:-$HERE}
export NCM_DEVICE=${NCM_DEVICE:-cpu}
export HF_HOME="$NCM_HOME/hf-home" HF_HUB_CACHE="$NCM_HOME/hf-home/hub" \
       TRANSFORMERS_CACHE="$NCM_HOME/hf-home/hub" HF_HUB_DISABLE_XET=1 \
       TOKENIZERS_PARALLELISM=false NCM_ROOT="$NCM_HOME/ncm"
PY=${PY:-$NCM_HOME/venv/bin/python}
[ -x "$PY" ] || { echo "no venv at $PY — see README reproduce steps"; exit 1; }
CK="$NCM_HOME/artifacts"; DD="$NCM_HOME/data"
OUT="$NCM_HOME/results/${NCM_DEVICE/./_}"
mkdir -p "$OUT"
cd "$NCM_HOME/ncm"
QUICK=${1:-}
shard() { $PY scripts/eval_shard.py --config "config/eval/$1" \
  --checkpoint-dir "$CK/$2" --data-dir "$DD/$3" \
  --n "$4" --bs "$5" --shard 0 --nshards 1 --out "$OUT/$6"; }

if [ "$QUICK" = "--quick" ]; then
  echo "== quick smoke =="
  N=8; BS=8
else
  N=200; BS=64
fi
{ time shard sudoku-hard.yaml sudoku/hard sudoku/hard "$N" "$BS" sudoku_shard.pkl ; } 2> "$OUT/sudoku_time.txt"
$PY scripts/eval_merge.py --config config/eval/sudoku-hard.yaml --glob "$OUT/sudoku_shard.pkl" | tee "$OUT/sudoku_score.txt"

if [ "$QUICK" = "--quick" ]; then
  { time shard xsum.yaml xsum xsum 8 1 xsum_q_shard.pkl ; } 2> "$OUT/xsum_quick_time.txt"
  $PY "$HERE/scripts/score_rouge_cpu.py" "$OUT/xsum_q_shard.pkl" | tee "$OUT/xsum_quick_rouge.txt"
  echo "quick done"; exit 0
fi

# XSum two-size timing (load cancels): bs=1 {4,16}, bs=8 {32,96}
for spec in "1 4" "1 16" "8 32" "8 96"; do
  set -- $spec; bs=$1; n=$2
  { time shard xsum.yaml xsum xsum "$n" "$bs" "xsum_bs${bs}_n${n}.pkl" ; } 2> "$OUT/xsum_time_bs${bs}_n${n}.txt"
  echo "bs=$bs n=$n $(grep real "$OUT/xsum_time_bs${bs}_n${n}.txt")" | tee -a "$OUT/xsum_run.meta"
done
$PY "$HERE/scripts/score_rouge_cpu.py" "$OUT/xsum_bs8_n96.pkl" | tee "$OUT/xsum_rouge_n96.txt"
echo "DONE $(date -u +%FT%TZ)" | tee "$OUT/suite.status"
