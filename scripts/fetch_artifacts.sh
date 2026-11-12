#!/usr/bin/env bash
# Fetch NCM checkpoints + eval data (explicit filenames — hf CLI v2 ignores multi --include globs).
# Pin every HF cache var IN-SCRIPT: machine-default HF_HOME may sit on a read-only mount,
# and HF_HOME alone does not override huggingface_hub v2 cache resolution.
set -Eeuo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
NCM_HOME=${NCM_HOME:-$HERE}
export HF_HOME="$NCM_HOME/hf-home" HF_HUB_CACHE="$NCM_HOME/hf-home/hub" \
       TRANSFORMERS_CACHE="$NCM_HOME/hf-home/hub" HF_HUB_DISABLE_XET=1
mkdir -p "$NCM_HOME/artifacts" "$NCM_HOME/data"
hf download hazyresearch/ncm-checkpoints \
  xsum/config.yaml xsum/generator.pt xsum/tokenizer.pt \
  sudoku/hard/config.yaml sudoku/hard/generator.pt sudoku/hard/tokenizer.pt \
  --local-dir "$NCM_HOME/artifacts"
hf download hazyresearch/ncm-datasets \
  xsum/passages.pickle xsum/reference_summaries.pickle \
  sudoku/hard/puzzles.pickle sudoku/hard/solutions.pickle \
  --repo-type dataset --local-dir "$NCM_HOME/data"
( cd "$NCM_HOME" && sha256sum -c "$NCM_HOME/SHA256SUMS" 2>/dev/null \
  || sha256sum -c "$NCM_HOME/../SHA256SUMS" ) \
  || echo "WARN: SHA256SUMS check did not run — verify manually: sha256sum -c SHA256SUMS"
