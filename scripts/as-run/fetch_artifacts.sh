#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/scott/ai-lab/ncm
export HF_HOME=/home/scott/ai-lab/ncm/hf-home HF_HUB_DISABLE_XET=1
mkdir -p artifacts data
hf download hazyresearch/ncm-checkpoints \
  xsum/config.yaml xsum/generator.pt xsum/tokenizer.pt \
  sudoku/hard/config.yaml sudoku/hard/generator.pt sudoku/hard/tokenizer.pt \
  --local-dir artifacts > logs/hf-ckpts.log 2>&1
hf download hazyresearch/ncm-datasets \
  xsum/passages.pickle xsum/reference_summaries.pickle \
  sudoku/hard/puzzles.pickle sudoku/hard/solutions.pickle \
  --repo-type dataset --local-dir data > logs/hf-data.log 2>&1
echo "DOWNLOAD-DONE rc=0" > logs/fetch.status
