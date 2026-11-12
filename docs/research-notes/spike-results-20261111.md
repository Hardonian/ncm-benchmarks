# NCM on home hardware — measured results (2026-11-11/12)

Paper: Next-Concept Modeling, arXiv 2610.14162, Stanford Hazy Research. MIT, weights on HF.
All numbers below are OUR runs, not the paper's. HX370 = Ryzen AI 9 HX 370, 16-core WSL2,
CPU-only, fp32, torch 2.15.0+cpu / transformers 5.22.2, vendored eval harness
(device-patched eval_shard.py, HF cache pinned to ~/ai-lab/ncm/hf-home).

## Sudoku hard (25M generator, 12 NFE) — CPU
- n=200 of their 2000-board hard test: **84.50% solved, cell 98.35%, valid 84.50%**
- Paper full test: 86.1%. Subset ±2.6pp sampling error → reproduces.
- Wall clock 200 boards incl. model load: 45s. Throughput ~5 boards/s on CPU.

## XSum summarization (88M generator, K=9 levels, bs=1 & 8) — CPU
- ROUGE (first 96 test docs): R-1 33.46 / R-2 12.33 / R-L 27.10, 20.9 words avg
  (paper, 11,334 docs: 35.0 / 12.8 / 28.0 — subset effect, consistent)
- Marginal latency 2.34 s/doc at bs=1; 2.38 s/doc at bs=8 → batching buys ~0% on CPU
  (compute-saturated at bs=1; consistent with ~900 GFLOPs/doc vs ~400 GFLOPS achieved)
- Model load ~5.7s. Sustained ~25 summaries/min on the laptop CPU.
- VERDICT: "sub-second on CPU" claim does NOT hold here. Honest claim: ~2.3 s local
  single-summary latency, 25/min sustained, zero GPU, 15W laptop, no cloud.

## EPYC V100 lane — PENDING (Tailscale auth blocked at run time)
- Planned: sudoku hard full 2000 + XSum n=1000 + bs=1 latency. Runner staged:
  ~/ai-lab/ncm/epyc_runbook.sh (+ patched repo ready to rsync).

## Repro artifacts (all committed-ready)
- results/sudoku_cpu/{score.txt,run.meta,time.txt,shard_0.pkl}
- results/xsum_cpu/{rouge.txt,run.meta,time_*.txt,bs*_n*.pkl}
- scripts: fetch_artifacts.sh, run_sudoku_cpu.sh, run_xsum_cpu.sh, score_rouge_cpu.py, epyc_runbook.sh
- vendored repo: ~/ai-lab/ncm/ncm @ c702742 + 7-line device patch (NCM_DEVICE/NCM_ROOT env)

## Pitfalls proven this run (worth a follow-a-paragraph)
1. Their harness hardcodes .cuda() — CPU needs a 7-point patch (or just don't claim CPU).
2. hf CLI v2 mishandles multiple --include globs; explicit filenames only.
3. xet backend + read-only G: drive HF cache = silent permission death; pin HF_HOME,
   HF_HUB_CACHE, TRANSFORMERS_CACHE all three.
4. Their own eval_merge rouge path pulls GPT-2 for genPPL on CUDA; subset runs need a
   CPU ROUGE scorer (8 lines of rouge_score).

## Post draft angle (fill V100 row, then publish)
"A Stanford paper said text diffusion is too slow. I ran its replacement on a laptop CPU
and a mining-era GPU." Table: paper claim vs our measured, per task. Include the batching
null-result (nobody reports that). Cite their repo honestly; our patch + runner scripts as
the pack sample.
