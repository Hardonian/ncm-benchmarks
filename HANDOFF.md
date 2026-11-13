# Handoff notes (for agents continuing this work)

- Everything measured is in `results/`; raw logs included. Re-run recipes:
  `bash run_all.sh` (CPU), `python scripts/benchmark_mbr.py` (MBR benchmark), or `bash scripts/as-run/epyc_runbook.sh` (V100).
- Delivered Milestones (2026-11-12):
  1. Full dedicated V100 reproduction (`results/v100-sxm2/results.json`, zero co-tenant warnings):
     - Sudoku Hard full 2,000-board test: **86.30%** solved (1,726/2,000), cell 98.35%, 26.0 s wall.
     - XSum full **11,334-doc census**: **34.92 / 12.71 / 27.93** (ROUGE-1/2/L) in 1,241.0 s (**9.13 docs/s**, bs=32), within **0.07–0.09 pp** of the paper's 35.0 / 12.8 / 28.0.
     - Clean V100 latency: **0.1073 s/doc** marginal (`bs=1`), **0.0974 s/doc** marginal (`bs=32`, **9.47 docs/s**).
  2. MBR decoding empirical benchmark: verified +1.82 R-L gain at N=4 on CPU (`results/hx370-cpu/mbr/results.json`).
  3. Zero-pickle Safetensors + INT8 parity + Sudoku Best-of-K ablations (`results/hx370-cpu/reality_checks.json`):
     - `xsum` and `sudoku/hard` convert cleanly to `.safetensors` + SHA-256 `inference_meta.json` and run end-to-end with `strict_checksums=True`.
     - Paired XSum FP32 vs INT8 (`n=32`, CPU `bs=8`): ROUGE-L **27.89 -> 27.65** (`-0.24 pp`) at **1.68x** faster wall time (`55.93s -> 33.26s`).
     - Sudoku-Hard (`n=100`, CPU): Greedy `84.0%` -> Clue-Clamp+Naked/Backtrack Repair `86.0%` -> **Best-of-4 + Bounded Repair `91.0%`** (cell `98.47%`, Wilson 95% CI `83.77-95.19%`).
  4. Clean-room Stage-1 VQ-VAE + Stage-2 Next-Concept Transformer trainers (`ncm_train/`), verified with 40-step `ncm-bench train-smoke` (`stage1_loss 194.08 -> 3.79`, `stage2_loss 3.56 -> 1.29`).
  5. Enterprise CLI & Hardened Async Service (`2.0.0`): `ncm_bench/longdoc.py` (offline PDF/text, PII redaction/restore, chunked map-reduce, SHA-256 audit receipts), async `service/summarize_server.py` (413/429/histogram/`/longdoc_summarize`/`/sudoku`/multi-model `/v1/chat/completions`/`/progressive_stream`/`/batch_mbr_summarize`/`--uds`), SDK `service/client.py`, `scripts/batch_triage.py --mode longdoc`, `scripts/fetch_artifacts.py --tasks`, systemd unit, Helm chart (`deploy/helm/ncm-summarizer/`), and pre-commit / CI coverage gate (`--cov-fail-under=70`).
  6. Test & Security Verification: `pytest --cov` (48/48 passing, 70.29% coverage), `ruff check` (0 issues), `bandit` (0 issues across 3,121 LOC), live E2E `.safetensors` INT8 service verification (`POST /summarize` `1,036 ms`, `POST /longdoc_summarize` 4 chunks `4,504 ms`).
- External / Hardware Boundaries still requiring operator action:
  1. External NVIDIA V100 / multi-GPU runs on EPYC (full 11,334-doc XSum MBR, WMT14/LM1B evaluation, large-scale `ncm_train` checkpoints).
  2. External publication & distribution (PyPI release, GHCR container push, publishing blog / Show HN post).
- Do not commit: `*.pkl`, `*.pt`, `*.safetensors`, `artifacts/`, `data/`, `venv*/`, `.venv*/`, `ncm/`, `hf-home/` (gitignored on purpose).
- Remote commits can arrive mid-task (desktop agents push to main) — fetch/rebase before push.
