# Handoff notes (for agents continuing this work)

- Everything measured is in `results/`; raw logs included. Re-run recipes:
  `bash run_all.sh` (CPU), `python scripts/benchmark_mbr.py` (MBR benchmark), or `bash scripts/as-run/epyc_runbook.sh` (V100).
- Delivered Milestones (2026-11-12):
  1. Full dedicated V100 reproduction (`results/v100-sxm2/results.json`, zero co-tenant warnings):
     - Sudoku Hard full 2,000-board test: **86.30%** solved (1,726/2,000), cell 98.35%, 26.0 s wall.
     - XSum full **11,334-doc census**: **34.92 / 12.71 / 27.93** (ROUGE-1/2/L) in 1,241.0 s (**9.13 docs/s**, bs=32), within **0.07–0.09 pp** of the paper's 35.0 / 12.8 / 28.0.
     - Clean V100 latency: **0.1073 s/doc** marginal (`bs=1`), **0.0974 s/doc** marginal (`bs=32`, **9.47 docs/s**).
  2. MBR decoding empirical benchmark: verified +1.82 R-L gain at N=4 on CPU (`results/hx370-cpu/mbr/results.json`).
  3. INT8 dynamic quantization: 1.32x speedup on CPU (2.89 s -> 2.19 s, 27.4 docs/min).
  4. Progressive Hierarchical Concept Decoder: Level 0 -> Level 8 coarse-to-fine inspection (`ncm_bench/progressive.py` + `artifacts/visualizer/index.html`), hardened with XSS escaping + CSP.
  5. Hardened Service & Unified Engine: `ncm_bench/inference.py` + `service/summarize_server.py` (pre-load SHA-256 verification, Pydantic v2 bounds, Bearer auth via `NCM_API_KEY`, bounded queue 503 backpressure, `/ready`, Prometheus `/metrics`, `/batch_summarize`), resilient `NCMClient` with retry/backoff, `scripts/batch_triage.py` (`--resume`, `--errors-out`, signal handling), and `ncm-bench` CLI (`sudoku`, `benchmark`, CUDA auto-detection fix).
  6. Test & Security Verification: `pytest tests/` (33/33 passing against real `create_app()`), `ruff check`, `bandit`, non-root read-only Docker Compose, and `.github/workflows/ci.yml`.
- Open items: (1) publish blog draft (EPYC `~/ai-lab/reports/content/drafts/2026-11-12-ncm-*`) + Show HN (copy in `docs/show-hn.md`).
- Do not commit: `*.pkl`, `artifacts/`, `data/`, `venv*/`, `.venv*/`, `ncm/`, `hf-home/` (gitignored on purpose).
- Remote commits can arrive mid-task (desktop agents push to main) — fetch/rebase before push.
