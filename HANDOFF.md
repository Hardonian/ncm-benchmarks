# Handoff notes (for agents continuing this work)

- Everything measured is in `results/`; raw logs included. Re-run recipes:
  `bash run_all.sh` (CPU), `python scripts/benchmark_mbr.py` (MBR benchmark), or `bash scripts/as-run/epyc_runbook.sh` (V100).
- Delivered Milestones (2026-11-11):
  1. MBR decoding empirical benchmark: verified +1.82 R-L gain at N=4 on CPU (`results/hx370-cpu/mbr/results.json`).
  2. INT8 dynamic quantization: 1.32x speedup on CPU (2.89s -> 2.19s, 27.4 docs/min).
  3. Progressive Hierarchical Concept Decoder: Level 0 -> Level 8 coarse-to-fine inspection (`ncm_bench/progressive.py` + `artifacts/visualizer/index.html`).
  4. Expanded local service: `service/summarize_server.py` with `/summarize`, `/mbr_summarize`, `/progressive_summarize`, `/visualize`, and `--quantize int8`.
  5. Test coverage: 13/13 unit and integration tests passing (`pytest tests/`).
- Open items: (1) dedicated-GPU re-run of bs=32 to drop the co-tenant caveat;
  (2) publish blog draft (EPYC `~/ai-lab/reports/content/drafts/2026-11-12-ncm-*`) + Show HN
  (copy in `docs/show-hn.md`); (3) optional: XSum full 11,334-doc pass on V100 (~18 min).
- Do not commit: *.pkl, artifacts/, data/, venv/, hf-home/ (gitignored on purpose).
- Remote commits can arrive mid-task (desktop agents push to main) — fetch/rebase before push.
