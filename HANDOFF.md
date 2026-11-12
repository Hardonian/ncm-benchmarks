# Handoff notes (for agents continuing this work)

- Everything measured is in `results/`; raw logs included. Re-run recipes:
  `bash run_all.sh` (CPU) or `bash scripts/as-run/epyc_runbook.sh` (V100, needs
  `$HOME/ai-lab/ncm` with venv+artifacts on the GPU box — already provisioned).
- Open items: (1) dedicated-GPU re-run of bs=32 to drop the co-tenant caveat;
  (2) publish blog draft (EPYC `~/ai-lab/reports/content/drafts/2026-11-12-ncm-*`) + Show HN
  (copy in `docs/show-hn.md`); (3) optional: XSum full 11,334-doc pass on V100 (~18 min).
- Do not commit: *.pkl, artifacts/, data/, venv/, hf-home/ (gitignored on purpose).
- Remote commits can arrive mid-task (desktop agents push to main) — fetch/rebase before push.
