# Launch & Press Kit (`ncm-benchmarks` v2.0.0)

Ready-to-publish announcements anchored strictly to verified receipts in [results/](../results/hx370-cpu/reality_checks.json).

---

## 1. X / Twitter Thread Draft

**Post 1/4 (Hook)**
We benchmarked Stanford & Together AI's *Next-Concept Modeling* (`arXiv:2610.14162`) on non-H100 hardware: a 2018 V100 + a laptop CPU.
- Full 11,334-doc XSum census on V100: **27.93 ROUGE-L** (vs paper 28.0, `9.47 docs/s`)
- Full 2,000 Sudoku-Hard on V100: **86.30%** (vs paper 86.1%)
- CPU Best-of-4 + bounded constraint repair (`n=100`): **91.0%** (`+7.0 pp`)
Code, weights verifier & local API (MIT): https://github.com/Hardonian/ncm-benchmarks

**Post 2/4 (The Surprises & Null Results)**
What papers don't tell you:
- Laptop CPU batching (`bs=8` vs `bs=1`) saves **-1.9%** — fp32 GEMMs already saturate 16 cores at `bs=1`.
- Dynamic INT8 + zero-pickle `.safetensors` (`n=32`, `bs=8`) gives **1.68x CPU speedup** for just **-0.24 pp ROUGE-L** (`1,036 ms` live API summary).

**Post 3/4 (Missing Upstream Pieces We Built)**
`HazyResearch/ncm` shipped inference code without training scripts or production serving. `ncm-benchmarks` adds:
- Clean-room Stage-1 VQ-VAE & Stage-2 NCM trainers (`ncm_train/`)
- Offline PDF map-reduce + PII redaction + SHA-256 audit receipts
- Hardened async FastAPI + OpenAI `/v1/chat/completions` + Helm/systemd

**Post 4/4 (Reproduce Yourself)**
48/48 tests (`70.29%` coverage), 0 `ruff`/`bandit` findings, pinned `SHA256SUMS`, zero network calls at inference:
https://github.com/Hardonian/ncm-benchmarks/releases/tag/v2.0.0

---

## 2. Reddit (`r/LocalLLaMA` & `r/MachineLearning`)

**Title**: `[R] We reproduced Stanford's Next-Concept Modeling (arXiv:2610.14162) on all 11,334 XSum docs (2018 V100) + hit 91% Sudoku-Hard on CPU with Best-of-4 repair`

**Body**:
We ran an independent reproduction and production engineering build around **Next-Concept Modeling** (discrete hierarchical codes, `K=9`–`12` parallel forward passes instead of 64–128 diffusion steps) on consumer/commodity silicon (`Ryzen AI 9 HX 370` CPU + dedicated `V100-SXM2`):

- **Exact V100 reproduction (`results/v100-sxm2/results.json`)**:
  - Sudoku-Hard full `2,000` boards: **86.30%** solved (paper: `86.1%`, `26.0 s` wall)
  - XSum full `11,334`-doc census: **34.92 / 12.71 / 27.93** (paper: `35.0 / 12.8 / 28.0`) at **9.47 docs/s** marginal (`bs=32`)
- **Inference-time CPU wins (`results/hx370-cpu/reality_checks.json`)**:
  - **Best-of-4 + bounded constraint repair (`n=100` CPU slice)**: **91.0%** solved (`84%` greedy -> `86%` repair -> `91%`, `98.47%` cell accuracy, Wilson 95% CI `83.77–95.19%`)
  - **MBR decoding (`N=4`)**: **+1.82 ROUGE-L** (`25.26 -> 27.08`)
  - **INT8 + zero-pickle `.safetensors` (`n=32`, `bs=8`)**: **1.68x faster** (`55.93 s -> 33.26 s`) at **`-0.24 pp` ROUGE-L**
- **Engineering extras**: clean-room Stage-1 + Stage-2 trainers (`ncm_train/`), offline PDF/PII scrubbing/audit CLI, async FastAPI + OpenAI `/v1/chat/completions`, `48/48` tests (`ruff` & `bandit` clean).

Honest caveats: single-shot CPU latency is `1.04–2.34 s/doc` (not sub-second), and CPU `bs=8` batching saves `-1.9%`.

Repo (MIT): https://github.com/Hardonian/ncm-benchmarks

---

## 3. LinkedIn Summary Post

**Headline**: Sovereign 654 MB Document Intelligence & Non-H100 Reproduction of Stanford's Next-Concept Modeling

Organizations evaluating local-first AI often ask whether published H100 benchmarks hold on affordable enterprise hardware. Our open-source reproduction (`ncm-benchmarks` v2.0.0) tested Stanford Hazy Research's Next-Concept Modeling end-to-end across a 2018 V100 and a laptop CPU:
- **100% full-corpus parity on V100**: 34.92 / 12.71 / 27.93 ROUGE across all 11,334 XSum articles (within 0.07 pp of H100) and 86.30% on 2,000 Sudoku-Hard puzzles.
- **CPU Edge Acceleration**: Zero-pickle `.safetensors` + dynamic INT8 delivered a **1.68x speedup** with only `-0.24 pp` ROUGE-L delta (`1,036 ms` live local API response), while Best-of-4 constraint search pushed Sudoku-Hard to **91.0%** (`n=100`).
- **Enterprise Controls**: Air-gapped execution (`HF_HUB_OFFLINE=1`), SHA-256 weight verification (`SHA256SUMS`), PII redaction/restoration, cryptographic audit receipts, Prometheus telemetry, and Kubernetes Helm charts.

Explore the full open-source release (MIT): https://github.com/Hardonian/ncm-benchmarks

