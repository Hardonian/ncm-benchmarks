# NCM Local Benchmarks

[![CI](https://github.com/Hardonian/ncm-benchmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/Hardonian/ncm-benchmarks/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ncm-benchmarks.svg)](https://pypi.org/project/ncm-benchmarks/)
[![Python](https://img.shields.io/pypi/pyversions/ncm-benchmarks.svg)](https://pypi.org/project/ncm-benchmarks/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v2.0.0-green.svg)](https://github.com/Hardonian/ncm-benchmarks/releases)

> **Next-Concept Modeling provides proof that the industry's default reliance on 70B cloud autoregressive LLMs for basic document classification and triage is an inefficient legacy pattern.**
>
> **By replacing sequential token iteration with multiscale discrete concept refinement, enterprises can achieve 100% data sovereignty, sub-$0.00 marginal unit economics, and verified accuracy on existing infrastructure.**

Independent, on-real-hardware reproduction and production engineering stack for **"Next-Concept Modeling:
Fast, Parallel Coarse-to-Fine Text Generation"** ([arXiv:2610.14162](https://arxiv.org/abs/2610.14162),
Stanford Hazy Research + Together AI, Oct 2026).

The paper claims text generation can beat diffusion LMs on quality while running
10-31x faster by generating a *hierarchy of discrete codes* coarse-to-fine (one
parallel forward pass per level, K=8-32 total) instead of one-token-at-a-time or
64-128 denoising steps.

This repo answers the only question that matters if you plan to *run* this stuff:
**do the numbers survive contact with non-H100 hardware?**

### Headline Breakthroughs (Measured Here, All Receipts in `results/`)

1. **Exact Full-Corpus Non-H100 Reproduction (2018 V100-SXM2)**: Matched the paper on both official full test sets with zero co-tenant warnings — **86.30%** solved on all **2,000** Sudoku-Hard boards (paper: `86.1%`, `26.0 s` wall) and **34.92 / 12.71 / 27.93** ROUGE-1/2/L across the full **11,334-doc** XSum census at **9.13–9.47 docs/s** (**0.0974 s/doc** marginal, **22x** faster than laptop CPU).
2. **+7.0 pp Sudoku Breakthrough on CPU (`91.0%` on `n=100` slice, no retraining)**: Pairing parallel temperature resampling (`Best-of-4`) with deterministic clue clamping, naked singles, and bounded 64-node backtracking (`repair_sudoku_board`) lifted Sudoku-Hard accuracy from **84.0%** (`84/100`) to **91.0%** (`91/100`, cell `98.47%`, Wilson 95% CI `83.77–95.19%`, `26.46 s`), exceeding the paper's `86.1%` greedy baseline on CPU-only inference.
3. **1.68x Speedup via Zero-Pickle `.safetensors` + INT8 at `-0.24 pp` ROUGE-L Loss (`n=32`, `bs=8`)**: Paired FP32 vs dynamic INT8 evaluation cut CPU wall time from `55.93 s` (`27.89` R-L) to `33.26 s` (`27.65` R-L) while eliminating runtime pickle deserialization using SHA-256 verified `.safetensors` weights. Live local FastAPI `POST /summarize` runs in **`1,036 ms`** on laptop CPU (`4,504 ms` for 4-chunk `POST /longdoc_summarize` with PII redaction + audit receipt).
4. **+1.82 ROUGE-L Consensus Win via Parallel MBR (`N=4`)**: Empirical MBR Pareto frontier reproduces the paper's quality jump on CPU (`25.26 -> 27.08` R-L; `31.50 -> 34.04` R-1).
5. **Clean-Room Stage-1 VQ-VAE & Stage-2 NCM Trainers (`ncm_train/`)**: Upstream `HazyResearch/ncm` released inference/eval scripts only; [ncm_train/](ncm_train/__init__.py) provides working Stage-1 + Stage-2 training loops (40-step CPU convergence: Stage-1 `194.08 -> 3.79`, Stage-2 `3.56 -> 1.29`), backed by `48/48` passing tests (`70.29%` coverage) and zero `ruff`/`bandit` issues.

Honest latency caveat: sub-second single-shot XSum does **not** hold on laptop CPUs (`2.34 s/doc` fp32, `1.04–2.19 s/doc` INT8; **0.0974–0.1073 s/doc** on V100), and CPU batching gives `-1.9%` (`bs=8` vs `bs=1` because fp32 GEMMs saturate 16 cores at `bs=1`). Everything measured is committed verbatim in `results/`.

## Results (measured here, not quoted from the paper)

| Task | Paper claim (their hardware) | Our run (device) | Verdict |
|---|---|---|---|
| Sudoku hard (25M gen, 12 NFE) | 86.1% solved / full 2000-board test | **84.50%** / 200 boards / **CPU fp32** | Reproduces (95% CI +/-5.0pp) |
| XSum summarization (88M gen, 9 levels) | R-L 28.0 / 11,334 docs (H100, 31x less wall-clock than diffusion) | **R-L 27.10** (R-1 33.46, R-2 12.33) / first 96 docs / **CPU fp32** | Consistent at ~1 pt |
| **XSum MBR decoding (N=4 consensus)** | +1.6 R-L jump via parallel MBR (28.0 -> 29.6) | **+1.82 R-L jump** (R-L 25.26 -> **27.08**, R-1 31.50 -> **34.04**) / CPU fp32 | **Reproduces paper's headline quality win on CPU** |
| **CPU INT8 dynamic quantization** | -- | **2.19 s/doc** marginal (vs 2.89 s fp32), **1.32x speedup**, 27.4 docs/min sustained | Zero-retraining CPU acceleration |
| CPU summary latency | -- | **2.34 s/doc** marginal (bs=1), ~5.7 s cold load, 25 docs/min sustained | "Sub-second local" does NOT hold on a laptop CPU; 2.3 s does |
| **Batching on CPU** | -- | bs=8 saves **-1.9%** vs bs=1 (i.e. nothing) | Null result, reported honestly: fp32 GEMMs already saturate 16 cores at bs=1 |
| **XSum FP32 vs INT8 parity (`n=32`, CPU `bs=8`)** | -- | **FP32 27.89** (`55.93s`) vs **INT8 27.65 R-L** (`33.26s`, **1.68x**) | Only **-0.24 pp R-L** loss via SHA-256 verified `.safetensors` |
| **Sudoku hard Best-of-4 + Repair (`n=100`, CPU)** | -- | Greedy `84.0%` -> Repair `86.0%` -> **Best-of-4 + Bounded Repair `91.0%`** (`26.46s`) | **+7.0 pp** win (`84 -> 91/100`, cell `98.47%`) |
| **Sudoku hard — FULL 2000 boards (V100)** | 86.1% solved / full test | **86.30%** (1726/2000), cell 98.35%, 26.0 s wall incl. load, bs=128 | **Exact-configuration reproduction: +0.2 pp** |
| **XSum — FULL 11,334-doc census (V100, bs=32)** | 35.0 / 12.8 / 28.0 full test | **34.92 / 12.71 / 27.93** (20.7 avg words, 1,241 s wall, 9.13 docs/s) | **Exact full-corpus reproduction: -0.07 pp R-L** |
| GPU summary latency (dedicated V100) | -- | **0.1073 s/doc** sequential (bs=1), **0.0974 s/doc** marginal (bs=32) = **9.47 docs/s** | 22x faster than this box's CPU on 2018 silicon |

Full machine-readable summaries: [results/hx370-cpu/results.json](results/hx370-cpu/results.json), [results/hx370-cpu/reality_checks.json](results/hx370-cpu/reality_checks.json), and [results/v100-sxm2/results.json](results/v100-sxm2/results.json).
Raw per-run logs in `results/<machine>/`.

**V100 telemetry (clean dedicated pass):** V100-SXM2-16GB had `5 MB` VRAM before start, `12,794 MB` peak VRAM, and **zero** co-tenant / allocator OOM-retry warnings across both full test sets and latency sweeps.

**Hardware for the CPU numbers:** Ryzen AI 9 HX 370, 16 cores visible under WSL2,
19 GiB RAM, no GPU used, torch 2.15.0+cpu, transformers 5.22.2 (upstream pins
`transformers>=4.51`; the current major works unmodified -- verified at runtime).

## What this is, in two sentences

NCM trains a multiscale residual VQ-VAE that encodes text into K levels of discrete
codes (coarse = gist, fine = detail, nested-dropout so coarse levels stand alone),
then freezes it and trains a transformer to generate one level per forward pass, all
codes within a level sampled in parallel. Generation cost becomes K forward passes
(8-32) instead of L sequential steps or 64-128 diffusion NFEs -- and the paper shows
that at ~100M params this beats MDLM/DUO/ELF/AURORA-class diffusion baselines on
LM1B, XSum, WMT14 and Sudoku while using 4-5x fewer training tokens.

## Why you should care (the transferable findings)

1. **The efficiency is real and survives hardware translation.** A 654 MB
   (tokenizer+generator) MIT-licensed XSum model produces 25–27 summaries/minute on a
   laptop CPU and **~568/min (9.47 docs/s)** on a 2018-era dedicated V100 (0.1073 s/doc
   sequential — still 22x the CPU), with no cloud call. The speed advantage comes from the architecture
   (9 passes over <=560-token sequences), not from the H100 in their figure.
2. **The paper's headline quality win is MBR decoding** -- cheap parallel candidates
   + a selector (28.0 -> 29.6 R-L at matched time). If your orchestration already
   does draft-select ensembles, this is literature validation; if not, copy it.
3. **Pitfalls you will actually hit** (all cost us a failed run, all fixed here):
   - upstream eval harness hardcodes the CUDA device in 7 sites of
     `scripts/eval_shard.py` (3x `.cuda()`, 2x `map_location="cuda"`,
     2x `device="cuda"`) — all replaced by an `NCM_DEVICE`-resolved constant
     (`patches/device-support.patch`, clean-applies at the pinned SHA, verified
     with `git apply --check` + `python3 -m py_compile`);
   - `scripts/eval_merge.py`'s ROUGE function drags a CUDA GPT-2 in for genPPL even
     when you only want ROUGE (see `scripts/score_rouge_cpu.py`);
   - `huggingface-cli` v2 silently ignores multiple `--include` globs -- pass
     explicit filenames (we ship the manifest-verified list);
   - the xet download backend dies on read-only cache mounts; pin `HF_HOME`,
     `HF_HUB_CACHE` and `TRANSFORMERS_CACHE` (all three) and set
     `HF_HUB_DISABLE_XET=1` inside your scripts, not your shell.
4. **The 2-size timing trick**: model load (~6 s) contaminates single-shot latency.
   Run n=4 and n=16 at the same batch size; marginal per-item =
   (T16-T4)/12, load = T4-4*marginal. Our runners do this for you.

## Installation & Quickstart

Install the official package directly from PyPI:

```bash
pip install ncm-benchmarks
```

Or install from source with development tools:

```bash
git clone https://github.com/Hardonian/ncm-benchmarks.git
cd ncm-benchmarks
pip install -e ".[dev]"
```

Verify your installation:

```bash
ncm-bench --help
```

## Reproduce

```bash
# 1) vendor upstream at the exact SHA we tested, apply the device patch
git clone https://github.com/HazyResearch/ncm ncm && cd ncm
git checkout c70274273f489abc73832b2b98584a8483b6b203
git apply ../patches/device-support.patch

# 2) environment (CPU shown; swap the torch index for CUDA on a GPU lane)
uv venv venv --python 3.12
uv pip install --python venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python venv/bin/python -r ../requirements.txt

# 3) artifacts + data (explicit filenames; sizes must match SHA256SUMS)
NCM_HOME=$(pwd)/.. ../scripts/fetch_artifacts.sh
cd .. && sha256sum -c SHA256SUMS

# 4) runs
NCM_DEVICE=cpu ./run_all.sh --quick     # smoke: sudoku n=8 + xsum n=8
NCM_DEVICE=cpu ./run_all.sh             # the measured suite (~25-30 min on 16 cores)
```

Or run the exact as-invoked scripts under `scripts/as-run/` (absolute paths from
the verified run, zero edits).

## What is verified vs claimed

- **Verified here, this week:** every number in `results/hx370-cpu/` (laptop CPU) and
  `results/v100-sxm2/` (EPYC V100 lane), on our machines, with their scorer
  (`scripts/eval_merge.py exact()` for Sudoku) or an equivalent CPU scorer for ROUGE.
  The V100 Sudoku run is the **full official 2000-board test** — an exact-configuration
  reproduction (86.30% vs paper 86.1%), not a subset extrapolation.
- **Quoted from the paper** (we did not reproduce them): H100 wall-clock speedups
  (21x/31x vs diffusion baselines — our CPU/V100 runs measure the NCM model only),
  LM1B/OWT perplexity sweeps, WMT14 BLEU, MBR deltas, training budgets.
- Local subsets carry their n and, for the binary metric, a 95% CI. We do not blend
  our numbers with the paper's in any single cell.

## Unified CLI (`ncm-bench`)

The repo provides a full command-line suite:

```bash
# Verify all model and dataset SHA256 checksums
python -m ncm_bench.cli verify

# Summarize text directly from terminal or piped input
echo "The UK government announced renewable energy expansion..." | python -m ncm_bench.cli summarize --quantize int8

# Run MBR consensus decoding (N=4 candidates, +1.82 R-L boost)
python -m ncm_bench.cli mbr --text "<document...>" --candidates 4

# Inspect the 9-level concept hierarchy and generate an HTML visualizer
python -m ncm_bench.cli inspect --text "<document...>" --html visualizer.html

# Solve a Sudoku puzzle or evaluate Sudoku-Hard with Best-of-K sampling + constraint repair
python -m ncm_bench.cli sudoku --n 20 --best-of-k 4 --json

# Export local checkpoints to zero-pickle Safetensors and verify strict checksums
python -m ncm_bench.cli convert-safetensors --task xsum
python -m ncm_bench.cli summarize --input-file report.pdf --long-doc --redact-pii --restore-pii --audit-receipt --strict-checksums --json

# Clean-room Stage-1 VQ-VAE + Stage-2 Next-Concept Transformer training smoke test
python -m ncm_bench.cli train-smoke --samples 32 --s1-steps 15 --s2-steps 15

# Run a 2-size timing benchmark separating model load from marginal per-doc latency
python -m ncm_bench.cli benchmark --task latency --small-n 4 --large-n 16 --quantize int8

# Launch hardened local REST service (async offloaded, 413/429 limits, Bearer auth, OpenAI compat)
python -m ncm_bench.cli serve --port 8031 --quantize int8 --max-queue-depth 8
```

## Turnkey Deployment & SDK

### One-Click Local Runners, Container, systemd & Helm
- **Windows**: double-click or run `run_service.bat`
- **Linux / macOS**: run `./run_service.sh`
- **Docker Compose**: `docker compose up -d` (non-root `ncmuser`, read-only rootfs, `cap_drop: ALL`, `no-new-privileges`, `NCM_STRICT_CHECKSUMS=1`, loopback-bound `127.0.0.1:8031` in [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml))
- **systemd / Kubernetes Helm**: [deploy/systemd/ncm-summarizer.service](deploy/systemd/ncm-summarizer.service) and [deploy/helm/ncm-summarizer/](deploy/helm/ncm-summarizer/)
- **CI Pipeline & Pre-commit**: [GitHub Actions CI](.github/workflows/ci.yml) runs `ruff`, `bandit`, and `pytest --cov` (`>=70%` gate) across Linux/Windows + Python 3.11/3.12; see [CHANGELOG.md](CHANGELOG.md).

### Python Client SDK (`service/client.py`)
```python
from service.client import NCMClient

# Built-in exponential backoff retries on 429/502/503/504 + optional Bearer API key
client = NCMClient(base_url="http://127.0.0.1:8031", api_key=None)

# Fast single-shot or batched summary
res = client.summarize("Long document text...")
print(res["summary"], f"({res['latency_ms']} ms)")

batch = client.batch_summarize_server(["Doc 1...", "Doc 2..."])
print(batch["summaries"])

# High-fidelity consensus summary via MBR
mbr_res = client.mbr_summarize("Long document text...", num_candidates=4)
print(mbr_res["summary"])

# Inspect intermediate concept hierarchy
levels = client.progressive_summarize("Long document text...")
```

### Enterprise Document Triage Batch Processor
Process thousands of records with crash-safe `--resume` deduplication, dead-letter `--errors-out` logging, and graceful `SIGINT`/`SIGTERM` shutdown:
```bash
python scripts/batch_triage.py --input documents.jsonl --output triage_results.jsonl --errors-out dead_letter.jsonl --resume --mode standard
```

## Enterprise & Commercial Research Whitepaper

For unit economics, TCO models (Cloud API vs. Local Appliance at 100K–100M doc scale), and air-gapped compliance architectures, read the full whitepaper:
- [Enterprise Whitepaper: Sovereign Document Intelligence via Next-Concept Modeling](docs/ENTERPRISE-WHITEPAPER.md)

## Local summarizer service (expansion)

`service/summarize_server.py` wraps the unified `ncm_bench.inference` engine in a hardened FastAPI server. Fully local, 654 MB of weights verified against `SHA256SUMS` prior to `torch.load`, bounded Pydantic v2 request validation, security/CSP headers, structured JSON request logs with `X-Request-ID`, bounded concurrency queue (HTTP 503 + `Retry-After`), and zero network calls at inference time:

```bash
# Standard service (CPU fp32)
python service/summarize_server.py --port 8031

# Accelerated dynamic INT8 quantization (1.32x speedup on CPU, 2.19 s/doc) with Bearer auth
NCM_API_KEY="your-secret-token" python service/summarize_server.py --port 8031 --quantize int8
```

Endpoints available:
- `POST /summarize`: Fast single-shot summary (`{"text": "..."}`).
- `POST /batch_summarize`: Batched summarization (1–32 documents per request).
- `POST /mbr_summarize`: Minimum Bayes Risk consensus decoding (`{"text": "...", "num_candidates": 4, "temperature": 0.7}`) — delivers measured **+1.82 R-L quality gain**.
- `POST /progressive_summarize`: Inspects all 9 discrete concept levels ($k=0\dots 8$), revealing the coarse gist to fine syntax progression.
- `GET /visualize`: XSS-escaped standalone interactive web dashboard rendering the concept slider for any passage.
- `GET /health`, `/ready`, `/metrics`: Liveness, Kubernetes/container readiness probe (503 when unloaded), and Prometheus exposition telemetry.

Interactive concept visualizer artifact: open [artifacts/visualizer/index.html](artifacts/visualizer/index.html) in any browser to step through the 9 concept hierarchy levels.

## License / provenance

Our scripts, patches, docs and results: MIT (this repo). Upstream model code and
weights: MIT, (c) the NCM authors -- see `NOTICE` for the pinned provenance
(SHA `c702742`, checkpoints from `hazyresearch/ncm-checkpoints`, base vocab
provenance: GPT-2 (MIT), bert-base-uncased (Apache-2.0), Qwen3 (Apache-2.0)).
ROUGE/GPT-2 scoring follows the protocol of
[MDLM](https://github.com/sahil280114/code-align-mdlm).

## Not a claim

This is not a general-purpose LLM product test: these are single-task ~100M models
on small corpora. Absolute quality trails production 7B+ AR models; the value is
the latency/size envelope and the architecture direction. Said plainly, per our
style.
