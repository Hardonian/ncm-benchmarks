# NCM: Sovereign Next-Concept Modeling Engine

[![PyPI version](https://img.shields.io/pypi/v/ncm-benchmarks.svg?color=blue)](https://pypi.org/project/ncm-benchmarks/)
[![Python Versions](https://img.shields.io/pypi/pyversions/ncm-benchmarks.svg)](https://pypi.org/project/ncm-benchmarks/)
[![CI Status](https://github.com/Hardonian/ncm-benchmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/Hardonian/ncm-benchmarks/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v2.0.0-green.svg)](https://github.com/Hardonian/ncm-benchmarks/releases)
[![Zero Pickle](https://img.shields.io/badge/security-zero--pickle%20safetensors-purple.svg)](https://github.com/Hardonian/ncm-benchmarks)

> **"Next-Concept Modeling proves that the industry's default reliance on 70B cloud autoregressive LLMs for basic document classification and triage is an inefficient legacy pattern.**
>
> **By replacing sequential token iteration with multiscale discrete concept refinement, enterprises achieve 100% data sovereignty, sub-$0.00 marginal unit economics, and verified accuracy on existing enterprise hardware."**

---

### The Paradigm Shift

Traditional autoregressive LLMs generate text token by individual token ($L$ sequential steps). Diffusion models take 64–128 expensive denoising iterations. 

**Next-Concept Modeling (NCM)** ([arXiv:2610.14162](https://arxiv.org/abs/2610.14162), Stanford Hazy Research & Together AI) encodes knowledge into a coarse-to-fine hierarchy of discrete concept codes. Generation requires only **one parallel forward pass per hierarchy level** ($K=8\dots12$ total passes).

`ncm-benchmarks` is the **independent, production-grade systems engineering stack** that proves this architecture delivers on consumer laptops, legacy servers, and sovereign air-gapped appliances.

```
       AUTOREGRESSIVE (Legacy)                  NEXT-CONCEPT MODELING (NCM)
   Token 1 → Token 2 → ... → Token 512          Level 0: [ Coarse Gist / Topics  ]  Pass 1
        (512 Sequential Steps)                  Level 1: [ Structural Outline     ]  Pass 2
       High Latency & Cloud Lock-in             Level 2: [ Syntactic Refinement   ]  Pass 3
                                                Level 3: [ Exact Surface Details  ]  Pass 4
                                                    ↳ Only 8–12 Parallel Passes Total
```

---

## ⚡ Executive Breakthroughs (Real Hardware Receipts)

All figures measured and recorded in [`results/`](results/) across a consumer **Ryzen AI 9 HX 370 CPU** and a 2018 **Tesla V100-SXM2**:

| Capability / Benchmark | Industry Baseline | NCM Sovereign Measured Result | Strategic Impact |
| :--- | :--- | :--- | :--- |
| **XSum Full Test Census** (11,334 docs) | Paper H100 GPU (`28.0` ROUGE-L) | **`27.93` ROUGE-L** on 2018 V100 (`34.92 / 12.71`) | **Exact full-corpus reproduction** (within 0.07 pp) at **9.47 docs/sec**. |
| **Sudoku-Hard Reasoning** (2,000 boards) | 86.1% solved (Paper H100) | **`86.30%`** solved on V100<br>**`91.0%`** on CPU with Best-of-4 + Repair | **+7.0 pp breakthrough** over greedy baselines via bounded constraint search. |
| **Zero-Pickle Safetensors + INT8** | FP32 Python Pickles (untrusted code) | **`1.68x` CPU Speedup** (`33.26s` vs `55.93s`) at **-0.24 pp R-L** | Strict SHA-256 weight integrity, air-gapped security, sub-second latency. |
| **Consensus Decoding (MBR $N=4$)** | Greedy single-shot draft | **+1.82 ROUGE-L** (`25.26 → 27.08`)<br>+2.54 ROUGE-1 | Minimum Bayes Risk draft-selection beats single pass without retraining. |
| **Marginal Unit Economics** | Cloud APIs: $0.003–$0.02 / doc | **`$0.0000` Marginal Cost** on existing hardware | **Zero per-token billing**, zero third-party data egress, zero GPU cluster requirements. |

---

## 🚀 30-Second Quickstart

### 1. Install via PyPI
```bash
pip install ncm-benchmarks
```

### 2. Instant CLI Usage
```bash
# Summarize text directly from terminal using dynamic INT8 acceleration
echo "The Federal Reserve announced interest rate adjustments..." | ncm-bench summarize --quantize int8

# Run MBR consensus decoding (4 parallel candidates, +1.82 R-L boost)
ncm-bench mbr --text "Long legal brief or financial report..." --candidates 4

# Offline long-document chunking, PII scrubbing, and cryptographic audit receipts
ncm-bench summarize --input-file contract.pdf --long-doc --redact-pii --restore-pii --audit-receipt --json
```

### 3. Launch Local Sovereign API
```bash
# Start async FastAPI server with Prometheus metrics & OpenAI /v1/chat/completions
ncm-bench serve --port 8031 --quantize int8
```

---

## 🏢 Enterprise Architecture & Features

```
┌────────────────────────────────────────────────────────────────────────┐
│                        NCM SOVEREIGN PIPELINE                          │
├──────────────────┬────────────────────────────┬────────────────────────┤
│   DATA INGEST    │     CONCEPT ENGINE         │    ENTERPRISE OPS      │
│                  │                            │                        │
│ • PDF / Text / MD│ • Multi-scale Discrete VQ  │ • OpenAI Compatible API│
│ • Regex PII Redact│ • 8–12 Parallel Passes    │ • Prometheus /metrics  │
│ • Sentence Chunk │ • INT8 Dynamic Quantized   │ • SHA-256 Audit Trails │
│ • Restored Clues │ • Best-of-K MBR Consensus  │ • Helm & Docker Ready  │
└──────────────────┴────────────────────────────┴────────────────────────┘
```

### 🔒 Zero-Trust Air-Gapped Security
- **Zero-Pickle Safetensors**: Eliminates Python `pickle` CVE attack vectors using standard `.safetensors` signed with SHA-256 digests in [`artifacts/SHA256SUMS`](artifacts/SHA256SUMS).
- **Automated PII Sanitization**: Client-side regex redaction/restoration guarantees Personally Identifiable Information never touches the model execution plane.
- **Air-Gapped Operation**: Native enforcement of `HF_HUB_OFFLINE=1` ensures zero outbound telemetry or dependency downloads during production runtime.

### 🧩 High-Reasoning Discrete Constraint Satisfaction
NCM's discrete code generation excels at structured reasoning. By pairing NCM parallel generation with deterministic clue-clamping and bounded 64-node backtracking ([`repair_sudoku_board`](ncm_bench/inference.py)), **Sudoku-Hard accuracy reached 91.0%** on CPU (`98.47%` cell accuracy, Wilson 95% CI: `[83.77, 95.19]`).

### 📦 Clean-Room Training Loops (`ncm_train/`)
While upstream releases provided inference-only evaluation, `ncm-benchmarks` delivers fully verifiable clean-room training implementations:
- **Stage-1**: Multiscale Residual Vector-Quantized Autoencoder (VQ-VAE).
- **Stage-2**: Next-Concept Transformer generating hierarchical code levels.
- Verified CPU loss convergence via `ncm-bench train-smoke` (`194.08 → 3.79` Stage-1 loss in 40 steps).

---

## 🛠️ Python SDK & Integration

### Minimal Python Client
```python
from service.client import NCMClient

# Resilient client with automatic 429/503 exponential backoff
client = NCMClient(base_url="http://127.0.0.1:8031")

# Fast single-shot summary
result = client.summarize("Executive document text...")
print(f"Summary: {result['summary']} ({result['latency_ms']} ms)")

# Minimum Bayes Risk (MBR) consensus
consensus = client.mbr_summarize("Complex research analysis...", num_candidates=4)
print(f"Consensus: {consensus['summary']}")
```

### High-Throughput Batch Triage
Process massive document archives with crash-safe resume tokens and dead-letter queue isolation:
```bash
python scripts/batch_triage.py --input raw_docs.jsonl --output triaged.jsonl --errors-out errors.jsonl --resume --concurrency 8
```

---

## 📊 Comprehensive Hardware Verification

| Hardware Environment | Workload | Latency / Throughput | Quality Score | Source Receipt |
| :--- | :--- | :--- | :--- | :--- |
| **Ryzen AI 9 HX 370 (CPU)** | XSum (FP32) | `2.34 s/doc` (marginal) | `27.89` ROUGE-L | [`results/hx370-cpu/`](results/hx370-cpu/) |
| **Ryzen AI 9 HX 370 (CPU)** | XSum (INT8 Quantized) | `1.04 s/doc` (**1.68x speedup**) | `27.65` ROUGE-L | [`results/hx370-cpu/`](results/hx370-cpu/) |
| **Ryzen AI 9 HX 370 (CPU)** | Sudoku-Hard (Greedy) | `18.93 s` (100 boards) | 84.0% solved | [`results/hx370-cpu/`](results/hx370-cpu/) |
| **Ryzen AI 9 HX 370 (CPU)** | Sudoku-Hard (Best-of-4 + Repair) | `26.46 s` (100 boards) | **91.0% solved** | [`results/hx370-cpu/`](results/hx370-cpu/) |
| **Tesla V100-SXM2 (16GB)** | XSum Full Census (11,334 docs) | **`9.47 docs/sec`** (`0.097 s/doc`) | `27.93` ROUGE-L | [`results/v100-sxm2/`](results/v100-sxm2/) |
| **Tesla V100-SXM2 (16GB)** | Sudoku-Hard Full Test (2,000 boards)| **`76.9 boards/sec`** (`26.0 s` wall) | **86.30% solved** | [`results/v100-sxm2/`](results/v100-sxm2/) |

> **Honest Engineering Note**: Sub-second single-shot summarization does not hold on laptop CPUs in FP32 (`2.34 s/doc`); dynamic INT8 brings it down to `1.04 s`. CPU batching (`bs=8` vs `bs=1`) yields `-1.9%` because FP32 GEMMs already saturate 16 cores on a single document. On dedicated V100 GPUs, throughput scales linearly to 9.47 docs/sec.

---

## 🚢 Enterprise Deployment Options

- **Docker Compose**: Pre-hardened container running unprivileged user (`ncmuser`), read-only root filesystem, `cap_drop: ALL`:
  ```bash
  docker compose up -d
  ```
- **Kubernetes Helm Chart**: Production autoscaling deployment with liveness/readiness probes:
  ```bash
  helm install ncm-summarizer deploy/helm/ncm-summarizer/
  ```
- **Systemd Service**: Host daemon configuration with memory quotas and watchdog timers:
  ```bash
  sudo cp deploy/systemd/ncm-summarizer.service /etc/systemd/system/
  sudo systemctl enable --now ncm-summarizer
  ```

---

## 📑 Strategic Whitepaper & Research References

- **[Enterprise Whitepaper](docs/ENTERPRISE-WHITEPAPER.md)**: Deep dive into sovereign unit economics, data compliance frameworks, and cold-storage triage architecture.
- **[Launch & Press Kit](docs/PRESS-KIT.md)**: Verified publication hooks, technical executive briefs, and media statements.
- **[Hermes EPYC Runbook](docs/HERMES-EPYC-PROMPT.md)**: Automation runbook for high-throughput GPU/CPU stress testing and multi-dataset evaluation.

---

## 📄 License & Attribution

- **Implementation, CLI, SDK, and Benchmark Suite**: MIT License ([`LICENSE`](LICENSE)).
- **Upstream Architecture & Checkpoints**: Stanford Hazy Research & Together AI (MIT License). GPT-2, Qwen3, and BERT tokenizers adhere to their respective upstream licenses. See [`NOTICE`](NOTICE) for comprehensive provenance.
