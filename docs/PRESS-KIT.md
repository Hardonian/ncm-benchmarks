# Launch & Press Kit (`ncm-benchmarks` v2.1.0)

Verified benchmark hooks, executive talking points, and publication templates anchored strictly to recorded hardware receipts in [`results/`](../results/).

---

## X / Twitter Thread Template

**Post 1/4**

Can an open, sovereign 149M-parameter Next-Concept Model match H100 summarization quality, solve 90.7% of Sudoku-Hard puzzles, and classify documents 3.38x faster via Level 0..2 early-exit on CPU — with zero cloud egress?

Meet `ncm-benchmarks` v2.1.0: [GitHub](https://github.com/Hardonian/ncm-benchmarks)

**Post 2/4**

Measured receipts on a single 2018 Tesla V100-SXM2 & Ryzen AI 9 HX 370 CPU:

- **XSum Full Census (11,334 docs)**: `27.93` ROUGE-L on V100 (`9.47 docs/sec`) vs `28.0` H100 baseline
- **XSum Batched (`bs=64`)**: `2,728.5 docs/min` (`0.022 s/doc` marginal on V100)
- **WMT14 En-De (3,003 sentences)**: `21.72` BLEU at `35.16 sent/sec` on V100
- **Sudoku Multi-Tier (Best-of-4 + Repair)**: `99.8%` Easy, `98.55%` Medium, `90.7%` Hard on V100 (`97.03 / 77.77 / 48.2 boards/s`)
- **Coarse Early-Exit (`L0..L2`)**: `3.38x` CPU speedup (`356.02 ms` vs `1205.13 ms/doc`)

**Post 3/4**

New in `v2.1.0` for sovereign enterprise AI teams:

- Multi-format zero-egress ingestion (`.pdf`, `.html`, `.csv`, `.eml`) + `API_KEY` / `IBAN` PII scrubbing
- Keyed `HMAC-SHA256` cryptographic audit receipts & `ncm-bench report` JSON/HTML compliance exports
- `POST /triage/feedback` online active learning (`ConceptClassifier.partial_fit`), `CVE` / `IP` extraction with `entity_spans` provenance, and Concept-Space DPO (`train_stage2_dpo`)

**Post 4/4**

Install in one command and inspect every receipt + interactive TCO calculator:

- `pip install ncm-benchmarks`
- [PyPI Package](https://pypi.org/project/ncm-benchmarks/)
- [Interactive Concept Visualizer & TCO Calculator](https://hardonian.github.io/ncm-benchmarks/)
- [Paper: arXiv:2610.14162](https://arxiv.org/abs/2610.14162)

---

## Reddit (`r/LocalLLaMA` / `r/MachineLearning`) Technical Brief

### [Project] ncm-benchmarks v2.1.0 — Sovereign Next-Concept Engine (V100 Multi-Tier Sudoku 99.8%/98.55%/90.7%, 2,728 docs/min XSum, Early-Exit Triage & Concept-DPO)

We released `ncm-benchmarks` v2.1.0 ([PyPI](https://pypi.org/project/ncm-benchmarks/), [GitHub](https://github.com/Hardonian/ncm-benchmarks)), a clean-room systems engineering, alignment, and benchmarking stack built around hierarchical **Next-Concept Modeling (NCM)** ([arXiv:2610.14162](https://arxiv.org/abs/2610.14162)).

Rather than predicting tokens autoregressively, NCM generates coarse-to-fine discrete codes across hierarchy levels in a bounded number of parallel forward passes.

#### Verified Hardware Receipts

- **Full XSum 11,334-doc census (V100 FP32)**: `27.93` ROUGE-L at `9.47 docs/sec`, scaling to `2,728.5 docs/min` (`0.022 s/doc` marginal) at `bs=64`
- **Full WMT14 En-De 3,003-sentence eval (V100 FP32)**: `21.72` BLEU at `35.16 sent/sec` (`85.4 s` total)
- **Sudoku Multi-Tier (V100 Best-of-4 + deterministic 64-node repair, 6,000 boards)**: Easy `99.8%` (`97.03 boards/sec`), Medium `98.55%` (`77.77 boards/sec`), Hard `90.7%` (`48.2 boards/sec`)
- **Ryzen AI 9 HX 370 CPU (INT8 `.safetensors`)**: `1.68x` speedup vs FP32 (`33.26s` vs `55.93s` at `-0.24 pp` ROUGE-L) and `3.38x` speedup (`356.02 ms` vs `1205.13 ms/doc`) using Level `0..2` early-exit triage
- **Clean-room Concept Training (`ncm_train/`)**: Stage-1 VQ (`194.08 -> 3.79`), Concept-GRPO (`+0.0938` reward delta on CPU), Teacher-Student Concept Distillation (`9.398 -> 6.3534`), and discrete Concept-Space DPO (`train_stage2_dpo`)

#### What Is New in v2.1.0

1. `.html`, `.csv`, and `.eml` extraction + `API_KEY` and `IBAN` PII redaction in [`ncm_bench/longdoc.py`](../ncm_bench/longdoc.py)
2. Keyed `HMAC-SHA256` audit signing/verification (`build_audit_receipt` / `verify_audit_receipt`) and `ncm-bench report` (`--html`)
3. Bounded LRU coarse code caching, `CVE`/`IP` extraction with character `entity_spans`, custom enum/required-field validation, and `POST /triage/feedback` (`ConceptClassifier.partial_fit`)
4. [100-Item Frontier Roadmap](ROADMAP-100.md) & interactive V100/CPU TCO calculator tab in [docs/index.html](index.html)

---

## LinkedIn Executive Announcement

**Sovereign Document AI Without $0.003–$0.02/Doc API Tax: `ncm-benchmarks` v2.1.0**

Enterprise AI deployments often route sensitive financial, legal, and security documents to expensive cloud APIs simply to classify, redact, and summarize text.

With **`ncm-benchmarks` v2.1.0**, teams can run verifiable **Next-Concept Modeling** completely offline on existing CPUs and commodity GPUs (`$0.0000` marginal API cost):

- **Verified Throughput & Quality**: `2,728.5 docs/min` batched summarization (`bs=64`) and `27.93` ROUGE-L full-census XSum accuracy on a 2018 Tesla V100, plus `3.38x` faster Level `0..2` early-exit routing on laptop CPUs.
- **Compliance by Construction**: Zero-pickle `.safetensors` weights, native `PDF`/`HTML`/`CSV`/`EML` scrubbing (PII, API keys, IBANs), `HMAC-SHA256` cryptographic audit receipts, and one-command `ncm-bench report` exports.
- **Closed-Loop Accuracy**: Online feedback tuning via `POST /triage/feedback` and discrete Concept-Space DPO (`train_stage2_dpo`) without cloud lock-in.

Install via `pip install ncm-benchmarks` or explore the interactive demo and hardware receipts at [GitHub](https://github.com/Hardonian/ncm-benchmarks).
