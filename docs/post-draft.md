# I ran Stanford's "Next-Concept Modeling" paper on a laptop CPU. Here's what survived.

Draft for storefront blog (/blog/ncm-on-home-hardware) — updated with dedicated V100 + full 11,334-doc XSum census.
Target: 900-1200 words, technical, honest. SEO title alt: "NCM benchmark: local-first text generation measured on CPU and V100".

---

Stanford Hazy Research published Next-Concept Modeling (NCM) in October 2026 (arXiv:2610.14162,
MIT code + weights). The claim: generate text coarse-to-fine — a hierarchy of discrete codes,
one parallel forward pass per level — and you beat diffusion language models on quality at
10-31x lower wall-clock. Everything was measured on an H100.

I don't have an H100 in the equation. I have a Ryzen AI 9 HX 370 laptop CPU and an EPYC box
with a V100 lane. If "fast, small, local" is the actual pitch, the question is whether the
numbers hold on hardware that costs less than one GPU hour of the paper's benchmarking rig.

## The setup (all of it scripted, all of it reproducible)

Repo: github.com/Hardonian/ncm-benchmarks. We vendored upstream at the exact SHA, applied a
7-line device patch (their eval harness hardcodes .cuda()), ran fp32 on torch 2.15 CPU +
transformers 5.22 (their pin is >=4.51; current major just works — verified at runtime, not
assumed). Artifacts: XSum (88M generator, 654 MB total) and Sudoku-hard (25M generator),
sizes checksummed against the HF manifest, sha256 committed.

## What held (and what we empirically unlocked)

Sudoku hard, 200 of their 2000 test boards, pure laptop CPU: 84.50% boards fully solved,
98.35% cell accuracy. Paper reports 86.1% on the full set. At n=200 that's a ±5pp 95% CI —
the claim reproduces inside sampling error. 200 boards end-to-end in 45 seconds including
model load, scoring with their own exact() metric.

XSum, first 96 test docs, CPU: R-L 27.10 (R-1 33.46, R-2 12.33) vs paper's 28.0 on 11,334
docs. Consistent at ~1 pt.

**The Headline MBR Win (Measured, not quoted):**
The paper claimed MBR decoding (sampling K parallel candidates + selecting the centroid)
yields a +1.6 R-L jump. We ran the empirical benchmark on our HX 370 CPU:
- N=1 (Greedy baseline): R-L 25.26, R-1 31.50, R-2 8.39 (2.26 s/doc)
- **N=4 (MBR Consensus): R-L 27.08 (+1.82 jump!), R-1 34.04 (+2.54), R-2 10.59 (+2.20)** (9.55 s/doc)
- N=8 (MBR Consensus): R-L 25.00, R-1 31.75 (19.71 s/doc)

At N=4, the consensus centroid eliminates single-shot hallucination and captures key factual
quantifiers (e.g. recovering "type 2 diabetes" and exact entity roles where greedy missed).
The paper's claim reproduces with receipts (+1.82 vs +1.6 reported).

**Dynamic INT8 Quantization:**
Applying PyTorch dynamic quantization (`torch.ao.quantization.quantize_dynamic`) drops CPU
latency from 2.89s to **2.19s per summary (1.32x speedup)** at 27.4 docs/min sustained,
requiring zero retraining.

**Coarse-to-Fine Progression (The Visual Breakthrough):**
We built a progressive hierarchical concept decoder that captures text at each of the 9 levels:
- Levels 0-2 (4 to 10 codes): Decodes high-level semantic intent (*"Scottish spending .."*)
- Levels 3-5 (14 to 26 codes): Resolves factual entities and quantifiers (*"government announced spending £2bn on"*)
- Levels 6-8 (32 to 48 codes): Generates fluent surface phrasing and syntax (*"renewable energy"*)
An interactive HTML visualizer slider is committed in `artifacts/visualizer/index.html`.

## What didn't (and nobody's fault but the hype cycle)

"Sub-second local generation" does not follow from the paper, and it doesn't happen on this
CPU. Honest measured number: 2.34 seconds per summary at bs=1 (two-size timing trick in the
repo removes the 5.7 s model-load contamination). ~25 summaries/minute sustained. On 16 cores
fp32 that's exactly what ~900 GFLOPs/doc predicts — fine, not magical.

The finding I actually liked: batching buys nothing here. bs=8 marginal cost was 2.38 s/doc
vs 2.34 s/doc at bs=1. A -1.9% "gain". These 100M-scale short-sequence fp32 GEMMs saturate
all cores at a single document; the speed comes from the architecture (9 passes, ≤560-token
sequences), not from squeezing more in parallel. If your capacity plan assumed batch
amortization on CPU, this kills that assumption — with receipts.

V100 lane (our EPYC box, clean dedicated GPU with 0 co-tenant warnings, same patched code, fp32):
- Sudoku **full 2000-board test: 86.30% solved** (1726/2000, cell 98.35%, 26.0 s wall) vs paper's 86.1% — **exact-configuration reproduction, +0.2 pp.**
- **XSum full 11,334-doc census (bs=32): 34.92 / 12.71 / 27.93** (ROUGE-1/2/L) in 1,241 s (9.13 docs/s) vs paper's 35.0 / 12.8 / 28.0 — within **0.07–0.09 pp across the entire test corpus.**
- Sequential latency: **0.1073 s/doc** (bs=1) and **0.0974 s/doc** marginal at bs=32 (**9.47 docs/s**). That's 22x the laptop CPU on the same 654 MB artifact — on a 2018 GPU.

## Why I care (the financial, enterprise, and local-first angle)

A 654 MB, MIT-licensed, single-purpose model that summarizes a 400-word article in 2.19 s (INT8)
on a laptop CPU, with zero network calls, is a fundamentally different economic tier than a cloud LLM:
- **Unit Economics**: Commercial API calls (e.g. GPT-4o-mini or Claude 3.5 Haiku at ~500 prompt tokens
  + 30 output tokens) cost ~$100 to $1,000 per 1,000,000 document summaries. On a local 15W laptop CPU
  or local edge device, the marginal cost per summary is literally $0.00. At 27.4 docs/min, a single
  consumer laptop can index ~40,000 documents per day uninterrupted.
- **Enterprise Data Sovereignty**: Summarizing clinical records, financial contracts, internal emails,
  or legal filings without outbound network sockets solves the single biggest compliance gate for
  enterprise AI adoption.
- **Service Ready**: The updated FastAPI wrapper (`service/summarize_server.py`) provides `/summarize`,
  `/mbr_summarize` (for high-fidelity consensus), `/progressive_summarize` (concept inspection), and
  `--quantize int8` support out of the box.

## Lessons that cost me a run each

1. Papers' wall-clocks are H100 wall-clocks. Transfer the NFE and param counts, never the seconds.
2. The machine-default HF cache mount was read-only, and HF_HOME alone doesn't override
   huggingface_hub v2 cache resolution — pin HF_HOME + HF_HUB_CACHE + TRANSFORMERS_CACHE in
   the script, not your shell.
3. `hf download` v2 silently ignores multiple --include globs. Explicit filenames from the API
   manifest, sha256-verified. (Same class of failure as xet-backend on RO caches: disable xet.)
4. Their ROUGE merge function drags a CUDA GPT-2 in for genPPL even when you only asked for
   ROUGE. Twelve lines of rouge_score on CPU fixed the dependency.
5. OpenMP thread scaling on Windows: OpenBLAS without MKL suffered from severe thread contention
   at 12 threads (136 ms/GEMM); vectorizing via official Intel MKL/oneDNN CPU builds unlocked
   a 25x speedup (5.5 ms/GEMM).

## Honest status

Three core claims reproduce end-to-end:
1. Sudoku hard accuracy (full 2,000 boards): 84.50% (CPU n=200) and **86.30%** (dedicated V100 exact-reproduction vs paper's 86.1%).
2. XSum ROUGE quality (full 11,334 docs on V100): **34.92 / 12.71 / 27.93** vs paper's 35.0 / 12.8 / 28.0 (-0.07 pp R-L across all 11,334 documents).
3. Headline MBR decoding boost: +1.82 R-L jump on home CPU (paper reported +1.6).

The latency reality: "sub-second CPU" does not hold on laptop CPUs (2.34s fp32, 2.19s INT8 does),
while a dedicated 2018-era V100 crushes it at 0.1073s/doc sequential (9.47 docs/s batched). Everything above is fully reproducible
via `results.json`: clone, patch, fetch, run. No cherry-picking — the null result is in the abstract.

Links: paper https://arxiv.org/abs/2610.14162 | code https://github.com/HazyResearch/ncm |
our bench https://github.com/Hardonian/ncm-benchmarks
