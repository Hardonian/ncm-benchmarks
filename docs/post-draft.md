# I ran Stanford's "Next-Concept Modeling" paper on a laptop CPU. Here's what survived.

Draft for storefront blog (/blog/ncm-on-home-hardware) — fill V100 row when EPYC leg runs.
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

## What held

Sudoku hard, 200 of their 2000 test boards, pure laptop CPU: 84.50% boards fully solved,
98.35% cell accuracy. Paper reports 86.1% on the full set. At n=200 that's a ±5pp 95% CI —
the claim reproduces inside sampling error. 200 boards end-to-end in 45 seconds including
model load, scoring with their own exact() metric.

XSum, first 96 test docs, CPU: R-L 27.10 (R-1 33.46, R-2 12.33) vs paper's 28.0 on 11,334
docs. Same distribution; subset, not CI-quantified — say so.

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

V100 lane: [PENDING — link with measured summaries/sec + full sudoku 2000 when EPYC leg lands]

## Why I care (the local-first angle)

A 654 MB, MIT-licensed, single-purpose model that summarizes a 400-word article in 2.3 s on
a laptop CPU, with zero network calls, is a different product category than "small LLM".
The repo includes a FastAPI wrapper (`service/summarize_server.py`) that loads once and serves
/summarize with latency included in the response. It's the honest version of "private
summarization appliance": no cloud, no per-token bill, the whole artifact fits in a backpack.

Second thing worth stealing from the paper itself: their headline quality win is MBR decoding
— cheap parallel candidates plus a similarity selector, +1.6 ROUGE-L at matched time. If your
agent orchestration already runs draft-select ensembles, this is literature validation from a
surprising direction.

## Lessons that cost me a run each

1. Papers' wall-clocks are H100 wall-clocks. Transfer the NFE and param counts, never the seconds.
2. The machine-default HF cache mount was read-only, and HF_HOME alone doesn't override
   huggingface_hub v2 cache resolution — pin HF_HOME + HF_HUB_CACHE + TRANSFORMERS_CACHE in
   the script, not your shell.
3. `hf download` v2 silently ignores multiple --include globs. Explicit filenames from the API
   manifest, sha256-verified. (Same class of failure as xet-backend on RO caches: disable xet.)
4. Their ROUGE merge function drags a CUDA GPT-2 in for genPPL even when you only asked for
   ROUGE. Twelve lines of rouge_score on CPU fixed the dependency.

## Honest status

Two claims reproduce. One (latency class) needs our hardware-specific number, which is still
good. GPU leg pending. Everything above is a `results.json` away from you: clone, patch,
fetch, run. No cherry-picking — the null result is in the abstract.

Links: paper https://arxiv.org/abs/2610.14162 | code https://github.com/HazyResearch/ncm |
our bench https://github.com/Hardonian/ncm-benchmarks
