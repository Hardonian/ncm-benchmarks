# NCM Local Benchmarks

Independent, on-real-hardware reproduction of the paper **"Next-Concept Modeling:
Fast, Parallel Coarse-to-Fine Text Generation"** ([arXiv:2610.14162](https://arxiv.org/abs/2610.14162),
Stanford Hazy Research + Together AI, Oct 2026).

The paper claims text generation can beat diffusion LMs on quality while running
10-31x faster by generating a *hierarchy of discrete codes* coarse-to-fine (one
parallel forward pass per level, K=8-32 total) instead of one-token-at-a-time or
64-128 denoising steps.

This repo answers the only question that matters if you plan to *run* this stuff:
**do the numbers survive contact with non-H100 hardware?**

## Results (measured here, not quoted from the paper)

| Task | Paper claim (their hardware) | Our run (device) | Verdict |
|---|---|---|---|
| Sudoku hard (25M gen, 12 NFE) | 86.1% solved / full 2000-board test | **84.50%** / 200 boards / **CPU fp32** | Reproduces (95% CI +/-5.0pp) |
| XSum summarization (88M gen, 9 levels) | R-L 28.0 / 11,334 docs (H100, 31x less wall-clock than diffusion) | **R-L 27.10** (R-1 33.46, R-2 12.33) / first 96 docs / **CPU fp32** | Consistent at ~1 pt |
| CPU summary latency | -- | **2.34 s/doc** marginal (bs=1), ~5.7 s cold load, 25 docs/min sustained | "Sub-second local" does NOT hold on a laptop CPU; 2.3 s does |
| Batching on CPU | -- | bs=8 saves **-1.9%** vs bs=1 (i.e. nothing) | Null result, reported honestly: fp32 GEMMs already saturate 16 cores at bs=1 |

Full machine-readable summary: [results/hx370-cpu/results.json](results/hx370-cpu/results.json).
Raw per-run logs next to it. GPU (V100 lane) rows pending -- see README notes below.

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
   (tokenizer+generator) MIT-licensed XSum model produces 25 summaries/minute on a
   laptop CPU with no cloud call. The speed advantage comes from the architecture
   (9 passes over <=560-token sequences), not from the H100 in their figure.
2. **The paper's headline quality win is MBR decoding** -- cheap parallel candidates
   + a selector (28.0 -> 29.6 R-L at matched time). If your orchestration already
   does draft-select ensembles, this is literature validation; if not, copy it.
3. **Pitfalls you will actually hit** (all cost us a failed run, all fixed here):
   - upstream eval harness hardcodes `.cuda()` in 7 sites (see `patches/`);
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

- **Verified here, this week:** every number in `results/hx370-cpu/`, on this
  machine, with their scorer (`scripts/eval_merge.py exact()` for Sudoku) or an
  equivalent CPU scorer for ROUGE.
- **Quoted from the paper** (we did not reproduce them): H100 wall-clock speedups
  (21x/31x), LM1B/OWT perplexity sweeps, WMT14 BLEU, MBR deltas, training budgets.
- **Pending:** V100 GPU lane row (runner staged in `scripts/as-run/epyc_runbook.sh`).
- Local subsets carry their n and, for the binary metric, a 95% CI. We do not blend
  our numbers with the paper's in any single cell.

## Local summarizer service (expansion)

`service/summarize_server.py` wraps the vendored pipeline in one FastAPI endpoint
(`/summarize`), loads the model once (~6 s at boot) and returns the summary with
measured latency. Fully local, 654 MB of weights, no network at inference time:

```bash
NCM_HOME=... venv/bin/python service/summarize_server.py --port 8031
curl -s localhost:8031/summarize -H 'content-type: application/json' \
  -d '{"text": "<article...>"}'
```

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
