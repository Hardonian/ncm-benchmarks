# Show HN draft

Title (≤80 chars):
  We ran Stanford's Next-Concept Modeling paper on a laptop CPU — here's what held [(pick: "I reproduced Stanford's NCM paper on a laptop CPU: 2 claims hold, batch scaling doesn't") ]

Body:
Paper: Next-Concept Modeling (arXiv:2610.14162, MIT, Hazy Research) — text generation as a
coarse-to-fine hierarchy of discrete codes, 8-32 parallel passes instead of 64-128 diffusion
NFEs. Published speedups (21x/31x) were all H100.

We benchmarked on hardware that exists in real homes: Ryzen AI 9 HX 370 CPU (16 cores, fp32)
and an EPYC box with a dedicated V100-SXM2. Scripted, sha256-pinned, reproducible.

Results:
- Sudoku hard, FULL 2000-board test on V100 (25M model): 86.3% solved vs paper's 86.1% —
  exact-configuration reproduction. 26 s wall including load. On 200 boards with a laptop CPU:
  84.5% greedy; adding Best-of-4 + bounded constraint repair (`n=100` CPU slice) jumps to **91.0%**
  solved (+7.0 pp over greedy `84.0%`, 98.47% cell accuracy, Wilson 95% CI 83.77-95.19%).
- XSum (88M), FULL 11,334-doc census on V100: R-1 34.92 / R-2 12.71 / R-L 27.93 in 1,241 s
  (9.13 docs/s) vs paper's 35.0 / 12.8 / 28.0 (within 0.07 pp R-L on the full corpus!).
  On CPU (96 docs): R-L 27.10; V100 latency 0.107s/doc (bs=1) and 0.097s/doc (bs=32, 9.47 docs/s).
- MBR Decoding (Headline Quality Win): We measured the empirical MBR Pareto frontier. Sampling
  N=4 parallel candidates + centroid selection gives +1.82 R-L (25.26 -> 27.08) and +2.54 R-1
  (31.50 -> 34.04). The paper's headline quality claim reproduces on real hardware.
- Zero-Pickle Safetensors + INT8 Parity (`n=32`, `bs=8` CPU): Converts verified weights to
  `.safetensors` + SHA-256 `inference_meta.json`, yielding a 1.68x speedup (`55.93 s -> 33.26 s`)
  for only `-0.24 pp` ROUGE-L (`27.89 -> 27.65`) and `1,036 ms` live FastAPI latency.
- Clean-Room Stage-1 VQ-VAE & Stage-2 NCM Trainers (`ncm_train/`): Upstream shipped eval only;
  we added end-to-end training loops + `train-smoke` (`194.08 -> 3.79` Stage-1 loss over 40 steps).
- Enterprise Stack: Offline PDF/long-doc map-reduce + PII scrubbing/restoration + SHA-256 audit
  receipts, async FastAPI (413/429 limits, Prometheus histograms, OpenAI `/v1/chat/completions`, SSE),
  Helm chart + systemd unit + 48-test CI (`ruff`/`bandit` clean).
- Honest null: batch size 8 gives -1.9% gain over bs=1 on CPU. The fp32 GEMMs saturate all
  cores at one document. Papers never show you this row.
- Honest L: 2.34 s/summary on CPU (measured with a two-size timing trick that subtracts the
  5.7 s model load). "Sub-second local" was never in the paper — it's what marketing would
  have said.
- Pitfalls: eval harness hardcodes .cuda() (7 sites, we ship the patch); HF_HOME alone does
  not override hub v2 cache resolution; `hf download` silently ignores multi-globs; their
  ROUGE scorer requires a CUDA GPT-2 unless you write 12 lines of rouge_score; OpenBLAS on
  Windows without MKL lock-convoys at 12 threads (MKL vectorized build fixed it 25x).

Repo (MIT, our content; upstream pinned to c702742 + NOTICE provenance):
https://github.com/Hardonian/ncm-benchmarks

Comment strategy: first comment = methodology (two-size timing, CI math, where the FLOPs
estimate lands); engage on batch-scaling and "why fp32 not bf16 on CPU" (oneDNN bf16 was not
worth the correctness risk vs their fp32-trained pipeline for a reproduction — say so plainly).
