# Show HN draft

Title (≤80 chars):
  We ran Stanford's Next-Concept Modeling paper on a laptop CPU — here's what held [(pick: "I reproduced Stanford's NCM paper on a laptop CPU: 2 claims hold, batch scaling doesn't") ]

Body:
Paper: Next-Concept Modeling (arXiv:2610.14162, MIT, Hazy Research) — text generation as a
coarse-to-fine hierarchy of discrete codes, 8-32 parallel passes instead of 64-128 diffusion
NFEs. Published speedups (21x/31x) were all H100.

We benchmarked on hardware that exists in real homes: Ryzen AI 9 HX 370 CPU (16 cores, fp32)
and an EPYC box with a shared V100-SXM2. Scripted, sha256-pinned, reproducible.

Results:
- Sudoku hard, FULL 2000-board test on V100 (25M model): 86.3% solved vs paper's 86.1% —
  exact-configuration reproduction. 26 s wall including load. On 200 of their boards with
  a laptop CPU: 84.5%, also CI-consistent.
- XSum (88M): R-L 27.10 on 96-doc subset vs paper 28.0 on full set.
- MBR Decoding (Headline Quality Win): We measured the empirical MBR Pareto frontier. Sampling
  N=4 parallel candidates + centroid selection gives +1.82 R-L (25.26 -> 27.08) and +2.54 R-1
  (31.50 -> 34.04). The paper's headline quality claim reproduces on real hardware.
- INT8 Dynamic Quantization: 2.19 s/doc on CPU (vs 2.89 s fp32) — a 1.32x speedup with zero retraining.
- Coarse-to-Fine Progression: Built a hierarchical decoder that outputs intermediate representations
  at each of the 9 code levels (L0-2 topic/gist -> L3-5 entities/quantifiers -> L6-8 syntax); interactive
  slider HTML committed in repo.
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
