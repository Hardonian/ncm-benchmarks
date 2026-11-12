# Show HN draft

Title (≤80 chars):
  We ran Stanford's Next-Concept Modeling paper on a laptop CPU — here's what held [(pick: "I reproduced Stanford's NCM paper on a laptop CPU: 2 claims hold, batch scaling doesn't") ]

Body:
Paper: Next-Concept Modeling (arXiv:2610.14162, MIT, Hazy Research) — text generation as a
coarse-to-fine hierarchy of discrete codes, 8-32 parallel passes instead of 64-128 diffusion
NFEs. Published speedups (21x/31x) were all H100.

We benchmarked on hardware that exists in real homes: Ryzen AI 9 HX 370 CPU (16 cores, fp32),
V100 lane pending. Scripted, sha256-pinned, reproducible.

Results:
- Sudoku hard (25M model): 84.5% solved on 200 of their 2000 boards vs paper's 86.1% —
  reproduces inside the ±5pp CI. 45 s total incl. load.
- XSum (88M): R-L 27.10 on 96-doc subset vs paper 28.0 on full set.
- Honest null: batch size 8 gives -1.9% gain over bs=1 on CPU. The fp32 GEMMs saturate all
  cores at one document. Papers never show you this row.
- Honest L: 2.34 s/summary on CPU (measured with a two-size timing trick that subtracts the
  5.7 s model load). "Sub-second local" was never in the paper — it's what marketing would
  have said.
- Pitfalls: eval harness hardcodes .cuda() (7 sites, we ship the patch); HF_HOME alone does
  not override hub v2 cache resolution; `hf download` silently ignores multi-globs; their
  ROUGE scorer requires a CUDA GPT-2 unless you write 12 lines of rouge_score.

Repo (MIT, our content; upstream pinned to c702742 + NOTICE provenance):
https://github.com/Hardonian/ncm-benchmarks

Comment strategy: first comment = methodology (two-size timing, CI math, where the FLOPs
estimate lands); engage on batch-scaling and "why fp32 not bf16 on CPU" (oneDNN bf16 was not
worth the correctness risk vs their fp32-trained pipeline for a reproduction — say so plainly).
