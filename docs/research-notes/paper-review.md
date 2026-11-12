# NCM — Next-Concept Modeling (arXiv 2610.14162, Oct 2026) — review + leverage notes

Source tarball reviewed 2026-11-11. Paper source: scratch/arxiv-2610.14162/
Repo cloned: scratch/ncm-repo (MIT, 1.8k stars in ~1 month, 5 commits, 2 contributors).
Authors: Kumbong, Khare et al., Stanford Hazy Research (Chris Ré) + Together AI (Mirhoseini, Arora).
Code: https://github.com/HazyResearch/ncm | Weights: HF hazyresearch/ncm-checkpoints | Data: hazyresearch/ncm-datasets | Blogs: hazyresearch.stanford.edu + scalingintelligence.stanford.edu.

## What it is (verified from paper)
Two-stage text generation, VAR-style (image next-scale prediction) ported to text:
1. Multiscale residual VQ-VAE tokenizer: encodes L tokens into K levels of discrete codes
   (l_1 < ... < l_K = L), residual quantization + nested dropout so coarse levels stand alone.
2. Frozen tokenizer; transformer generates one level per forward pass, all codes within a
   level sampled in parallel, conditioned on cumulative reconstruction. Generation = K NFE
   (K = 8–32) vs 64–128 for diffusion baselines and L sequential steps for AR.
Training tricks: conditioning corruption (random code replacement, fixes exposure bias),
level-weighted CE (alpha), block-diagonal/prefix-LM attention, BF16 + torch.compile.
ELBO formulation in appendix/theory.tex.

## Headline results at ~100M params (verified in tables)
- LM1B: Gen PPL 93.1 vs DUO 97.3 (99M-class), 11x fewer NFE, 21x less wall-clock, 4.9x fewer
  train tokens (13.36B vs 66B full-length positions incl. tokenizer).
- XSum: ROUGE-L 28.0 unguided at 8 NFE (beats AURORA-LM 26.6 @128 NFE); +MBR 29.6 @128 NFE
  = best in table; 31x faster than strongest DLM.
- WMT14 De-En: 22.6 BLEU @11 NFE (non-AR runner-up to E2D2 24.8 @64); +MBR 25.2.
- Sudoku (25M gen): 99.5/96.5/86.1% solved easy/med/hard @12 NFE — beats FMLM+ (81.3 hard @16),
  crushes AR (1.0 hard).
- Scaling: 97/317/725M on LM1B improves PPL at fixed token budget; trend not saturated.
- Tokenizer recon exactness: 100% Sudoku, 99.79% XSum, 98.79% WMT14.

## Verified facts about the release (important for us)
- MIT license, code AND checkpoints+datasets on HF. Base tokenizer vocabs are GPT-2 (MIT),
  bert-base-uncased (Apache-2.0), Qwen3-0.6B (Apache-2.0) → wheelhouse SPDX allowlist clean
  (confirm against Foundry licensing.py list, but MIT/Apache both standard-allow).
- EVAL-ONLY repo: models/ + evaluate.py + per-GPU shards. NO training loop released.
  Reproducing any model = reimplementing tokenizer + generator training (days of work).
- Checkpoint sizes: XSum tok+gen 654 MB; Sudoku 162 MB/difficulty; LM1B large 2.9 GB;
  WMT14 1.1 GB. Tiny enough for any lane.
- No model card maturity concerns: single-purpose 25–120M models on small corpora.

## Leverage (ranked)
1. Local-first fast summarizer/translator demo + microservice (EPYC V100 lane; CPU smoke
   test on HX370). XSum 88M model, 8 NFE: privacy-first, tiny, fast. Feeds $49/$297 audit
   narrative ("local summarization that doesn't need a 7B model") and a Gumroad
   "Local-First Diffusion-Era NLM" pack. MUST measure actual tok/s; paper's H100 wall-clocks
   don't transfer.
2. Content/visibility spike NOW (paper is trending, 1.8k stars): "ran a Stanford NCM paper on
   home hardware" benchmark post + honest explainer. Pairs with Foundry PyPI publish / Show HN
   window (star-magnet plan).
3. Validation of our orchestration pattern: paper's headline quality win is MBR (sample K
   candidates, select) at matched wall-clock — exactly our draft-select K=8 ensemble
   (0.625 -> 0.875). Cite this for triage-decide pipelines: efficiency gains should be spent
   on more candidates, not banked.
4. Foundry: NOT in the LoRA lane (from-scratch regime, no training code released). Adjacent
   fit: constrained structured-output tasks (Sudoku-like) where coarse-to-fine parallel gen
   beats AR commit-without-revision. Possible future "training-from-scratch" gate work only
   if we write the trainer; do not promise reproduction.
5. JEV triage research note: nested dropout (each partial hierarchy must reconstruct the
   whole) + Sudoku global-refinement result = ideas worth stealing for progressive-representation
   triage. Do not pivot the JEPA spike on this paper's account.

## Honest caveats
- 100M-class on LM1B/XSum/WMT14: quality is far below production 7B+ AR models; the claim is
  speed+size, not SOTA text. Sample artifacts at coarse-level transitions (their own limitation).
- Per-dataset retraining required (tokenizer is not universal); fixed K/ell_l schedules.
- Training costs (our estimate, 6*N*T FLOPs, NOT from paper): XSum ~1.3e19 FLOPs -> ~3 days
  on a V100-class lane; Sudoku ~8e17 -> hours. P40 lane (CC 6.1, no bf16, inference-only) can
  serve fp32/fp16(-ish) checkpoints but cannot retrain the paper's recipe as-is.
- Repo youth: 5 commits, 2 contributors, no releases. Vendor the ~2k LOC rather than depend on it.

## Next actions (uncharged backlog)
- [ ] Download xsum + sudoku hard checkpoints (HF), run evaluate.py on EPYC V100; record
      summaries/sec + ROUGE to verify the paper's speed story at our hardware tier.
- [ ] CPU-only inference spike on HX370 (fp32; batch>1) — decide if "sub-second local summary"
      claim survives.
- [ ] Check requirements (torch, transformers>=4.51) against lane fleet envs; wheelhouse v5
  transformers pin noted elsewhere — NCM repo may need its own venv.
- [ ] If both pass: draft the benchmark post + pack outline.
