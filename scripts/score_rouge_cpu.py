"""CPU-only scorer for NCM eval shards (skips their GPT-2 genPPL which needs CUDA).
Usage: score_rouge_cpu.py shard1.pkl [shard2.pkl ...]
"""
import pickle  # nosec B403 - local self-produced shards, see rationale below
import sys

import numpy as np
from rouge_score import rouge_scorer

rows = []
for f in sys.argv[1:]:
    # Safe: these .pkl files are produced locally by our own eval_shard.py run
    # (their eval format), not fetched from any remote source.
    d = pickle.load(open(f, "rb"))  # nosec B301 - file produced by our own eval_shard run
    rows += list(zip(d["idxs"], d["hyps"], d["refs"]))
rows.sort(key=lambda r: r[0])
hyps = [h for _, h, _ in rows]
refs = [r for _, _, r in rows]

sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
R = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
for h, r in zip(hyps, refs):
    v = sc.score(r, h)
    for k in R:
        R[k].append(v[k].fmeasure)

print(f"n={len(hyps)}")
print(f"R-1  R-2  R-L  words")
print(
    f"{100 * np.mean(R['rouge1']):5.2f} {100 * np.mean(R['rouge2']):5.2f} "
    f"{100 * np.mean(R['rougeL']):5.2f} {np.mean([len(h.split()) for h in hyps]):6.1f}"
)
