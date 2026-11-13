"""Minimum Bayes Risk (MBR) Decoding Engine for Next-Concept Modeling.

Implements batched parallel candidate generation and consensus selection:
h_MBR = argmax_{h_i} sum_{j=1}^K Utility(h_i, h_j)

Where Utility is ROUGE-L F1 or Token Jaccard overlap.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Literal

import torch
from rouge_score import rouge_scorer

from ncm_bench.inference import decode_logits_to_texts, prepare_prefix_batch


def compute_pairwise_rouge_l(candidates: list[str]) -> list[float]:
    """Computes total ROUGE-L utility for each candidate against all other candidates."""
    K = len(candidates)
    if K == 1:
        return [1.0]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [0.0] * K
    for i in range(K):
        for j in range(i + 1, K):
            score = scorer.score(candidates[j], candidates[i])["rougeL"].fmeasure
            scores[i] += score
            scores[j] += score
    return scores


def compute_pairwise_jaccard(candidates: list[str]) -> list[float]:
    """Fast token-set Jaccard similarity utility matrix."""
    K = len(candidates)
    if K == 1:
        return [1.0]

    token_sets = [set(c.lower().split()) for c in candidates]
    scores = [0.0] * K
    for i in range(K):
        for j in range(i + 1, K):
            s_i, s_j = token_sets[i], token_sets[j]
            union = len(s_i | s_j)
            sim = len(s_i & s_j) / union if union > 0 else 0.0
            scores[i] += sim
            scores[j] += sim
    return scores


def compute_pairwise_cosine(candidates: list[str]) -> list[float]:
    """Fast bag-of-words TF cosine similarity utility matrix."""
    K = len(candidates)
    if K == 1:
        return [1.0]

    vecs = [Counter(c.lower().split()) for c in candidates]
    norms = [math.sqrt(sum(v * v for v in vec.values())) for vec in vecs]
    scores = [0.0] * K
    for i in range(K):
        for j in range(i + 1, K):
            if norms[i] == 0.0 or norms[j] == 0.0:
                sim = 0.0
            else:
                dot = sum(vecs[i][w] * vecs[j][w] for w in vecs[i] if w in vecs[j])
                sim = dot / (norms[i] * norms[j])
            scores[i] += sim
            scores[j] += sim
    return scores


def select_mbr_centroid(
    candidates: list[str],
    utility: Literal["rougeL", "jaccard", "cosine"] = "rougeL",
) -> tuple[str, int, list[float]]:
    """Selects the candidate with the highest consensus utility.

    Returns:
        (best_candidate, best_index, candidate_scores)
    """
    if not candidates:
        raise ValueError("Candidate pool is empty")
    if len(candidates) == 1:
        return candidates[0], 0, [1.0]

    if utility == "rougeL":
        scores = compute_pairwise_rouge_l(candidates)
    elif utility == "jaccard":
        scores = compute_pairwise_jaccard(candidates)
    elif utility == "cosine":
        scores = compute_pairwise_cosine(candidates)
    else:
        raise ValueError(f"Unknown utility metric: {utility}")

    best_idx = max(range(len(candidates)), key=lambda i: scores[i])
    return candidates[best_idx], best_idx, scores


def generate_mbr_summary(
    model_bundle: dict,
    text: str | list[int],
    num_candidates: int = 4,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    utility: Literal["rougeL", "jaccard", "cosine"] = "rougeL",
) -> dict:
    """Generates K parallel candidates with temperature sampling and selects MBR centroid."""
    if num_candidates < 1:
        raise ValueError("num_candidates must be >= 1")

    S = model_bundle
    x_single = prepare_prefix_batch(S, [text])
    x_batch = x_single.expand(num_candidates, -1).contiguous()
    eos = S["EOS"] if S["EOS"] is not None else 50256

    with torch.inference_mode():
        g = S["net"].generate(
            None,
            x_batch,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_scale=0.0,
            cfg_null_eos=eos,
        )
        _, zl = S["tk"].vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        logits = S["tk"].vqvae.decoder(zl[-1])

    candidates = decode_logits_to_texts(logits, S)
    best_cand, best_idx, scores = select_mbr_centroid(candidates, utility=utility)
    return {
        "summary": best_cand,
        "selected_index": best_idx,
        "num_candidates": num_candidates,
        "candidates": candidates,
        "scores": [round(s, 4) for s in scores],
    }
