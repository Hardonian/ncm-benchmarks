"""Minimum Bayes Risk (MBR) Decoding Engine for Next-Concept Modeling.

Implements candidate generation and consensus selection:
h_MBR = argmax_{h_i} sum_{j=1}^K Utility(h_i, h_j)

Where Utility is ROUGE-L F1, ROUGE-1 F1, or Token Jaccard overlap.
"""
from typing import Literal
import torch
from rouge_score import rouge_scorer


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


def select_mbr_centroid(
    candidates: list[str],
    utility: Literal["rougeL", "jaccard"] = "rougeL"
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
    utility: Literal["rougeL", "jaccard"] = "rougeL"
) -> dict:
    """Generates K parallel candidates with temperature sampling and selects MBR centroid."""
    S = model_bundle
    if isinstance(text, str):
        ids = S["src"](text)["input_ids"]
    else:
        ids = list(text)
    x_single = (ids[:S["P"]] + [S["PAD"]] * S["P"])[:S["P"]]
    
    # Batch candidates together in a single tensor of shape [K, prefix_len]
    x_batch = torch.tensor([x_single] * num_candidates, device=S["device"])
    
    with torch.no_grad():
        g = S["net"].generate(
            None,
            x_batch,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_scale=0.0,
            cfg_null_eos=S["EOS"]
        )
        _, zl = S["tk"].vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        logits = S["tk"].vqvae.decoder(zl[-1])
    
    # Decode candidates
    B, T, V = logits.shape
    lg = logits.double()
    allow = (S["allow"][:V] * T) if S["allow"] is not None else None
    al = S["adj"][0] if S["adj"] else 0.0
    aw = int(S["adj"][1]) if S["adj"] and len(S["adj"]) > 1 else 1
    ad = S["adj"][2] if S["adj"] and len(S["adj"]) > 2 else 1.0
    
    candidates = []
    for b in range(B):
        cnt = torch.zeros(1, V, dtype=torch.float64, device=lg.device)
        out = torch.empty(1, T, dtype=torch.long, device=lg.device)
        hist = []
        for t in range(T):
            step = lg[b:b+1, t, :].clone()
            if S["freq"] > 0:
                excess = (cnt - allow).clamp_min(0.0) if allow is not None else cnt
                step -= S["freq"] * excess
                if S["mu"] > 0 and allow is not None:
                    step += S["mu"] * ((T - t) / T) * (allow - cnt).clamp_min(0.0)
            for j, h in enumerate(reversed(hist[-aw:])):
                step.scatter_add_(1, h[:, None], torch.full((1, 1), -al * ad ** j, dtype=torch.float64, device=lg.device))
            p = step.argmax(-1)
            out[:, t] = p
            hist.append(p)
            cnt.scatter_add_(1, p[:, None], torch.ones(1, 1, dtype=torch.float64, device=lg.device))
        
        row = out[0].tolist()
        if S["EOS"] in row:
            row = row[:row.index(S["EOS"])]
        cand = S["tk"].pre_tokenizer.decode([tok for tok in row if tok not in (S["EOS"], S["PAD"])], skip_special_tokens=True)
        candidates.append(cand.strip())
    
    best_cand, best_idx, scores = select_mbr_centroid(candidates, utility=utility)
    return {
        "summary": best_cand,
        "selected_index": best_idx,
        "num_candidates": num_candidates,
        "candidates": candidates,
        "scores": scores,
    }

