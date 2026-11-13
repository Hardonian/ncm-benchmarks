"""Metrics and statistical verification utilities for NCM benchmarks.

Implements:
1. ROUGE-1, ROUGE-2, ROUGE-L scoring without CUDA GPT-2 dependencies.
2. Sudoku exact match, cell-level accuracy, and validity checking.
3. Wilson score 95% confidence intervals for binomial metrics.
4. Two-size timing isolation (subtracts model load from marginal latency).
"""
import math
from typing import Any

from rouge_score import rouge_scorer


def score_rouge(hyps: list[str], refs: list[str]) -> dict[str, float]:
    """Computes mean ROUGE-1, ROUGE-2, and ROUGE-L F1 scores across hyps and refs."""
    assert len(hyps) == len(refs), f"Length mismatch: {len(hyps)} hyps vs {len(refs)} refs"
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for h, r in zip(hyps, refs, strict=True):
        s = scorer.score(r, h)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    n = max(len(hyps), 1)
    return {
        "rouge1": round(sum(r1) / n * 100.0, 2),
        "rouge2": round(sum(r2) / n * 100.0, 2),
        "rougeL": round(sum(rl) / n * 100.0, 2),
        "n_samples": len(hyps),
    }


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> dict[str, float]:
    """Calculates Wilson score interval for binomial proportions.
    
    Returns:
        dict containing 'p', 'lower', 'upper', and 'margin_pp' (percentage points).
    """
    if n == 0:
        return {"p": 0.0, "lower": 0.0, "upper": 0.0, "margin_pp": 0.0}
    # For 95% confidence, z ~ 1.95996
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-4 else math.sqrt(2.0) * math.erf(confidence)
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * n**2)))
    low = max(0.0, center - spread)
    high = min(1.0, center + spread)
    margin = (high - low) / 2.0 * 100.0
    return {
        "p": round(p * 100.0, 2),
        "lower": round(low * 100.0, 2),
        "upper": round(high * 100.0, 2),
        "margin_pp": round(margin, 2),
    }


def is_valid_sudoku_board(board: str | list[int]) -> bool:
    """Checks whether an 81-cell Sudoku board satisfies all row, column, and 3x3 box constraints.

    Matches upstream `scripts/eval_merge.py exact()` validity logic.
    """
    if isinstance(board, list):
        # Strip pipe separators (10) and BOS (11) if raw token sequence
        cells = "".join(str(x) for x in board if 0 <= int(x) <= 9)
    else:
        cells = board.replace("|", "")
    if len(cells) != 81 or not cells.isdigit() or "0" in cells:
        return False
    rows = [cells[i * 9 : (i + 1) * 9] for i in range(9)]
    cols = ["".join(r[j] for r in rows) for j in range(9)]
    boxes = [
        "".join(rows[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3))
        for br in (0, 3, 6)
        for bc in (0, 3, 6)
    ]
    target = list("123456789")
    return all(sorted(group) == target for group in rows + cols + boxes)


def score_sudoku(preds: list[list[int] | str], solutions: list[list[int] | str]) -> dict[str, Any]:
    """Evaluates Sudoku board predictions against ground truth solutions."""
    assert len(preds) == len(solutions), f"Mismatch: {len(preds)} preds vs {len(solutions)} solutions"
    n = len(preds)
    exact_matches = 0
    total_cells = 0
    correct_cells = 0
    constraint_valid = 0
    format_valid = 0

    def _to_cells(seq: list[int] | str) -> list[int]:
        if isinstance(seq, str):
            return [int(c) if c.isdigit() else 0 for c in seq.replace("|", "")]
        return [int(x) for x in seq if int(x) != 10]

    for pred, sol in zip(preds, solutions, strict=True):
        p_cells = _to_cells(pred)
        s_cells = _to_cells(sol)
        if len(p_cells) == len(s_cells):
            format_valid += 1
        total_cells += len(s_cells)
        matches = sum(1 for p, s in zip(p_cells, s_cells, strict=False) if p == s)
        correct_cells += matches
        if matches == len(s_cells) and len(p_cells) == len(s_cells):
            exact_matches += 1
        if is_valid_sudoku_board(pred):
            constraint_valid += 1

    ci = wilson_ci(exact_matches, n)
    return {
        "n_boards": n,
        "solved_pct": round(exact_matches / max(n, 1) * 100.0, 2),
        "cell_pct": round(correct_cells / max(total_cells, 1) * 100.0, 2),
        "valid_pct": round(constraint_valid / max(n, 1) * 100.0, 2),
        "format_valid_pct": round(format_valid / max(n, 1) * 100.0, 2),
        "solved_count": exact_matches,
        "ci_95": ci,
    }


def two_size_timing(t_small: float, n_small: int, t_large: float, n_large: int) -> dict[str, float]:
    """Calculates load time and marginal per-item time using the two-size timing trick.
    
    T(n) = Load + n * Marginal
    Marginal = (T_large - T_small) / (N_large - N_small)
    Load = T_small - N_small * Marginal
    """
    assert n_large > n_small, "n_large must be strictly greater than n_small"
    marginal = (t_large - t_small) / (n_large - n_small)
    load = max(0.0, t_small - n_small * marginal)
    throughput = 60.0 / marginal if marginal > 0 else 0.0
    return {
        "model_load_s": round(load, 3),
        "marginal_s_per_doc": round(marginal, 3),
        "items_per_min": round(throughput, 1),
    }


def score_bleu(hyps: list[str], refs: list[str]) -> dict[str, Any]:
    """Computes corpus BLEU score (WMT14 De-En evaluation) via sacrebleu."""
    import sacrebleu

    if len(hyps) != len(refs):
        raise ValueError(f"Length mismatch: {len(hyps)} hyps vs {len(refs)} refs")
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return {
        "bleu": round(float(bleu.score), 2),
        "precisions": [round(float(p), 2) for p in bleu.precisions],
        "bp": round(float(bleu.bp), 4),
    }


