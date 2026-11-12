import pytest
from ncm_bench.metrics import score_rouge, score_sudoku, wilson_ci, two_size_timing


def test_wilson_ci_known_values():
    # 84.5% on n=200 boards (169 / 200)
    res = wilson_ci(169, 200)
    assert res["p"] == 84.5
    assert 78.0 <= res["lower"] <= 80.0
    assert 88.0 <= res["upper"] <= 90.0
    assert 4.5 <= res["margin_pp"] <= 5.5


def test_two_size_timing():
    # Example: t4 = 15.0s, t16 = 43.0s -> marginal = (43-15)/12 = 2.333s, load = 15 - 4*2.333 = 5.667s
    timing = two_size_timing(15.0, 4, 43.0, 16)
    assert round(timing["marginal_s_per_doc"], 2) == 2.33
    assert round(timing["model_load_s"], 2) == 5.67
    assert 25.0 <= timing["items_per_min"] <= 26.0


def test_score_rouge():
    hyps = ["The quick brown fox jumps over the lazy dog.", "Hello world."]
    refs = ["The fast brown fox jumps over a lazy dog.", "Hello there world."]
    scores = score_rouge(hyps, refs)
    assert "rouge1" in scores
    assert "rouge2" in scores
    assert "rougeL" in scores
    assert scores["rouge1"] > 50.0
    assert scores["rougeL"] > 50.0


def test_score_sudoku():
    sol = list(range(81))
    # 1 perfect board, 1 board with 1 wrong cell
    preds = [list(range(81)), list(range(80)) + [99]]
    sols = [sol, sol]
    res = score_sudoku(preds, sols)
    assert res["n_boards"] == 2
    assert res["solved_pct"] == 50.0
    assert res["solved_count"] == 1
    assert res["cell_pct"] == round(161 / 162 * 100.0, 2)

