"""Tests for NCM vs Autoregressive Token LM, Symbolic, and Extractive comparative benchmarks."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

from ncm_bench.cli import main
from ncm_bench.compare import (
    CausalTokenLM,
    benchmark_step_scaling_ncm_vs_ar,
    run_full_comparison_suite,
)


def test_causal_token_lm_generate_autoregressive_shape() -> None:
    """CausalTokenLM generates exactly gen_len tokens sequentially."""
    model = CausalTokenLM(vocab_size=16, max_len=32, n_embd=32, n_head=4, n_layer=1).eval()
    prefix = torch.randint(0, 16, (2, 6))
    out = model.generate_autoregressive(prefix, gen_len=5)
    assert out.shape == (2, 5)
    assert out.min().item() >= 0
    assert out.max().item() < 16


def test_benchmark_step_scaling_ncm_vs_ar_structure() -> None:
    """Step scaling confirms constant NCM passes vs linear O(T) autoregressive passes."""
    res = benchmark_step_scaling_ncm_vs_ar(seq_lengths=(16, 32))
    assert len(res["rows"]) == 2
    assert res["rows"][0]["ncm_passes"] == 4
    assert res["rows"][0]["ar_passes"] == 16
    assert res["rows"][1]["ncm_passes"] == 4
    assert res["rows"][1]["ar_passes"] == 32
    assert res["rows"][1]["pass_reduction_x"] == 8.0


def test_run_full_comparison_suite_fast_smoke() -> None:
    """Fast comparison suite executes all 4 arenas cleanly without external artifacts."""
    report = run_full_comparison_suite(n_sudoku=4, n_xsum=4, fast=True)
    assert report["schema"] == "ncm-vs-baselines-v1"
    assert report["fast_mode"] is True

    games = report["games_arena"]
    assert games["sudoku_9x9_hard"]["skipped"] is True
    assert "mini_sudoku_4x4" in games
    assert "mastermind_4peg" in games
    assert "dyck2_bracket_closure" in games
    assert (
        games["mini_sudoku_4x4"]["methods"]["ncm_concept_4levels_plus_repair"]["sequential_passes"] == 4
    )
    assert games["mini_sudoku_4x4"]["methods"]["causal_token_lm_16steps"]["sequential_passes"] == 16

    xsum = report["xsum_dataset_arena"]
    assert xsum["dataset"] == "synthetic_news_smoke"
    assert "lead_1_extractive" in xsum["methods"]
    assert "tfidf_centroid_extractive" in xsum["methods"]

    ent = report["enterprise_use_cases"]
    sec = ent["sovereign_compliance_and_finqa"]
    assert sec["ncm_sovereign_pii_protected_pct"] == 100.0
    assert sec["ncm_sovereign_entity_restored_pct"] == 100.0
    assert sec["ncm_detached_sig_verified_pct"] == 100.0
    assert sec["finqa_discrepancy_caught"] is True


def test_cli_compare_fast_writes_json(tmp_path: Path) -> None:
    """`ncm-bench compare --fast` writes a valid comparison JSON receipt."""
    out_file = tmp_path / "compare_fast.json"
    with patch.object(
        sys,
        "argv",
        ["ncm-bench", "compare", "--fast", "--output-file", str(out_file)],
    ):
        main()
    assert out_file.exists()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["schema"] == "ncm-vs-baselines-v1"
    assert "enterprise_use_cases" in payload
