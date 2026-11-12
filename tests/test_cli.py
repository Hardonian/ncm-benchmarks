import json
import sys
from unittest.mock import patch

import pytest

from ncm_bench.cli import main


def test_cli_verify():
    with patch("scripts.fetch_artifacts.load_checksums") as mock_sums:
        mock_sums.return_value = {}
        with patch.object(sys, "argv", ["ncm-bench", "verify"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1


def test_cli_summarize_mocked(capsys):
    with patch("ncm_bench.cli.get_model_bundle") as mock_bundle:
        mock_bundle.return_value = {"mock": True}
        with patch("ncm_bench.inference.summarize_texts") as mock_sum:
            mock_sum.return_value = ["A mocked summary output."]
            with patch.object(sys, "argv", ["ncm-bench", "summarize", "--text", "Sample article text.", "--json"]):
                main()
                mock_sum.assert_called_once()
                out = json.loads(capsys.readouterr().out)
                assert out["summary"] == "A mocked summary output."


def test_cli_mbr_mocked():
    with patch("ncm_bench.cli.get_model_bundle") as mock_bundle:
        mock_bundle.return_value = {"mock": True}
        with patch("ncm_bench.mbr.generate_mbr_summary") as mock_mbr:
            mock_mbr.return_value = {
                "summary": "Best MBR.",
                "selected_index": 0,
                "candidates": ["Best MBR."],
                "scores": [1.0],
            }
            with patch.object(sys, "argv", ["ncm-bench", "mbr", "--text", "Sample text", "--candidates", "2"]):
                main()
                mock_mbr.assert_called_once()


def test_cli_sudoku_single_puzzle_mocked(capsys):
    solved_81 = (
        "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
    )
    with patch("ncm_bench.cli.get_model_bundle") as mock_bundle:
        mock_bundle.return_value = {"mock": True}
        with patch("ncm_bench.inference.solve_sudoku") as mock_solve:
            mock_solve.return_value = [{"board": solved_81, "cells": solved_81, "valid": True}]
            with patch.object(
                sys,
                "argv",
                ["ncm-bench", "sudoku", "--puzzle", "530070000" + "600195000" + ("0" * 63), "--json"],
            ):
                main()
                out = json.loads(capsys.readouterr().out)
                assert out["valid"] is True
                assert out["cells"] == solved_81
