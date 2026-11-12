import pytest
from unittest.mock import patch, MagicMock
from ncm_bench.cli import main
import sys


def test_cli_verify():
    with patch("scripts.fetch_artifacts.load_checksums") as mock_sums:
        mock_sums.return_value = {}
        with patch.object(sys, "argv", ["ncm-bench", "verify"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1


def test_cli_summarize_mocked():
    with patch("ncm_bench.cli.get_model_bundle") as mock_bundle:
        mock_bundle.return_value = {"mock": True}
        with patch("service.summarize_server.summarize") as mock_sum:
            mock_sum.return_value = "A mocked summary output."
            with patch.object(sys, "argv", ["ncm-bench", "summarize", "--text", "Sample article text."]):
                main()
                mock_sum.assert_called_once()


def test_cli_mbr_mocked():
    with patch("ncm_bench.cli.get_model_bundle") as mock_bundle:
        mock_bundle.return_value = {"mock": True}
        with patch("ncm_bench.mbr.generate_mbr_summary") as mock_mbr:
            mock_mbr.return_value = {
                "summary": "Best MBR.",
                "selected_index": 0,
                "candidates": ["Best MBR."],
                "scores": [1.0]
            }
            with patch.object(sys, "argv", ["ncm-bench", "mbr", "--text", "Sample text", "--candidates", "2"]):
                main()
                mock_mbr.assert_called_once()

