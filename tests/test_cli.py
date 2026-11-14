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


def test_cli_verify_artifacts_only_skips_unmounted_data(tmp_path):
    art = tmp_path / "artifacts" / "xsum" / "config.yaml"
    art.parent.mkdir(parents=True)
    art.write_text("cfg: 1\n", encoding="utf-8")
    import hashlib

    h = hashlib.sha256(art.read_bytes()).hexdigest()
    sums = {"artifacts/xsum/config.yaml": h, "data/xsum/passages.pickle": "deadbeef"}
    with (
        patch("ncm_bench.cli.ROOT_DIR", tmp_path),
        patch("scripts.fetch_artifacts.load_checksums", return_value=sums),
        patch.object(sys, "argv", ["ncm-bench", "verify", "--artifacts-only"]),
    ):
        main()


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


def test_cli_longdoc_pii_audit_and_file_io(tmp_path, capsys):
    src = tmp_path / "doc.txt"
    out_file = tmp_path / "summary.json"
    src.write_text(
        ("Email alice@example.com about quarterly earnings in London. " * 40),
        encoding="utf-8",
    )
    with patch("ncm_bench.cli.get_model_bundle") as mock_bundle:
        mock_bundle.return_value = {"task": "xsum", "device": "cpu", "quantize": "none"}
        with patch("ncm_bench.longdoc.summarize_texts", side_effect=lambda b, texts, **kw: ["Email [EMAIL_1] summary."] * len(texts)):
            with patch.object(
                sys,
                "argv",
                [
                    "ncm-bench",
                    "summarize",
                    "--input-file",
                    str(src),
                    "--output-file",
                    str(out_file),
                    "--long-doc",
                    "--redact-pii",
                    "--restore-pii",
                    "--audit-receipt",
                    "--json",
                ],
            ):
                main()
                capsys.readouterr()
                payload = json.loads(out_file.read_text(encoding="utf-8"))
                assert "alice@example.com" in payload["summary"]
                assert payload["audit_receipt"]["pii_redactions"] >= 1


def test_cli_translate_convert_and_train_smoke(tmp_path, capsys):
    with patch("ncm_bench.cli.get_model_bundle", return_value={"mock": True}):
        with patch("ncm_bench.inference.summarize_texts", return_value=["Hello Germany."]):
            with patch.object(sys, "argv", ["ncm-bench", "translate", "--text", "Hallo Deutschland.", "--json"]):
                main()
                out = json.loads(capsys.readouterr().out)
                assert out["translation"] == "Hello Germany."

    with patch("ncm_bench.inference.export_safetensors", return_value={"vqvae": tmp_path / "vq.safetensors"}):
        with patch.object(sys, "argv", ["ncm-bench", "convert-safetensors", "--task", "xsum"]):
            main()
            out = json.loads(capsys.readouterr().out)
            assert "vqvae" in out

    smoke_dir = tmp_path / "smoke"
    with patch.object(
        sys,
        "argv",
        ["ncm-bench", "train-smoke", "--samples", "16", "--s1-steps", "3", "--s2-steps", "3", "--out-dir", str(smoke_dir)],
    ):
        main()
        raw_out = capsys.readouterr().out
        out = json.loads(raw_out[raw_out.index("{") :])
        assert "stage1_final_loss" in out
        assert "stage2_final_loss" in out
        assert (smoke_dir / "tokenizer.pt").exists()
        assert (smoke_dir / "generator.pt").exists()


def test_cli_report_json_and_html(tmp_path, capsys):
    from ncm_bench.longdoc import build_audit_receipt

    r1 = build_audit_receipt("Input doc 1", "Summary 1", {"EMAIL_1": "a@b.com"}, hmac_secret="k")
    r2 = build_audit_receipt("Input doc 2", "Summary 2", {})
    log_path = tmp_path / "audit.jsonl"
    log_path.write_text(json.dumps(r1) + "\n" + json.dumps({"audit_receipt": r2}) + "\n", encoding="utf-8")

    rep_json_path = tmp_path / "report.json"
    with patch.object(sys, "argv", ["ncm-bench", "report", "--input", str(log_path), "--output-file", str(rep_json_path)]):
        main()
    capsys.readouterr()
    rep_data = json.loads(rep_json_path.read_text(encoding="utf-8"))
    assert rep_data["total_docs"] == 2
    assert rep_data["audited_docs"] == 1
    assert rep_data["schema_valid_pct"] == 0.0

    html_path = tmp_path / "report.html"
    with patch.object(
        sys,
        "argv",
        ["ncm-bench", "report", "--input", str(log_path), "--output-file", str(html_path), "--html"],
    ):
        main()
    capsys.readouterr()
    html_text = html_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text
    assert "NCM Sovereign Compliance Report" in html_text
    assert "Total Documents: <strong>2</strong>" in html_text
    assert "Audited Receipts: <strong>1</strong>" in html_text

