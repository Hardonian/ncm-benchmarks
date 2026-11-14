import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.batch_triage import dispatch_document, load_completed_ids, main


def test_load_completed_ids_handles_valid_and_corrupt_lines(tmp_path: Path):
    out_path = tmp_path / "results.jsonl"
    out_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "doc-1", "summary": "ok"}),
                "{corrupt-json-line",
                "",
                json.dumps({"id": "doc-2", "summary": "ok 2"}),
                json.dumps({"no_id": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = load_completed_ids(out_path)
    assert completed == {"doc-1", "doc-2"}


def test_load_completed_ids_missing_file(tmp_path: Path):
    assert load_completed_ids(tmp_path / "does_not_exist.jsonl") == set()


def test_dispatch_document_modes():
    client = MagicMock()
    base = {
        "candidates": 3,
        "redact_pii": True,
        "restore_pii": False,
        "audit_receipt": True,
        "labels": "finance,security",
        "max_level": 2,
        "coarse_only": True,
        "adaptive_compute": True,
    }
    dispatch_document(client, "hello", argparse.Namespace(mode="standard", **base))
    client.summarize.assert_called_once_with("hello")

    dispatch_document(client, "hello", argparse.Namespace(mode="mbr", **base))
    client.mbr_summarize.assert_called_once_with("hello", num_candidates=3)

    dispatch_document(client, "hello", argparse.Namespace(mode="longdoc", **base))
    client.longdoc_summarize.assert_called_once()

    dispatch_document(client, "hello", argparse.Namespace(mode="triage", **base))
    client.triage.assert_called_once_with(
        "hello",
        labels=["finance", "security"],
        max_level=2,
        best_of_k=3,
        coarse_only=True,
        adaptive_compute=True,
        redact_pii=True,
        restore_pii=False,
        audit_receipt=True,
    )


def test_batch_triage_cli_concurrency_and_limit(tmp_path: Path):
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    err_path = tmp_path / "err.jsonl"
    in_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "text": "Urgent breach report on auth gateway."}),
                json.dumps({"id": "b", "text": "Quarterly budget revenue update."}),
                json.dumps({"id": "c", "text": "Ignored beyond limit"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fake_client = MagicMock()
    fake_client.health.return_value = {"model": "xsum", "device": "cpu", "quantize": "int8"}
    fake_client.triage.side_effect = lambda text, **kw: {
        "summary": text[:20],
        "category": "security",
        "priority": "high",
        "confidence": 0.9,
        "schema_valid": True,
        "compute_tier": "coarse_early_exit",
        "audit_receipt": {"sha256": "abc"},
    }

    with patch("scripts.batch_triage.NCMClient", return_value=fake_client):
        with patch.object(
            sys,
            "argv",
            [
                "batch_triage.py",
                "--input",
                str(in_path),
                "--output",
                str(out_path),
                "--errors-out",
                str(err_path),
                "--mode",
                "triage",
                "--coarse-only",
                "--audit-receipt",
                "--concurrency",
                "2",
                "--limit",
                "2",
            ],
        ):
            main()

    lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert {row["id"] for row in lines} == {"a", "b"}
    assert all(row["compute_tier"] == "coarse_early_exit" for row in lines)
