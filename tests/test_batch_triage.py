import json
from pathlib import Path

from scripts.batch_triage import load_completed_ids


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

