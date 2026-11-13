import hashlib
from pathlib import Path

import pytest
import torch

from ncm_bench.inference import (
    decode_logits_to_texts,
    normalize_sudoku_input,
    verify_file_integrity,
)
from ncm_bench.metrics import is_valid_sudoku_board, score_sudoku


def test_verify_file_integrity_passes_and_detects_tampering(tmp_path: Path):
    target_dir = tmp_path / "artifacts"
    target_dir.mkdir(parents=True)
    weights = target_dir / "model.safetensors"
    weights.write_bytes(b"clean-ncm-weights-payload")

    digest = hashlib.sha256(b"clean-ncm-weights-payload").hexdigest()
    (tmp_path / "SHA256SUMS").write_text(
        f"{digest}  artifacts/model.safetensors\n", encoding="utf-8"
    )

    verify_file_integrity(weights, root_dir=tmp_path)

    weights.write_bytes(b"tampered-payload")
    with pytest.raises(RuntimeError, match="Checkpoint integrity verification failed"):
        verify_file_integrity(weights, root_dir=tmp_path)


def test_verify_file_integrity_allows_custom_unlisted_file(tmp_path: Path):
    weights = tmp_path / "custom.pt"
    weights.write_bytes(b"abc")
    (tmp_path / "SHA256SUMS").write_text(
        f"{'0' * 64}  artifacts/other.pt\n", encoding="utf-8"
    )
    verify_file_integrity(weights, root_dir=tmp_path)


def test_normalize_sudoku_input_accepts_multiple_formats():
    board_81 = "530070000" + "600195000" + ("0" * 63)
    tokens_plain = normalize_sudoku_input(board_81)
    assert len(tokens_plain) == 91
    assert tokens_plain[0] == 11  # BOS
    assert tokens_plain[1:10] == [5, 3, 0, 0, 7, 0, 0, 0, 0]
    assert tokens_plain[10] == 10  # row separator
    assert tokens_plain[-1] == 10  # trailing separator

    dots_and_pipes = (
        "53..7....|6..195...|" + (".........|" * 7)
    )
    tokens_dots = normalize_sudoku_input(dots_and_pipes)
    assert tokens_dots == tokens_plain

    assert normalize_sudoku_input(tokens_plain) == tokens_plain

    with pytest.raises(ValueError):
        normalize_sudoku_input("12345")


def test_is_valid_sudoku_board_validates_rows_cols_and_boxes():
    valid_81 = (
        "534678912"
        "672195348"
        "198342567"
        "859761423"
        "426853791"
        "713924856"
        "961537284"
        "287419635"
        "345286179"
    )
    assert is_valid_sudoku_board(valid_81) is True

    # Break first row uniqueness while keeping digits 1-9
    bad_row = "334678912" + valid_81[9:]
    assert is_valid_sudoku_board(bad_row) is False

    # Wrong length / contains zero
    assert is_valid_sudoku_board("0" * 81) is False
    assert is_valid_sudoku_board("123") is False

    res = score_sudoku([valid_81, bad_row], [valid_81, valid_81])
    assert res["n_boards"] == 2
    assert res["solved_count"] == 1
    assert res["solved_pct"] == 50.0
    assert res["valid_pct"] == 50.0


def test_decode_logits_to_texts_strips_eos_and_pad():
    class FakePreTokenizer:
        eos_token_id = 2
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            mapping = {10: "hello", 11: "world", 12: "!"}
            return " ".join(
                mapping.get(i, "") for i in ids if not skip_special_tokens or i not in (0, 2)
            ).strip()

    class FakeTokenizer:
        pre_tokenizer = FakePreTokenizer()

    # Vocab size 15, batch=2, target_len=5
    logits = torch.full((2, 5, 15), -10.0)
    # Row 0: [10, 11, 2 (eos), 12, 0]
    for pos, tok in enumerate([10, 11, 2, 12, 0]):
        logits[0, pos, tok] = 10.0
    # Row 1: [10, 12, 0 (pad), 11, 2]
    for pos, tok in enumerate([10, 12, 0, 11, 2]):
        logits[1, pos, tok] = 10.0

    bundle = {
        "cfg": {"prefix_len": 4},
        "tk": FakeTokenizer(),
        "EOS": 2,
        "PAD": 0,
    }
    texts = decode_logits_to_texts(logits, bundle)
    assert texts == ["hello world", "hello ! world"]


def test_verify_file_integrity_strict_mode_fails_when_missing(tmp_path: Path):
    weights = tmp_path / "artifacts" / "unlisted.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"payload")
    with pytest.raises(RuntimeError, match="SHA256SUMS missing"):
        verify_file_integrity(weights, root_dir=tmp_path, strict=True)

    (tmp_path / "SHA256SUMS").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA256SUMS"):
        verify_file_integrity(weights, root_dir=tmp_path, strict=True)


def test_repair_sudoku_board_fixes_single_missing_cell_and_preserves_clues():
    from unittest.mock import patch

    from ncm_bench.inference import repair_sudoku_board, solve_sudoku

    valid_81 = (
        "534678912"
        "672195348"
        "198342567"
        "859761423"
        "426853791"
        "713924856"
        "961537284"
        "287419635"
        "345286179"
    )
    clue = "0" + valid_81[1:]
    corrupted_pred = "9" + valid_81[1:]
    repaired = repair_sudoku_board(clue, corrupted_pred)
    assert repaired == valid_81
    assert is_valid_sudoku_board(repaired)

    with patch("ncm_bench.inference.summarize_texts", return_value=[corrupted_pred]):
        res = solve_sudoku({"device": "cpu"}, [clue], best_of_k=2, repair=True)
        assert len(res) == 1
        assert res[0]["valid"] is True
        assert res[0]["cells"] == valid_81

