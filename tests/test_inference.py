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

    # Test multi-cell ambiguity (>2 missing cells) via bounded backtracking
    multi_clue = "000" + valid_81[3:]
    multi_pred = "999" + valid_81[3:]
    repaired_multi = repair_sudoku_board(multi_clue, multi_pred)
    assert repaired_multi == valid_81
    assert is_valid_sudoku_board(repaired_multi)

    with patch("ncm_bench.inference.summarize_texts", return_value=[corrupted_pred]):
        res = solve_sudoku({"device": "cpu"}, [clue], best_of_k=2, repair=True)
        assert len(res) == 1
        assert res[0]["valid"] is True
        assert res[0]["cells"] == valid_81


def test_resolve_device_and_dtype_and_nested_load():
    from ncm_bench.inference import resolve_device, resolve_dtype

    assert resolve_device("cpu") == "cpu"
    assert resolve_dtype("fp32", "cpu") == torch.float32
    assert resolve_dtype("bf16", "cpu") == torch.bfloat16
    assert resolve_dtype("fp16", "cpu") == torch.float32


def test_decode_logits_with_frequency_and_adjacency_penalties_and_prefix_batch(tmp_path: Path):
    from types import SimpleNamespace

    from ncm_bench.inference import ensure_offline_env, prepare_prefix_batch

    home = ensure_offline_env(tmp_path)
    assert home == tmp_path.resolve()

    class FakePreTokenizer:
        eos_token_id = 2
        pad_token_id = 0

        def encode(self, _text):
            return [4, 5]

        def decode(self, ids, skip_special_tokens=True):
            vocab = {4: "alpha", 5: "beta", 6: "gamma"}
            return " ".join(vocab.get(i, "") for i in ids if not skip_special_tokens or i not in (0, 2)).strip()

    # Batch of 1, sequence len 3, vocab 8: token 4 starts strong then triggers adjacency/freq penalty
    logits = torch.full((1, 3, 8), -5.0)
    logits[0, :, 4] = 4.0
    logits[0, :, 5] = 3.5
    logits[0, :, 6] = 3.0
    allow = torch.zeros((1, 8))
    allow[0, 5] = 1.0

    bundle = {
        "P": 4,
        "EOS": 2,
        "PAD": 0,
        "device": "cpu",
        "src": lambda text: {"input_ids": [4, 6] if text else []},
        "tk": SimpleNamespace(pre_tokenizer=FakePreTokenizer()),
        "freq": 5.0,
        "mu": 2.0,
        "allow": allow,
        "adj": [4.0, 2, 0.5],
        "cut_at_eos": True,
    }
    texts = decode_logits_to_texts(logits, bundle)
    assert len(texts) == 1 and texts[0]

    batch_src = prepare_prefix_batch(bundle, ["hello", [5, 6]])
    assert batch_src.tolist() == [[4, 6, 0, 0], [5, 6, 0, 0]]

    bundle["src"] = None
    batch_pretok = prepare_prefix_batch(bundle, ["hello"])
    assert batch_pretok.tolist() == [[4, 5, 0, 0]]


def test_export_safetensors_and_load_bundle_mocked(tmp_path: Path, monkeypatch):
    import sys
    import types

    from ncm_bench.inference import export_safetensors, load_bundle

    (tmp_path / "ncm").mkdir(parents=True)
    art_dir = tmp_path / "artifacts" / "xsum"
    art_dir.mkdir(parents=True)

    cfg_yaml = (
        "checkpoints:\n"
        "  tokenizer: tok.pt\n"
        "  generator: gen.pt\n"
        "tokenizer:\n"
        "  pre_tokenizer:\n"
        "    name: hf\n"
        "    model_id: gpt2\n"
        "  vqvae:\n"
        "    dim: 4\n"
        "model:\n"
        "  prefix_len: 4\n"
        "generation:\n"
        "  frequency_allowance: true\n"
        "  adjacency_penalty: '1.0,2.0'\n"
        "  temperature: 0.7\n"
        "  top_k: 4\n"
        "  cut_at_eos: true\n"
    )
    cfg_path = art_dir / "config.yaml"
    cfg_path.write_text(cfg_yaml, encoding="utf-8")

    tok_pt = art_dir / "tok.pt"
    gen_pt = art_dir / "gen.pt"
    torch.save({"model": {"lin.weight": torch.randn(2, 2), "lin.bias": torch.zeros(2)}}, tok_pt)
    torch.save(
        {
            "model": {"lin.weight": torch.randn(2, 2), "lin.bias": torch.zeros(2)},
            "inference": {"token_unigram": torch.ones(4)},
        },
        gen_pt,
    )

    sums_lines = []
    for p, rel in [
        (cfg_path, "artifacts/xsum/config.yaml"),
        (tok_pt, "artifacts/xsum/tok.pt"),
        (gen_pt, "artifacts/xsum/gen.pt"),
    ]:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        sums_lines.append(f"{h}  {rel}")
    (tmp_path / "SHA256SUMS").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")

    exported = export_safetensors("xsum", root_dir=tmp_path, verify_checksums=True)
    assert exported["tokenizer"].exists()
    assert exported["generator"].exists()
    assert exported["meta"].exists()

    class _FakeVQVAE(torch.nn.Module):
        def __init__(self, **_kw):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

    class _FakeGen(torch.nn.Module):
        def __init__(self, vqvae_local=None, config=None):
            super().__init__()
            del vqvae_local, config
            self.lin = torch.nn.Linear(2, 2)

    class _FakePreTok:
        eos_token_id = 2
        pad_token_id = 0
        bos_token_id = 1

        @classmethod
        def from_config(cls, _cfg):
            return cls()

    class _FakeVQTok:
        def __init__(self, pre_tok, vqvae):
            self.pre_tokenizer = pre_tok
            self.vqvae = vqvae

        @classmethod
        def from_config(cls, _cfg):
            return cls(_FakePreTok(), _FakeVQVAE())

    mod_hydra = types.ModuleType("hydra")
    mod_hydra_utils = types.ModuleType("hydra.utils")
    mod_hydra_utils.get_class = lambda _path: _FakePreTok
    mod_hydra.utils = mod_hydra_utils

    mod_models = types.ModuleType("models")
    mod_models_ncm = types.ModuleType("models.next_concept")
    mod_models_ncm.NextConceptTransformer = _FakeGen
    mod_models_vq = types.ModuleType("models.vqvae")
    mod_models_vq.VQVAE = _FakeVQVAE

    mod_tok = types.ModuleType("tokenization")
    mod_tok_vq = types.ModuleType("tokenization.vqvae_tokenizer")
    mod_tok_vq.VQVAETokenizer = _FakeVQTok

    mod_utils = types.ModuleType("utils")
    mod_utils_reg = types.ModuleType("utils.registry")
    mod_utils_reg.tokenizers = {"hf": "fake.PreTok"}

    mod_tf = types.ModuleType("transformers")

    class _FakeAutoTok:
        @staticmethod
        def from_pretrained(_model_id, revision=None):
            return {"revision": revision}

    mod_tf.AutoTokenizer = _FakeAutoTok

    for name, mod in {
        "hydra": mod_hydra,
        "hydra.utils": mod_hydra_utils,
        "models": mod_models,
        "models.next_concept": mod_models_ncm,
        "models.vqvae": mod_models_vq,
        "tokenization": mod_tok,
        "tokenization.vqvae_tokenizer": mod_tok_vq,
        "utils": mod_utils,
        "utils.registry": mod_utils_reg,
        "transformers": mod_tf,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    st_bundle = load_bundle(
        "xsum",
        device="cpu",
        quantize="int8",
        verify_checksums=True,
        prefer_safetensors=True,
        root_dir=tmp_path,
    )
    assert st_bundle["ckpt_format"] == "safetensors"
    assert st_bundle["quantize"] == "int8"
    assert st_bundle["adj"] == [1.0, 2.0]

    pt_bundle = load_bundle(
        "xsum",
        device="cpu",
        quantize="none",
        verify_checksums=True,
        prefer_safetensors=False,
        root_dir=tmp_path,
    )
    assert pt_bundle["ckpt_format"] == "pt"
    assert pt_bundle["quantize"] == "none"


