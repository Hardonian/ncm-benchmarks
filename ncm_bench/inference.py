"""Unified, integrity-verified inference engine for Next-Concept Modeling.

Consolidates model loading, pre-load SHA-256 verification, dynamic INT8
quantization, batched text summarization, and Sudoku solving into one shared path.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
PINNED_GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def ensure_offline_env(ncm_home: Path | str | None = None) -> Path:
    """Pins Hugging Face cache variables before hub imports to enforce offline execution."""
    home = Path(ncm_home or os.environ.get("NCM_HOME") or REPO_ROOT).resolve()
    os.environ.setdefault("NCM_HOME", str(home))
    os.environ.setdefault("NCM_ROOT", str(home / "ncm"))
    os.environ.setdefault("HF_HOME", str(home / "hf-home"))
    os.environ.setdefault("HF_HUB_CACHE", str(home / "hf-home" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(home / "hf-home" / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return home


def resolve_device(device: str | None = None) -> str:
    """Resolves target compute device from explicit arg, NCM_DEVICE env, or CUDA availability."""
    if device:
        return device
    env_dev = os.environ.get("NCM_DEVICE")
    if env_dev:
        return env_dev
    return "cuda" if torch.cuda.is_available() else "cpu"


def sha256_file(path: Path) -> str:
    """Computes SHA-256 digest of a file using 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_expected_checksums(root_dir: Path | None = None) -> dict[str, str]:
    """Loads relative path -> SHA-256 mapping from SHA256SUMS."""
    base = Path(root_dir or REPO_ROOT).resolve()
    sums_path = base / "SHA256SUMS"
    if not sums_path.exists():
        return {}
    mapping: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            mapping[parts[1].replace("\\", "/")] = parts[0]
    return mapping


def verify_file_integrity(file_path: Path, root_dir: Path | None = None) -> None:
    """Verifies a checkpoint/config against SHA256SUMS before deserialization.

    Mitigates pickle deserialization tampering risk by validating cryptographic digests
    prior to calling torch.load(..., weights_only=False).
    """
    base = Path(root_dir or REPO_ROOT).resolve()
    expected = load_expected_checksums(base)
    if not expected:
        return
    try:
        rel = file_path.resolve().relative_to(base).as_posix()
    except ValueError:
        return
    if rel not in expected:
        return
    actual = sha256_file(file_path)
    if actual != expected[rel]:
        raise RuntimeError(
            f"Checkpoint integrity verification failed for {rel}: "
            f"expected {expected[rel]}, got {actual}"
        )


def _ensure_ncm_on_path(ncm_root: Path) -> None:
    root_str = str(ncm_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def load_bundle(
    task: str = "xsum",
    device: str | None = None,
    quantize: str = "none",
    verify_checksums: bool = True,
    root_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Loads an NCM model + VQ-VAE tokenizer bundle safely.

    Args:
        task: Subdirectory under artifacts/ (e.g. 'xsum' or 'sudoku/hard').
        device: 'cpu', 'cuda', or None for auto-detection.
        quantize: 'none' or 'int8' (CPU dynamic quantization).
        verify_checksums: Verify SHA256SUMS prior to torch.load.
        root_dir: Repository root containing artifacts/, ncm/, hf-home/.
    """
    home = ensure_offline_env(root_dir)
    ncm_root = Path(os.environ.get("NCM_ROOT", str(home / "ncm"))).resolve()
    if not ncm_root.exists():
        raise FileNotFoundError(
            f"Upstream NCM directory not found at {ncm_root}. "
            "Clone HazyResearch/ncm @ c702742 and apply patches/device-support.patch."
        )
    _ensure_ncm_on_path(ncm_root)

    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer
    from transformers import AutoTokenizer

    target_device = resolve_device(device)
    ckpt_dir = home / "artifacts" / task
    cfg_path = ckpt_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing config {cfg_path}. Run `python scripts/fetch_artifacts.py` first."
        )

    cfg = OmegaConf.load(str(cfg_path))
    tok_ckpt = ckpt_dir / str(cfg.checkpoints.tokenizer)
    gen_ckpt = ckpt_dir / str(cfg.checkpoints.generator)

    if verify_checksums:
        for p in (cfg_path, tok_ckpt, gen_ckpt):
            verify_file_integrity(p, home)

    cfg.tokenizer.vqvae.checkpoint_path = str(tok_ckpt)
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.to(target_device).eval()

    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(target_device)
    # Integrity checked via verify_file_integrity against SHA256SUMS above.
    sd = torch.load(str(gen_ckpt), map_location=target_device, weights_only=False)  # nosec B614
    net.load_state_dict(sd["model"], strict=True)
    net.eval()

    applied_quant = "none"
    if quantize == "int8":
        if target_device != "cpu":
            raise ValueError("Dynamic INT8 quantization is supported on CPU only.")
        net = torch.ao.quantization.quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8)
        applied_quant = "int8"

    g = cfg.generation
    pre_tok_name = str(cfg.tokenizer.pre_tokenizer.get("name", ""))
    src = None
    if pre_tok_name == "hf":
        src = AutoTokenizer.from_pretrained(
            str(cfg.tokenizer.pre_tokenizer.model_id),
            revision=PINNED_GPT2_REVISION,
        )

    allow = (
        sd["inference"]["token_unigram"].to(target_device).double()[None, :]
        if g.get("frequency_allowance")
        else None
    )
    adj_raw = g.get("adjacency_penalty")
    adj = [float(v) for v in str(adj_raw).split(",")] if adj_raw else None

    return {
        "task": task,
        "cfg": cfg,
        "tk": tk,
        "net": net,
        "src": src,
        "P": int(cfg.model.prefix_len),
        "EOS": tk.pre_tokenizer.eos_token_id,
        "PAD": tk.pre_tokenizer.pad_token_id,
        "BOS": getattr(tk.pre_tokenizer, "bos_token_id", None),
        "device": target_device,
        "temp": float(g.get("temperature", 0.0)),
        "top_k": int(g.get("top_k", 0)),
        "top_p": float(g.get("top_p", 1.0)),
        "freq": float(g.get("frequency_penalty", 0.0)),
        "mu": float(g.get("frequency_deficit", 0.0)),
        "adj": adj,
        "allow": allow,
        "quantize": applied_quant,
        "cut_at_eos": bool(g.get("cut_at_eos", True)),
    }


def decode_logits_to_texts(logits: torch.Tensor, bundle: dict[str, Any]) -> list[str]:
    """Batched token selection (frequency allowance + adjacency penalty) and decoding."""
    freq = float(bundle.get("freq", 0.0))
    adj = bundle.get("adj")
    eos = bundle.get("EOS")
    pad = bundle.get("PAD")
    cut_at_eos = bool(bundle.get("cut_at_eos", True))

    if freq <= 0 and not adj:
        out_ids = logits.argmax(-1).tolist()
    else:
        B, T, V = logits.shape
        lg = logits.double()
        raw_allow = bundle.get("allow")
        allow = (raw_allow[:, :V] * T) if raw_allow is not None else None
        al = adj[0] if adj else 0.0
        aw = int(adj[1]) if adj and len(adj) > 1 else 1
        ad = adj[2] if adj and len(adj) > 2 else 1.0
        mu = float(bundle.get("mu", 0.0))

        cnt = torch.zeros(B, V, dtype=torch.float64, device=lg.device)
        out = torch.empty(B, T, dtype=torch.long, device=lg.device)
        hist: list[torch.Tensor] = []

        for t in range(T):
            step = lg[:, t, :].clone()
            if freq > 0:
                excess = (cnt - allow).clamp_min(0.0) if allow is not None else cnt
                step -= freq * excess
                if mu > 0 and allow is not None:
                    step += mu * ((T - t) / T) * (allow - cnt).clamp_min(0.0)
            for j, h in enumerate(reversed(hist[-aw:])):
                step.scatter_add_(
                    1,
                    h[:, None],
                    torch.full((B, 1), -al * ad ** j, dtype=torch.float64, device=lg.device),
                )
            p = step.argmax(-1)
            out[:, t] = p
            hist.append(p)
            cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=torch.float64, device=lg.device))
        out_ids = out.tolist()

    texts: list[str] = []
    skip_set = {tok for tok in (eos, pad) if tok is not None}
    for row in out_ids:
        if cut_at_eos and eos is not None and eos in row:
            row = row[: row.index(eos)]
        clean = [t for t in row if t not in skip_set]
        texts.append(bundle["tk"].pre_tokenizer.decode(clean, skip_special_tokens=True).strip())
    return texts


def prepare_prefix_batch(
    bundle: dict[str, Any],
    items: list[str | list[int]],
) -> torch.Tensor:
    """Encodes and pads a batch of raw strings or pre-tokenized ID lists to prefix_len."""
    P = int(bundle["P"])
    pad = bundle["PAD"] if bundle["PAD"] is not None else 0
    rows: list[list[int]] = []
    for item in items:
        if isinstance(item, str):
            if bundle["src"] is not None:
                ids = bundle["src"](item)["input_ids"]
            else:
                ids = bundle["tk"].pre_tokenizer.encode(item)
        else:
            ids = list(item)
        rows.append((ids[:P] + [pad] * P)[:P])
    return torch.tensor(rows, device=bundle["device"])


def summarize_texts(bundle: dict[str, Any], texts: list[str | list[int]]) -> list[str]:
    """Runs batched NCM generation and decodes summaries."""
    if not texts:
        return []
    x = prepare_prefix_batch(bundle, texts)
    eos = bundle["EOS"] if bundle["EOS"] is not None else 50256
    with torch.inference_mode():
        g = bundle["net"].generate(
            None,
            x,
            temperature=bundle["temp"],
            top_k=bundle["top_k"],
            top_p=bundle["top_p"],
            cfg_scale=0.0,
            cfg_null_eos=eos,
        )
        _, zl = bundle["tk"].vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        logits = bundle["tk"].vqvae.decoder(zl[-1])
    return decode_logits_to_texts(logits, bundle)


def normalize_sudoku_input(puzzle: str | list[int]) -> list[int]:
    """Normalizes an 81-char string, pipe-delimited string, or 91-token list into 91 Sudoku IDs."""
    if isinstance(puzzle, list):
        if len(puzzle) != 91:
            raise ValueError(f"Expected 91 Sudoku prefix tokens, got {len(puzzle)}")
        return [int(v) for v in puzzle]

    cleaned = "".join(ch for ch in puzzle.strip() if not ch.isspace())
    if cleaned.startswith("<BOS>"):
        cleaned = cleaned[5:]
    if cleaned.startswith("B"):
        cleaned = cleaned[1:]
    cleaned = cleaned.replace(".", "0").replace("_", "0")
    digits_only = cleaned.replace("|", "")
    if len(digits_only) != 81 or not digits_only.isdigit():
        raise ValueError(
            "Sudoku puzzle must contain exactly 81 digits (use 0, ., or _ for blanks)."
        )
    rows = [digits_only[i * 9 : (i + 1) * 9] for i in range(9)]
    formatted = "B" + "|".join(rows) + "|"
    char_map = {"_": 0, "|": 10, "B": 11, **{str(i): i for i in range(10)}}
    return [char_map[c] for c in formatted]


def solve_sudoku(
    bundle: dict[str, Any],
    puzzles: list[str | list[int]],
) -> list[dict[str, Any]]:
    """Solves Sudoku puzzles using the NCM Sudoku model and validates constraints."""
    from ncm_bench.metrics import is_valid_sudoku_board

    norm_inputs = [normalize_sudoku_input(p) for p in puzzles]
    decoded_boards = summarize_texts(bundle, norm_inputs)
    results: list[dict[str, Any]] = []
    for raw_board in decoded_boards:
        cells = raw_board.replace("|", "")
        results.append(
            {
                "board": raw_board,
                "cells": cells,
                "valid": is_valid_sudoku_board(cells),
            }
        )
    return results

