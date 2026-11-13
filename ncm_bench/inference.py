"""Unified, integrity-verified inference engine for Next-Concept Modeling.

Consolidates model loading (.safetensors and .pt), pre-load SHA-256 verification,
dtype/quantization selection, batched summarization, and Sudoku solving + repair.
"""
from __future__ import annotations

import hashlib
import json
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
    """Resolves compute device from explicit arg, NCM_DEVICE env, CUDA, or Apple MPS."""
    if device:
        return device
    env_dev = os.environ.get("NCM_DEVICE")
    if env_dev:
        return env_dev
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str = "fp32", device: str = "cpu") -> torch.dtype:
    """Maps CLI/env dtype selection to torch.dtype safely per target device."""
    norm = dtype.lower()
    if norm == "fp16":
        return torch.float16 if device != "cpu" else torch.float32
    if norm == "bf16":
        return torch.bfloat16
    return torch.float32


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


def verify_file_integrity(
    file_path: Path,
    root_dir: Path | None = None,
    *,
    strict: bool = False,
) -> None:
    """Verifies a checkpoint/config against SHA256SUMS before deserialization.

    When strict=True (or NCM_STRICT_CHECKSUMS=1), fails closed if SHA256SUMS is missing
    or does not contain the file.
    """
    strict_env = os.environ.get("NCM_STRICT_CHECKSUMS", "").strip() == "1"
    fail_closed = strict or strict_env

    base = Path(root_dir or REPO_ROOT).resolve()
    expected = load_expected_checksums(base)
    if not expected:
        if fail_closed:
            raise RuntimeError(f"Strict checksum mode enabled but SHA256SUMS missing at {base}")
        return
    try:
        rel = file_path.resolve().relative_to(base).as_posix()
    except ValueError as exc:
        if fail_closed:
            raise RuntimeError(f"Artifact path {file_path} lies outside root {base}") from exc
        return
    if rel not in expected:
        if fail_closed:
            raise RuntimeError(f"Strict checksum mode: {rel} is not listed in SHA256SUMS")
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


def export_safetensors(
    task: str = "xsum",
    *,
    root_dir: Path | str | None = None,
    verify_checksums: bool = True,
) -> dict[str, Path]:
    """Converts SHA-256 verified .pt NCM checkpoints into zero-pickle .safetensors files."""
    from safetensors.torch import save_file

    home = Path(root_dir or REPO_ROOT).resolve()
    ckpt_dir = home / "artifacts" / task
    cfg_path = ckpt_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config {cfg_path}")

    cfg = OmegaConf.load(str(cfg_path))
    tok_pt = ckpt_dir / str(cfg.checkpoints.tokenizer)
    gen_pt = ckpt_dir / str(cfg.checkpoints.generator)

    if verify_checksums:
        for p in (cfg_path, tok_pt, gen_pt):
            verify_file_integrity(p, home)

    tok_sd = torch.load(str(tok_pt), map_location="cpu", weights_only=False)["model"]  # nosec B614
    gen_raw = torch.load(str(gen_pt), map_location="cpu", weights_only=False)  # nosec B614

    # Clone shared/tied tensors so safetensors serializes cleanly without aliasing errors
    tok_tensors = {k: v.detach().clone().contiguous() for k, v in tok_sd.items() if torch.is_tensor(v)}
    gen_tensors = {
        k: v.detach().clone().contiguous() for k, v in gen_raw["model"].items() if torch.is_tensor(v)
    }

    tok_out = ckpt_dir / "tokenizer.safetensors"
    gen_out = ckpt_dir / "generator.safetensors"
    meta_out = ckpt_dir / "inference_meta.json"

    save_file(tok_tensors, str(tok_out))
    save_file(gen_tensors, str(gen_out))

    meta: dict[str, Any] = {}
    if "inference" in gen_raw and "token_unigram" in gen_raw["inference"]:
        meta["token_unigram"] = gen_raw["inference"]["token_unigram"].detach().cpu().tolist()
    meta_out.write_text(json.dumps(meta), encoding="utf-8")

    return {"tokenizer": tok_out, "generator": gen_out, "meta": meta_out}


def load_bundle(
    task: str = "xsum",
    device: str | None = None,
    quantize: str = "none",
    dtype: str = "fp32",
    verify_checksums: bool = True,
    strict_checksums: bool = False,
    prefer_safetensors: bool = True,
    compile_model: bool = False,
    root_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Loads an NCM model + VQ-VAE tokenizer bundle safely (.safetensors preferred over .pt)."""
    home = ensure_offline_env(root_dir)
    ncm_root = Path(os.environ.get("NCM_ROOT", str(home / "ncm"))).resolve()
    if not ncm_root.exists():
        raise FileNotFoundError(
            f"Upstream NCM directory not found at {ncm_root}. "
            "Clone HazyResearch/ncm @ c702742 and apply patches/device-support.patch."
        )
    _ensure_ncm_on_path(ncm_root)

    import hydra
    from models.next_concept import NextConceptTransformer
    from models.vqvae import VQVAE
    from safetensors.torch import load_file as st_load_file
    from tokenization.vqvae_tokenizer import VQVAETokenizer
    from transformers import AutoTokenizer
    from utils.registry import tokenizers as ncm_tokenizers

    target_device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, target_device)
    ckpt_dir = home / "artifacts" / task
    cfg_path = ckpt_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing config {cfg_path}. Run `python scripts/fetch_artifacts.py` first."
        )

    cfg = OmegaConf.load(str(cfg_path))
    tok_pt = ckpt_dir / str(cfg.checkpoints.tokenizer)
    gen_pt = ckpt_dir / str(cfg.checkpoints.generator)
    tok_st = ckpt_dir / "tokenizer.safetensors"
    gen_st = ckpt_dir / "generator.safetensors"
    meta_json = ckpt_dir / "inference_meta.json"

    use_safetensors = (
        prefer_safetensors and tok_st.exists() and gen_st.exists() and meta_json.exists()
    )

    if verify_checksums:
        verify_file_integrity(cfg_path, home, strict=strict_checksums)
        if not use_safetensors:
            verify_file_integrity(tok_pt, home, strict=strict_checksums)
            verify_file_integrity(gen_pt, home, strict=strict_checksums)

    if use_safetensors:
        PreTok = hydra.utils.get_class(ncm_tokenizers[cfg.tokenizer.pre_tokenizer.name])
        pre_tok = PreTok.from_config(cfg.tokenizer.pre_tokenizer)
        vqvae = VQVAE(**cfg.tokenizer.vqvae)
        vqvae.load_state_dict(st_load_file(str(tok_st), device="cpu"), strict=True)
        tk = VQVAETokenizer(pre_tok, vqvae)
        ckpt_format = "safetensors"
    else:
        cfg.tokenizer.vqvae.checkpoint_path = str(tok_pt)
        tk = VQVAETokenizer.from_config(cfg.tokenizer)
        ckpt_format = "pt"

    tk.vqvae = tk.vqvae.to(device=target_device, dtype=torch_dtype).eval()

    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(
        device=target_device, dtype=torch_dtype
    )

    if use_safetensors:
        gen_sd = st_load_file(str(gen_st), device=str(target_device))
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        token_unigram = (
            torch.tensor(meta["token_unigram"], dtype=torch.float32, device=target_device)
            if "token_unigram" in meta
            else None
        )
    else:
        # Integrity verified against SHA256SUMS prior to deserialization.
        sd = torch.load(str(gen_pt), map_location=target_device, weights_only=False)  # nosec B614
        gen_sd = sd["model"]
        token_unigram = sd.get("inference", {}).get("token_unigram")

    net.load_state_dict(gen_sd, strict=True)
    net.eval()

    applied_quant = "none"
    if quantize == "int8":
        if target_device != "cpu":
            raise ValueError("Dynamic INT8 quantization is supported on CPU only.")
        net = torch.ao.quantization.quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8)
        applied_quant = "int8"

    if compile_model and hasattr(torch, "compile"):
        net = torch.compile(net)

    g = cfg.generation
    pre_tok_name = str(cfg.tokenizer.pre_tokenizer.get("name", ""))
    src = None
    if pre_tok_name == "hf":
        model_id = str(cfg.tokenizer.pre_tokenizer.model_id)
        rev = PINNED_GPT2_REVISION if model_id == "gpt2" else None
        src = AutoTokenizer.from_pretrained(model_id, revision=rev)  # nosec B615

    allow = (
        token_unigram.to(target_device).double()[None, :]
        if (g.get("frequency_allowance") and token_unigram is not None)
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
        "dtype": str(torch_dtype).replace("torch.", ""),
        "ckpt_format": ckpt_format,
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


def summarize_texts(
    bundle: dict[str, Any],
    texts: list[str | list[int]],
    *,
    temperature: float | None = None,
) -> list[str]:
    """Runs batched NCM generation and decodes outputs."""
    if not texts:
        return []
    x = prepare_prefix_batch(bundle, texts)
    eos = bundle["EOS"] if bundle["EOS"] is not None else 50256
    temp = bundle["temp"] if temperature is None else temperature
    with torch.inference_mode():
        g = bundle["net"].generate(
            None,
            x,
            temperature=temp,
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


def _sudoku_tokens_to_clues(tokens: list[int]) -> str:
    return "".join(str(t) for t in tokens if 0 <= int(t) <= 9)


def repair_sudoku_board(clue_digits: str, pred_digits: str) -> str:
    """Repairs small Sudoku prediction conflicts via clue clamping and naked-single propagation."""
    from ncm_bench.metrics import is_valid_sudoku_board

    if len(clue_digits) != 81 or len(pred_digits) != 81:
        return pred_digits
    if is_valid_sudoku_board(pred_digits):
        return pred_digits

    grid = list(pred_digits)
    # 1) Restore immutable puzzle clues
    for idx, ch in enumerate(clue_digits):
        if ch in "123456789":
            grid[idx] = ch

    def _peers(idx: int) -> set[int]:
        r, c = divmod(idx, 9)
        br, bc = (r // 3) * 3, (c // 3) * 3
        p = {r * 9 + cc for cc in range(9)} | {rr * 9 + c for rr in range(9)}
        p |= {(br + dr) * 9 + (bc + dc) for dr in range(3) for dc in range(3)}
        p.discard(idx)
        return p

    # 2) Blank out conflicting non-clue cells and iteratively fill naked singles
    conflicted = set()
    for idx in range(81):
        val = grid[idx]
        if val not in "123456789":
            conflicted.add(idx)
            continue
        if clue_digits[idx] == "0":
            for p_idx in _peers(idx):
                if grid[p_idx] == val:
                    conflicted.add(idx)
                    break

    if not conflicted:
        return "".join(grid)
    for idx in conflicted:
        if clue_digits[idx] == "0":
            grid[idx] = "0"

    digits = set("123456789")
    for _ in range(20):
        changed = False
        for idx in range(81):
            if grid[idx] != "0":
                continue
            used = {grid[p] for p in _peers(idx) if grid[p] in digits}
            avail = digits - used
            if len(avail) == 1:
                grid[idx] = next(iter(avail))
                changed = True
        if not changed:
            break

    return "".join(grid)


def _format_board_pipes(cells_81: str) -> str:
    if len(cells_81) != 81:
        return cells_81
    return "|".join(cells_81[i * 9 : (i + 1) * 9] for i in range(9)) + "|"


def solve_sudoku(
    bundle: dict[str, Any],
    puzzles: list[str | list[int]],
    *,
    best_of_k: int = 1,
    temperature: float = 0.4,
    repair: bool = True,
) -> list[dict[str, Any]]:
    """Solves Sudoku puzzles using greedy or Best-of-K sampling + optional constraint repair."""
    from ncm_bench.metrics import is_valid_sudoku_board

    norm_inputs = [normalize_sudoku_input(p) for p in puzzles]
    clues = [_sudoku_tokens_to_clues(toks) for toks in norm_inputs]

    if best_of_k <= 1:
        decoded_boards = summarize_texts(bundle, norm_inputs)
        results: list[dict[str, Any]] = []
        for clue, raw_board in zip(clues, decoded_boards, strict=True):
            cells = raw_board.replace("|", "")
            valid = is_valid_sudoku_board(cells)
            if not valid and repair and len(clue) == 81 and len(cells) == 81:
                repaired = repair_sudoku_board(clue, cells)
                if is_valid_sudoku_board(repaired):
                    cells = repaired
                    raw_board = _format_board_pipes(cells)
                    valid = True
            results.append({"board": raw_board, "cells": cells, "valid": valid})
        return results

    # Greedy attempt first; only resample boards that fail initial constraint verification
    greedy_boards = summarize_texts(bundle, norm_inputs, temperature=0.0)
    results = []
    for norm_in, clue, raw_board in zip(norm_inputs, clues, greedy_boards, strict=True):
        cells = raw_board.replace("|", "")
        valid = is_valid_sudoku_board(cells)
        if not valid:
            cand_boards = summarize_texts(
                bundle, [norm_in] * (best_of_k - 1), temperature=temperature
            )
            for cand in cand_boards:
                cand_cells = cand.replace("|", "")
                if repair and len(clue) == 81 and len(cand_cells) == 81:
                    cand_cells = repair_sudoku_board(clue, cand_cells)
                if is_valid_sudoku_board(cand_cells):
                    cells = cand_cells
                    raw_board = _format_board_pipes(cells)
                    valid = True
                    break
        results.append({"board": raw_board, "cells": cells, "valid": valid})
    return results
