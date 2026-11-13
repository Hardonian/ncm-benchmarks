"""Cross-platform Python artifact and dataset fetcher for NCM benchmarks.

Fetches pinned checkpoints and datasets from Hugging Face Hub,
places them in artifacts/ and data/, and verifies SHA-256 digests against SHA256SUMS.
Supports default verified artifacts (xsum, sudoku/hard) as well as optional upstream tasks
(--tasks wmt14 sudoku/easy sudoku/medium lm1b/small).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
HF_HOME = ROOT_DIR / "hf-home"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

CKPT_REPO = "hazyresearch/ncm-checkpoints"
DATA_REPO = "hazyresearch/ncm-datasets"

CKPT_REVISION = "d342ce8aaea58830eef3b8ad473664673e41f692"
DATA_REVISION = "6580ba41a4b0a976dde876fbe84d25327617b0e6"

TASK_SPECS: dict[str, dict[str, list[str]]] = {
    "xsum": {
        "ckpts": ["xsum/config.yaml", "xsum/generator.pt", "xsum/tokenizer.pt"],
        "data": ["xsum/passages.pickle", "xsum/reference_summaries.pickle"],
        "tokenizers": ["gpt2"],
    },
    "sudoku/hard": {
        "ckpts": [
            "sudoku/hard/config.yaml",
            "sudoku/hard/generator.pt",
            "sudoku/hard/tokenizer.pt",
        ],
        "data": ["sudoku/hard/puzzles.pickle", "sudoku/hard/solutions.pickle"],
        "tokenizers": [],
    },
    "sudoku/easy": {
        "ckpts": [
            "sudoku/easy/config.yaml",
            "sudoku/easy/generator.pt",
            "sudoku/easy/tokenizer.pt",
        ],
        "data": ["sudoku/easy/puzzles.pickle", "sudoku/easy/solutions.pickle"],
        "tokenizers": [],
    },
    "sudoku/medium": {
        "ckpts": [
            "sudoku/medium/config.yaml",
            "sudoku/medium/generator.pt",
            "sudoku/medium/tokenizer.pt",
        ],
        "data": ["sudoku/medium/puzzles.pickle", "sudoku/medium/solutions.pickle"],
        "tokenizers": [],
    },
    "wmt14": {
        "ckpts": ["wmt14/config.yaml", "wmt14/generator.pt", "wmt14/tokenizer.pt"],
        "data": ["wmt14/sources.pickle", "wmt14/reference_translations.pickle"],
        "tokenizers": ["Qwen/Qwen3-0.6B"],
    },
    "lm1b/small": {
        "ckpts": ["lm1b/tokenizer.pt", "lm1b/small/config.yaml", "lm1b/small/generator.pt"],
        "data": [],
        "tokenizers": ["bert-base-uncased"],
    },
    "cnn_dailymail": {
        "ckpts": ["cnn_dailymail/config.yaml", "cnn_dailymail/generator.pt", "cnn_dailymail/tokenizer.pt"],
        "data": ["cnn_dailymail/passages.pickle", "cnn_dailymail/reference_summaries.pickle"],
        "tokenizers": ["gpt2"],
    },
    "squad": {
        "ckpts": ["squad/config.yaml", "squad/generator.pt", "squad/tokenizer.pt"],
        "data": ["squad/questions.pickle", "squad/contexts.pickle", "squad/answers.pickle"],
        "tokenizers": ["gpt2"],
    },
    "gsm8k": {
        "ckpts": ["gsm8k/config.yaml", "gsm8k/generator.pt", "gsm8k/tokenizer.pt"],
        "data": ["gsm8k/questions.pickle", "gsm8k/answers.pickle"],
        "tokenizers": ["gpt2"],
    },
    "humaneval": {
        "ckpts": ["humaneval/config.yaml", "humaneval/generator.pt", "humaneval/tokenizer.pt"],
        "data": ["humaneval/problems.pickle", "humaneval/solutions.pickle"],
        "tokenizers": ["gpt2"],
    },
}

# Preserve module-level lists for backward compatibility
CKPT_FILES = TASK_SPECS["xsum"]["ckpts"] + TASK_SPECS["sudoku/hard"]["ckpts"]
DATA_FILES = TASK_SPECS["xsum"]["data"] + TASK_SPECS["sudoku/hard"]["data"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_checksums() -> dict[str, str]:
    sums_file = ROOT_DIR / "SHA256SUMS"
    if not sums_file.exists():
        return {}
    out = {}
    for line in sums_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            out[parts[1].replace("\\", "/")] = parts[0]
    return out


def main():
    parser = argparse.ArgumentParser(description="Fetch pinned NCM checkpoints and datasets.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["xsum", "sudoku/hard"],
        choices=sorted(TASK_SPECS.keys()),
        help="Tasks to fetch (default: xsum sudoku/hard; extended: wmt14 sudoku/easy sudoku/medium lm1b/small cnn_dailymail squad gsm8k humaneval)",
    )
    args = parser.parse_args()

    print(f"[fetch] Root dir: {ROOT_DIR}")
    expected_sums = load_checksums()

    ckpts_to_fetch: list[str] = []
    data_to_fetch: list[str] = []
    tokenizers_to_fetch: list[str] = []
    for task in args.tasks:
        spec = TASK_SPECS[task]
        ckpts_to_fetch.extend(spec["ckpts"])
        data_to_fetch.extend(spec["data"])
        for tok in spec["tokenizers"]:
            if tok not in tokenizers_to_fetch:
                tokenizers_to_fetch.append(tok)

    artifacts_dir = ROOT_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    for fname in ckpts_to_fetch:
        target = artifacts_dir / fname
        rel_path = f"artifacts/{fname}"
        if target.exists() and rel_path in expected_sums and sha256_file(target) == expected_sums[rel_path]:
            print(f"[fetch] Checkpoint verified: {rel_path}")
            continue
        print(f"[fetch] Downloading checkpoint: {fname}...")
        downloaded = hf_hub_download(
            repo_id=CKPT_REPO, filename=fname, revision=CKPT_REVISION, local_dir=str(artifacts_dir)
        )
        print(f"[fetch] Downloaded {fname} -> {downloaded}")

    data_dir = ROOT_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    for fname in data_to_fetch:
        target = data_dir / fname
        rel_path = f"data/{fname}"
        if target.exists() and rel_path in expected_sums and sha256_file(target) == expected_sums[rel_path]:
            print(f"[fetch] Dataset verified: {rel_path}")
            continue
        print(f"[fetch] Downloading dataset: {fname}...")
        downloaded = hf_hub_download(
            repo_id=DATA_REPO,
            repo_type="dataset",
            filename=fname,
            revision=DATA_REVISION,
            local_dir=str(data_dir),
        )
        print(f"[fetch] Downloaded {fname} -> {downloaded}")

    for tok_id in tokenizers_to_fetch:
        print(f"[fetch] Caching HF tokenizer {tok_id}...")
        rev = "607a30d783dfa663caf39e06633721c8d4cfcd7e" if tok_id == "gpt2" else None
        AutoTokenizer.from_pretrained(tok_id, revision=rev, cache_dir=str(HF_HOME / "hub"))  # nosec B615

    print("\n[fetch] Verifying pinned SHA256SUMS...")
    all_ok = True
    for rel_path, exp_hash in expected_sums.items():
        p = ROOT_DIR / rel_path
        if not p.exists():
            print(f"[ERROR] Missing file: {rel_path}")
            all_ok = False
            continue
        actual_hash = sha256_file(p)
        if actual_hash == exp_hash:
            print(f"[OK] {rel_path}")
        else:
            print(f"[MISMATCH] {rel_path}: expected {exp_hash}, got {actual_hash}")
            all_ok = False

    if not all_ok:
        sys.exit(1)
    print("\nAll requested artifacts and pinned SHA256SUMS verified!")


if __name__ == "__main__":
    main()
