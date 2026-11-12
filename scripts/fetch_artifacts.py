"""Cross-platform Python artifact and dataset fetcher for NCM benchmarks.

Fetches the exact pinned checkpoints and datasets from Hugging Face Hub,
places them in artifacts/ and data/, and verifies their sha256 against SHA256SUMS.
Runs seamlessly on Windows, Linux, and macOS without requiring bash or WSL.
"""
import hashlib
import os
import sys
from pathlib import Path

# Pin cache environment variables before importing huggingface_hub
ROOT_DIR = Path(__file__).resolve().parent.parent
HF_HOME = ROOT_DIR / "hf-home"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

CKPT_REPO = "hazyresearch/ncm-checkpoints"
DATA_REPO = "hazyresearch/ncm-datasets"

CKPT_REVISION = "d342ce8aaea58830eef3b8ad473664673e41f692"  # hazyresearch/ncm-checkpoints @ 2026-11-11
DATA_REVISION = "6580ba41a4b0a976dde876fbe84d25327617b0e6"   # hazyresearch/ncm-datasets @ 2026-11-11

CKPT_FILES = [
    "xsum/config.yaml",
    "xsum/generator.pt",
    "xsum/tokenizer.pt",
    "sudoku/hard/config.yaml",
    "sudoku/hard/generator.pt",
    "sudoku/hard/tokenizer.pt",
]

DATA_FILES = [
    "xsum/passages.pickle",
    "xsum/reference_summaries.pickle",
    "sudoku/hard/puzzles.pickle",
    "sudoku/hard/solutions.pickle",
]


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
    print(f"[fetch] Root dir: {ROOT_DIR}")
    expected_sums = load_checksums()

    # 1. Checkpoints
    artifacts_dir = ROOT_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    for fname in CKPT_FILES:
        target = artifacts_dir / fname
        rel_path = f"artifacts/{fname}"
        if target.exists() and rel_path in expected_sums and sha256_file(target) == expected_sums[rel_path]:
            print(f"[fetch] Checkpoint verified: {rel_path}")
            continue
        print(f"[fetch] Downloading checkpoint: {fname}...")
        downloaded = hf_hub_download(repo_id=CKPT_REPO, filename=fname, revision=CKPT_REVISION, local_dir=str(artifacts_dir))
        print(f"[fetch] Downloaded {fname} -> {downloaded}")

    # 2. Datasets
    data_dir = ROOT_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    for fname in DATA_FILES:
        target = data_dir / fname
        rel_path = f"data/{fname}"
        if target.exists() and rel_path in expected_sums and sha256_file(target) == expected_sums[rel_path]:
            print(f"[fetch] Dataset verified: {rel_path}")
            continue
        print(f"[fetch] Downloading dataset: {fname}...")
        downloaded = hf_hub_download(repo_id=DATA_REPO, repo_type="dataset", filename=fname, revision=DATA_REVISION, local_dir=str(data_dir))
        print(f"[fetch] Downloaded {fname} -> {downloaded}")

    # 3. Pinned GPT-2 Tokenizer (needed for offline inference)
    print("[fetch] Downloading and caching pinned GPT-2 tokenizer...")
    AutoTokenizer.from_pretrained(
        "gpt2",
        revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
        cache_dir=str(HF_HOME / "hub")
    )
    print("[fetch] Pinned GPT-2 tokenizer cached.")

    # 4. Checksum verification
    print("\n[fetch] Verifying SHA256 checksums...")
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

    if all_ok:
        print("\nAll files successfully fetched and verified against SHA256SUMS!")
    else:
        print("\nVerification failed for one or more files.")
        sys.exit(1)


if __name__ == "__main__":
    main()

