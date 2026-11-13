"""Enterprise document utilities: offline file/PDF ingestion, long-doc map-reduce, PII redaction, and audit receipts."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ncm_bench.inference import summarize_texts
from ncm_bench.mbr import generate_mbr_summary

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\w)")),
    ("IPV4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def extract_text_from_file(path: Path | str) -> str:
    """Extracts UTF-8 plain text from .txt, .md, .json/.jsonl, or .pdf files locally."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(p))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()
    return p.read_text(encoding="utf-8", errors="replace").strip()


def redact_pii(text: str) -> tuple[str, dict[str, str]]:
    """Replaces common PII/PHI spans with deterministic typed placeholders for local scrubbing."""
    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}
    scrubbed = text

    for label, pattern in _PII_PATTERNS:
        def _repl(match: re.Match[str], _lbl: str = label) -> str:
            idx = counters.get(_lbl, 0) + 1
            counters[_lbl] = idx
            token = f"[{_lbl}_{idx}]"
            mapping[token] = match.group(0)
            return token

        scrubbed = pattern.sub(_repl, scrubbed)
    return scrubbed, mapping


def restore_pii(text: str, mapping: dict[str, str]) -> str:
    """Restores previously redacted placeholders in generated outputs when authorized."""
    out = text
    for placeholder, original in mapping.items():
        out = out.replace(placeholder, original)
    return out


def chunk_text_for_ncm(
    text: str,
    max_chunk_chars: int = 1600,
    overlap_chars: int = 160,
) -> list[str]:
    """Splits long documents along sentence/word boundaries into <=512-token friendly chunks."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= max_chunk_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    step = max(max_chunk_chars - overlap_chars, 200)
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chunk_chars)
        if end < len(cleaned):
            split_at = cleaned.rfind(". ", start + max_chunk_chars // 2, end)
            if split_at == -1:
                split_at = cleaned.rfind(" ", start + max_chunk_chars // 2, end)
            if split_at > start:
                end = split_at + 1
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start += step
    return chunks


def summarize_long_document(
    bundle: dict[str, Any],
    text: str,
    *,
    max_chunk_chars: int = 1600,
    overlap_chars: int = 160,
    use_mbr: bool = False,
    num_candidates: int = 4,
) -> dict[str, Any]:
    """Hierarchical map-reduce NCM summarization for multi-page documents."""
    chunks = chunk_text_for_ncm(text, max_chunk_chars=max_chunk_chars, overlap_chars=overlap_chars)
    if not chunks:
        raise ValueError("Input document is empty")

    if len(chunks) == 1:
        if use_mbr:
            mbr = generate_mbr_summary(bundle, chunks[0], num_candidates=num_candidates)
            final = mbr["summary"]
        else:
            final = summarize_texts(bundle, chunks)[0]
        return {"summary": final, "num_chunks": 1, "chunk_summaries": [final]}

    if use_mbr:
        chunk_summaries = [
            generate_mbr_summary(bundle, c, num_candidates=num_candidates)["summary"]
            for c in chunks
        ]
    else:
        chunk_summaries = summarize_texts(bundle, chunks)

    merged = " ".join(chunk_summaries)
    if use_mbr:
        final = generate_mbr_summary(bundle, merged, num_candidates=num_candidates)["summary"]
    else:
        final = summarize_texts(bundle, [merged])[0]

    return {
        "summary": final,
        "num_chunks": len(chunks),
        "chunk_summaries": chunk_summaries,
    }


def build_audit_receipt(
    input_text: str,
    output_text: str,
    bundle: dict[str, Any],
    *,
    pii_redactions: int = 0,
    mode: str = "standard",
) -> dict[str, Any]:
    """Creates a cryptographic SHA-256 compliance audit receipt without storing raw PII."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": bundle.get("task", "xsum"),
        "device": bundle.get("device", "cpu"),
        "quantize": bundle.get("quantize", "none"),
        "dtype": bundle.get("dtype", "fp32"),
        "format": bundle.get("ckpt_format", "pt"),
        "mode": mode,
        "pii_redactions": pii_redactions,
        "input_chars": len(input_text),
        "output_words": len(output_text.split()),
        "input_sha256": hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
        "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
    }

