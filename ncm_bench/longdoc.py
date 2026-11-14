"""Enterprise document utilities: offline file/PDF ingestion, long-doc map-reduce, PII redaction, and audit receipts."""

from __future__ import annotations

import csv
import email
import hashlib
import hmac
import html
import io
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ncm_bench.inference import summarize_texts
from ncm_bench.mbr import generate_mbr_summary

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("API_KEY", re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,})\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\w)")),
    ("IPV4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def extract_text_from_file(path: Path | str, *, ocr_fallback=None) -> str:
    """Extracts UTF-8 plain text from .txt, .md, .json/.jsonl, .html, .csv, .eml, .docx, .xlsx, or .pdf files locally."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(p))
        pages = [page.extract_text() or "" for page in reader.pages]
        joined = "\n\n".join(pages).strip()
        if not joined and ocr_fallback is not None:
            return str(ocr_fallback(p)).strip()
        return joined

    if suffix == ".docx":
        with zipfile.ZipFile(p, "r") as zf:
            xml_raw = zf.read("word/document.xml").decode("utf-8", errors="replace")
        runs = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml_raw, flags=re.DOTALL)
        if runs:
            return " ".join(html.unescape(r) for r in runs if r.strip()).strip()
        return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", xml_raw)).split()).strip()

    if suffix == ".xlsx":
        extracted_cells: list[str] = []
        with zipfile.ZipFile(p, "r") as zf:
            names = sorted(zf.namelist())
            for member in names:
                if member == "xl/sharedStrings.xml" or member.startswith("xl/worksheets/sheet"):
                    xml_raw = zf.read(member).decode("utf-8", errors="replace")
                    if member == "xl/sharedStrings.xml":
                        vals = re.findall(r"<t[^>]*>(.*?)</t>", xml_raw, flags=re.DOTALL)
                    else:
                        vals = re.findall(r"<v>(.*?)</v>", xml_raw, flags=re.DOTALL)
                    extracted_cells.extend(html.unescape(v).strip() for v in vals if v.strip())
        return " ".join(extracted_cells).strip()

    raw = p.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        no_scripts = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
        stripped = re.sub(r"<[^>]+>", " ", no_scripts)
        return " ".join(html.unescape(stripped).split()).strip()
    if suffix == ".csv":
        rows = [
            ", ".join(cell.strip() for cell in row if cell.strip()) for row in csv.reader(io.StringIO(raw))
        ]
        return "\n".join(r for r in rows if r).strip()
    if suffix == ".eml":
        msg = email.message_from_string(raw)
        parts: list[str] = []
        subject = str(msg.get("Subject", "")).strip()
        if subject:
            parts.append(f"Subject: {subject}")
        if msg.is_multipart():
            for part in msg.walk():
                if (
                    part.get_content_type() == "text/plain"
                    and "attachment" not in str(part.get("Content-Disposition", "")).lower()
                ):
                    payload = part.get_payload(decode=True)
                    text_part = (
                        payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                        if isinstance(payload, (bytes, bytearray))
                        else str(part.get_payload() or "")
                    )
                    if text_part.strip():
                        parts.append(text_part.strip())
        else:
            payload = msg.get_payload(decode=True)
            body = (
                payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
                if isinstance(payload, (bytes, bytearray))
                else str(msg.get_payload() or "")
            )
            if body.strip():
                parts.append(body.strip())
        return "\n\n".join(parts).strip()
    return raw.strip()


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
            generate_mbr_summary(bundle, c, num_candidates=num_candidates)["summary"] for c in chunks
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
    hmac_secret: str | None = None,
) -> dict[str, Any]:
    """Creates a cryptographic SHA-256 compliance audit receipt without storing raw PII."""
    receipt: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": str(bundle.get("task", "xsum")),
        "device": str(bundle.get("device", "cpu")),
        "quantize": str(bundle.get("quantize", "none")),
        "dtype": str(bundle.get("dtype", "fp32")),
        "format": str(bundle.get("ckpt_format", "pt")),
        "mode": mode,
        "pii_redactions": pii_redactions,
        "input_chars": len(input_text),
        "output_words": len(output_text.split()),
        "input_sha256": hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
        "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
    }
    secret = hmac_secret if hmac_secret is not None else os.environ.get("NCM_AUDIT_HMAC_KEY")
    if secret:
        canonical = (
            f"{receipt['timestamp_utc']}|{receipt['task']}|{receipt['mode']}|"
            f"{receipt['input_sha256']}|{receipt['output_sha256']}|{receipt['pii_redactions']}"
        )
        receipt["hmac_sha256"] = hmac.new(
            secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256
        ).hexdigest()
    return receipt


def verify_audit_receipt(receipt: dict[str, Any], hmac_secret: str) -> bool:
    """Verifies the HMAC-SHA256 signature on an audit receipt."""
    canonical = (
        f"{receipt['timestamp_utc']}|{receipt['task']}|{receipt['mode']}|"
        f"{receipt['input_sha256']}|{receipt['output_sha256']}|{receipt['pii_redactions']}"
    )
    expected = hmac.new(hmac_secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(str(receipt.get("hmac_sha256", "")), expected)
