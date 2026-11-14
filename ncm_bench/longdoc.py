"""Enterprise document utilities: offline file/PDF ingestion, long-doc map-reduce, PII redaction, and audit receipts."""

from __future__ import annotations

import csv
import email
import hashlib
import hmac
import html
import importlib
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
    ("MRN", re.compile(r"\bMRN[:\-\s]?\d{5,10}\b")),
    ("CARD", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\w)")),
    ("IPV4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def _check_archive_quota(zf: zipfile.ZipFile, max_archive_bytes: int) -> None:
    total_size = sum(max(0, info.file_size) for info in zf.infolist())
    if total_size > max_archive_bytes:
        raise ValueError(
            f"Archive uncompressed size ({total_size} bytes) exceeds max_archive_bytes={max_archive_bytes}"
        )


def _extract_docx_xml_text(xml_raw: str) -> str:
    blocks: list[str] = []
    # Extract structured tables row-by-row first, then remaining body text
    table_pattern = re.compile(r"<w:tbl\b[^>]*>(.*?)</w:tbl>", flags=re.DOTALL)
    pos = 0
    for tbl_match in table_pattern.finditer(xml_raw):
        before = xml_raw[pos : tbl_match.start()]
        runs = re.findall(r"<w:t[^>]*>(.*?)</w:t>", before, flags=re.DOTALL)
        if runs:
            txt = " ".join(html.unescape(r) for r in runs if r.strip()).strip()
            if txt:
                blocks.append(txt)
        for tr_xml in re.findall(r"<w:tr\b[^>]*>(.*?)</w:tr>", tbl_match.group(1), flags=re.DOTALL):
            cells: list[str] = []
            for tc_xml in re.findall(r"<w:tc\b[^>]*>(.*?)</w:tc>", tr_xml, flags=re.DOTALL):
                tc_runs = re.findall(r"<w:t[^>]*>(.*?)</w:t>", tc_xml, flags=re.DOTALL)
                cell_txt = " ".join(html.unescape(r).strip() for r in tc_runs if r.strip()).strip()
                cells.append(cell_txt)
            if any(cells):
                blocks.append(" | ".join(cells))
        pos = tbl_match.end()
    tail = xml_raw[pos:]
    tail_runs = re.findall(r"<w:t[^>]*>(.*?)</w:t>", tail, flags=re.DOTALL)
    if tail_runs:
        txt = " ".join(html.unescape(r) for r in tail_runs if r.strip()).strip()
        if txt:
            blocks.append(txt)
    if blocks:
        return "\n".join(blocks).strip()
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", xml_raw)).split()).strip()


def _extract_xlsx_text(zf: zipfile.ZipFile) -> str:
    names = sorted(zf.namelist())
    shared: list[str] = []
    if "xl/sharedStrings.xml" in names:
        ss_xml = zf.read("xl/sharedStrings.xml").decode("utf-8", errors="replace")
        si_blocks = re.findall(r"<si\b[^>]*>(.*?)</si>", ss_xml, flags=re.DOTALL)
        if si_blocks:
            for si in si_blocks:
                parts = re.findall(r"<t[^>]*>(.*?)</t>", si, flags=re.DOTALL)
                shared.append("".join(html.unescape(p) for p in parts).strip())
        else:
            vals = re.findall(r"<t[^>]*>(.*?)</t>", ss_xml, flags=re.DOTALL)
            shared.extend(html.unescape(v).strip() for v in vals if v.strip())

    used_shared_indices: set[int] = set()
    lines: list[str] = []
    for member in names:
        if not member.startswith("xl/worksheets/sheet"):
            continue
        sheet_xml = zf.read(member).decode("utf-8", errors="replace")
        row_blocks = re.findall(r"<row\b[^>]*>(.*?)</row>", sheet_xml, flags=re.DOTALL)
        if row_blocks:
            for row_xml in row_blocks:
                row_vals: list[str] = []
                for attrs, body in re.findall(r"<c\b([^>]*)>(.*?)</c>", row_xml, flags=re.DOTALL):
                    v_match = re.search(r"<v>(.*?)</v>", body, flags=re.DOTALL)
                    if re.search(r'\bt=["\']s["\']', attrs) and v_match is not None:
                        try:
                            s_idx = int(v_match.group(1).strip())
                            if 0 <= s_idx < len(shared):
                                used_shared_indices.add(s_idx)
                                if shared[s_idx]:
                                    row_vals.append(shared[s_idx])
                                continue
                        except ValueError:
                            pass
                    t_parts = re.findall(r"<t[^>]*>(.*?)</t>", body, flags=re.DOTALL)
                    if t_parts:
                        val = "".join(html.unescape(p) for p in t_parts).strip()
                        if val:
                            row_vals.append(val)
                    elif v_match is not None:
                        val = html.unescape(v_match.group(1)).strip()
                        if val:
                            row_vals.append(val)
                if row_vals:
                    lines.append(" | ".join(row_vals))
        else:
            vals = re.findall(r"<v>(.*?)</v>", sheet_xml, flags=re.DOTALL)
            lines.extend(html.unescape(v).strip() for v in vals if v.strip())

    unreferenced_shared = [s for idx, s in enumerate(shared) if idx not in used_shared_indices and s]
    return "\n".join(unreferenced_shared + lines).strip()


def extract_text_from_file(
    path: Path | str,
    *,
    ocr_fallback=None,
    max_archive_bytes: int = 50 * 1024 * 1024,
) -> str:
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
            _check_archive_quota(zf, max_archive_bytes)
            xml_raw = zf.read("word/document.xml").decode("utf-8", errors="replace")
        return _extract_docx_xml_text(xml_raw)

    if suffix == ".xlsx":
        with zipfile.ZipFile(p, "r") as zf:
            _check_archive_quota(zf, max_archive_bytes)
            return _extract_xlsx_text(zf)

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


def default_contextual_ner(text: str) -> list[tuple[str, str]]:
    """Extracts contextual PERSON names, street ADDRESS spans, and API_KEY secrets without external models."""
    findings: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    person_pat = re.compile(
        r"(?:Patient|Dr\.|Mr\.|Mrs\.|Ms\.|Attending|Signed by|Prepared for|Claimant|Defendant|Contact)"
        r":?\s+([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){1,2})"
    )
    address_pat = re.compile(
        r"\b\d{1,5}[ \t]+[A-Z][a-zA-Z0-9 \t]{2,25}[ \t]+"
        r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Way)\b"
    )
    secret_pat = re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16})\b")

    for label, pat, group_idx in (
        ("PERSON", person_pat, 1),
        ("ADDRESS", address_pat, 0),
        ("API_KEY", secret_pat, 0),
    ):
        for match in pat.finditer(text):
            span = match.group(group_idx).strip()
            item = (label, span)
            if span and item not in seen:
                seen.add(item)
                findings.append(item)

    try:
        presidio_mod = importlib.import_module("presidio_analyzer")
        engine = presidio_mod.AnalyzerEngine()
        for ent in engine.analyze(text=text, language="en", entities=["PERSON", "LOCATION"]):
            span = text[ent.start : ent.end].strip()
            lbl = "ADDRESS" if ent.entity_type == "LOCATION" else str(ent.entity_type).upper()
            item = (lbl, span)
            if span and item not in seen:
                seen.add(item)
                findings.append(item)
    except Exception:
        pass

    return findings


def redact_pii(
    text: str,
    *,
    ner_fn=None,
    enable_contextual_ner: bool = False,
) -> tuple[str, dict[str, str]]:
    """Replaces PII/PHI/secret spans via regex + optional/built-in NER hook with deterministic placeholders."""
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

    active_ner = ner_fn if ner_fn is not None else (default_contextual_ner if enable_contextual_ner else None)
    if active_ner is not None:
        for raw_label, raw_span in active_ner(scrubbed):
            span = str(raw_span)
            lbl = str(raw_label).upper().strip() or "ENTITY"
            if span and span in scrubbed and not span.startswith("["):
                idx = counters.get(lbl, 0) + 1
                counters[lbl] = idx
                token = f"[{lbl}_{idx}]"
                mapping[token] = span
                scrubbed = scrubbed.replace(span, token, 1)

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
