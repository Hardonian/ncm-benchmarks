"""Unit tests for long-document chunking, PII redaction, audit receipts, progressive diff & metrics."""

from __future__ import annotations

from pathlib import Path

from ncm_bench.longdoc import (
    build_audit_receipt,
    chunk_text_for_ncm,
    extract_text_from_file,
    redact_pii,
    restore_pii,
    summarize_long_document,
)
from ncm_bench.mbr import compute_pairwise_cosine, select_mbr_centroid
from ncm_bench.metrics import score_bleu
from ncm_bench.progressive import render_progressive_html


def test_pii_redact_and_restore_roundtrip() -> None:
    src = (
        "Contact Alice via alice@example.com or +1 (555) 123-4567. "
        "SSN 123-45-6789 card 4111-2222-3333-4444 from 192.168.1.42."
    )
    scrubbed, mapping = redact_pii(src)
    assert "alice@example.com" not in scrubbed
    assert "123-45-6789" not in scrubbed
    assert "4111-2222-3333-4444" not in scrubbed
    assert "192.168.1.42" not in scrubbed
    assert "[EMAIL_1]" in scrubbed
    assert restore_pii(scrubbed, mapping) == src


def test_chunk_text_for_ncm_splits_on_sentence_boundaries() -> None:
    text = ". ".join(f"Sentence number {i} has several informative tokens" for i in range(25)) + "."
    chunks = chunk_text_for_ncm(text, max_chunk_chars=240, overlap_chars=40)
    assert len(chunks) >= 3
    assert all(len(c) > 0 for c in chunks)


def test_summarize_long_document_and_audit_receipt(monkeypatch) -> None:
    monkeypatch.setattr(
        "ncm_bench.longdoc.summarize_texts",
        lambda bundle, texts, **kwargs: [f"Summary({len(t)})" for t in texts],
    )
    monkeypatch.setattr(
        "ncm_bench.longdoc.generate_mbr_summary",
        lambda bundle, text, **kwargs: {"summary": f"MBR({len(text)})"},
    )
    long_text = " ".join(["Alpha paragraph sentence." for _ in range(120)])
    bundle = {"task": "xsum", "device": "cpu", "quantize": "none"}

    out = summarize_long_document(bundle, long_text, max_chunk_chars=400, overlap_chars=50)
    assert out["num_chunks"] >= 2
    assert "summary" in out
    assert len(out["chunk_summaries"]) == out["num_chunks"]

    out_mbr = summarize_long_document(bundle, long_text, max_chunk_chars=400, overlap_chars=50, use_mbr=True)
    assert out_mbr["summary"].startswith("MBR(")

    receipt = build_audit_receipt(
        long_text,
        out["summary"],
        bundle,
        pii_redactions=3,
        mode="longdoc-mbr",
    )
    assert receipt["pii_redactions"] == 3
    assert len(receipt["input_sha256"]) == 64
    assert len(receipt["output_sha256"]) == 64


def test_extract_text_from_file_txt(tmp_path: Path) -> None:
    p = tmp_path / "doc.txt"
    p.write_text("Hello enterprise NCM.", encoding="utf-8")
    assert extract_text_from_file(p) == "Hello enterprise NCM."


def test_mbr_cosine_and_bleu_and_progressive_html() -> None:
    cands = [
        "stocks rose sharply on Tuesday after earnings beat estimates",
        "stocks rose sharply on Tuesday following strong earnings",
        "unrelated weather forecast for mountain regions",
    ]
    scores = compute_pairwise_cosine(cands)
    assert len(scores) == 3
    best_cand, best_idx, cand_scores = select_mbr_centroid(cands, utility="cosine")
    assert best_idx in (0, 1)
    assert best_cand == cands[best_idx]
    assert len(cand_scores) == 3

    bleu = score_bleu(["the cat sat on the mat"], ["the cat is on the mat"])
    assert bleu["bleu"] >= 0.0

    html_out = render_progressive_html(
        {
            "num_levels": 2,
            "final_summary": "Hello world today friend",
            "levels": [
                {"level": 0, "codes_at_level": 2, "total_codes_so_far": 2, "text": "Hello", "words": 1},
                {
                    "level": 1,
                    "codes_at_level": 4,
                    "total_codes_so_far": 6,
                    "text": "Hello world today friend",
                    "words": 4,
                },
            ],
        },
        "source",
    )
    assert "tok-added" in html_out
    assert "diff-row" in html_out
    assert "Hello world today friend" in html_out
