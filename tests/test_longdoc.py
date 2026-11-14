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
    verify_audit_receipt,
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


def test_pii_api_key_iban_multiformat_and_hmac_receipt(tmp_path: Path) -> None:
    src = (
        "Key sk-ant-api03-1234567890abcdef123456 and ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456 "
        "plus IBAN DE89370400440532013000 transferred."
    )
    scrubbed, mapping = redact_pii(src)
    assert "sk-ant-api03-" not in scrubbed
    assert "ghp_ABCDEFG" not in scrubbed
    assert "DE89370400440532013000" not in scrubbed
    assert len(mapping) >= 3
    assert restore_pii(scrubbed, mapping) == src

    html_p = tmp_path / "memo.html"
    html_p.write_text(
        "<html><head><style>body{color:red}</style><script>alert(1)</script></head>"
        "<body><h1>Brief</h1><p>Alpha &amp; Beta report.</p></body></html>",
        encoding="utf-8",
    )
    html_txt = extract_text_from_file(html_p)
    assert "Brief" in html_txt and "Alpha & Beta report." in html_txt
    assert "alert(1)" not in html_txt

    csv_p = tmp_path / "ledger.csv"
    csv_p.write_text("dept,amount\nFinance,480000\nSecurity,12000\n", encoding="utf-8")
    csv_txt = extract_text_from_file(csv_p)
    assert "dept, amount" in csv_txt and "Finance, 480000" in csv_txt

    eml_p = tmp_path / "alert.eml"
    eml_p.write_text(
        "From: ops@example.com\nTo: sec@example.com\nSubject: Incident Notice\n"
        "MIME-Version: 1.0\nContent-Type: multipart/alternative; boundary=\"b1\"\n\n"
        "--b1\nContent-Type: text/plain; charset=utf-8\n\nPlain body alert.\n"
        "--b1\nContent-Type: text/html; charset=utf-8\n\n<p>HTML body</p>\n--b1--\n",
        encoding="utf-8",
    )
    eml_txt = extract_text_from_file(eml_p)
    assert "Subject: Incident Notice" in eml_txt
    assert "Plain body alert." in eml_txt

    receipt = build_audit_receipt(src, "Sanitized summary", mapping, hmac_secret="secret-key-21")
    assert "hmac_sha256" in receipt
    assert verify_audit_receipt(receipt, "secret-key-21") is True
    assert verify_audit_receipt(receipt, "wrong-key") is False

    tampered = dict(receipt)
    tampered["output_sha256"] = "0" * 64
    assert verify_audit_receipt(tampered, "secret-key-21") is False

    unsigned = build_audit_receipt(src, "Sanitized summary", mapping)
    assert verify_audit_receipt(unsigned, "secret-key-21") is False
