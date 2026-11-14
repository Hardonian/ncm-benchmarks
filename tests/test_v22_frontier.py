"""Comprehensive unit tests for v2.2.0 frontier roadmap capabilities across all 10 pillars."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from ncm_bench.accel import (
    compile_static_concept_decoder,
    dequantize_weights_int4_groupwise,
    export_ncm_onnx_manifest,
    fused_residual_vq_lookup,
    probe_accelerator_capabilities,
    quantize_weights_int4_groupwise,
)
from ncm_bench.cli import main
from ncm_bench.compliance import (
    AuditSinkSQLite,
    decrypt_field_envelope,
    encrypt_field_envelope,
    export_evidence_bundle,
    generate_signing_keypair,
    sign_detached_receipt,
    verify_detached_receipt,
    verify_evidence_bundle,
)
from ncm_bench.frontier import (
    encode_layout_text_joint_concepts,
    estimate_energy_telemetry,
    evaluate_legal_clause_redaction,
    extract_python_ast_concepts,
    generate_synthetic_dpo_preferences,
    run_longcontext_summarization_harness,
    score_cross_lingual_concept_alignment,
    validate_receipt_submission,
    verify_finqa_table_arithmetic,
)
from ncm_bench.longdoc import extract_text_from_file
from ncm_bench.mbr import select_diverse_mbr_centroid
from ncm_bench.reasoning import (
    adaptive_entropy_early_exit,
    apply_json_grammar_mask,
    solve_symbolic_constraints,
    speculative_coarse_to_fine_verify,
)
from ncm_bench.triage import ConceptClassifier
from ncm_train import (
    CallableRewardPlugin,
    CodebookMonitor,
    DDPConfig,
    RegexRewardPlugin,
    Stage1Config,
    Stage2Config,
    inject_concept_lora,
    train_stage1_vqvae,
    train_stage2_ncm,
    wrap_model_distributed,
)
from service.client import (
    AsyncNCMClient,
    DynamicBatchCoalescer,
    HybridCircuitBreakerClient,
    NCMClient,
    decode_binary_frame,
    encode_binary_frame,
)
from service.summarize_server import create_app


def test_pillar1_docx_xlsx_and_ocr_fallback(tmp_path: Path) -> None:
    docx_p = tmp_path / "contract.docx"
    with zipfile.ZipFile(docx_p, "w") as zf:
        zf.writestr(
            "word/document.xml",
            "<w:document><w:body><w:p><w:r><w:t>Sovereign Contract Clause</w:t></w:r></w:p></w:body></w:document>",
        )
    assert "Sovereign Contract Clause" in extract_text_from_file(docx_p)

    xlsx_p = tmp_path / "metrics.xlsx"
    with zipfile.ZipFile(xlsx_p, "w") as zf:
        zf.writestr("xl/sharedStrings.xml", "<sst><si><t>Revenue</t></si><si><t>Margin</t></si></sst>")
        zf.writestr("xl/worksheets/sheet1.xml", "<worksheet><sheetData><row><c><v>480000</v></c></row></sheetData></worksheet>")
    xlsx_txt = extract_text_from_file(xlsx_p)
    assert "Revenue" in xlsx_txt and "480000" in xlsx_txt

    pdf_p = tmp_path / "scanned.pdf"
    pdf_p.write_bytes(b"%PDF-1.4\n%%EOF\n")
    with patch("pypdf.PdfReader", side_effect=lambda _s: type("R", (), {"pages": []})()):
        assert extract_text_from_file(pdf_p, ocr_fallback=lambda p: f"OCR:{p.name}") == "OCR:scanned.pdf"


def test_pillar2_compliance_detached_sqlite_envelope_and_merkle(tmp_path: Path) -> None:
    priv, _pub = generate_signing_keypair(b"ncm-test-seed")
    receipt = {"input_sha256": "abc123", "output_sha256": "def456", "pii_redactions": 2}
    sig = sign_detached_receipt(receipt, priv)
    assert verify_detached_receipt(receipt, sig) is True
    assert verify_detached_receipt({"input_sha256": "tampered"}, sig) is False

    sink = AuditSinkSQLite(tmp_path / "audit.db")
    h1 = sink.append(receipt)
    h2 = sink.append({"input_sha256": "xyz", "output_sha256": "uvw", "pii_redactions": 0})
    assert len(h1) == 64 and h1 != h2
    assert sink.verify_chain() is True
    assert len(sink.list_receipts()) == 2

    env = encrypt_field_envelope("Confidential board memo", "master-key-32")
    assert decrypt_field_envelope(env, "master-key-32") == "Confidential board memo"
    bad_env = dict(env, tag="0" * 64)
    with pytest.raises(ValueError, match="Envelope authentication failed"):
        decrypt_field_envelope(bad_env, "master-key-32")

    bundle_p = tmp_path / "evidence.json"
    export_evidence_bundle([receipt], bundle_p, hmac_secret="audit-hmac", frameworks=["SOC2", "GDPR"])
    ver_ok = verify_evidence_bundle(bundle_p, hmac_secret="audit-hmac")
    assert ver_ok["valid"] is True and ver_ok["total_receipts"] == 1
    ver_bad = verify_evidence_bundle(bundle_p, hmac_secret="wrong-secret")
    assert ver_bad["valid"] is False


def test_pillar3_and_pillar4_conformal_diverse_mbr_and_reasoning() -> None:
    clf = ConceptClassifier()
    s1 = {"histogram": {1: 1.0, 2: 0.2}}
    s2 = {"histogram": {3: 1.0, 4: 0.2}}
    clf.fit([s1, s2], ["security", "finance"])
    q_hat = clf.calibrate_conformal([s1, s2], ["security", "finance"], alpha=0.10)
    assert 0.0 <= q_hat <= 1.0
    pset = clf.predict_conformal_set({"histogram": {1: 0.7, 3: 0.7}}, q_hat=0.8)
    assert "security" in pset["prediction_set"]

    diverse = select_diverse_mbr_centroid(
        ["markets rallied sharply today", "markets rallied sharply today", "storm hits coastal town"],
        cluster_hashes=["h1", "h1", "h2"],
        utility="jaccard",
    )
    assert diverse["diverse_pool_size"] == 2

    draft = torch.tensor([[1, 2, 0, 1, 2, 3]])
    lvl0 = torch.zeros(1, 2, 4)
    lvl0[0, 0, 1] = 8.0
    lvl0[0, 1, 2] = 8.0
    lvl1 = torch.zeros(1, 4, 4)
    lvl1[0, :, 3] = 8.0
    spec = speculative_coarse_to_fine_verify(draft, [lvl0, lvl1])
    assert spec["accepted_levels"] >= 1
    assert spec["verified_codes"].shape == draft.shape

    masked = apply_json_grammar_mask(torch.zeros(1, 5), [1, 3])
    assert masked[0, 0].item() < -1e8 and masked[0, 1].item() == 0.0

    sat_out = solve_symbolic_constraints([(1, 2), (-1, 3), (-2, -3)], task="sat")
    assert sat_out["satisfiable"] is True
    sudoku_board = [
        5, 3, 4, 6, 7, 8, 9, 1, 2,
        6, 7, 2, 1, 9, 5, 3, 4, 8,
        1, 9, 8, 3, 4, 2, 5, 6, 7,
        8, 5, 9, 7, 6, 1, 4, 2, 3,
        4, 2, 6, 8, 5, 3, 7, 9, 1,
        7, 1, 3, 9, 2, 4, 8, 5, 6,
        9, 6, 1, 5, 3, 7, 2, 8, 4,
        2, 8, 7, 4, 1, 9, 6, 3, 5,
        3, 4, 5, 2, 8, 6, 1, 7, 9,
    ]
    sud_out = solve_symbolic_constraints(sudoku_board, task="sudoku")
    assert sud_out["satisfiable"] is True

    confident_logits = torch.full((1, 4, 8), -8.0)
    confident_logits[..., 0] = 8.0
    ee = adaptive_entropy_early_exit([confident_logits, confident_logits, confident_logits], entropy_threshold=0.5, min_levels=2)
    assert ee["exit_level"] == 1 and ee["passes_saved"] == 1


def test_pillar5_and_pillar6_adapters_quantization_onnx_and_accel(tmp_path: Path) -> None:
    lin_model = torch.nn.Sequential(torch.nn.Linear(8, 8))
    dist_info = wrap_model_distributed(lin_model, DDPConfig())
    assert dist_info["mode"] == "single_process"

    lora_res = inject_concept_lora(torch.nn.ModuleList([torch.nn.Linear(8, 4)]), rank=2)
    assert lora_res["adapters_injected"] == 1 and lora_res["trainable_params"] > 0

    mon = CodebookMonitor(vocab_size=8, ema_decay=0.5)
    stats = mon.update(torch.tensor([0, 0, 1, 1]))
    assert 0.0 < stats["utilization_ratio"] <= 1.0
    cb = torch.randn(8, 4)
    revived = mon.reset_dead_codes(cb, torch.ones(4, 4), threshold=0.2)
    assert revived >= 0

    rx = RegexRewardPlugin([r"CVE-\d+"], forbidden_patterns=[r"ERROR"])
    assert rx.score_texts(["Patched CVE-2026-1", "ERROR failed"]).tolist() == [1.0, 0.0]
    cr = CallableRewardPlugin(lambda toks: 1.0 if 0 in toks else 0.0)
    logits = torch.zeros(1, 3, 4)
    logits[..., 0] = 5.0
    assert float(cr(logits, torch.zeros(1, 2)).item()) == 1.0

    w = torch.randn(32, 32)
    pack = quantize_weights_int4_groupwise(w, group_size=16)
    rec = dequantize_weights_int4_groupwise(pack)
    assert pack["compression_ratio"] > 3.0
    assert rec.shape == w.shape and (w - rec).abs().mean().item() < 0.25

    idx, q = fused_residual_vq_lookup(torch.randn(2, 4, 8), torch.randn(16, 8))
    assert idx.shape == (2, 4) and q.shape == (2, 4, 8)

    manifest = export_ncm_onnx_manifest(lin_model, tmp_path / "ncm_onnx.json")
    assert manifest["param_count"] == 72 and "CPUExecutionProvider" in manifest["providers_detected"]
    comp = compile_static_concept_decoder(lin_model)
    assert "compiled" in comp
    hw = probe_accelerator_capabilities()
    assert hw["active_device"] in ("cpu", "cuda", "mps")


@pytest.mark.anyio
async def test_pillar7_async_transport_traceparent_and_circuit_breaker() -> None:
    app = create_app({"net": None, "device": "cpu"})
    client = TestClient(app)
    r = client.get("/health", headers={"traceparent": "00-11112222333344445555666677778888-1111222233334444-01"})
    assert r.status_code == 200
    assert r.headers["X-Trace-ID"] == "11112222333344445555666677778888"

    async def _mock_async_post(path: str, payload: dict) -> dict:
        if path == "/triage":
            return {"category": "security", "confidence": 0.91, "escalate_to_frontier": False, "summary": payload["text"]}
        return {"summary": "Async summary", "latency_ms": 1.2}

    aclient = AsyncNCMClient("http://test", request_fn=_mock_async_post)
    out = await aclient.summarize("Test document")
    assert out.summary == "Async summary"
    tout = await aclient.triage("Alert")
    assert tout.category == "security"

    frame = encode_binary_frame({"ok": True, "score": 0.99})
    assert decode_binary_frame(frame) == {"ok": True, "score": 0.99}

    coalescer = DynamicBatchCoalescer(lambda batch: [{"text": b.upper()} for b in batch], max_batch_size=2)
    batched = coalescer.process(["a", "b", "c"])
    assert batched == [{"text": "A"}, {"text": "B"}, {"text": "C"}] and coalescer.batches_executed == 2

    fake_local = NCMClient("http://127.0.0.1:8031")
    with patch.object(fake_local, "triage", side_effect=RuntimeError("down")):
        cb = HybridCircuitBreakerClient(fake_local, lambda t: {"summary": f"fallback:{t}"}, failure_threshold=1)
        res1 = cb.triage_with_escalation("urgent")
        res2 = cb.triage_with_escalation("urgent")
    assert res1["routed_via"] == "error_fallback"
    assert res2["routed_via"] == "circuit_open_fallback"


def test_pillar9_and_pillar10_frontier_harnesses_and_cli(tmp_path: Path, capsys) -> None:
    torch.manual_seed(3)
    vocab, seq_len = 16, 16
    seqs = torch.randint(0, vocab, (16, seq_len), dtype=torch.long)
    pfxs = torch.randint(0, vocab, (16, seq_len), dtype=torch.long)
    s1 = Stage1Config(in_vocab_size=vocab, context_length=seq_len, embed_dim=32, levels=(2, 4, 8, 16), vq_vocab_size=16, n_layers=1, batch_size=8, steps=2, device="cpu")
    vqvae = train_stage1_vqvae(seqs, s1)["model"]
    s2 = Stage2Config(prefix_len=seq_len, n_layer=1, n_head=2, n_embd=32, batch_size=8, steps=2, query_level0=True, device="cpu")
    policy = train_stage2_ncm(pfxs, seqs, vqvae, s1, s2)["model"]

    prefs = generate_synthetic_dpo_preferences(pfxs[:4], vqvae, policy, lambda logits, _p: logits.mean(dim=(-1, -2)), top_k=4)
    assert prefs["chosen_codes"].shape == prefs["rejected_codes"].shape

    bundle = {"mock": True}
    with patch("ncm_bench.longdoc.summarize_texts", side_effect=lambda _b, txts, **_kw: ["Brief."] * len(txts)):
        lc = run_longcontext_summarization_harness(bundle, ["Sentence one. Sentence two."], max_chunk_chars=15)
    assert lc["docs_evaluated"] == 1 and lc["total_chunks"] >= 1

    legal = evaluate_legal_clause_redaction(["Pay ops@acme.io IBAN DE89370400440532013000"])
    assert legal["exact_restoration_rate"] == 1.0 and legal["total_redactions"] >= 2

    fin = verify_finqa_table_arithmetic("item,amount\na,$100.50\nb,$200.00\n", "amount", 300.0)
    assert fin["computed_sum"] == 300.5 and fin["exact_match"] is False and fin["repaired_delta"] == 0.5

    ast_res = extract_python_ast_concepts("def add(a, b):\n    return a + b\n")
    assert "FunctionDef" in ast_res["node_counts"] and len(ast_res["concept_hash"]) == 16

    xling = score_cross_lingual_concept_alignment({"histogram": {1: 1.0, 2: 1.0}}, {"histogram": {1: 1.0, 3: 1.0}})
    assert xling["cosine_similarity"] == 0.5 and xling["shared_concepts"] == 1

    energy = estimate_energy_telemetry(10.0, 100)
    assert energy["wh_per_1k_docs"] > 0.0 and energy["joules_per_doc"] > 0.0

    joint = encode_layout_text_joint_concepts([10, 20], [(0, 0, 200, 100), (500, 500, 900, 900)])
    assert len(joint) == 2

    rec_p = tmp_path / "receipt.json"
    rec_p.write_text(
        json.dumps({"hardware": "V100", "workload": "xsum", "throughput": 9.47, "metric_name": "rougeL", "metric_value": 27.93}),
        encoding="utf-8",
    )
    assert validate_receipt_submission(json.loads(rec_p.read_text(encoding="utf-8")))["valid"] is True

    capsys.readouterr()
    with patch.object(sys, "argv", ["ncm-bench", "validate-receipt", "--input", str(rec_p)]):
        main()
        assert json.loads(capsys.readouterr().out)["valid"] is True

    with patch.object(sys, "argv", ["ncm-bench", "hardware-probe"]):
        main()
        assert "active_device" in json.loads(capsys.readouterr().out)

    off_dir = tmp_path / "offline_dist"
    with patch.object(sys, "argv", ["ncm-bench", "bundle-offline", "--out-dir", str(off_dir)]):
        main()
        b_out = json.loads(capsys.readouterr().out)
        assert Path(b_out["tar_bundle"]).exists() and Path(b_out["standalone_pyz"]).exists()
