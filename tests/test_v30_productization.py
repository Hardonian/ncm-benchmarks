"""Comprehensive verification suite for v3.0 Enterprise Productization upgrades."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from ncm_bench.accel import Int4Linear, quantize_model_int4
from ncm_bench.compliance import (
    AuditSinkSQLite,
    EnvKMSProvider,
    ExternalKMSHookProvider,
    LocalKeyringProvider,
    decrypt_field_envelope,
    encrypt_field_envelope,
)
from ncm_bench.frontier import estimate_energy_telemetry
from ncm_bench.inference import _apply_dynamic_int8
from ncm_bench.longdoc import (
    default_contextual_ner,
    extract_text_from_file,
    redact_pii,
    restore_pii,
)
from ncm_bench.reasoning import JSONGrammarStateMachine
from ncm_bench.triage import ConceptClassifier, SQLiteClassifierStore
from ncm_train import (
    Stage1Config,
    Stage2Config,
    extract_lora_state_dict,
    inject_concept_lora,
    load_lora_state_dict,
    train_stage1_vqvae,
    train_stage2_ncm,
)
from service.summarize_server import create_app
from service.transport import DynamicBatchCoalescer


def test_trainer_hardening_resume_and_safetensors(tmp_path: Path) -> None:
    seqs = torch.randint(0, 12, (24, 16))
    s1_path = tmp_path / "vqvae.safetensors"
    s1_cfg = Stage1Config(
        in_vocab_size=12,
        context_length=16,
        embed_dim=32,
        levels=(2, 4, 8, 16),
        vq_vocab_size=16,
        n_layers=1,
        batch_size=8,
        steps=4,
        ema_decay=0.85,
        dropout=0.0,
        grad_accum_steps=2,
        device="cpu",
    )
    res1 = train_stage1_vqvae(seqs, s1_cfg, out_path=s1_path)
    assert s1_path.exists()
    assert len(res1["history"]) == 4

    s1_resumed = Stage1Config(**{**s1_cfg.__dict__, "steps": 2, "resume_from": s1_path})
    res1_resumed = train_stage1_vqvae(seqs, s1_resumed)
    assert len(res1_resumed["history"]) == 2

    s2_path = tmp_path / "ncm.safetensors"
    s2_cfg = Stage2Config(
        prefix_len=16,
        n_layer=1,
        n_head=4,
        n_embd=32,
        batch_size=8,
        steps=4,
        grad_accum_steps=2,
        device="cpu",
    )
    res2 = train_stage2_ncm(seqs, seqs, res1["model"], s1_cfg, s2_cfg, out_path=s2_path)
    assert s2_path.exists()
    assert s2_path.with_suffix(".meta.json").exists()
    assert len(res2["history"]) == 4

    s2_resumed = Stage2Config(**{**s2_cfg.__dict__, "steps": 2, "resume_from": s2_path})
    res2_resumed = train_stage2_ncm(seqs, seqs, res1["model"], s1_cfg, s2_resumed)
    assert len(res2_resumed["history"]) == 2


def test_recursive_lora_injection_and_state_roundtrip() -> None:
    class NestedBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn_proj = nn.Linear(16, 16)
            self.mlp_up = nn.Linear(16, 32)
            self.mlp_down = nn.Linear(32, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.mlp_down(torch.relu(self.mlp_up(self.attn_proj(x))))

    model = nn.Sequential(NestedBlock(), nn.Linear(16, 8))
    info = inject_concept_lora(model, rank=2, alpha=4.0, target_modules=("attn_proj", "mlp_up", "1"))
    assert info["adapters_injected"] == 3
    assert len(info["adapter_paths"]) == 3

    x = torch.randn(2, 16)
    out = model(x)
    assert out.shape == (2, 8)

    sd = extract_lora_state_dict(model)
    assert len(sd) == 6
    with torch.no_grad():
        for k in sd:
            sd[k].fill_(0.25)
    loaded = load_lora_state_dict(model, sd)
    assert loaded == 6


def test_aes_gcm_key_rotation_and_sqlite_wal_streaming(tmp_path: Path) -> None:
    keyring = {"v1": "old-key-secret", "v2": "active-key-secret"}
    env = encrypt_field_envelope("Sensitive merger payload", keyring["v2"], key_id="v2")
    assert env["alg"] in {"AES-256-GCM", "PBKDF2-SHA256-HMAC"}
    assert env["key_id"] == "v2"
    assert decrypt_field_envelope(env, keyring) == "Sensitive merger payload"

    with pytest.raises(ValueError, match="Unknown key_id"):
        decrypt_field_envelope(env, {"v1": "old-key-secret"})

    with pytest.raises(ValueError, match="Envelope authentication failed"):
        decrypt_field_envelope(dict(env, tag="00" * 16), keyring)

    db_path = tmp_path / "wal_audit.sqlite"
    sink = AuditSinkSQLite(db_path)
    with sqlite3.connect(db_path) as conn:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert str(mode).lower() == "wal"

    for i in range(15):
        sink.append({"seq_id": i, "msg": f"event-{i}"})
    assert sink.verify_chain(batch_size=4) is True


def test_longdoc_archive_quota_table_xlsx_and_ner_hook(tmp_path: Path) -> None:
    docx_tbl = tmp_path / "table.docx"
    with zipfile.ZipFile(docx_tbl, "w") as zf:
        zf.writestr(
            "word/document.xml",
            "<w:document><w:body><w:p><w:t>Header</w:t></w:p>"
            "<w:tbl><w:tr><w:tc><w:p><w:t>ColA</w:t></w:p></w:tc>"
            "<w:tc><w:p><w:t>123</w:t></w:p></w:tc></w:tr></w:tbl></w:body></w:document>",
        )
    docx_txt = extract_text_from_file(docx_tbl)
    assert "Header" in docx_txt and "ColA | 123" in docx_txt

    with pytest.raises(ValueError, match="max_archive_bytes"):
        extract_text_from_file(docx_tbl, max_archive_bytes=10)

    xlsx_ref = tmp_path / "structured.xlsx"
    with zipfile.ZipFile(xlsx_ref, "w") as zf:
        zf.writestr("xl/sharedStrings.xml", "<sst><si><t>Quarter</t></si><si><t>Q4</t></si></sst>")
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet><sheetData><row><c t="s"><v>0</v></c><c t="s"><v>1</v></c><c><v>99.5</v></c></row></sheetData></worksheet>',
        )
    xlsx_txt = extract_text_from_file(xlsx_ref)
    assert "Quarter | Q4 | 99.5" in xlsx_txt

    raw = "Patient Jane Doe MRN-123456 contacted alice@acme.io."
    scrubbed, pii_map = redact_pii(raw, ner_fn=lambda _t: [("PERSON", "Jane Doe")])
    assert "[MRN_1]" in scrubbed and "[PERSON_1]" in scrubbed and "[EMAIL_1]" in scrubbed
    assert "Jane Doe" not in scrubbed
    assert restore_pii(scrubbed, pii_map) == raw

    telemetry = estimate_energy_telemetry(2.0, 10)
    assert telemetry["telemetry_source"] in {"nvml", "rapl", "tdp_estimate"}


def test_triage_persistence_hybrid_predict_and_async_batcher(tmp_path: Path) -> None:
    clf = ConceptClassifier().fit(
        [{"histogram": {1: 1.0}}, {"histogram": {2: 1.0}}],
        ["security", "finance"],
    )
    hybrid = clf.predict_one({"histogram": {1: 0.5, 2: 0.5}}, text_hint="ransomware phishing breach")
    assert hybrid["label"] == "security"

    state_file = tmp_path / "clf_state.json"
    clf.save(state_file)
    reloaded = ConceptClassifier.load(state_file)
    assert reloaded.centroids == clf.centroids

    fake_bundle = {
        "device": "cpu",
        "dtype": "fp32",
        "ckpt_format": "safetensors",
        "quantize": "none",
    }
    app = create_app(fake_bundle, triage_state_path=state_file)
    from unittest.mock import patch

    client = TestClient(app)
    with patch(
        "ncm_bench.triage.generate_coarse_codes",
        return_value=[{"histogram": {9: 1.0}, "concept_hash": "h9", "passes_executed": 3}],
    ):
        resp = client.post("/triage/feedback", json={"text": "New legal NDA review", "label": "legal"})
    assert resp.status_code == 200
    assert resp.json()["persisted"] is True
    assert "legal" in json.loads(state_file.read_text(encoding="utf-8"))["centroids"]

    calls: list[int] = []

    def _batch(chunk: list[str]) -> list[dict[str, str]]:
        calls.append(len(chunk))
        return [{"echo": item} for item in chunk]

    coalescer = DynamicBatchCoalescer(_batch, max_batch_size=3, max_wait_ms=5.0)
    outs = asyncio.run(coalescer.process_async(["a", "b", "c", "d"]))
    assert [o["echo"] for o in outs] == ["a", "b", "c", "d"]
    assert coalescer.batches_executed == 2 and calls == [3, 1]


def test_int4_linear_dynamic_int8_and_json_fsm() -> None:
    net = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))
    x = torch.randn(3, 32)
    q_int8 = _apply_dynamic_int8(nn.Sequential(nn.Linear(32, 8)))
    assert q_int8(x).shape == (3, 8)

    res = quantize_model_int4(net, group_size=16)
    assert res["layers_quantized"] == 2
    assert isinstance(net[0], Int4Linear)
    assert net(x).shape == (3, 4)

    fsm = JSONGrammarStateMachine()
    id_to_tok = {0: "{", 1: '"ok"', 2: ":", 3: "true", 4: "}", 5: "invalid"}
    raw_logits = torch.zeros(6)
    m0 = fsm.mask_logits(raw_logits, id_to_tok)
    assert int(m0.argmax().item()) == 0
    assert fsm.advance("{") == "expect_key_or_close"
    assert fsm.advance('"ok"') == "expect_colon"
    assert fsm.advance(":") == "expect_value"
    assert fsm.advance("true") == "expect_comma_or_close"
    assert fsm.advance("}") == "complete"


def test_int4_lut_cache_kms_providers_sqlite_store_and_contextual_ner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(7)
    base_lin = nn.Linear(32, 16)
    lin_nocache = Int4Linear.from_linear(base_lin, group_size=16, cache_dequantized=False)
    lin_cached = Int4Linear.from_linear(base_lin, group_size=16, cache_dequantized=True)
    x = torch.randn(4, 32)
    y_nocache = lin_nocache(x)
    y_cached_first = lin_cached(x)
    y_cached_second = lin_cached(x)
    assert torch.allclose(y_nocache, y_cached_first, atol=1e-6)
    assert torch.allclose(y_cached_first, y_cached_second, atol=1e-6)
    assert lin_cached._cached_weight is not None

    q_res = quantize_model_int4(nn.Sequential(nn.Linear(16, 8)), group_size=16, cache_dequantized=True)
    assert q_res["cache_dequantized"] is True

    # LocalKeyringProvider, EnvKMSProvider, and ExternalKMSHookProvider roundtrips
    local_kms = LocalKeyringProvider({"v1": "alpha-secret", "v2": "beta-secret"})
    env_local = encrypt_field_envelope("board-minutes", local_kms)
    assert env_local["key_id"] == "v2"
    assert decrypt_field_envelope(env_local, local_kms) == "board-minutes"

    monkeypatch.setenv("NCM_KMS_KEY_ACTIVE", "env-master-key")
    secret_file = tmp_path / "vault_v1"
    secret_file.write_text("mounted-secret-key", encoding="utf-8")
    env_kms = EnvKMSProvider(secrets_dir=tmp_path, active_key_id="active")
    e_env = encrypt_field_envelope("env-payload", env_kms)
    assert decrypt_field_envelope(e_env, env_kms) == "env-payload"
    e_file = encrypt_field_envelope("file-payload", env_kms, key_id="vault_v1")
    assert decrypt_field_envelope(e_file, env_kms) == "file-payload"

    master_mask = bytes([0x5A] * 32)
    ext_kms = ExternalKMSHookProvider(
        wrap_fn=lambda dek, _kid: bytes(a ^ b for a, b in zip(dek, master_mask, strict=True)),
        unwrap_fn=lambda wdek, _kid: bytes(a ^ b for a, b in zip(wdek, master_mask, strict=True)),
        key_id="kms-eu-1",
    )
    e_ext = encrypt_field_envelope("hipaa-record", ext_kms)
    assert "wrapped_dek" in e_ext and e_ext["key_id"] == "kms-eu-1"
    assert decrypt_field_envelope(e_ext, ext_kms) == "hipaa-record"
    with pytest.raises(ValueError, match="unwrap_dek"):
        decrypt_field_envelope(e_ext, "plain-key")

    # SQLiteClassifierStore WAL persistence
    db_file = tmp_path / "triage_store.sqlite"
    clf = ConceptClassifier().fit(
        [{"histogram": {1: 1.0}}, {"histogram": {2: 1.0}}],
        ["security", "finance"],
    )
    clf.save(db_file)
    reloaded = ConceptClassifier.load(db_file)
    assert reloaded.centroids == clf.centroids
    store = SQLiteClassifierStore(db_file)
    store.save(clf, state_id="tenant-a")
    assert store.load(state_id="tenant-a").centroids == clf.centroids
    assert store.load(state_id="unseeded").centroids == {}

    # Built-in contextual NER + restore_pii roundtrip
    clinical = (
        "Patient Elena Rostova lives at 742 Evergreen Terrace Ave "
        "using key sk-testsecret1234567890abcd and alice@clinic.org."
    )
    findings = default_contextual_ner(clinical)
    labels_found = {lbl for lbl, _ in findings}
    assert {"PERSON", "ADDRESS", "API_KEY"}.issubset(labels_found)
    scrubbed, pii_map = redact_pii(clinical, enable_contextual_ner=True)
    assert "Elena Rostova" not in scrubbed
    assert "742 Evergreen Terrace Ave" not in scrubbed
    assert "sk-testsecret1234567890abcd" not in scrubbed
    assert "alice@clinic.org" not in scrubbed
    assert restore_pii(scrubbed, pii_map) == clinical

    # Helm chart & v3.0.0 version consistency
    root = Path(__file__).resolve().parent.parent
    assert 'version = "3.0.0"' in (root / "pyproject.toml").read_text(encoding="utf-8")
    for tpl in ("deployment.yaml", "service.yaml", "hpa.yaml", "networkpolicy.yaml"):
        assert (root / "deploy" / "helm" / "ncm-benchmarks" / "templates" / tpl).is_file()
    assert 'appVersion: "3.0.0"' in (
        root / "deploy" / "helm" / "ncm-benchmarks" / "Chart.yaml"
    ).read_text(encoding="utf-8")
    assert 'appVersion: "3.0.0"' in (
        root / "deploy" / "helm" / "ncm-summarizer" / "Chart.yaml"
    ).read_text(encoding="utf-8")
