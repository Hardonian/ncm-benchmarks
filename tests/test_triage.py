"""Tests for coarse-level concept triage, semantic indexing, and verified structured extraction."""
from __future__ import annotations

import json
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from ncm_bench.cli import main
from ncm_bench.triage import (
    ConceptClassifier,
    ConceptIndex,
    extract_structured_triage,
    generate_coarse_codes,
    validate_and_repair_triage_record,
)
from ncm_train import Stage1Config, Stage2Config, train_stage1_vqvae, train_stage2_ncm
from service.summarize_server import create_app


class _FakeCoarseNet:
    def __init__(self):
        self.levels = [2, 4, 8, 16]

    def generate(self, idx, prefix_tokens, **kwargs):
        del idx, kwargs
        bs = prefix_tokens.shape[0]
        row = torch.arange(sum(self.levels), dtype=torch.long) % 7
        out = row.unsqueeze(0).repeat(bs, 1)
        for i in range(bs):
            out[i, :4] = (out[i, :4] + int(prefix_tokens[i, 0].item())) % 11
        return out


class _FakeTokenizer:
    def encode(self, text: str):
        return [sum(ord(c) for c in text) % 15 + 1, len(text) % 15 + 1, 3, 4]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"tok{i}" for i in ids)


def _fake_bundle():
    tok = _FakeTokenizer()
    return {
        "net": _FakeCoarseNet(),
        "tk": SimpleNamespace(pre_tokenizer=tok),
        "src": None,
        "device": torch.device("cpu"),
        "EOS": 0,
        "BOS": 1,
        "PAD": 2,
        "P": 8,
        "adj": None,
        "freq": 0.0,
    }


def test_generate_coarse_codes_and_concept_index_deduplication():
    bundle = _fake_bundle()
    sigs = generate_coarse_codes(
        bundle,
        ["Alpha financial quarter report", "Alpha financial quarter report", "Zeta breach alert"],
        max_level=1,
    )
    assert len(sigs) == 3
    assert sigs[0]["max_level"] == 1
    assert sigs[0]["passes_executed"] == 2
    assert sigs[0]["total_levels"] == 4
    assert sigs[0]["concept_hash"] == sigs[1]["concept_hash"]

    idx = ConceptIndex()
    idx.add("doc-0", sigs[0], metadata={"tag": "finance"})
    assert len(idx) == 1
    assert idx.is_duplicate(sigs[1], threshold=0.99) is True
    hits = idx.query(sigs[1], top_k=1)
    assert hits[0]["doc_id"] == "doc-0"
    assert hits[0]["similarity"] >= 0.99


def test_early_exit_fast_path_on_real_ncm_transformer():
    torch.manual_seed(0)
    sequences = torch.randint(0, 16, (16, 16), dtype=torch.long)
    prefixes = torch.randint(0, 16, (16, 16), dtype=torch.long)
    s1 = Stage1Config(
        in_vocab_size=16,
        context_length=16,
        embed_dim=32,
        levels=(2, 4, 8, 16),
        vq_vocab_size=16,
        n_layers=1,
        batch_size=8,
        steps=2,
        device="cpu",
    )
    vqvae = train_stage1_vqvae(sequences, s1)["model"]
    s2 = Stage2Config(
        prefix_len=16,
        n_layer=1,
        n_head=2,
        n_embd=32,
        batch_size=8,
        steps=2,
        query_level0=True,
        device="cpu",
    )
    net = train_stage2_ncm(prefixes, sequences, vqvae, s1, s2)["model"]
    real_bundle = {
        "net": net,
        "tk": SimpleNamespace(vqvae=vqvae, pre_tokenizer=_FakeTokenizer()),
        "src": None,
        "device": torch.device("cpu"),
        "EOS": 0,
        "BOS": 1,
        "PAD": 0,
        "P": 16,
        "adj": None,
        "freq": 0.0,
    }
    res = generate_coarse_codes(real_bundle, ["Security incident report", "Quarterly earnings"], max_level=1, decode_preview=True)
    assert len(res) == 2
    assert res[0]["passes_executed"] == 2
    assert res[0]["total_levels"] == 4
    assert [len(lv) for lv in res[0]["levels"]] == [2, 4]
    assert isinstance(res[0]["preview_text"], str)


def test_concept_classifier_and_validation_repair():
    fin_sig = {
        "concept_hash": "aaa",
        "passes_executed": 2,
        "histogram": {0: 0.9, 1: 0.43},
    }
    sec_sig = {
        "concept_hash": "bbb",
        "passes_executed": 2,
        "histogram": {500: 0.9, 501: 0.43},
    }
    clf = ConceptClassifier(escalation_confidence=0.6).fit(
        [fin_sig, sec_sig],
        ["finance", "security"],
    )
    pred = clf.predict_one(fin_sig)
    assert pred["label"] == "finance"
    assert pred["confidence"] > 0.7
    assert pred["escalate"] is False

    repaired = validate_and_repair_triage_record(
        {"category": "invalid-cat", "priority": "unknown", "summary": ""},
        source_text="Urgent security breach & CVE vulnerability at alice@acme.org cost $250,000 on 2026-12-01.",
        allowed_labels=["finance", "security", "legal"],
    )
    assert repaired["schema_valid"] is True
    assert repaired["category"] == "security"
    assert repaired["priority"] == "high"
    assert "alice@acme.org" in repaired["entities"]["emails"]
    assert "$250,000" in repaired["entities"]["money"]
    assert "2026-12-01" in repaired["entities"]["dates"]


def test_extract_structured_triage_endpoint_and_cli(capsys, tmp_path):
    bundle = _fake_bundle()
    sample = (
        "Urgent security breach CVE vulnerability noticed by alice@corp.com involving $120,000 on 2026-11-12. "
        "Please audit https://status.corp.com immediately."
    )
    with patch(
        "ncm_bench.triage.summarize_texts",
        side_effect=lambda b, texts, **kw: ["Security breach email [EMAIL_1]."] * len(texts),
    ):
        res = extract_structured_triage(
            bundle,
            sample,
            labels=["finance", "security", "legal"],
            max_level=1,
            best_of_k=3,
            redact=True,
            restore=True,
        )
    assert res["schema_valid"] is True
    assert res["category"] == "security"
    assert res["priority"] == "high"
    assert res["pii_redactions"] >= 1
    assert "alice@corp.com" in res["summary"]
    assert "$120,000" in res["entities"]["money"]

    # FastAPI /triage endpoint check
    app = create_app(bundle)
    client = TestClient(app)
    with patch(
        "ncm_bench.triage.summarize_texts",
        return_value=["Security breach summary."],
    ):
        http_res = client.post(
            "/triage",
            json={"text": sample, "labels": ["finance", "security"], "best_of_k": 2, "audit_receipt": True},
        )
        assert http_res.status_code == 200
        payload = http_res.json()
        assert payload["category"] == "security"
        assert "audit_receipt" in payload

    # CLI triage check
    with patch("ncm_bench.cli._load_bundle_for_args", return_value=bundle):
        with patch("ncm_bench.triage.summarize_texts", return_value=["CLI triage summary."]):
            with patch.object(
                sys,
                "argv",
                ["ncm-bench", "triage", "--text", sample, "--labels", "finance,security", "--best-of-k", "1", "--json"],
            ):
                main()
                cli_out = json.loads(capsys.readouterr().out)
                assert cli_out["schema_valid"] is True
                assert cli_out["category"] == "security"

    # CLI early-exit-bench check (self-contained without repo data/ mount for CI)
    data_dir = tmp_path / "data" / "xsum"
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "passages.pickle", "wb") as f:
        pickle.dump(["Doc one", "Doc two"], f)

    with patch("ncm_bench.cli.ROOT_DIR", tmp_path):
        with patch("ncm_bench.cli._load_bundle_for_args", return_value=bundle):
            with patch("ncm_bench.inference.summarize_texts", return_value=["Doc summary", "Doc summary"]):
                with patch.object(
                    sys,
                    "argv",
                    ["ncm-bench", "early-exit-bench", "--n", "2", "--max-level", "1"],
                ):
                    main()
                    bench_out = json.loads(capsys.readouterr().out)
                    assert bench_out["n"] == 2
                    assert bench_out["passes_early_exit"] == 2
                    assert bench_out["passes_full"] == 4
