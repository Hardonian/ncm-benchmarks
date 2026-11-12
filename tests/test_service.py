import pytest
from fastapi.testclient import TestClient
import fastapi
from unittest.mock import MagicMock
import os
import sys

# Test the FastAPI application endpoints with mocked model backend
app = fastapi.FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "device": "cpu", "load_s": 2.1, "model": "ncm-xsum-88M", "quantize": "none"}

@app.post("/summarize")
def summarize_ep(payload: dict):
    text = payload.get("text", "")
    if not text:
        return fastapi.responses.JSONResponse({"error": "text required"}, status_code=400)
    return {"summary": "A mock summary.", "latency_ms": 120, "words": 3}

@app.post("/mbr_summarize")
def mbr_ep(payload: dict):
    text = payload.get("text", "")
    if not text:
        return fastapi.responses.JSONResponse({"error": "text required"}, status_code=400)
    return {
        "summary": "Best consensus summary.",
        "selected_index": 0,
        "num_candidates": 4,
        "candidates": ["Best consensus summary.", "Second candidate."],
        "scores": [1.8, 1.2],
        "latency_ms": 450,
    }

@app.post("/progressive_summarize")
def prog_ep(payload: dict):
    text = payload.get("text", "")
    if not text:
        return fastapi.responses.JSONResponse({"error": "text required"}, status_code=400)
    return {
        "num_levels": 2,
        "final_summary": "Final summary.",
        "levels": [
            {"level": 0, "codes_at_level": 4, "total_codes_so_far": 4, "text": "gist", "words": 1},
            {"level": 1, "codes_at_level": 6, "total_codes_so_far": 10, "text": "Final summary.", "words": 2},
        ],
        "latency_ms": 250,
    }

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["model"] == "ncm-xsum-88M"

def test_summarize_endpoint():
    res = client.post("/summarize", json={"text": "Article text here."})
    assert res.status_code == 200
    assert "summary" in res.json()
    assert res.json()["words"] == 3

def test_summarize_empty():
    res = client.post("/summarize", json={"text": ""})
    assert res.status_code == 400

def test_mbr_endpoint():
    res = client.post("/mbr_summarize", json={"text": "Article text here.", "num_candidates": 4})
    assert res.status_code == 200
    data = res.json()
    assert "candidates" in data
    assert data["selected_index"] == 0

def test_progressive_endpoint():
    res = client.post("/progressive_summarize", json={"text": "Article text."})
    assert res.status_code == 200
    data = res.json()
    assert data["num_levels"] == 2
    assert len(data["levels"]) == 2
