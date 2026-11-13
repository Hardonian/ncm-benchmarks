import threading
from unittest.mock import patch

from fastapi.testclient import TestClient

from service.summarize_server import create_app


def _dummy_bundle():
    return {"device": "cpu", "quantize": "int8"}


def test_health_ready_and_metrics():
    app = create_app(_dummy_bundle(), load_s=1.4)
    client = TestClient(app)

    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["ok"] is True
    assert res.json()["quantize"] == "int8"
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.headers["x-frame-options"] == "DENY"
    assert "x-request-id" in res.headers

    ready_res = client.get("/ready")
    assert ready_res.status_code == 200
    assert ready_res.json()["ready"] is True

    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    assert "ncm_requests_total" in metrics_res.text
    assert "ncm_model_loaded 1" in metrics_res.text


def test_unloaded_model_returns_503():
    app = create_app(None)
    client = TestClient(app)
    assert client.get("/ready").status_code == 503
    assert client.post("/summarize", json={"text": "hello"}).status_code == 503


def test_input_validation_bounds():
    app = create_app(_dummy_bundle())
    client = TestClient(app)

    assert client.post("/summarize", json={"text": ""}).status_code == 422
    assert client.post("/summarize", json={"text": "   \n  "}).status_code == 422
    assert client.post("/mbr_summarize", json={"text": "valid", "num_candidates": 999}).status_code == 422
    assert client.post("/mbr_summarize", json={"text": "valid", "temperature": 9.0}).status_code == 422
    assert client.post("/batch_summarize", json={"texts": []}).status_code == 422


def test_bearer_auth_enforcement():
    app = create_app(_dummy_bundle(), api_key="secret-token-123")
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.post("/summarize", json={"text": "hello"}).status_code == 401
    assert client.post(
        "/summarize",
        json={"text": "hello"},
        headers={"Authorization": "Bearer wrong-token"},
    ).status_code == 403

    with patch("service.summarize_server.summarize_texts", return_value=["Mock summary."]):
        ok = client.post(
            "/summarize",
            json={"text": "hello"},
            headers={"Authorization": "Bearer secret-token-123"},
        )
        assert ok.status_code == 200
        assert ok.json()["summary"] == "Mock summary."


def test_batch_mbr_and_progressive_endpoints():
    app = create_app(_dummy_bundle())
    client = TestClient(app)

    with patch("service.summarize_server.summarize_texts", return_value=["Sum 1.", "Sum 2."]):
        res = client.post("/batch_summarize", json={"texts": ["Doc 1", "Doc 2"]})
        assert res.status_code == 200
        assert res.json()["count"] == 2
        assert res.json()["summaries"] == ["Sum 1.", "Sum 2."]

    with patch(
        "service.summarize_server.generate_mbr_summary",
        return_value={
            "summary": "MBR consensus.",
            "selected_index": 0,
            "num_candidates": 4,
            "candidates": ["MBR consensus."],
            "scores": [1.0],
        },
    ):
        res = client.post("/mbr_summarize", json={"text": "Article", "num_candidates": 4})
        assert res.status_code == 200
        assert res.json()["summary"] == "MBR consensus."


def test_visualize_escapes_xss_payload():
    app = create_app(_dummy_bundle())
    client = TestClient(app)

    payload = "<script>alert('xss')</script>"
    prog_result = {
        "num_levels": 1,
        "final_summary": "<img src=x onerror=alert(1)>",
        "levels": [
            {
                "level": 0,
                "codes_at_level": 4,
                "total_codes_so_far": 4,
                "text": "<img src=x onerror=alert(1)>",
                "words": 2,
            }
        ],
    }
    with patch("service.summarize_server.progressive_decode", return_value=prog_result):
        res = client.get("/visualize", params={"text": payload})
        assert res.status_code == 200
        assert payload not in res.text
        assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in res.text
        assert "<img src=x" not in res.text
        assert "content-security-policy" in res.headers


def test_queue_backpressure_returns_503():
    app = create_app(_dummy_bundle(), max_queue_depth=1)
    client = TestClient(app)
    entered = threading.Event()
    release = threading.Event()

    def slow_summarize(_b, _texts):
        entered.set()
        release.wait(timeout=3.0)
        return ["done"]

    with patch("service.summarize_server.summarize_texts", side_effect=slow_summarize):
        holder = {}

        def _bg():
            holder["res"] = client.post("/summarize", json={"text": "first"})

        t = threading.Thread(target=_bg)
        t.start()
        assert entered.wait(timeout=2.0)

        second = client.post("/summarize", json={"text": "second"})
        assert second.status_code == 503
        assert second.headers.get("retry-after") == "2"

        release.set()
        t.join(timeout=2.0)
        assert holder["res"].status_code == 200


def test_sudoku_openai_batch_mbr_and_progressive_stream() -> None:
    bundle = _dummy_bundle()
    puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    with (
        patch(
            "service.summarize_server.solve_sudoku",
            return_value=[{"board": "5" * 81, "cells": "5" * 81, "valid": True}],
        ),
        patch(
            "service.summarize_server.summarize_texts",
            side_effect=lambda b, texts, **kw: [f"Sum:{t}" for t in texts],
        ),
        patch(
            "service.summarize_server.generate_mbr_summary",
            side_effect=lambda b, text, **kw: {"summary": f"MBR:{text}", "selected_index": 0, "num_candidates": 2},
        ),
        patch(
            "service.summarize_server.progressive_decode",
            return_value={
                "num_levels": 1,
                "final_summary": "Stream",
                "levels": [{"level": 0, "codes_at_level": 2, "total_codes_so_far": 2, "text": "Stream", "words": 1}],
            },
        ),
    ):
        app = create_app(bundle=bundle, sudoku_bundle=bundle)
        with TestClient(app) as client:
            models_resp = client.get("/v1/models")
            assert models_resp.status_code == 200
            model_ids = [m["id"] for m in models_resp.json()["data"]]
            assert "ncm-xsum-88M" in model_ids
            assert "ncm-sudoku-25M" in model_ids

            chat_resp = client.post(
                "/v1/chat/completions",
                json={"model": "ncm-xsum-88M", "messages": [{"role": "user", "content": "Hello world"}]},
            )
            assert chat_resp.status_code == 200
            chat_data = chat_resp.json()
            assert chat_data["object"] == "chat.completion"
            assert chat_data["choices"][0]["message"]["content"] == "Sum:Hello world"

            batch_resp = client.post(
                "/batch_mbr_summarize",
                json={"texts": ["Doc one", "Doc two"], "num_candidates": 2},
            )
            assert batch_resp.status_code == 200
            assert len(batch_resp.json()["results"]) == 2

            sudoku_resp = client.post("/sudoku", json={"puzzle": puzzle, "best_of_k": 2, "repair": True})
            assert sudoku_resp.status_code == 200
            sdata = sudoku_resp.json()
            assert sdata["valid"] is True

            stream_resp = client.post("/progressive_stream", json={"text": "Stream test"})
            assert stream_resp.status_code == 200
            assert "text/event-stream" in stream_resp.headers["content-type"]
            assert "event: done" in stream_resp.text


def test_body_limit_rate_limit_strict_auth_and_histogram() -> None:
    bundle = _dummy_bundle()
    with patch("service.summarize_server.summarize_texts", return_value=["ok"]):
        app = create_app(
            bundle=bundle,
            max_body_bytes=64,
            rate_limit_rpm=2,
            api_key="sec-token",
            strict_auth=True,
        )
        with TestClient(app) as client:
            assert client.get("/metrics").status_code == 401

            big = client.post(
                "/summarize",
                json={"text": "x" * 200},
                headers={"Authorization": "Bearer sec-token"},
            )
            assert big.status_code == 413

            r1 = client.post("/summarize", json={"text": "a"}, headers={"Authorization": "Bearer sec-token"})
            r2 = client.post("/summarize", json={"text": "b"}, headers={"Authorization": "Bearer sec-token"})
            r3 = client.post("/summarize", json={"text": "c"}, headers={"Authorization": "Bearer sec-token"})
            assert r1.status_code == 200
            assert r2.status_code == 200
            assert r3.status_code == 429
            assert "retry-after" in r3.headers

            metrics_ok = client.get("/metrics", headers={"Authorization": "Bearer sec-token"})
            assert metrics_ok.status_code == 200
            assert "ncm_inference_duration_seconds_bucket" in metrics_ok.text

