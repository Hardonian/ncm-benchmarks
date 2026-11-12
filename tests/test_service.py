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
