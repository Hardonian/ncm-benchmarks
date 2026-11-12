from unittest.mock import MagicMock, patch

import pytest

from service.client import NCMAuthError, NCMClient, NCMServerBusyError


def test_client_health_and_ready():
    client = NCMClient(base_url="http://localhost:8031", api_key="secret-token")
    assert client.session.headers["Authorization"] == "Bearer secret-token"

    with patch.object(client.session, "request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True, "ready": True, "model": "ncm-xsum-88M"}
        mock_resp.raise_for_status.return_value = None
        mock_req.return_value = mock_resp

        res = client.health()
        assert res["ok"] is True
        assert res["model"] == "ncm-xsum-88M"
        assert client.ready()["ready"] is True
        assert mock_req.call_count == 2


def test_client_summarize():
    client = NCMClient()
    with patch.object(client.session, "request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"summary": "A mock summary.", "latency_ms": 2100}
        mock_resp.raise_for_status.return_value = None
        mock_req.return_value = mock_resp

        res = client.summarize("Article text.")
        assert res["summary"] == "A mock summary."
        mock_req.assert_called_once()


def test_client_batch_summarize_server():
    client = NCMClient()
    with patch.object(client.session, "request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "summaries": ["Sum 1", "Sum 2"],
            "count": 2,
            "latency_ms": 150,
        }
        mock_resp.raise_for_status.return_value = None
        mock_req.return_value = mock_resp

        res = client.batch_summarize_server(["Doc 1", "Doc 2"])
        assert res["summaries"] == ["Sum 1", "Sum 2"]
        assert res["count"] == 2


def test_client_mbr():
    client = NCMClient()
    with patch.object(client.session, "request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"summary": "MBR summary.", "num_candidates": 4, "selected_index": 0}
        mock_resp.raise_for_status.return_value = None
        mock_req.return_value = mock_resp

        res = client.mbr_summarize("Article text.", num_candidates=4)
        assert res["summary"] == "MBR summary."
        assert res["num_candidates"] == 4


def test_client_batch():
    client = NCMClient()
    with patch.object(client.session, "request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"summary": "Sum", "latency_ms": 100}
        mock_resp.raise_for_status.return_value = None
        mock_req.return_value = mock_resp

        batch_res = client.batch_summarize(["Doc 1", "Doc 2"])
        assert len(batch_res) == 2
        assert batch_res[0]["summary"] == "Sum"
        assert "client_elapsed_s" in batch_res[0]


def test_client_auth_and_busy_errors():
    client = NCMClient()
    with patch.object(client.session, "request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Invalid token"
        mock_req.return_value = mock_resp
        with pytest.raises(NCMAuthError):
            client.summarize("text")

        mock_resp.status_code = 503
        mock_resp.text = "Server overloaded"
        with pytest.raises(NCMServerBusyError):
            client.summarize("text")
