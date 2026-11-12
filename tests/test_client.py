import pytest
from unittest.mock import MagicMock, patch
from service.client import NCMClient


def test_client_health():
    client = NCMClient(base_url="http://localhost:8031")
    with patch.object(client.session, "get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "model": "ncm-xsum-88M"}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        res = client.health()
        assert res["ok"] is True
        assert res["model"] == "ncm-xsum-88M"
        mock_get.assert_called_once_with("http://localhost:8031/health", timeout=60.0)


def test_client_summarize():
    client = NCMClient()
    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"summary": "A mock summary.", "latency_ms": 2100}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        res = client.summarize("Article text.")
        assert res["summary"] == "A mock summary."
        mock_post.assert_called_once()


def test_client_mbr():
    client = NCMClient()
    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"summary": "MBR summary.", "num_candidates": 4, "selected_index": 0}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        res = client.mbr_summarize("Article text.", num_candidates=4)
        assert res["summary"] == "MBR summary."
        assert res["num_candidates"] == 4


def test_client_batch():
    client = NCMClient()
    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"summary": "Sum", "latency_ms": 100}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        batch_res = client.batch_summarize(["Doc 1", "Doc 2"])
        assert len(batch_res) == 2
        assert batch_res[0]["summary"] == "Sum"
        assert "client_elapsed_s" in batch_res[0]
