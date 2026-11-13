"""Resilient Python Client SDK for the Next-Concept Modeling Sovereign Summarizer Service."""
from __future__ import annotations

import time
from typing import Any, Literal

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class NCMClientError(RuntimeError):
    """Base error raised by NCMClient."""


class NCMAuthError(NCMClientError):
    """Raised on 401/403 authentication failures."""


class NCMServerBusyError(NCMClientError):
    """Raised when the server rejects a request with 503 queue backpressure."""


class NCMClient:
    """Resilient client for interacting with the local or remote NCM microservice."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8031",
        timeout: float = 60.0,
        api_key: str | None = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        try:
            resp = self.session.request(
                method,
                f"{self.base_url}{path}",
                timeout=self.timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise NCMClientError(f"Connection error calling {path}: {exc}") from exc

        if resp.status_code in (401, 403):
            raise NCMAuthError(f"Authentication failed ({resp.status_code}): {resp.text}")
        if resp.status_code == 503:
            raise NCMServerBusyError(f"Server busy ({resp.status_code}): {resp.text}")
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise NCMClientError(f"HTTP {resp.status_code} calling {path}: {resp.text}") from exc
        return resp.json()

    def health(self) -> dict[str, Any]:
        """Queries service health, device, and quantization mode."""
        return self._request("GET", "/health")

    def ready(self) -> dict[str, Any]:
        """Queries readiness probe."""
        return self._request("GET", "/ready")

    def summarize(self, text: str) -> dict[str, Any]:
        """Generates a single-shot summary of the provided text."""
        return self._request("POST", "/summarize", json={"text": text})

    def batch_summarize_server(self, texts: list[str]) -> dict[str, Any]:
        """Sends up to 32 documents in a single batched POST /batch_summarize call."""
        return self._request("POST", "/batch_summarize", json={"texts": texts})

    def mbr_summarize(
        self,
        text: str,
        num_candidates: int = 4,
        temperature: float = 0.7,
        utility: Literal["rougeL", "jaccard", "cosine"] = "rougeL",
    ) -> dict[str, Any]:
        """Generates candidates and selects the MBR consensus centroid."""
        return self._request(
            "POST",
            "/mbr_summarize",
            json={
                "text": text,
                "num_candidates": num_candidates,
                "temperature": temperature,
                "utility": utility,
            },
        )

    def progressive_summarize(self, text: str) -> dict[str, Any]:
        """Extracts text across all 9 discrete concept levels (k=0..8)."""
        return self._request("POST", "/progressive_summarize", json={"text": text})

    def batch_mbr_summarize_server(
        self,
        texts: list[str],
        num_candidates: int = 4,
        temperature: float = 0.7,
        utility: Literal["rougeL", "jaccard", "cosine"] = "rougeL",
    ) -> dict[str, Any]:
        """Sends up to 16 documents to POST /batch_mbr_summarize."""
        return self._request(
            "POST",
            "/batch_mbr_summarize",
            json={
                "texts": texts,
                "num_candidates": num_candidates,
                "temperature": temperature,
                "utility": utility,
            },
        )

    def longdoc_summarize(
        self,
        text: str,
        max_chunk_chars: int = 1600,
        overlap_chars: int = 160,
        use_mbr: bool = False,
        num_candidates: int = 4,
        redact_pii: bool = False,
        restore_pii: bool = False,
        audit_receipt: bool = False,
    ) -> dict[str, Any]:
        """Summarizes multi-page documents via POST /longdoc_summarize with optional PII & audit receipt."""
        return self._request(
            "POST",
            "/longdoc_summarize",
            json={
                "text": text,
                "max_chunk_chars": max_chunk_chars,
                "overlap_chars": overlap_chars,
                "use_mbr": use_mbr,
                "num_candidates": num_candidates,
                "redact_pii": redact_pii,
                "restore_pii": restore_pii,
                "audit_receipt": audit_receipt,
            },
        )

    def sudoku(
        self,
        puzzle: str,
        best_of_k: int = 1,
        temperature: float = 0.4,
        repair: bool = True,
    ) -> dict[str, Any]:
        """Solves and validates an 81-char Sudoku puzzle via POST /sudoku."""
        return self._request(
            "POST",
            "/sudoku",
            json={
                "puzzle": puzzle,
                "best_of_k": best_of_k,
                "temperature": temperature,
                "repair": repair,
            },
        )

    def chat_completions(self, prompt: str, model: str = "ncm-xsum-88M") -> dict[str, Any]:
        """Calls the OpenAI-compatible POST /v1/chat/completions endpoint."""
        return self._request(
            "POST",
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        )

    def batch_summarize(
        self,
        texts: list[str],
        mode: Literal["standard", "mbr"] = "standard",
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Processes a list of documents sequentially and collects latency metrics."""
        results = []
        for idx, text in enumerate(texts):
            t0 = time.perf_counter()
            if mode == "mbr":
                res = self.mbr_summarize(text, **kwargs)
            else:
                res = self.summarize(text)
            res["client_elapsed_s"] = round(time.perf_counter() - t0, 3)
            res["doc_index"] = idx
            results.append(res)
        return results
