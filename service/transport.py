"""Async httpx SDK client, Pydantic schemas, binary VPC framing, dynamic batcher & circuit-breaker escalation."""
from __future__ import annotations

import importlib
import json
import struct
import time
import zlib
from collections.abc import Awaitable, Callable
from typing import Any

import anyio
import requests
from pydantic import BaseModel


class SummarizeResponseModel(BaseModel):
    summary: str
    latency_ms: float = 0.0


class TriageResponseModel(BaseModel):
    category: str
    confidence: float
    escalate_to_frontier: bool
    summary: str
    schema_valid: bool = True
    latency_ms: float = 0.0


class FeedbackResponseModel(BaseModel):
    status: str
    label: str
    known_labels: list[str]
    latency_ms: float = 0.0


class AsyncNCMClient:
    """Async sovereign NCM client supporting optional httpx AsyncClient or threaded requests transport."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8031",
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        transport: Any = None,
        request_fn: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout
        self.transport = transport
        self.request_fn = request_fn

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.request_fn is not None:
            return await self.request_fn(path, payload)
        try:
            httpx = importlib.import_module("httpx")
        except ImportError:
            httpx = None

        if httpx is not None:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                resp = await client.post(path, json=payload)
                resp.raise_for_status()
                return resp.json()

        def _sync_post() -> dict[str, Any]:
            resp = requests.post(
                f"{self.base_url}{path}",
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()

        return await anyio.to_thread.run_sync(_sync_post)

    async def summarize(self, text: str) -> SummarizeResponseModel:
        data = await self._post_json("/summarize", {"text": text})
        return SummarizeResponseModel(**data)

    async def triage(
        self,
        text: str,
        *,
        labels: list[str] | None = None,
        max_level: int = 2,
    ) -> TriageResponseModel:
        data = await self._post_json("/triage", {"text": text, "labels": labels, "max_level": max_level})
        return TriageResponseModel(**data)


def encode_binary_frame(payload: dict[str, Any]) -> bytes:
    """Packs JSON payload into a zlib-compressed length-prefixed binary RPC frame."""
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    compressed = zlib.compress(raw, level=6)
    return struct.pack(">I", len(compressed)) + compressed


def decode_binary_frame(frame: bytes) -> dict[str, Any]:
    """Unpacks a length-prefixed zlib-compressed binary RPC frame into a dictionary."""
    if len(frame) < 4:
        raise ValueError("Frame too short for 4-byte length header")
    (length,) = struct.unpack(">I", frame[:4])
    if len(frame) - 4 != length:
        raise ValueError(f"Frame length mismatch: expected {length}, got {len(frame) - 4}")
    raw = zlib.decompress(frame[4:])
    return json.loads(raw.decode("utf-8"))


class DynamicBatchCoalescer:
    """Coalesces multiple document inputs into bounded micro-batches and preserves per-item ordering."""

    def __init__(
        self,
        batch_fn: Callable[[list[str]], list[dict[str, Any]]],
        *,
        max_batch_size: int = 16,
    ):
        self.batch_fn = batch_fn
        self.max_batch_size = max(1, int(max_batch_size))
        self.batches_executed = 0

    def process(self, items: list[str]) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for idx in range(0, len(items), self.max_batch_size):
            chunk = items[idx : idx + self.max_batch_size]
            res = self.batch_fn(chunk)
            if len(res) != len(chunk):
                raise RuntimeError("Batch handler returned mismatched result count")
            outputs.extend(res)
            self.batches_executed += 1
        return outputs


class HybridCircuitBreakerClient:
    """Routes traffic to local sovereign NCM and fails over to a frontier fallback when open or escalated."""

    def __init__(
        self,
        local_client,
        fallback_fn: Callable[[str], dict[str, Any]],
        *,
        failure_threshold: int = 3,
        recovery_timeout_s: float = 30.0,
    ):
        self.local_client = local_client
        self.fallback_fn = fallback_fn
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_timeout_s = float(recovery_timeout_s)
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if (time.monotonic() - self.opened_at) >= self.recovery_timeout_s:
            self.opened_at = None
            self.failures = 0
            return False
        return True

    def triage_with_escalation(self, text: str, *, labels: list[str] | None = None) -> dict[str, Any]:
        if self.is_open:
            fb = dict(self.fallback_fn(text))
            fb["routed_via"] = "circuit_open_fallback"
            return fb
        try:
            res = dict(self.local_client.triage(text, labels=labels))
            self.failures = 0
            if res.get("escalate_to_frontier"):
                fb = dict(self.fallback_fn(text))
                fb["routed_via"] = "confidence_escalation"
                fb["local_preview"] = res
                return fb
            res["routed_via"] = "sovereign_ncm"
            return res
        except Exception:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = time.monotonic()
            fb = dict(self.fallback_fn(text))
            fb["routed_via"] = "error_fallback"
            return fb
