"""Hardened local-first NCM inference, streaming concept inspection, and OpenAI-compatible microservice.

Endpoints:
- POST /summarize: Single-shot summarization.
- POST /batch_summarize: Batched summarization (1..32 docs).
- POST /mbr_summarize: Minimum Bayes Risk consensus decoding.
- POST /batch_mbr_summarize: Batched MBR consensus decoding (up to 16 docs).
- POST /progressive_summarize: Coarse-to-fine inspection across all code levels.
- POST /progressive_stream: Server-Sent Events (SSE) stream yielding levels 0..K-1 incrementally.
- POST /sudoku: Sudoku board solver + row/col/box validity checker.
- POST /v1/chat/completions & GET /v1/models: OpenAI-compatible drop-in adapter.
- GET  /visualize: XSS-safe interactive HTML concept progression explorer.
- GET  /health, /ready, /metrics: Liveness, readiness, and Prometheus metrics + latency histogram.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import anyio  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402

from ncm_bench.inference import (  # noqa: E402
    ensure_offline_env,
    load_bundle,
    resolve_device,
    solve_sudoku,
    summarize_texts,
)
from ncm_bench.mbr import generate_mbr_summary  # noqa: E402
from ncm_bench.progressive import progressive_decode, render_progressive_html  # noqa: E402

ensure_offline_env()

try:
    from opentelemetry import trace as _otel_trace

    _TRACER = _otel_trace.get_tracer("ncm-benchmarks")
except ImportError:  # pragma: no cover
    _TRACER = None

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)

    @field_validator("text")
    @classmethod
    def _must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must not be empty or whitespace-only")
        return stripped


class BatchSummarizeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=32)

    @field_validator("texts")
    @classmethod
    def _validate_items(cls, items: list[str]) -> list[str]:
        cleaned = [t.strip() for t in items]
        if any(not t for t in cleaned):
            raise ValueError("every item in texts must be non-empty")
        if any(len(t) > 100_000 for t in cleaned):
            raise ValueError("individual text items must not exceed 100,000 characters")
        return cleaned


class MBRRequest(SummarizeRequest):
    num_candidates: int = Field(default=4, ge=1, le=16)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    utility: Literal["rougeL", "jaccard", "cosine"] = "rougeL"


class BatchMBRRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=16)
    num_candidates: int = Field(default=4, ge=1, le=16)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    utility: Literal["rougeL", "jaccard", "cosine"] = "rougeL"


class ProgressiveRequest(SummarizeRequest):
    use_penalties: bool = True


class SudokuRequest(BaseModel):
    puzzle: str = Field(..., min_length=81, max_length=256)
    best_of_k: int = Field(default=1, ge=1, le=16)
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    repair: bool = True


class ChatMessage(BaseModel):
    role: str
    content: str = Field(..., min_length=1, max_length=100_000)


class OpenAIChatRequest(BaseModel):
    model: str = "ncm-xsum-88M"
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=32)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class LongDocRequest(SummarizeRequest):
    max_chunk_chars: int = Field(default=1600, ge=256, le=8000)
    overlap_chars: int = Field(default=160, ge=0, le=1000)
    use_mbr: bool = False
    num_candidates: int = Field(default=4, ge=1, le=16)
    redact_pii: bool = False
    restore_pii: bool = False
    audit_receipt: bool = False


class TriageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    labels: list[str] | None = Field(default=None, max_length=32)
    max_level: int = Field(default=2, ge=0, le=16)
    best_of_k: int = Field(default=3, ge=1, le=16)
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    coarse_only: bool = False
    adaptive_compute: bool = False
    redact_pii: bool = False
    restore_pii: bool = False
    audit_receipt: bool = False


class BatchTriageRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)
    labels: list[str] | None = Field(default=None, max_length=32)
    max_level: int = Field(default=2, ge=0, le=16)
    best_of_k: int = Field(default=3, ge=1, le=16)
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    coarse_only: bool = False
    adaptive_compute: bool = False
    deduplicate: bool = True
    dedup_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    redact_pii: bool = False
    restore_pii: bool = False
    audit_receipt: bool = False


CKPT: dict[str, Any] | None = None


def load(
    cfg_root: str,
    ck: str,
    device: str,
    quantize: str = "none",
    verify_checksums: bool = True,
) -> dict[str, Any]:
    """Backwards-compatible loader used by legacy scripts."""
    del cfg_root
    ck_path = Path(ck).resolve()
    parts = ck_path.parts
    if "artifacts" in parts:
        idx = len(parts) - 1 - parts[::-1].index("artifacts")
        home = Path(*parts[:idx])
        rel_task = "/".join(parts[idx + 1 :]) or "xsum"
    else:
        home = REPO_ROOT
        rel_task = "xsum"
    return load_bundle(
        task=rel_task,
        device=device,
        quantize=quantize,
        verify_checksums=verify_checksums,
        root_dir=home,
    )


def summarize(S: dict[str, Any], text: str | list[int]) -> str:
    """Single-document summarization helper using unified batched inference path."""
    return summarize_texts(S, [text])[0]


def create_app(
    bundle: dict[str, Any] | None = None,
    *,
    sudoku_bundle: dict[str, Any] | None = None,
    load_s: float = 0.0,
    max_queue_depth: int = 8,
    request_timeout_s: float = 60.0,
    max_body_bytes: int = 2_000_000,
    rate_limit_rpm: int = 0,
    api_key: str | None = None,
    strict_auth: bool | None = None,
    cors_origins: list[str] | None = None,
):
    """Builds the hardened FastAPI application with dependency-injected bundles."""
    import fastapi
    from fastapi import Depends, Header, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

    configured_key = (
        api_key if api_key is not None else os.environ.get("NCM_API_KEY", "").strip() or None
    )
    enforce_strict_auth = (
        strict_auth
        if strict_auth is not None
        else os.environ.get("NCM_STRICT_AUTH", "").strip() == "1"
    )
    origins = (
        cors_origins
        if cors_origins is not None
        else [
            o.strip()
            for o in os.environ.get("NCM_CORS_ORIGINS", "").split(",")
            if o.strip()
        ]
    )

    state: dict[str, Any] = {
        "bundle": bundle,
        "sudoku_bundle": sudoku_bundle,
        "load_s": load_s,
        "requests_total": 0,
        "errors_total": 0,
        "rejected_503_total": 0,
        "rate_limited_429_total": 0,
        "latency_seconds_sum": 0.0,
        "latency_seconds_count": 0,
        "active_inferences": 0,
        "hist_buckets": {b: 0 for b in LATENCY_BUCKETS},
        "shutting_down": False,
    }
    model_lock = threading.Lock()
    queue_semaphore = threading.BoundedSemaphore(max(1, max_queue_depth))
    metrics_lock = threading.Lock()
    rate_lock = threading.Lock()
    rate_buckets: dict[str, deque[float]] = {}

    @asynccontextmanager
    async def lifespan(_app):
        yield
        state["shutting_down"] = True

    app = fastapi.FastAPI(
        title="Next-Concept Modeling Sovereign Service",
        description="Local-first coarse-to-fine concept generation, MBR, and Sudoku microservice",
        version="2.0.0",
        lifespan=lifespan,
    )

    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )

    def _inc_metric(key: str, delta: float = 1.0) -> None:
        with metrics_lock:
            state[key] += delta

    def _record_latency(elapsed_s: float) -> None:
        with metrics_lock:
            state["latency_seconds_sum"] += elapsed_s
            state["latency_seconds_count"] += 1
            for b in LATENCY_BUCKETS:
                if elapsed_s <= b:
                    state["hist_buckets"][b] += 1

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not configured_key:
            return
        if not authorization or not authorization[:7].lower() == "bearer ":
            raise fastapi.HTTPException(status_code=401, detail="Missing Bearer token")
        token = authorization[7:].strip()
        if not hmac.compare_digest(token.encode("utf-8"), configured_key.encode("utf-8")):
            raise fastapi.HTTPException(status_code=403, detail="Invalid API key")

    def require_optional_strict_auth(authorization: str | None = Header(default=None)) -> None:
        if enforce_strict_auth:
            require_auth(authorization)

    async def _run_serialized(fn, *, target_key: str = "bundle"):
        if state["shutting_down"]:
            raise fastapi.HTTPException(status_code=503, detail="Server shutting down")
        active_bundle = state[target_key]
        if active_bundle is None:
            raise fastapi.HTTPException(status_code=503, detail=f"Model '{target_key}' not loaded")
        if not queue_semaphore.acquire(blocking=False):
            _inc_metric("rejected_503_total")
            raise fastapi.HTTPException(
                status_code=503,
                detail="Server busy: inference queue full",
                headers={"Retry-After": "2"},
            )

        def _exec():
            t0 = time.perf_counter()
            with model_lock:
                if _TRACER is not None:
                    with _TRACER.start_as_current_span(f"ncm.inference.{target_key}"):
                        res = fn(active_bundle)
                else:
                    res = fn(active_bundle)
            return res, time.perf_counter() - t0

        _inc_metric("active_inferences", 1)
        try:
            res, elapsed = await asyncio.wait_for(
                anyio.to_thread.run_sync(_exec),
                timeout=max(0.5, request_timeout_s),
            )
            _record_latency(elapsed)
            return res, int(elapsed * 1000)
        except TimeoutError as exc:
            _inc_metric("errors_total")
            raise fastapi.HTTPException(status_code=504, detail="Inference timed out") from exc
        finally:
            _inc_metric("active_inferences", -1)
            queue_semaphore.release()

    @app.middleware("http")
    async def security_and_logging_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        _inc_metric("requests_total")

        # 1) Content-Length size guard before buffering request body
        raw_cl = request.headers.get("content-length")
        if raw_cl and raw_cl.isdigit() and int(raw_cl) > max_body_bytes:
            return JSONResponse(
                {"detail": f"Request exceeds max body size of {max_body_bytes} bytes"},
                status_code=413,
                headers={"X-Request-ID": req_id},
            )

        # 2) Optional sliding-window rate limiter per client IP
        if rate_limit_rpm > 0 and request.url.path not in ("/health", "/ready", "/metrics"):
            client_ip = request.client.host if request.client else "unknown"
            now = time.monotonic()
            with rate_lock:
                q = rate_buckets.setdefault(client_ip, deque())
                while q and (now - q[0]) > 60.0:
                    q.popleft()
                if len(q) >= rate_limit_rpm:
                    _inc_metric("rate_limited_429_total")
                    return JSONResponse(
                        {"detail": "Rate limit exceeded"},
                        status_code=429,
                        headers={"Retry-After": "10", "X-Request-ID": req_id},
                    )
                q.append(now)

        try:
            response = await call_next(request)
        except Exception:
            _inc_metric("errors_total")
            raise
        if response.status_code >= 500:
            _inc_metric("errors_total")

        response.headers["X-Request-ID"] = req_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path == "/visualize":
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"
            )
        log_line = {
            "request_id": req_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": int((time.perf_counter() - t0) * 1000),
        }
        print(json.dumps(log_line), file=sys.stderr, flush=True)
        return response

    @app.get("/health")
    def health():
        active_bundle = state["bundle"]
        return {
            "ok": active_bundle is not None,
            "device": active_bundle["device"] if active_bundle else "unloaded",
            "dtype": active_bundle.get("dtype", "fp32") if active_bundle else "unloaded",
            "format": active_bundle.get("ckpt_format", "none") if active_bundle else "none",
            "load_s": state["load_s"],
            "model": "ncm-xsum-88M",
            "sudoku_loaded": state["sudoku_bundle"] is not None,
            "quantize": active_bundle.get("quantize", "none") if active_bundle else "none",
            "auth_enabled": configured_key is not None,
        }

    @app.get("/ready")
    def ready():
        if state["bundle"] is None or state["shutting_down"]:
            return JSONResponse({"ready": False}, status_code=503)
        return {"ready": True}

    @app.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(require_optional_strict_auth)])
    def metrics():
        with metrics_lock:
            snap = dict(state)
            buckets = dict(state["hist_buckets"])
        lines = [
            "# HELP ncm_requests_total Total HTTP requests handled",
            "# TYPE ncm_requests_total counter",
            f"ncm_requests_total {int(snap['requests_total'])}",
            "# HELP ncm_request_errors_total Total 5xx responses",
            "# TYPE ncm_request_errors_total counter",
            f"ncm_request_errors_total {int(snap['errors_total'])}",
            "# HELP ncm_rejected_503_total Requests rejected by queue backpressure",
            "# TYPE ncm_rejected_503_total counter",
            f"ncm_rejected_503_total {int(snap['rejected_503_total'])}",
            "# HELP ncm_rate_limited_429_total Requests rejected by rate limiting",
            "# TYPE ncm_rate_limited_429_total counter",
            f"ncm_rate_limited_429_total {int(snap['rate_limited_429_total'])}",
            "# HELP ncm_active_inferences Current in-flight/queued inferences",
            "# TYPE ncm_active_inferences gauge",
            f"ncm_active_inferences {int(snap['active_inferences'])}",
            "# HELP ncm_model_loaded Whether the NCM bundle is loaded (1 or 0)",
            "# TYPE ncm_model_loaded gauge",
            f"ncm_model_loaded {1 if snap['bundle'] is not None else 0}",
            "# HELP ncm_inference_duration_seconds Inference latency distribution",
            "# TYPE ncm_inference_duration_seconds histogram",
        ]
        for b, count in buckets.items():
            lines.append(f'ncm_inference_duration_seconds_bucket{{le="{b}"}} {count}')
        lines.append(
            f'ncm_inference_duration_seconds_bucket{{le="+Inf"}} {int(snap["latency_seconds_count"])}'
        )
        lines.append(f"ncm_inference_duration_seconds_sum {snap['latency_seconds_sum']:.6f}")
        lines.append(f"ncm_inference_duration_seconds_count {int(snap['latency_seconds_count'])}")
        return "\n".join(lines) + "\n"

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    def list_models():
        models = []
        if state["bundle"] is not None:
            models.append({"id": "ncm-xsum-88M", "object": "model", "owned_by": "hardonian"})
        if state["sudoku_bundle"] is not None:
            models.append({"id": "ncm-sudoku-25M", "object": "model", "owned_by": "hardonian"})
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
    async def openai_chat_completions(req: OpenAIChatRequest):
        prompt = req.messages[-1].content
        if "sudoku" in req.model.lower():
            res, latency_ms = await _run_serialized(
                lambda b: solve_sudoku(b, [prompt], best_of_k=4, repair=True)[0],
                target_key="sudoku_bundle",
            )
            content = res["board"]
        else:
            content, latency_ms = await _run_serialized(lambda b: summarize_texts(b, [prompt])[0])
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len(prompt.split()) + len(content.split()),
            },
            "ncm_latency_ms": latency_ms,
        }

    @app.post("/longdoc_summarize", dependencies=[Depends(require_auth)])
    async def longdoc_summarize_endpoint(req: LongDocRequest):
        from ncm_bench.longdoc import (
            build_audit_receipt,
            redact_pii,
            restore_pii,
            summarize_long_document,
        )

        pii_map: dict[str, str] = {}
        proc_text = req.text
        if req.redact_pii:
            proc_text, pii_map = redact_pii(req.text)

        res, latency_ms = await _run_serialized(
            lambda b: summarize_long_document(
                b,
                proc_text,
                max_chunk_chars=req.max_chunk_chars,
                overlap_chars=req.overlap_chars,
                use_mbr=req.use_mbr,
                num_candidates=req.num_candidates,
            )
        )
        summary = res["summary"]
        if req.restore_pii and pii_map:
            summary = restore_pii(summary, pii_map)

        payload: dict[str, Any] = {
            "summary": summary,
            "num_chunks": res["num_chunks"],
            "pii_redactions": len(pii_map),
            "latency_ms": latency_ms,
        }
        if req.audit_receipt:
            active_bundle = state["bundle"] or {}
            payload["audit_receipt"] = build_audit_receipt(
                req.text,
                summary,
                active_bundle,
                pii_redactions=len(pii_map),
                mode="longdoc-mbr" if req.use_mbr else "longdoc",
            )
        return payload

    @app.post("/triage", dependencies=[Depends(require_auth)])
    async def triage_endpoint(req: TriageRequest):
        from ncm_bench.longdoc import build_audit_receipt
        from ncm_bench.triage import extract_structured_triage

        res, latency_ms = await _run_serialized(
            lambda b: extract_structured_triage(
                b,
                req.text,
                labels=req.labels,
                max_level=req.max_level,
                best_of_k=req.best_of_k,
                temperature=req.temperature,
                coarse_only=req.coarse_only,
                adaptive_compute=req.adaptive_compute,
                redact=req.redact_pii,
                restore=req.restore_pii,
            )
        )
        res["latency_ms"] = latency_ms
        if req.audit_receipt:
            active_bundle = state["bundle"] or {}
            res["audit_receipt"] = build_audit_receipt(
                req.text,
                res["summary"],
                active_bundle,
                pii_redactions=res.get("pii_redactions", 0),
                mode="triage",
            )
        return res

    @app.post("/batch_triage", dependencies=[Depends(require_auth)])
    async def batch_triage_endpoint(req: BatchTriageRequest):
        from ncm_bench.longdoc import build_audit_receipt
        from ncm_bench.triage import batch_extract_structured_triage

        res, latency_ms = await _run_serialized(
            lambda b: batch_extract_structured_triage(
                b,
                req.texts,
                labels=req.labels,
                max_level=req.max_level,
                best_of_k=req.best_of_k,
                temperature=req.temperature,
                coarse_only=req.coarse_only,
                adaptive_compute=req.adaptive_compute,
                deduplicate=req.deduplicate,
                dedup_threshold=req.dedup_threshold,
                redact=req.redact_pii,
                restore=req.restore_pii,
            )
        )
        res["latency_ms"] = latency_ms
        if req.audit_receipt:
            active_bundle = state["bundle"] or {}
            for src_text, item in zip(req.texts, res["results"], strict=True):
                item["audit_receipt"] = build_audit_receipt(
                    src_text,
                    item["summary"],
                    active_bundle,
                    pii_redactions=item.get("pii_redactions", 0),
                    mode="batch_triage",
                )
        return res

    @app.post("/summarize", dependencies=[Depends(require_auth)])
    async def summarize_endpoint(req: SummarizeRequest):
        summary, latency_ms = await _run_serialized(lambda b: summarize_texts(b, [req.text])[0])
        return {"summary": summary, "latency_ms": latency_ms, "words": len(summary.split())}

    @app.post("/batch_summarize", dependencies=[Depends(require_auth)])
    async def batch_summarize_endpoint(req: BatchSummarizeRequest):
        summaries, latency_ms = await _run_serialized(lambda b: summarize_texts(b, req.texts))
        return {"summaries": summaries, "count": len(summaries), "latency_ms": latency_ms}

    @app.post("/mbr_summarize", dependencies=[Depends(require_auth)])
    async def mbr_endpoint(req: MBRRequest):
        res, latency_ms = await _run_serialized(
            lambda b: generate_mbr_summary(
                b,
                req.text,
                num_candidates=req.num_candidates,
                temperature=req.temperature,
                utility=req.utility,
            )
        )
        res["latency_ms"] = latency_ms
        return res

    @app.post("/batch_mbr_summarize", dependencies=[Depends(require_auth)])
    async def batch_mbr_endpoint(req: BatchMBRRequest):
        results, latency_ms = await _run_serialized(
            lambda b: [
                generate_mbr_summary(
                    b,
                    t,
                    num_candidates=req.num_candidates,
                    temperature=req.temperature,
                    utility=req.utility,
                )
                for t in req.texts
            ]
        )
        return {"results": results, "count": len(results), "latency_ms": latency_ms}

    @app.post("/progressive_summarize", dependencies=[Depends(require_auth)])
    async def progressive_endpoint(req: ProgressiveRequest):
        res, latency_ms = await _run_serialized(
            lambda b: progressive_decode(b, req.text, use_penalties=req.use_penalties)
        )
        res["latency_ms"] = latency_ms
        return res

    @app.post("/progressive_stream", dependencies=[Depends(require_auth)])
    async def progressive_stream_endpoint(req: ProgressiveRequest):
        res, _ = await _run_serialized(
            lambda b: progressive_decode(b, req.text, use_penalties=req.use_penalties)
        )

        async def _sse_gen() -> AsyncIterator[str]:
            for lvl in res["levels"]:
                yield f"event: level\ndata: {json.dumps(lvl)}\n\n"
            yield f"event: done\ndata: {json.dumps({'final_summary': res['final_summary']})}\n\n"

        return StreamingResponse(_sse_gen(), media_type="text/event-stream")

    @app.post("/sudoku", dependencies=[Depends(require_auth)])
    async def sudoku_endpoint(req: SudokuRequest):
        res, latency_ms = await _run_serialized(
            lambda b: solve_sudoku(
                b,
                [req.puzzle],
                best_of_k=req.best_of_k,
                temperature=req.temperature,
                repair=req.repair,
            )[0],
            target_key="sudoku_bundle",
        )
        res["latency_ms"] = latency_ms
        return res

    @app.get("/visualize", response_class=HTMLResponse, dependencies=[Depends(require_optional_strict_auth)])
    async def visualize_endpoint(text: str = Query(default="", max_length=10_000)):
        sample = text.strip() or (
            "The UK government has announced a major investment into renewable energy infrastructure, "
            "aiming to double wind and solar capacity by 2030. Ministers said the move will lower bills "
            "and cut carbon emissions across England and Wales."
        )
        prog, _ = await _run_serialized(lambda b: progressive_decode(b, sample))
        return render_progressive_html(prog, prompt_text=sample)

    return app


def main():
    global CKPT
    ap = argparse.ArgumentParser(description="NCM Local Summarization & Concept Server")
    ap.add_argument("--port", type=int, default=8031)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--uds", default=None)
    ap.add_argument("--quantize", choices=["none", "int8"], default="none")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-queue-depth", type=int, default=8)
    ap.add_argument("--request-timeout-s", type=float, default=60.0)
    ap.add_argument("--max-body-bytes", type=int, default=2_000_000)
    ap.add_argument("--rate-limit-rpm", type=int, default=0)
    ap.add_argument("--with-sudoku", action="store_true", help="Also load sudoku/hard model for POST /sudoku")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--skip-checksum", action="store_true")
    ap.add_argument("--strict-checksums", action="store_true")
    a = ap.parse_args()

    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.uds:
        if not os.environ.get("NCM_API_KEY"):
            print(
                f"[WARNING] Binding to non-loopback host '{a.host}' without NCM_API_KEY set!",
                file=sys.stderr,
                flush=True,
            )

    home = ensure_offline_env()
    device = resolve_device(a.device)
    t0 = time.perf_counter()
    CKPT = load_bundle(
        task="xsum",
        device=device,
        quantize=a.quantize,
        dtype=a.dtype,
        verify_checksums=not a.skip_checksum,
        strict_checksums=a.strict_checksums,
        compile_model=a.compile,
        root_dir=home,
    )
    sudoku_ckpt = None
    if a.with_sudoku:
        sudoku_ckpt = load_bundle(
            task="sudoku/hard",
            device=device,
            quantize=a.quantize,
            dtype=a.dtype,
            verify_checksums=not a.skip_checksum,
            strict_checksums=a.strict_checksums,
            root_dir=home,
        )
    load_s = round(time.perf_counter() - t0, 1)
    print(f"[ncm-summarizer] loaded in {load_s}s on {device} (quantize={a.quantize}, dtype={a.dtype})", flush=True)

    import uvicorn

    app = create_app(
        CKPT,
        sudoku_bundle=sudoku_ckpt,
        load_s=load_s,
        max_queue_depth=a.max_queue_depth,
        request_timeout_s=a.request_timeout_s,
        max_body_bytes=a.max_body_bytes,
        rate_limit_rpm=a.rate_limit_rpm,
    )
    if a.uds:
        uvicorn.run(app, uds=a.uds, log_level="warning")
    else:
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
