"""Local-first summarizer and concept inspection service around Next-Concept Modeling (XSum).

One MIT-licensed 88M-param generator + 60M tokenizer (654 MB total), loaded once,
serves:
- POST /summarize: Standard single-shot summarization.
- POST /batch_summarize: Batched summarization (1..32 docs per call).
- POST /mbr_summarize: Minimum Bayes Risk consensus decoding (higher quality).
- POST /progressive_summarize: Hierarchical coarse-to-fine concept inspection across levels (k=0..8).
- GET  /visualize: XSS-safe interactive web UI slider visualizer.
- GET  /health, /ready, /metrics: Liveness, readiness, and Prometheus telemetry.

Hardened for sovereign/air-gapped deployments:
- Strict Pydantic v2 input bounds (prevents OOM/DoS via uncapped payloads)
- Pre-deserialization SHA-256 checkpoint verification
- Optional constant-time Bearer token auth via NCM_API_KEY
- Bounded concurrency queue with HTTP 503 backpressure
- Security headers + X-Request-ID correlation + structured JSON logs
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydantic import BaseModel, Field, field_validator  # noqa: E402

from ncm_bench.inference import (  # noqa: E402
    ensure_offline_env,
    load_bundle,
    resolve_device,
    summarize_texts,
)
from ncm_bench.mbr import generate_mbr_summary  # noqa: E402
from ncm_bench.progressive import progressive_decode, render_progressive_html  # noqa: E402

ensure_offline_env()


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
    utility: Literal["rougeL", "jaccard"] = "rougeL"


class ProgressiveRequest(SummarizeRequest):
    use_penalties: bool = True


CKPT: dict[str, Any] | None = None  # populated at CLI startup


def load(
    cfg_root: str,
    ck: str,
    device: str,
    quantize: str = "none",
    verify_checksums: bool = True,
) -> dict[str, Any]:
    """Backward-compatible loader delegating to ncm_bench.inference.load_bundle."""
    del cfg_root  # retained for signature compatibility with existing callers
    ck_path = Path(ck).resolve()
    artifacts_root = ck_path.parent
    # Derive task relative to artifacts/ when possible; fallback to standard 'xsum'
    try:
        rel_task = ck_path.relative_to(artifacts_root).as_posix()
    except ValueError:
        rel_task = "xsum"
    home = artifacts_root.parent if artifacts_root.name == "artifacts" else REPO_ROOT
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
    load_s: float = 0.0,
    max_queue_depth: int = 8,
    api_key: str | None = None,
):
    """Builds the hardened FastAPI application with dependency-injected model bundle."""
    import fastapi
    from fastapi import Depends, Header, Query, Request
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

    configured_key = api_key if api_key is not None else os.environ.get("NCM_API_KEY", "").strip() or None

    app = fastapi.FastAPI(
        title="Next-Concept Modeling Local Service",
        description="Local-first coarse-to-fine concept generation & summarization API",
        version="2.1.0",
    )

    state: dict[str, Any] = {
        "bundle": bundle,
        "load_s": load_s,
        "requests_total": 0,
        "errors_total": 0,
        "rejected_503_total": 0,
        "latency_seconds_sum": 0.0,
        "active_inferences": 0,
    }
    model_lock = threading.Lock()
    queue_semaphore = threading.BoundedSemaphore(max(1, max_queue_depth))
    metrics_lock = threading.Lock()

    def _inc_metric(key: str, delta: float = 1.0) -> None:
        with metrics_lock:
            state[key] += delta

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not configured_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise fastapi.HTTPException(status_code=401, detail="Missing Bearer token")
        token = authorization[7:].strip()
        if not hmac.compare_digest(token.encode("utf-8"), configured_key.encode("utf-8")):
            raise fastapi.HTTPException(status_code=403, detail="Invalid API key")

    def _run_serialized(fn):
        active_bundle = state["bundle"]
        if active_bundle is None:
            raise fastapi.HTTPException(status_code=503, detail="Model not loaded")
        if not queue_semaphore.acquire(blocking=False):
            _inc_metric("rejected_503_total")
            raise fastapi.HTTPException(
                status_code=503,
                detail="Server busy: inference queue full",
                headers={"Retry-After": "2"},
            )
        t0 = time.perf_counter()
        _inc_metric("active_inferences", 1)
        try:
            with model_lock:
                res = fn(active_bundle)
            elapsed = time.perf_counter() - t0
            _inc_metric("latency_seconds_sum", elapsed)
            return res, int(elapsed * 1000)
        finally:
            _inc_metric("active_inferences", -1)
            queue_semaphore.release()

    @app.middleware("http")
    async def security_and_logging_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        _inc_metric("requests_total")
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
            "load_s": state["load_s"],
            "model": "ncm-xsum-88M",
            "quantize": active_bundle.get("quantize", "none") if active_bundle else "none",
            "auth_enabled": configured_key is not None,
        }

    @app.get("/ready")
    def ready():
        if state["bundle"] is None:
            return JSONResponse({"ready": False}, status_code=503)
        return {"ready": True}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        with metrics_lock:
            snap = dict(state)
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
            "# HELP ncm_inference_latency_seconds_sum Cumulative inference seconds",
            "# TYPE ncm_inference_latency_seconds_sum counter",
            f"ncm_inference_latency_seconds_sum {snap['latency_seconds_sum']:.6f}",
            "# HELP ncm_active_inferences Current in-flight/queued inferences",
            "# TYPE ncm_active_inferences gauge",
            f"ncm_active_inferences {int(snap['active_inferences'])}",
            "# HELP ncm_model_loaded Whether the NCM bundle is loaded (1 or 0)",
            "# TYPE ncm_model_loaded gauge",
            f"ncm_model_loaded {1 if snap['bundle'] is not None else 0}",
        ]
        return "\n".join(lines) + "\n"

    @app.post("/summarize", dependencies=[Depends(require_auth)])
    def summarize_endpoint(req: SummarizeRequest):
        summary, latency_ms = _run_serialized(lambda b: summarize_texts(b, [req.text])[0])
        return {"summary": summary, "latency_ms": latency_ms, "words": len(summary.split())}

    @app.post("/batch_summarize", dependencies=[Depends(require_auth)])
    def batch_summarize_endpoint(req: BatchSummarizeRequest):
        summaries, latency_ms = _run_serialized(lambda b: summarize_texts(b, req.texts))
        return {
            "summaries": summaries,
            "count": len(summaries),
            "latency_ms": latency_ms,
        }

    @app.post("/mbr_summarize", dependencies=[Depends(require_auth)])
    def mbr_endpoint(req: MBRRequest):
        res, latency_ms = _run_serialized(
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

    @app.post("/progressive_summarize", dependencies=[Depends(require_auth)])
    def progressive_endpoint(req: ProgressiveRequest):
        res, latency_ms = _run_serialized(
            lambda b: progressive_decode(b, req.text, use_penalties=req.use_penalties)
        )
        res["latency_ms"] = latency_ms
        return res

    @app.get("/visualize", response_class=HTMLResponse)
    def visualize_endpoint(text: str = Query(default="", max_length=10_000)):
        sample = text.strip() or (
            "The UK government has announced a major investment into renewable energy infrastructure, "
            "aiming to double wind and solar capacity by 2030. Ministers said the move will lower bills "
            "and cut carbon emissions across England and Wales."
        )
        prog, _ = _run_serialized(lambda b: progressive_decode(b, sample))
        return render_progressive_html(prog, prompt_text=sample)

    return app


def main():
    global CKPT
    ap = argparse.ArgumentParser(description="NCM Local Summarization & Concept Server")
    ap.add_argument("--port", type=int, default=8031)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--quantize", choices=["none", "int8"], default="none", help="Apply dynamic INT8 quantization on CPU")
    ap.add_argument("--device", default=None, help="Target device (cpu, cuda)")
    ap.add_argument("--max-queue-depth", type=int, default=8, help="Max queued/concurrent requests before HTTP 503")
    ap.add_argument("--skip-checksum", action="store_true", help="Skip SHA-256 verification before loading checkpoints")
    a = ap.parse_args()

    home = ensure_offline_env()
    device = resolve_device(a.device)
    t0 = time.perf_counter()
    CKPT = load_bundle(
        task="xsum",
        device=device,
        quantize=a.quantize,
        verify_checksums=not a.skip_checksum,
        root_dir=home,
    )
    load_s = round(time.perf_counter() - t0, 1)
    print(f"[ncm-summarizer] loaded in {load_s}s on {device} (quantize={a.quantize})", flush=True)

    import uvicorn

    app = create_app(CKPT, load_s=load_s, max_queue_depth=a.max_queue_depth)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
