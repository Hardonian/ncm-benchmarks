"""Resilient high-throughput enterprise document triage processor using local NCM inference."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service.client import NCMClient, NCMClientError  # noqa: E402


def load_completed_ids(out_path: Path) -> set[str]:
    """Reads already-processed document IDs from an existing output JSONL for --resume."""
    done: set[str] = set()
    if not out_path.exists():
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "id" in rec:
                    done.add(str(rec["id"]))
            except json.JSONDecodeError:
                continue
    return done


def dispatch_document(client: NCMClient, text: str, args: argparse.Namespace) -> dict[str, Any]:
    """Dispatches a single document to the configured NCMClient endpoint."""
    mode = getattr(args, "mode", "standard")
    if mode == "mbr":
        return client.mbr_summarize(text, num_candidates=args.candidates)
    if mode == "longdoc":
        return client.longdoc_summarize(
            text,
            num_candidates=args.candidates,
            redact_pii=args.redact_pii,
            restore_pii=args.restore_pii,
            audit_receipt=args.audit_receipt,
        )
    if mode == "triage":
        raw_labels = getattr(args, "labels", None)
        labels = [s.strip() for s in raw_labels.split(",") if s.strip()] if raw_labels else None
        return client.triage(
            text,
            labels=labels,
            max_level=getattr(args, "max_level", 2),
            best_of_k=args.candidates,
            coarse_only=bool(getattr(args, "coarse_only", False)),
            adaptive_compute=bool(getattr(args, "adaptive_compute", False)),
            redact_pii=args.redact_pii,
            restore_pii=args.restore_pii,
            audit_receipt=args.audit_receipt,
        )
    return client.summarize(text)


def _format_record(doc_id: Any, res: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    summary = str(res.get("summary", ""))
    record: dict[str, Any] = {
        "id": doc_id,
        "summary": summary,
        "latency_s": round(elapsed_s, 3),
        "words": len(summary.split()),
    }
    for key in ("category", "priority", "confidence", "schema_valid", "compute_tier", "entities"):
        if key in res:
            record[key] = res[key]
    if "selected_index" in res:
        record["mbr_selected_idx"] = res["selected_index"]
    if "audit_receipt" in res:
        record["audit_receipt"] = res["audit_receipt"]
    return record


def main():
    parser = argparse.ArgumentParser(description="NCM Sovereign Batch Triage Engine")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file containing documents ('text' field)")
    parser.add_argument("--output", "-o", default="triage_results.jsonl", help="Output JSONL destination")
    parser.add_argument("--errors-out", default="triage_errors.jsonl", help="Dead-letter JSONL for failed documents")
    parser.add_argument(
        "--mode",
        choices=["standard", "mbr", "longdoc", "triage"],
        default="standard",
        help="Generation mode",
    )
    parser.add_argument("--candidates", type=int, default=4, help="MBR or triage candidate count")
    parser.add_argument("--labels", default=None, help="Comma-separated labels for --mode triage")
    parser.add_argument("--max-level", type=int, default=2, help="Coarse exit level for --mode triage")
    parser.add_argument("--coarse-only", action="store_true", help="Exit after coarse levels in --mode triage")
    parser.add_argument("--adaptive-compute", action="store_true", help="Adapt candidate count in --mode triage")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel client worker threads")
    parser.add_argument("--limit", type=int, default=0, help="Max pending documents to process (0 = all)")
    parser.add_argument("--redact-pii", action="store_true", help="Scrub PII during longdoc/triage")
    parser.add_argument("--restore-pii", action="store_true", help="Restore PII placeholders in output")
    parser.add_argument("--audit-receipt", action="store_true", help="Include SHA-256 compliance receipts")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8031", help="NCM service endpoint")
    parser.add_argument("--api-key", default=os.environ.get("NCM_API_KEY"), help="Optional Bearer API key")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSONL without duplicating IDs")
    args = parser.parse_args()

    client = NCMClient(base_url=args.endpoint, api_key=args.api_key)
    try:
        health = client.health()
        print(
            f"[batch-triage] Connected to NCM server: {health['model']} "
            f"on {health['device']} (quantize={health['quantize']})"
        )
    except Exception as e:
        print(f"[batch-triage] Error connecting to service at {args.endpoint}: {e}", file=sys.stderr)
        print("Tip: Start the service first with `run_service.bat` or `python service/summarize_server.py`", file=sys.stderr)
        sys.exit(1)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: Input file {in_path} not found.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    err_path = Path(args.errors_out)
    done_ids = load_completed_ids(out_path) if args.resume else set()

    documents = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[batch-triage] Skipping malformed JSON at line {line_no}: {exc}", file=sys.stderr)
                continue
            doc_id = str(doc.get("id", line_no - 1))
            if doc_id in done_ids:
                continue
            doc["_resolved_id"] = doc.get("id", line_no - 1)
            documents.append(doc)

    if args.limit and args.limit > 0:
        documents = documents[: args.limit]

    if args.resume and done_ids:
        print(f"[batch-triage] Resume active: skipped {len(done_ids)} completed docs.")
    print(f"[batch-triage] Processing {len(documents)} documents in mode='{args.mode}' (concurrency={args.concurrency})...")

    stop_requested = False

    def _handle_signal(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"\n[batch-triage] Signal {signum} received; finishing active work and flushing...", file=sys.stderr)

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    latencies: list[float] = []
    failed_count = 0
    file_mode = "a" if args.resume else "w"

    def _process_one(doc: dict[str, Any]) -> tuple[Any, dict[str, Any] | None, float, str | None]:
        text = str(doc.get("text", "")).strip()
        doc_id = doc["_resolved_id"]
        if not text:
            return doc_id, None, 0.0, "empty text"
        t0 = time.perf_counter()
        try:
            res = dispatch_document(client, text, args)
            return doc_id, res, time.perf_counter() - t0, None
        except NCMClientError as exc:
            return doc_id, None, time.perf_counter() - t0, str(exc)

    def _record_outcome(out_f, err_f, idx_done: int, outcome: tuple[Any, dict[str, Any] | None, float, str | None]) -> None:
        nonlocal failed_count
        doc_id, res, elapsed_s, err = outcome
        if err is not None or res is None:
            failed_count += 1
            err_f.write(json.dumps({"id": doc_id, "error": err}) + "\n")
            err_f.flush()
            print(f"  [ERR] doc_id={doc_id}: {err}", file=sys.stderr)
            return
        latencies.append(elapsed_s)
        record = _format_record(doc_id, res, elapsed_s)
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()
        rate = 60.0 / (sum(latencies) / len(latencies))
        print(f"  [{idx_done:3d}/{len(documents)}] doc_id={doc_id} | {elapsed_s:.2f}s | {rate:.1f} docs/min | {record['summary'][:60]}...")

    with open(out_path, file_mode, encoding="utf-8") as out_f, open(err_path, file_mode, encoding="utf-8") as err_f:
        if args.concurrency <= 1:
            for idx, doc in enumerate(documents):
                if stop_requested:
                    break
                _record_outcome(out_f, err_f, idx + 1, _process_one(doc))
        else:
            workers = max(1, min(args.concurrency, len(documents) or 1))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_process_one, doc) for doc in documents]
                for idx_done, fut in enumerate(as_completed(futures), start=1):
                    _record_outcome(out_f, err_f, idx_done, fut.result())

    print("\n" + "=" * 55)
    print("BATCH TRIAGE EXECUTION SUMMARY")
    print("=" * 55)
    n = len(latencies)
    print(f"Succeeded:                 {n}")
    print(f"Failed (dead-lettered):    {failed_count}")
    if n > 0:
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[int(n * 0.50)]
        p90 = latencies_sorted[min(int(n * 0.90), n - 1)]
        p99 = latencies_sorted[min(int(n * 0.99), n - 1)]
        print(f"Mean latency per document: {sum(latencies)/n:.2f}s")
        print(f"Latency p50:               {p50:.2f}s")
        print(f"Latency p90:               {p90:.2f}s")
        print(f"Latency p99:               {p99:.2f}s")
        print(f"Throughput:                {60.0/(sum(latencies)/n):.1f} documents/minute")
    print(f"Results saved to:          {out_path.resolve()}")
    if failed_count > 0:
        print(f"Dead-letter log:           {err_path.resolve()}")
    print("=" * 55)


if __name__ == "__main__":
    main()
