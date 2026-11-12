"""High-throughput enterprise document triage processor using local NCM inference."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from service.client import NCMClient


def main():
    parser = argparse.ArgumentParser(description="NCM Sovereign Batch Triage Engine")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file containing documents (with 'text' field)")
    parser.add_argument("--output", "-o", default="triage_results.jsonl", help="Output JSONL destination")
    parser.add_argument("--mode", choices=["standard", "mbr"], default="standard", help="Generation mode")
    parser.add_argument("--candidates", type=int, default=4, help="MBR candidate count")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8031", help="NCM service endpoint")
    args = parser.parse_args()

    client = NCMClient(base_url=args.endpoint)
    try:
        health = client.health()
        print(f"[batch-triage] Connected to NCM server: {health['model']} on {health['device']} (quantize={health['quantize']})")
    except Exception as e:
        print(f"[batch-triage] Error connecting to service at {args.endpoint}: {e}", file=sys.stderr)
        print("Tip: Start the service first with `run_service.bat` or `python service/summarize_server.py`", file=sys.stderr)
        sys.exit(1)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: Input file {in_path} not found.", file=sys.stderr)
        sys.exit(1)

    documents = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))

    print(f"[batch-triage] Loaded {len(documents)} documents. Starting processing in mode='{args.mode}'...")
    latencies = []
    out_path = Path(args.output)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for idx, doc in enumerate(documents):
            text = doc.get("text", "")
            doc_id = doc.get("id", idx)
            t0 = time.perf_counter()
            if args.mode == "mbr":
                res = client.mbr_summarize(text, num_candidates=args.candidates)
            else:
                res = client.summarize(text)
            elapsed_s = time.perf_counter() - t0
            latencies.append(elapsed_s)

            record = {
                "id": doc_id,
                "summary": res["summary"],
                "latency_s": round(elapsed_s, 3),
                "words": len(res["summary"].split()),
            }
            if "selected_index" in res:
                record["mbr_selected_idx"] = res["selected_index"]
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            rate = 60.0 / (sum(latencies) / len(latencies))
            print(f"  [{idx+1:3d}/{len(documents)}] doc_id={doc_id} | {elapsed_s:.2f}s | {rate:.1f} docs/min | {res['summary'][:60]}...")

    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.50)]
    p90 = latencies[int(n * 0.90)]
    p99 = latencies[min(int(n * 0.99), n - 1)]

    print("\n" + "=" * 55)
    print("BATCH TRIAGE EXECUTION SUMMARY")
    print("=" * 55)
    print(f"Total documents processed: {n}")
    print(f"Mean latency per document: {sum(latencies)/n:.2f}s")
    print(f"Latency p50:               {p50:.2f}s")
    print(f"Latency p90:               {p90:.2f}s")
    print(f"Latency p99:               {p99:.2f}s")
    print(f"Throughput:                {60.0/(sum(latencies)/n):.1f} documents/minute")
    print(f"Results saved to:          {out_path.resolve()}")
    print("=" * 55)


if __name__ == "__main__":
    main()
