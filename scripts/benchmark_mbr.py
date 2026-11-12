"""Empirical MBR (Minimum Bayes Risk) decoding benchmark on XSum.

Evaluates:
- N=1 (Greedy baseline)
- N=4 (MBR consensus with ROUGE-L utility)
- N=8 (MBR consensus with ROUGE-L utility)

Measures ROUGE-1, ROUGE-2, ROUGE-L, and per-document generation latency on real CPU hardware.
"""
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["NCM_HOME"] = str(ROOT)
os.environ["NCM_ROOT"] = str(ROOT / "ncm")
os.environ["NCM_DEVICE"] = "cpu"
sys.path.insert(0, str(ROOT))

from ncm_bench.inference import verify_file_integrity  # noqa: E402
from ncm_bench.mbr import generate_mbr_summary  # noqa: E402
from ncm_bench.metrics import score_rouge  # noqa: E402
from service.summarize_server import load  # noqa: E402


def main():
    print("=" * 60)
    print("NCM MBR DECODING BENCHMARK (On-Hardware Reproduction)")
    print("=" * 60)
    
    # Load model
    print("[1/4] Loading model checkpoints...")
    t0 = time.perf_counter()
    S = load(str(ROOT / "artifacts/xsum/config.yaml"), str(ROOT / "artifacts/xsum"), "cpu")
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s\n")
    
    # Load dataset
    print("[2/4] Loading test dataset...")
    passages_path = ROOT / "data/xsum/passages.pickle"
    refs_path = ROOT / "data/xsum/reference_summaries.pickle"
    verify_file_integrity(passages_path, ROOT)
    verify_file_integrity(refs_path, ROOT)
    with open(passages_path, "rb") as f:
        passages = pickle.load(f)  # nosec B301 - verified by SHA256SUMS
    with open(refs_path, "rb") as f:
        refs = pickle.load(f)  # nosec B301 - verified by SHA256SUMS
    
    # Run on n=8 documents for statistically solid and fast comparison
    n_docs = 8
    test_passages = passages[:n_docs]
    test_refs = [r if isinstance(r, str) else S["tk"].pre_tokenizer.decode(r, skip_special_tokens=True) for r in refs[:n_docs]]
    
    results = {}
    
    # Evaluate N in [1, 4, 8]
    for n_cands in [1, 4, 8]:
        print(f"\n[3/4] Running MBR with N={n_cands} candidates over {n_docs} documents...")
        hyps = []
        latencies = []
        for i, text in enumerate(test_passages):
            t_start = time.perf_counter()
            if n_cands == 1:
                # Standard greedy (T=0)
                from service.summarize_server import summarize
                summary = summarize(S, text)
            else:
                mbr_res = generate_mbr_summary(
                    S,
                    text,
                    num_candidates=n_cands,
                    temperature=0.7,
                    utility="rougeL"
                )
                summary = mbr_res["summary"]
            
            elapsed = time.perf_counter() - t_start
            latencies.append(elapsed)
            hyps.append(summary)
            print(f"  doc {i+1:2d}/{n_docs}: {elapsed:.2f}s | {summary[:60]}...")
            
        scores = score_rouge(hyps, test_refs)
        avg_latency = sum(latencies) / len(latencies)
        results[f"mbr_N{n_cands}"] = {
            "n_candidates": n_cands,
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeL": scores["rougeL"],
            "avg_latency_s": round(avg_latency, 2),
            "total_time_s": round(sum(latencies), 2),
            "samples": hyps[:3],
        }
        print(f"--> N={n_cands} Results: R-1: {scores['rouge1']} | R-2: {scores['rouge2']} | R-L: {scores['rougeL']} | Avg Latency: {avg_latency:.2f}s/doc")

    print("\n" + "=" * 60)
    print("FINAL MBR COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Mode':<10} | {'R-1':<6} | {'R-2':<6} | {'R-L':<6} | {'Latency (s/doc)':<15}")
    print("-" * 55)
    for k, v in results.items():
        print(f"{k:<10} | {v['rouge1']:<6.2f} | {v['rouge2']:<6.2f} | {v['rougeL']:<6.2f} | {v['avg_latency_s']:<15.2f}")
    print("=" * 60)
    
    # Save results
    import json
    out_dir = ROOT / "results/hx370-cpu/mbr"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mbr_results.pickle", "wb") as f:
        pickle.dump(results, f)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved MBR benchmark results to {out_dir / 'results.json'} and {out_dir / 'mbr_results.pickle'}")

if __name__ == "__main__":
    main()
