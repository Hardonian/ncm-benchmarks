"""Unified command-line interface for NCM benchmarks, sovereign inference, and concept inspection."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def get_model_bundle(
    task: str = "xsum",
    quantize: str = "none",
    device: str | None = None,
    verify_checksums: bool = True,
):
    from ncm_bench.inference import load_bundle

    return load_bundle(
        task=task,
        device=device,
        quantize=quantize,
        verify_checksums=verify_checksums,
        root_dir=ROOT_DIR,
    )


def cmd_serve(args):
    from service.summarize_server import main as serve_main

    argv = [
        "summarize_server.py",
        "--port",
        str(args.port),
        "--host",
        args.host,
        "--quantize",
        args.quantize,
        "--max-queue-depth",
        str(args.max_queue_depth),
    ]
    if args.device:
        argv.extend(["--device", args.device])
    if args.skip_checksum:
        argv.append("--skip-checksum")
    sys.argv = argv
    serve_main()


def cmd_summarize(args):
    text = args.text
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if not text:
        print("Error: No text provided via --text or standard input.", file=sys.stderr)
        sys.exit(1)

    print(f"[ncm-bench] Loading model (quantize={args.quantize})...", file=sys.stderr)
    bundle = get_model_bundle(task="xsum", quantize=args.quantize, device=args.device)
    from ncm_bench.inference import summarize_texts

    t0 = time.perf_counter()
    summary = summarize_texts(bundle, [text])[0]
    latency_ms = int((time.perf_counter() - t0) * 1000)
    if args.json:
        print(
            json.dumps(
                {"summary": summary, "words": len(summary.split()), "latency_ms": latency_ms},
                indent=2,
            )
        )
    else:
        print(summary)


def cmd_mbr(args):
    text = args.text
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if not text:
        print("Error: No text provided via --text or standard input.", file=sys.stderr)
        sys.exit(1)

    print(f"[ncm-bench] Running MBR decoding with N={args.candidates} candidates...", file=sys.stderr)
    bundle = get_model_bundle(task="xsum", quantize=args.quantize, device=args.device)
    from ncm_bench.mbr import generate_mbr_summary

    res = generate_mbr_summary(
        bundle,
        text,
        num_candidates=args.candidates,
        temperature=args.temperature,
        utility=args.utility,
    )
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"Selected Summary (Cand #{res['selected_index']}):\n{res['summary']}\n")
        print(f"Candidate Pool Scores (Utility={args.utility}):")
        for i, (c, score) in enumerate(zip(res["candidates"], res["scores"], strict=True)):
            mark = "-> [BEST]" if i == res["selected_index"] else "  "
            print(f"  {mark} #{i} (score: {score:.3f}): {c[:80]}...")


def cmd_inspect(args):
    text = args.text
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if not text:
        text = (
            "The UK government has announced a major investment into renewable energy infrastructure, "
            "aiming to double wind and solar capacity by 2030. Ministers said the move will lower bills "
            "and cut carbon emissions across England and Wales."
        )

    print("[ncm-bench] Inspecting concept hierarchy across all 9 levels...", file=sys.stderr)
    bundle = get_model_bundle(task="xsum", quantize=args.quantize, device=args.device)
    from ncm_bench.progressive import progressive_decode, render_progressive_html

    res = progressive_decode(bundle, text)

    if args.html:
        html_doc = render_progressive_html(res, prompt_text=text)
        out_path = Path(args.html)
        out_path.write_text(html_doc, encoding="utf-8")
        print(f"[ncm-bench] Wrote interactive visualizer to {out_path.resolve()}", file=sys.stderr)
    elif args.json:
        print(json.dumps(res, indent=2))
    else:
        print("=" * 70)
        print("HIERARCHICAL COARSE-TO-FINE PROGRESSION (Levels 0 -> 8)")
        print("=" * 70)
        for lvl in res["levels"]:
            print(
                f"Level {lvl['level']:2d} [{lvl['codes_at_level']:2d} codes, "
                f"{lvl['total_codes_so_far']:2d} accum]: {lvl['text'] or '(initial gist)'}"
            )
        print("=" * 70)
        print(f"Final Summary: {res['final_summary']}")


def cmd_sudoku(args):
    from ncm_bench.inference import solve_sudoku, verify_file_integrity
    from ncm_bench.metrics import score_sudoku

    bundle = get_model_bundle(
        task="sudoku/hard",
        quantize=args.quantize,
        device=args.device,
        verify_checksums=not args.skip_checksum,
    )
    if args.puzzle:
        t0 = time.perf_counter()
        res = solve_sudoku(bundle, [args.puzzle])[0]
        res["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"Valid Sudoku: {res['valid']} ({res['latency_ms']} ms)")
            for row in res["board"].split("|"):
                if row:
                    print("  " + " ".join(row))
        return

    puzzles_path = ROOT_DIR / "data/sudoku/hard/puzzles.pickle"
    solutions_path = ROOT_DIR / "data/sudoku/hard/solutions.pickle"
    if not args.skip_checksum:
        verify_file_integrity(puzzles_path, ROOT_DIR)
        verify_file_integrity(solutions_path, ROOT_DIR)
    with open(puzzles_path, "rb") as f:
        puzzles = pickle.load(f)[: args.n]  # nosec B301 - verified by SHA256SUMS
    with open(solutions_path, "rb") as f:
        solutions = pickle.load(f)[: args.n]  # nosec B301 - verified by SHA256SUMS

    t0 = time.perf_counter()
    preds_raw = []
    for start in range(0, len(puzzles), args.batch_size):
        batch = puzzles[start : start + args.batch_size]
        preds_raw.extend(r["board"] for r in solve_sudoku(bundle, batch))
    elapsed = time.perf_counter() - t0

    refs_str = [bundle["tk"].pre_tokenizer.decode(s, skip_special_tokens=True) for s in solutions]
    stats = score_sudoku(preds_raw, refs_str)
    stats["wall_s"] = round(elapsed, 3)
    stats["boards_per_sec"] = round(len(puzzles) / max(elapsed, 1e-6), 2)
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        ci = stats["ci_95"]
        print(
            f"Sudoku Hard (n={stats['n_boards']}): "
            f"solved={stats['solved_pct']}% ({stats['solved_count']}/{stats['n_boards']}, "
            f"95% CI +/-{ci['margin_pp']}pp) | cell={stats['cell_pct']}% | "
            f"valid={stats['valid_pct']}% | {stats['wall_s']}s ({stats['boards_per_sec']} boards/s)"
        )


def cmd_benchmark(args):
    from ncm_bench.inference import summarize_texts, verify_file_integrity
    from ncm_bench.metrics import two_size_timing

    if args.task == "sudoku":
        args.puzzle = None
        cmd_sudoku(args)
        return

    passages_path = ROOT_DIR / "data/xsum/passages.pickle"
    if not args.skip_checksum:
        verify_file_integrity(passages_path, ROOT_DIR)
    with open(passages_path, "rb") as f:
        passages = pickle.load(f)  # nosec B301 - verified by SHA256SUMS

    def _timed_run(n_items: int) -> float:
        t0 = time.perf_counter()
        b = get_model_bundle(task="xsum", quantize=args.quantize, device=args.device)
        for i in range(0, n_items, args.batch_size):
            summarize_texts(b, passages[i : min(i + args.batch_size, n_items)])
        return time.perf_counter() - t0

    t_small = _timed_run(args.small_n)
    t_large = _timed_run(args.large_n)
    timing = two_size_timing(t_small, args.small_n, t_large, args.large_n)
    timing.update(
        {
            "small_n": args.small_n,
            "small_s": round(t_small, 3),
            "large_n": args.large_n,
            "large_s": round(t_large, 3),
            "batch_size": args.batch_size,
            "quantize": args.quantize,
        }
    )
    if args.json:
        print(json.dumps(timing, indent=2))
    else:
        print(
            f"2-Size Latency Benchmark (quantize={args.quantize}, bs={args.batch_size}):\n"
            f"  T({args.small_n})={timing['small_s']}s, T({args.large_n})={timing['large_s']}s\n"
            f"  Load={timing['model_load_s']}s | Marginal={timing['marginal_s_per_doc']}s/doc "
            f"({timing['items_per_min']} docs/min)"
        )


def cmd_verify(args):
    from scripts.fetch_artifacts import load_checksums, sha256_file

    del args
    sums = load_checksums()
    if not sums:
        print("Error: SHA256SUMS file not found or empty.", file=sys.stderr)
        sys.exit(1)
    all_ok = True
    print(f"Verifying {len(sums)} artifact checksums...")
    for rel_path, exp_hash in sums.items():
        p = ROOT_DIR / rel_path
        if not p.exists():
            print(f"  [MISSING] {rel_path}")
            all_ok = False
            continue
        actual = sha256_file(p)
        if actual == exp_hash:
            print(f"  [OK]      {rel_path}")
        else:
            print(f"  [FAILED]  {rel_path} (expected {exp_hash[:10]}..., got {actual[:10]}...)")
            all_ok = False
    if all_ok:
        print("\nAll artifact checksums successfully verified against pinned SHA256SUMS!")
    else:
        print("\nVerification failed.", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="ncm-bench",
        description="NCM Sovereign Inference Engine, Benchmark Suite, & Concept Inspector",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # serve
    p_serve = sub.add_parser("serve", help="Run local REST API server")
    p_serve.add_argument("--port", type=int, default=8031, help="Port to bind (default: 8031)")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    p_serve.add_argument("--quantize", choices=["none", "int8"], default="none", help="Inference quantization")
    p_serve.add_argument("--device", default=None, help="Target device (cpu, cuda)")
    p_serve.add_argument("--max-queue-depth", type=int, default=8, help="Max queued requests before HTTP 503")
    p_serve.add_argument("--skip-checksum", action="store_true", help="Skip SHA-256 verification before load")
    p_serve.set_defaults(func=cmd_serve)

    # summarize
    p_sum = sub.add_parser("summarize", help="Summarize text single-shot")
    p_sum.add_argument("--text", "-t", default=None, help="Input text to summarize")
    p_sum.add_argument("--quantize", choices=["none", "int8"], default="none")
    p_sum.add_argument("--device", default=None)
    p_sum.add_argument("--json", action="store_true", help="Output JSON response")
    p_sum.set_defaults(func=cmd_summarize)

    # mbr
    p_mbr = sub.add_parser("mbr", help="Run MBR consensus decoding")
    p_mbr.add_argument("--text", "-t", default=None, help="Input text to summarize")
    p_mbr.add_argument("--candidates", "-k", type=int, default=4, help="Number of candidates (default: 4)")
    p_mbr.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    p_mbr.add_argument("--utility", choices=["rougeL", "jaccard"], default="rougeL", help="Utility consensus metric")
    p_mbr.add_argument("--quantize", choices=["none", "int8"], default="none")
    p_mbr.add_argument("--device", default=None)
    p_mbr.add_argument("--json", action="store_true", help="Output JSON response")
    p_mbr.set_defaults(func=cmd_mbr)

    # inspect
    p_insp = sub.add_parser("inspect", help="Progressively inspect concept hierarchy (Levels 0..8)")
    p_insp.add_argument("--text", "-t", default=None, help="Input text")
    p_insp.add_argument("--html", help="Path to write standalone HTML visualizer slider")
    p_insp.add_argument("--quantize", choices=["none", "int8"], default="none")
    p_insp.add_argument("--device", default=None)
    p_insp.add_argument("--json", action="store_true", help="Output JSON response")
    p_insp.set_defaults(func=cmd_inspect)

    # sudoku
    p_sud = sub.add_parser("sudoku", help="Solve a Sudoku puzzle or run accuracy evaluation")
    p_sud.add_argument("--puzzle", "-p", default=None, help="81-char Sudoku puzzle (0, ., or _ for blanks)")
    p_sud.add_argument("--n", type=int, default=20, help="Number of test boards when --puzzle is omitted")
    p_sud.add_argument("--batch-size", type=int, default=16)
    p_sud.add_argument("--quantize", choices=["none", "int8"], default="none")
    p_sud.add_argument("--device", default=None)
    p_sud.add_argument("--skip-checksum", action="store_true", help="Skip SHA-256 verification before load")
    p_sud.add_argument("--json", action="store_true")
    p_sud.set_defaults(func=cmd_sudoku)

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run timed latency or accuracy benchmarks")
    p_bench.add_argument("--task", choices=["latency", "sudoku"], default="latency")
    p_bench.add_argument("--small-n", type=int, default=4)
    p_bench.add_argument("--large-n", type=int, default=16)
    p_bench.add_argument("--n", type=int, default=20)
    p_bench.add_argument("--batch-size", type=int, default=1)
    p_bench.add_argument("--quantize", choices=["none", "int8"], default="none")
    p_bench.add_argument("--device", default=None)
    p_bench.add_argument("--skip-checksum", action="store_true", help="Skip SHA-256 verification before load")
    p_bench.add_argument("--json", action="store_true")
    p_bench.set_defaults(func=cmd_benchmark)

    # verify
    p_ver = sub.add_parser("verify", help="Verify model and data SHA256 checksums")
    p_ver.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
