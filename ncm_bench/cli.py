"""Unified command-line interface for NCM benchmarks, sovereign inference, and concept inspection."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def get_model_bundle(quantize: str = "none", device: str = None):
    from service.summarize_server import load
    target_device = device or os.environ.get("NCM_DEVICE") or ("cuda" if False else "cpu")
    config_path = str(ROOT_DIR / "artifacts/xsum/config.yaml")
    ckpt_dir = str(ROOT_DIR / "artifacts/xsum")
    return load(config_path, ckpt_dir, target_device, quantize=quantize)


def cmd_serve(args):
    from service.summarize_server import main as serve_main
    sys.argv = ["summarize_server.py", "--port", str(args.port), "--host", args.host, "--quantize", args.quantize]
    if args.device:
        sys.argv.extend(["--device", args.device])
    serve_main()


def cmd_summarize(args):
    text = args.text
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if not text:
        print("Error: No text provided via --text or standard input.", file=sys.stderr)
        sys.exit(1)

    print(f"[ncm-bench] Loading model (quantize={args.quantize})...", file=sys.stderr)
    bundle = get_model_bundle(quantize=args.quantize, device=args.device)
    from service.summarize_server import summarize
    summary = summarize(bundle, text)
    if args.json:
        print(json.dumps({"summary": summary, "words": len(summary.split())}, indent=2))
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
    bundle = get_model_bundle(quantize=args.quantize, device=args.device)
    from ncm_bench.mbr import generate_mbr_summary
    res = generate_mbr_summary(bundle, text, num_candidates=args.candidates, temperature=args.temperature, utility=args.utility)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"Selected Summary (Cand #{res['selected_index']}):\n{res['summary']}\n")
        print(f"Candidate Pool Scores (Utility={args.utility}):")
        for i, (c, score) in enumerate(zip(res["candidates"], res["scores"])):
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

    print(f"[ncm-bench] Inspecting concept hierarchy across all 9 levels...", file=sys.stderr)
    bundle = get_model_bundle(quantize=args.quantize, device=args.device)
    from ncm_bench.progressive import progressive_decode, render_progressive_html
    res = progressive_decode(bundle, text)

    if args.html:
        html = render_progressive_html(res, prompt_text=text)
        out_path = Path(args.html)
        out_path.write_text(html, encoding="utf-8")
        print(f"[ncm-bench] Wrote interactive visualizer to {out_path.resolve()}", file=sys.stderr)
    elif args.json:
        print(json.dumps(res, indent=2))
    else:
        print("=" * 70)
        print("HIERARCHICAL COARSE-TO-FINE PROGRESSION (Levels 0 -> 8)")
        print("=" * 70)
        for lvl in res["levels"]:
            print(f"Level {lvl['level']:2d} [{lvl['codes_at_level']:2d} codes, {lvl['total_codes_so_far']:2d} accum]: {lvl['text'] or '(initial gist)'}")
        print("=" * 70)
        print(f"Final Summary: {res['final_summary']}")


def cmd_verify(args):
    from scripts.fetch_artifacts import load_checksums, sha256_file
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
        description="NCM Sovereign Inference Engine, Benchmark Suite, & Concept Inspector"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # serve
    p_serve = sub.add_parser("serve", help="Run local REST API server")
    p_serve.add_argument("--port", type=int, default=8031, help="Port to bind (default: 8031)")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    p_serve.add_argument("--quantize", choices=["none", "int8"], default="none", help="Inference quantization")
    p_serve.add_argument("--device", default=None, help="Target device (cpu, cuda)")
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

    # verify
    p_ver = sub.add_parser("verify", help="Verify model and data SHA256 checksums")
    p_ver.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

