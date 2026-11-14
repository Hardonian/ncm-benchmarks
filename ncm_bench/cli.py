"""Unified command-line interface for NCM benchmarks, sovereign inference, training, and inspection."""
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
    strict_checksums: bool = False,
    dtype: str = "fp32",
    compile_model: bool = False,
):
    from ncm_bench.inference import load_bundle

    return load_bundle(
        task=task,
        device=device,
        quantize=quantize,
        dtype=dtype,
        verify_checksums=verify_checksums,
        strict_checksums=strict_checksums,
        compile_model=compile_model,
        root_dir=ROOT_DIR,
    )


def _read_text_arg(args) -> str:
    if getattr(args, "input_file", None):
        from ncm_bench.longdoc import extract_text_from_file

        return extract_text_from_file(Path(args.input_file))
    text = getattr(args, "text", None)
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    return (text or "").strip()


def _emit_output(payload: str, args) -> None:
    out_file = getattr(args, "output_file", None)
    if out_file:
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        print(f"[ncm-bench] Wrote output to {out_path}", file=sys.stderr)
    else:
        print(payload)


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
        "--dtype",
        args.dtype,
        "--max-queue-depth",
        str(args.max_queue_depth),
        "--request-timeout-s",
        str(args.request_timeout_s),
        "--max-body-bytes",
        str(args.max_body_bytes),
        "--rate-limit-rpm",
        str(args.rate_limit_rpm),
    ]
    if args.device:
        argv.extend(["--device", args.device])
    if args.uds:
        argv.extend(["--uds", args.uds])
    if getattr(args, "with_sudoku", False):
        argv.append("--with-sudoku")
    if args.skip_checksum:
        argv.append("--skip-checksum")
    if args.strict_checksums:
        argv.append("--strict-checksums")
    if args.compile:
        argv.append("--compile")
    sys.argv = argv
    serve_main()


def cmd_summarize(args):
    text = _read_text_arg(args)
    if not text:
        print("Error: No text provided via --text, --input-file, or standard input.", file=sys.stderr)
        sys.exit(1)

    from ncm_bench.longdoc import (
        build_audit_receipt,
        redact_pii,
        restore_pii,
        summarize_long_document,
    )

    pii_map: dict[str, str] = {}
    proc_text = text
    if args.redact_pii:
        proc_text, pii_map = redact_pii(text)

    print(f"[ncm-bench] Loading model (quantize={args.quantize}, dtype={args.dtype})...", file=sys.stderr)
    bundle = get_model_bundle(
        task="xsum",
        quantize=args.quantize,
        device=args.device,
        verify_checksums=not args.skip_checksum,
        strict_checksums=args.strict_checksums,
        dtype=args.dtype,
        compile_model=args.compile,
    )

    t0 = time.perf_counter()
    if args.long_doc or len(proc_text) > 1800:
        res = summarize_long_document(bundle, proc_text, use_mbr=False)
        summary = res["summary"]
        num_chunks = res["num_chunks"]
    else:
        from ncm_bench.inference import summarize_texts

        summary = summarize_texts(bundle, [proc_text])[0]
        num_chunks = 1
    latency_ms = int((time.perf_counter() - t0) * 1000)

    if args.restore_pii and pii_map:
        summary = restore_pii(summary, pii_map)

    if args.json or args.audit_receipt:
        payload: dict[str, object] = {
            "summary": summary,
            "words": len(summary.split()),
            "latency_ms": latency_ms,
            "num_chunks": num_chunks,
            "pii_redactions": len(pii_map),
        }
        if args.audit_receipt:
            payload["audit_receipt"] = build_audit_receipt(
                text,
                summary,
                bundle,
                pii_redactions=len(pii_map),
                mode="long-doc" if num_chunks > 1 else "standard",
            )
        _emit_output(json.dumps(payload, indent=2), args)
    else:
        _emit_output(summary, args)


def cmd_mbr(args):
    text = _read_text_arg(args)
    if not text:
        print("Error: No text provided via --text, --input-file, or standard input.", file=sys.stderr)
        sys.exit(1)

    print(f"[ncm-bench] Running MBR decoding with N={args.candidates} candidates...", file=sys.stderr)
    bundle = get_model_bundle(
        task="xsum",
        quantize=args.quantize,
        device=args.device,
        verify_checksums=not args.skip_checksum,
        strict_checksums=args.strict_checksums,
        dtype=args.dtype,
        compile_model=args.compile,
    )
    from ncm_bench.mbr import generate_mbr_summary

    res = generate_mbr_summary(
        bundle,
        text,
        num_candidates=args.candidates,
        temperature=args.temperature,
        utility=args.utility,
    )
    if args.json:
        _emit_output(json.dumps(res, indent=2), args)
    else:
        lines = [
            f"Selected Summary (Cand #{res['selected_index']}):",
            res["summary"],
            "",
            f"Candidate Pool Scores (Utility={args.utility}):",
        ]
        for i, (c, score) in enumerate(zip(res["candidates"], res["scores"], strict=True)):
            mark = "-> [BEST]" if i == res["selected_index"] else "  "
            lines.append(f"  {mark} #{i} (score: {score:.3f}): {c[:80]}...")
        _emit_output("\n".join(lines), args)


def cmd_inspect(args):
    text = _read_text_arg(args)
    if not text:
        text = (
            "The UK government has announced a major investment into renewable energy infrastructure, "
            "aiming to double wind and solar capacity by 2030. Ministers said the move will lower bills "
            "and cut carbon emissions across England and Wales."
        )

    print("[ncm-bench] Inspecting concept hierarchy across all 9 levels...", file=sys.stderr)
    bundle = get_model_bundle(
        task="xsum",
        quantize=args.quantize,
        device=args.device,
        verify_checksums=not args.skip_checksum,
        strict_checksums=args.strict_checksums,
        dtype=args.dtype,
        compile_model=args.compile,
    )
    from ncm_bench.progressive import progressive_decode, render_progressive_html

    res = progressive_decode(bundle, text)

    if args.html:
        html_doc = render_progressive_html(res, prompt_text=text)
        out_path = Path(args.html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html_doc, encoding="utf-8")
        print(f"[ncm-bench] Wrote interactive visualizer to {out_path.resolve()}", file=sys.stderr)
    if args.json:
        _emit_output(json.dumps(res, indent=2), args)
    elif not args.html:
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

    task_name = f"sudoku/{args.difficulty}"
    bundle = get_model_bundle(
        task=task_name,
        quantize=args.quantize,
        device=args.device,
        verify_checksums=not args.skip_checksum,
        strict_checksums=args.strict_checksums,
        dtype=args.dtype,
        compile_model=args.compile,
    )
    repair = not args.no_repair
    if args.puzzle:
        t0 = time.perf_counter()
        res = solve_sudoku(
            bundle,
            [args.puzzle],
            best_of_k=args.best_of_k,
            temperature=args.temperature,
            repair=repair,
        )[0]
        res["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        if args.json:
            _emit_output(json.dumps(res, indent=2), args)
        else:
            print(f"Valid Sudoku: {res['valid']} ({res['latency_ms']} ms)")
            for row in res["board"].split("|"):
                if row:
                    print("  " + " ".join(row))
        return

    puzzles_path = ROOT_DIR / f"data/sudoku/{args.difficulty}/puzzles.pickle"
    solutions_path = ROOT_DIR / f"data/sudoku/{args.difficulty}/solutions.pickle"
    if not args.skip_checksum:
        verify_file_integrity(puzzles_path, ROOT_DIR, strict=args.strict_checksums)
        verify_file_integrity(solutions_path, ROOT_DIR, strict=args.strict_checksums)
    with open(puzzles_path, "rb") as f:
        puzzles = pickle.load(f)[: args.n]  # nosec B301 - verified by SHA256SUMS
    with open(solutions_path, "rb") as f:
        solutions = pickle.load(f)[: args.n]  # nosec B301 - verified by SHA256SUMS

    t0 = time.perf_counter()
    preds_raw = []
    for start in range(0, len(puzzles), args.batch_size):
        batch = puzzles[start : start + args.batch_size]
        preds_raw.extend(
            r["board"]
            for r in solve_sudoku(
                bundle,
                batch,
                best_of_k=args.best_of_k,
                temperature=args.temperature,
                repair=repair,
            )
        )
    elapsed = time.perf_counter() - t0

    refs_str = [bundle["tk"].pre_tokenizer.decode(s, skip_special_tokens=True) for s in solutions]
    stats = score_sudoku(preds_raw, refs_str)
    stats["wall_s"] = round(elapsed, 3)
    stats["boards_per_sec"] = round(len(puzzles) / max(elapsed, 1e-6), 2)
    stats["best_of_k"] = args.best_of_k
    stats["repair"] = repair
    stats["difficulty"] = args.difficulty
    if args.json:
        _emit_output(json.dumps(stats, indent=2), args)
    else:
        ci = stats["ci_95"]
        print(
            f"Sudoku {args.difficulty} (n={stats['n_boards']}, K={args.best_of_k}, repair={repair}): "
            f"solved={stats['solved_pct']}% ({stats['solved_count']}/{stats['n_boards']}, "
            f"95% CI +/-{ci['margin_pp']}pp) | cell={stats['cell_pct']}% | "
            f"valid={stats['valid_pct']}% | {stats['wall_s']}s ({stats['boards_per_sec']} boards/s)"
        )


def cmd_translate(args):
    text = _read_text_arg(args)
    if not text:
        print("Error: No German source text provided via --text, --input-file, or stdin.", file=sys.stderr)
        sys.exit(1)
    bundle = get_model_bundle(
        task="wmt14",
        quantize=args.quantize,
        device=args.device,
        verify_checksums=not args.skip_checksum,
        strict_checksums=args.strict_checksums,
        dtype=args.dtype,
        compile_model=args.compile,
    )
    from ncm_bench.inference import summarize_texts

    t0 = time.perf_counter()
    translation = summarize_texts(bundle, [text])[0]
    latency_ms = int((time.perf_counter() - t0) * 1000)
    if args.json:
        _emit_output(json.dumps({"translation": translation, "latency_ms": latency_ms}, indent=2), args)
    else:
        _emit_output(translation, args)


def cmd_benchmark(args):
    from ncm_bench.inference import summarize_texts, verify_file_integrity
    from ncm_bench.metrics import two_size_timing

    if args.task == "sudoku":
        args.puzzle = None
        args.difficulty = getattr(args, "difficulty", "hard")
        args.best_of_k = getattr(args, "best_of_k", 1)
        args.temperature = getattr(args, "temperature", 0.4)
        args.no_repair = False
        cmd_sudoku(args)
        return

    passages_path = ROOT_DIR / "data/xsum/passages.pickle"
    if not args.skip_checksum:
        verify_file_integrity(passages_path, ROOT_DIR, strict=args.strict_checksums)
    with open(passages_path, "rb") as f:
        passages = pickle.load(f)  # nosec B301 - verified by SHA256SUMS

    def _timed_run(n_items: int) -> float:
        t0 = time.perf_counter()
        b = get_model_bundle(
            task="xsum",
            quantize=args.quantize,
            device=args.device,
            verify_checksums=not args.skip_checksum,
            strict_checksums=args.strict_checksums,
            dtype=args.dtype,
            compile_model=args.compile,
        )
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
            "dtype": args.dtype,
        }
    )

    if args.baseline_results:
        base_data = json.loads(Path(args.baseline_results).read_text(encoding="utf-8"))
        ref_marginal = base_data.get("runs", {}).get("latency_cpu", {}).get("marginal_s_per_doc_bs1")
        if ref_marginal:
            timing["baseline_marginal_s"] = ref_marginal
            timing["delta_pct"] = round(
                (timing["marginal_s_per_doc"] - ref_marginal) / ref_marginal * 100.0, 2
            )

    if args.json:
        _emit_output(json.dumps(timing, indent=2), args)
    else:
        print(
            f"2-Size Latency Benchmark (quantize={args.quantize}, dtype={args.dtype}, bs={args.batch_size}):\n"
            f"  T({args.small_n})={timing['small_s']}s, T({args.large_n})={timing['large_s']}s\n"
            f"  Load={timing['model_load_s']}s | Marginal={timing['marginal_s_per_doc']}s/doc "
            f"({timing['items_per_min']} docs/min)"
        )


def cmd_convert_safetensors(args):
    from ncm_bench.inference import export_safetensors

    paths = export_safetensors(
        task=args.task,
        root_dir=ROOT_DIR,
        verify_checksums=not args.skip_checksum,
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


def cmd_train_smoke(args):
    import torch

    from ncm_train import Stage1Config, Stage2Config, train_stage1_vqvae, train_stage2_ncm

    torch.manual_seed(42)
    # Deterministic synthetic structured sequence-reversal dataset (real gradients, zero mocks)
    n_samples = args.samples
    L = 16
    vocab = 16
    prefixes = torch.randint(1, vocab, (n_samples, L))
    targets = torch.flip(prefixes, dims=[1])

    s1 = Stage1Config(
        in_vocab_size=vocab,
        context_length=L,
        embed_dim=64,
        levels=(2, 4, 8, 16),
        vq_vocab_size=32,
        n_layers=2,
        batch_size=min(16, n_samples),
        steps=args.s1_steps,
        device="cpu",
    )
    res1 = train_stage1_vqvae(targets, s1, out_path=Path(args.out_dir) / "tokenizer.pt")

    s2 = Stage2Config(
        prefix_len=L,
        n_layer=2,
        n_head=4,
        n_embd=64,
        batch_size=min(16, n_samples),
        steps=args.s2_steps,
        device="cpu",
    )
    res2 = train_stage2_ncm(
        prefixes,
        targets,
        res1["model"],
        s1,
        s2,
        out_path=Path(args.out_dir) / "generator.pt",
    )

    with torch.inference_mode():
        codes = res2["model"].generate(None, prefixes[:2], temperature=0.0)
        recon_logits = res1["model"].decode(codes)
        assert recon_logits.shape[:2] == (2, L)

    print(
        json.dumps(
            {
                "stage1_initial_loss": round(res1["history"][0]["loss"], 4),
                "stage1_final_loss": round(res1["final_loss"], 4),
                "stage2_initial_loss": round(res2["history"][0]["loss"], 4),
                "stage2_final_loss": round(res2["final_loss"], 4),
                "out_dir": str(Path(args.out_dir).resolve()),
            },
            indent=2,
        )
    )


def cmd_rl_smoke(args):
    import torch

    from ncm_train import (
        GRPOConfig,
        Stage1Config,
        Stage2Config,
        train_stage1_vqvae,
        train_stage2_grpo,
        train_stage2_ncm,
    )

    torch.manual_seed(0)
    vocab = 16
    seq_len = 16
    sequences = torch.randint(0, vocab, (args.samples, seq_len), dtype=torch.long)
    prefixes = torch.randint(0, vocab, (args.samples, seq_len), dtype=torch.long)

    s1 = Stage1Config(
        in_vocab_size=vocab,
        context_length=seq_len,
        embed_dim=32,
        levels=(2, 4, 8, 16),
        vq_vocab_size=16,
        n_layers=1,
        batch_size=min(8, args.samples),
        steps=4,
        device=args.device or "cpu",
    )
    res1 = train_stage1_vqvae(sequences, s1)
    s2 = Stage2Config(
        prefix_len=seq_len,
        n_layer=1,
        n_head=2,
        n_embd=32,
        batch_size=min(8, args.samples),
        steps=4,
        query_level0=True,
        device=args.device or "cpu",
    )
    res2 = train_stage2_ncm(prefixes, sequences, res1["model"], s1, s2)

    # Deterministic verifiable reward: fraction of tokens staying within target lower-half token range [0, vocab//2)
    def _verifiable_reward(decoded_logits: torch.Tensor, prompt_rep: torch.Tensor) -> torch.Tensor:
        del prompt_rep
        preds = decoded_logits.argmax(dim=-1)
        return (preds < (vocab // 2)).float().mean(dim=-1)

    grpo_cfg = GRPOConfig(
        group_size=args.group_size,
        batch_size=min(4, args.samples),
        steps=args.steps,
        temperature=0.8,
        kl_beta=0.04,
        device=args.device or "cpu",
    )
    res_grpo = train_stage2_grpo(
        prefixes,
        _verifiable_reward,
        res1["model"],
        res2["model"],
        grpo_cfg,
        out_path=Path(args.out_dir) / "generator_grpo.pt",
    )
    print(
        json.dumps(
            {
                "initial_reward": round(res_grpo["initial_reward"], 4),
                "final_reward": round(res_grpo["final_reward"], 4),
                "delta_reward": round(res_grpo["final_reward"] - res_grpo["initial_reward"], 4),
                "steps": args.steps,
                "group_size": args.group_size,
                "out_dir": str(Path(args.out_dir).resolve()),
            },
            indent=2,
        )
    )


def cmd_distill_smoke(args):
    import torch

    from ncm_train import (
        DistillConfig,
        Stage1Config,
        Stage2Config,
        distill_teacher_into_ncm,
        train_stage1_vqvae,
        train_stage2_ncm,
    )

    torch.manual_seed(0)
    vocab = 16
    seq_len = 16
    teacher_targets = torch.randint(0, vocab, (args.samples, seq_len), dtype=torch.long)
    prefixes = torch.randint(0, vocab, (args.samples, seq_len), dtype=torch.long)

    s1 = Stage1Config(
        in_vocab_size=vocab,
        context_length=seq_len,
        embed_dim=32,
        levels=(2, 4, 8, 16),
        vq_vocab_size=16,
        n_layers=1,
        batch_size=min(8, args.samples),
        steps=4,
        device=args.device or "cpu",
    )
    res1 = train_stage1_vqvae(teacher_targets, s1)
    teacher_s2 = Stage2Config(
        prefix_len=seq_len,
        n_layer=1,
        n_head=4,
        n_embd=64,
        batch_size=min(8, args.samples),
        steps=4,
        query_level0=True,
        device=args.device or "cpu",
    )
    teacher_res = train_stage2_ncm(prefixes, teacher_targets, res1["model"], s1, teacher_s2)
    student_s2 = Stage2Config(
        prefix_len=seq_len,
        n_layer=1,
        n_head=2,
        n_embd=32,
        batch_size=min(8, args.samples),
        steps=args.steps,
        query_level0=True,
        device=args.device or "cpu",
    )
    res_d = distill_teacher_into_ncm(
        prefixes,
        teacher_targets,
        res1["model"],
        s1,
        student_s2,
        DistillConfig(
            batch_size=min(8, args.samples),
            steps=args.steps,
            device=args.device or "cpu",
        ),
        teacher_net=teacher_res["model"],
        out_path=Path(args.out_dir) / "generator_distilled.pt",
    )
    print(
        json.dumps(
            {
                "initial_loss": round(res_d["initial_loss"], 4),
                "final_loss": round(res_d["final_loss"], 4),
                "steps": args.steps,
                "out_dir": str(Path(args.out_dir).resolve()),
            },
            indent=2,
        )
    )


def _load_bundle_for_args(task: str, args):
    return get_model_bundle(
        task=task,
        quantize=getattr(args, "quantize", "none"),
        device=getattr(args, "device", None),
        verify_checksums=not getattr(args, "skip_checksum", False),
        strict_checksums=getattr(args, "strict_checksums", False),
        dtype=getattr(args, "dtype", "fp32"),
        compile_model=getattr(args, "compile", False),
    )


def cmd_triage(args):
    from ncm_bench.longdoc import build_audit_receipt
    from ncm_bench.triage import extract_structured_triage

    text = _read_text_arg(args)
    if not text:
        print("Error: provide --text, --input-file, or pipe input via stdin.", file=sys.stderr)
        sys.exit(1)

    bundle = _load_bundle_for_args("xsum", args)
    labels = [s.strip() for s in args.labels.split(",") if s.strip()] if args.labels else None
    t0 = time.perf_counter()
    res = extract_structured_triage(
        bundle,
        text,
        labels=labels,
        max_level=args.max_level,
        best_of_k=args.best_of_k,
        temperature=args.temperature,
        coarse_only=bool(getattr(args, "coarse_only", False)),
        adaptive_compute=bool(getattr(args, "adaptive_compute", False)),
        redact=bool(args.redact_pii),
        restore=bool(args.restore_pii),
    )
    res["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    if args.audit_receipt:
        res["audit_receipt"] = build_audit_receipt(
            text,
            res["summary"],
            bundle,
            pii_redactions=res.get("pii_redactions", 0),
            mode="triage",
        )
    if args.json:
        _emit_output(json.dumps(res, indent=2), args)
    else:
        _emit_output(
            f"[{res['category']} | priority={res.get('priority')} | schema_valid={res['schema_valid']}]\n{res['summary']}",
            args,
        )


def cmd_early_exit_bench(args):
    from ncm_bench.inference import summarize_texts
    from ncm_bench.triage import generate_coarse_codes

    data_file = ROOT_DIR / "data" / "xsum" / "passages.pickle"
    if not data_file.exists():
        print(f"Error: dataset not found at {data_file}", file=sys.stderr)
        sys.exit(1)

    with open(data_file, "rb") as f:
        passages = pickle.load(f)[: args.n]  # nosec B301 - verified by SHA256SUMS

    bundle = _load_bundle_for_args("xsum", args)
    # Warm up once
    _ = generate_coarse_codes(bundle, [passages[0]], max_level=args.max_level)

    t0 = time.perf_counter()
    coarse_out = generate_coarse_codes(bundle, passages, max_level=args.max_level)
    coarse_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    _ = summarize_texts(bundle, passages)
    full_s = time.perf_counter() - t1

    payload = {
        "n": len(passages),
        "max_level": args.max_level,
        "passes_early_exit": coarse_out[0]["passes_executed"] if coarse_out else 0,
        "passes_full": coarse_out[0]["total_levels"] if coarse_out else 0,
        "early_exit_total_s": round(coarse_s, 4),
        "early_exit_per_doc_ms": round((coarse_s / max(len(passages), 1)) * 1000.0, 2),
        "full_decode_total_s": round(full_s, 4),
        "full_decode_per_doc_ms": round((full_s / max(len(passages), 1)) * 1000.0, 2),
        "speedup": round(full_s / max(coarse_s, 1e-6), 2),
    }
    print(json.dumps(payload, indent=2))


def cmd_verify(args):
    from scripts.fetch_artifacts import load_checksums, sha256_file

    artifacts_only = bool(getattr(args, "artifacts_only", False))
    sums = load_checksums()
    if not sums:
        print("Error: SHA256SUMS file not found or empty.", file=sys.stderr)
        sys.exit(1)
    entries = (
        {k: v for k, v in sums.items() if k.startswith("artifacts/")}
        if artifacts_only
        else sums
    )
    all_ok = True
    print(f"Verifying {len(entries)} artifact checksums...")
    for rel_path, exp_hash in entries.items():
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


def _add_common_flags(p, *, with_io: bool = True):
    if with_io:
        p.add_argument("--input-file", "-f", default=None, help="Read input from .txt/.md/.json/.pdf")
        p.add_argument("--output-file", "-o", default=None, help="Write output payload to file")
    p.add_argument("--quantize", choices=["none", "int8"], default="none")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--device", default=None)
    p.add_argument("--skip-checksum", action="store_true", help="Skip SHA-256 verification before load")
    p.add_argument("--strict-checksums", action="store_true", help="Fail closed on unlisted artifacts")
    p.add_argument("--compile", action="store_true", help="Apply torch.compile optimization")


def main():
    parser = argparse.ArgumentParser(
        prog="ncm-bench",
        description="NCM Sovereign Inference Engine, Training Suite, Benchmark CLI & Inspector",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # serve
    p_serve = sub.add_parser("serve", help="Run local REST API server")
    p_serve.add_argument("--port", type=int, default=8031)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--uds", default=None, help="Bind to a Unix domain socket path instead of TCP")
    p_serve.add_argument("--max-queue-depth", type=int, default=8)
    p_serve.add_argument("--request-timeout-s", type=float, default=60.0)
    p_serve.add_argument("--max-body-bytes", type=int, default=2_000_000)
    p_serve.add_argument("--rate-limit-rpm", type=int, default=0, help="Requests/min per client IP (0=disabled)")
    p_serve.add_argument("--with-sudoku", action="store_true", help="Also load sudoku/hard model for POST /sudoku")
    _add_common_flags(p_serve, with_io=False)
    p_serve.set_defaults(func=cmd_serve)

    # summarize
    p_sum = sub.add_parser("summarize", help="Summarize text or document file offline")
    p_sum.add_argument("--text", "-t", default=None, help="Input text to summarize")
    p_sum.add_argument("--long-doc", action="store_true", help="Enable map-reduce chunked summarization")
    p_sum.add_argument("--redact-pii", action="store_true", help="Scrub PII before encoding")
    p_sum.add_argument("--restore-pii", action="store_true", help="Restore PII placeholders in output")
    p_sum.add_argument("--audit-receipt", action="store_true", help="Include SHA-256 compliance audit receipt")
    p_sum.add_argument("--json", action="store_true", help="Output JSON response")
    _add_common_flags(p_sum, with_io=True)
    p_sum.set_defaults(func=cmd_summarize)

    # mbr
    p_mbr = sub.add_parser("mbr", help="Run MBR consensus decoding")
    p_mbr.add_argument("--text", "-t", default=None, help="Input text to summarize")
    p_mbr.add_argument("--candidates", "-k", type=int, default=4)
    p_mbr.add_argument("--temperature", type=float, default=0.7)
    p_mbr.add_argument("--utility", choices=["rougeL", "jaccard", "cosine"], default="rougeL")
    p_mbr.add_argument("--json", action="store_true")
    _add_common_flags(p_mbr, with_io=True)
    p_mbr.set_defaults(func=cmd_mbr)

    # inspect
    p_insp = sub.add_parser("inspect", help="Progressively inspect concept hierarchy (Levels 0..8)")
    p_insp.add_argument("--text", "-t", default=None, help="Input text")
    p_insp.add_argument("--html", help="Path to write standalone HTML visualizer slider")
    p_insp.add_argument("--json", action="store_true")
    _add_common_flags(p_insp, with_io=True)
    p_insp.set_defaults(func=cmd_inspect)

    # sudoku
    p_sud = sub.add_parser("sudoku", help="Solve Sudoku puzzles or run constraint-verified evaluation")
    p_sud.add_argument("--puzzle", "-p", default=None, help="81-char Sudoku puzzle (0, ., or _ for blanks)")
    p_sud.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="hard")
    p_sud.add_argument("--n", type=int, default=20)
    p_sud.add_argument("--batch-size", type=int, default=16)
    p_sud.add_argument("--best-of-k", type=int, default=1, help="Resample failing boards up to K candidates")
    p_sud.add_argument("--temperature", type=float, default=0.4)
    p_sud.add_argument("--no-repair", action="store_true", help="Disable post-prediction Sudoku constraint repair")
    p_sud.add_argument("--json", action="store_true")
    _add_common_flags(p_sud, with_io=True)
    p_sud.set_defaults(func=cmd_sudoku)

    # translate
    p_tr = sub.add_parser("translate", help="Translate German to English via NCM WMT14")
    p_tr.add_argument("--text", "-t", default=None)
    p_tr.add_argument("--json", action="store_true")
    _add_common_flags(p_tr, with_io=True)
    p_tr.set_defaults(func=cmd_translate)

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run timed latency or accuracy benchmarks")
    p_bench.add_argument("--task", choices=["latency", "sudoku"], default="latency")
    p_bench.add_argument("--small-n", type=int, default=4)
    p_bench.add_argument("--large-n", type=int, default=16)
    p_bench.add_argument("--n", type=int, default=20)
    p_bench.add_argument("--batch-size", type=int, default=1)
    p_bench.add_argument("--baseline-results", default=None, help="Compare against baseline results.json")
    p_bench.add_argument("--json", action="store_true")
    _add_common_flags(p_bench, with_io=False)
    p_bench.set_defaults(func=cmd_benchmark)

    # convert-safetensors
    p_st = sub.add_parser("convert-safetensors", help="Convert verified .pt weights to zero-pickle .safetensors")
    p_st.add_argument("--task", default="xsum", help="Artifact subdirectory under artifacts/ (e.g. xsum, sudoku/hard)")
    p_st.add_argument("--skip-checksum", action="store_true")
    p_st.set_defaults(func=cmd_convert_safetensors)

    # train-smoke
    p_ts = sub.add_parser("train-smoke", help="Run clean-room Stage-1 + Stage-2 NCM training smoke check")
    p_ts.add_argument("--samples", type=int, default=32)
    p_ts.add_argument("--s1-steps", type=int, default=10)
    p_ts.add_argument("--s2-steps", type=int, default=10)
    p_ts.add_argument("--out-dir", default="artifacts/train_smoke")
    p_ts.set_defaults(func=cmd_train_smoke)

    # rl-smoke
    p_rl = sub.add_parser("rl-smoke", help="Run verifiable-reward GRPO/RLVR over discrete NCM codebooks")
    p_rl.add_argument("--samples", type=int, default=16)
    p_rl.add_argument("--steps", type=int, default=6)
    p_rl.add_argument("--group-size", type=int, default=4)
    p_rl.add_argument("--device", default="cpu")
    p_rl.add_argument("--out-dir", default="artifacts/rl_smoke")
    p_rl.set_defaults(func=cmd_rl_smoke)

    # distill-smoke
    p_ds = sub.add_parser("distill-smoke", help="Run teacher-student concept distillation smoke check")
    p_ds.add_argument("--samples", type=int, default=16)
    p_ds.add_argument("--steps", type=int, default=8)
    p_ds.add_argument("--device", default="cpu")
    p_ds.add_argument("--out-dir", default="artifacts/distill_smoke")
    p_ds.set_defaults(func=cmd_distill_smoke)

    # triage
    p_tri = sub.add_parser("triage", help="Early-exit coarse concept classification + verified structured extraction")
    p_tri.add_argument("--text", "-t", default=None)
    p_tri.add_argument("--labels", default=None, help="Comma-separated allowed labels")
    p_tri.add_argument("--max-level", type=int, default=2, help="Coarse concept exit level (0..8)")
    p_tri.add_argument("--best-of-k", type=int, default=3, help="Parallel candidates for schema/consensus scoring")
    p_tri.add_argument("--temperature", type=float, default=0.5)
    p_tri.add_argument("--coarse-only", action="store_true", help="Exit after coarse Level 0..k without full generation")
    p_tri.add_argument("--adaptive-compute", action="store_true", help="Use greedy decode when coarse confidence is high")
    p_tri.add_argument("--redact-pii", action="store_true")
    p_tri.add_argument("--restore-pii", action="store_true")
    p_tri.add_argument("--audit-receipt", action="store_true")
    p_tri.add_argument("--json", action="store_true")
    _add_common_flags(p_tri, with_io=True)
    p_tri.set_defaults(func=cmd_triage)

    # early-exit-bench
    p_eeb = sub.add_parser("early-exit-bench", help="Benchmark coarse Level 0..k early-exit vs full NCM decode")
    p_eeb.add_argument("--n", type=int, default=8)
    p_eeb.add_argument("--max-level", type=int, default=2)
    _add_common_flags(p_eeb, with_io=False)
    p_eeb.set_defaults(func=cmd_early_exit_bench)

    # verify
    p_ver = sub.add_parser("verify", help="Verify model and data SHA256 checksums")
    p_ver.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Verify pinned model weights under artifacts/ only (skip optional data/ datasets)",
    )
    p_ver.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
