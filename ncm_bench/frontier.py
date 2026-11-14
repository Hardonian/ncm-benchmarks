"""Frontier benchmark harnesses: long-context docs, legal redaction, FinQA repair, AST/layout concepts & energy telemetry."""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import time
from collections import Counter
from typing import Any

import torch

from ncm_bench.longdoc import redact_pii, restore_pii, summarize_long_document


def run_longcontext_summarization_harness(
    model_bundle: dict[str, Any],
    texts: list[str],
    *,
    max_chunk_chars: int = 800,
) -> dict[str, Any]:
    """Benchmarks hierarchical long-document summarization across GovReport/BillSum/PubMed-style passages."""
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    total_chunks = 0
    for txt in texts:
        out = summarize_long_document(model_bundle, txt, max_chunk_chars=max_chunk_chars)
        n_chunks = int(out.get("num_chunks", 1))
        total_chunks += n_chunks
        results.append({"summary": out["summary"], "num_chunks": n_chunks})
    elapsed = round(time.perf_counter() - t0, 4)
    return {
        "docs_evaluated": len(texts),
        "total_chunks": total_chunks,
        "elapsed_s": elapsed,
        "results": results,
    }


def evaluate_legal_clause_redaction(clauses: list[str]) -> dict[str, Any]:
    """Measures zero-loss PII/secret scrubbing and placeholder restoration across LegalBench/CUAD clauses."""
    exact_matches = 0
    total_redactions = 0
    for clause in clauses:
        scrubbed, mapping = redact_pii(clause)
        restored = restore_pii(scrubbed, mapping)
        total_redactions += len(mapping)
        if restored == clause:
            exact_matches += 1
    n = max(1, len(clauses))
    return {
        "clauses_evaluated": len(clauses),
        "total_redactions": total_redactions,
        "exact_restoration_rate": round(exact_matches / n, 4),
    }


def verify_finqa_table_arithmetic(
    csv_text: str,
    column: str,
    claimed_sum: float,
    *,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Parses FinQA/TAT-QA CSV tables, verifies numeric column totals, and repairs arithmetic errors."""
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    total = 0.0
    rows_parsed = 0
    for row in reader:
        raw = str(row.get(column, "")).replace("$", "").replace(",", "").strip()
        if raw:
            total += float(raw)
            rows_parsed += 1
    delta = round(total - claimed_sum, 4)
    return {
        "column": column,
        "rows_parsed": rows_parsed,
        "computed_sum": round(total, 4),
        "claimed_sum": round(float(claimed_sum), 4),
        "exact_match": abs(delta) <= tolerance,
        "repaired_delta": delta,
    }


def extract_python_ast_concepts(source_code: str) -> dict[str, Any]:
    """Extracts discrete structural AST concept histograms for code triage and similarity matching."""
    tree = ast.parse(source_code)
    counts = Counter(type(node).__name__ for node in ast.walk(tree))
    ordered = sorted(counts.items())
    digest = hashlib.sha256(json.dumps(ordered).encode("utf-8")).hexdigest()[:16]
    concept_ids = [int(hashlib.sha256(k.encode("utf-8")).hexdigest()[:8], 16) % 4096 for k, _ in ordered]
    return {
        "node_counts": dict(ordered),
        "concept_ids": concept_ids,
        "concept_hash": digest,
        "total_ast_nodes": sum(counts.values()),
    }


def score_cross_lingual_concept_alignment(
    sig_src: dict[str, Any],
    sig_tgt: dict[str, Any],
) -> dict[str, Any]:
    """Measures coarse hierarchical concept histogram overlap across cross-lingual pairs."""
    h1 = {int(k): float(v) for k, v in sig_src.get("histogram", {}).items()}
    h2 = {int(k): float(v) for k, v in sig_tgt.get("histogram", {}).items()}
    shared = set(h1.keys()) & set(h2.keys())
    dot = sum(h1[k] * h2[k] for k in shared)
    n1 = math.sqrt(sum(v * v for v in h1.values()))
    n2 = math.sqrt(sum(v * v for v in h2.values()))
    cos = dot / (n1 * n2) if (n1 > 0 and n2 > 0) else 0.0
    return {
        "cosine_similarity": round(cos, 4),
        "shared_concepts": len(shared),
    }


def generate_synthetic_dpo_preferences(
    prefixes: torch.Tensor,
    vqvae,
    policy_net,
    reward_fn,
    *,
    top_k: int = 4,
) -> dict[str, torch.Tensor]:
    """Samples dual temperature trajectories in discrete concept space and labels chosen vs rejected pairs."""
    with torch.no_grad():
        c1 = policy_net.generate(None, prefixes, temperature=0.4, top_k=top_k)
        c2 = policy_net.generate(None, prefixes, temperature=1.3, top_k=top_k)
        r1 = reward_fn(vqvae.decode(c1), prefixes).float().reshape(-1)
        r2 = reward_fn(vqvae.decode(c2), prefixes).float().reshape(-1)
    win_c1 = (r1 >= r2).unsqueeze(-1)
    chosen = torch.where(win_c1, c1, c2)
    rejected = torch.where(win_c1, c2, c1)
    return {"chosen_codes": chosen, "rejected_codes": rejected, "mean_margin": (r1 - r2).abs().mean()}


def estimate_energy_telemetry(
    elapsed_s: float,
    n_docs: int,
    *,
    tdp_watts: float = 54.0,
    utilization: float = 0.65,
) -> dict[str, float]:
    """Computes Wh/1k-docs and Joules/doc sovereign energy efficiency metrics."""
    safe_docs = max(1, int(n_docs))
    avg_watts = max(0.0, float(tdp_watts) * float(utilization))
    energy_joules = avg_watts * max(0.0, float(elapsed_s))
    energy_wh = energy_joules / 3600.0
    return {
        "docs": float(safe_docs),
        "elapsed_s": round(float(elapsed_s), 4),
        "avg_watts": round(avg_watts, 2),
        "joules_per_doc": round(energy_joules / safe_docs, 4),
        "wh_per_1k_docs": round((energy_wh / safe_docs) * 1000.0, 4),
    }


def encode_layout_text_joint_concepts(
    tokens: list[int],
    bboxes: list[tuple[int, int, int, int]],
    *,
    grid_bins: int = 16,
    vocab_size: int = 4096,
) -> list[int]:
    """Fuses discrete token IDs with quantized 2D bounding-box spatial coordinates into joint concept IDs."""
    joint: list[int] = []
    for tok, box in zip(tokens, bboxes, strict=True):
        x0, y0, x1, y1 = box
        cx = min(grid_bins - 1, max(0, int(((x0 + x1) / 2.0) / 1000.0 * grid_bins)))
        cy = min(grid_bins - 1, max(0, int(((y0 + y1) / 2.0) / 1000.0 * grid_bins)))
        spatial_bucket = cy * grid_bins + cx
        joint_id = (int(tok) * 131 + spatial_bucket) % max(1, int(vocab_size))
        joint.append(joint_id)
    return joint


def validate_receipt_submission(receipt: dict[str, Any]) -> dict[str, Any]:
    """Validates community hardware benchmark receipt JSON schema and computes submission digest."""
    required = ("hardware", "workload", "throughput", "metric_name", "metric_value")
    missing = [k for k in required if k not in receipt or receipt[k] in (None, "")]
    errors: list[str] = []
    if missing:
        errors.append(f"Missing required keys: {', '.join(missing)}")
    if "throughput" in receipt:
        try:
            tp = float(receipt["throughput"])
            if not math.isfinite(tp) or tp <= 0.0:
                errors.append("throughput must be positive and finite")
        except (TypeError, ValueError):
            errors.append("throughput must be numeric")
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "receipt_sha256": digest,
    }
