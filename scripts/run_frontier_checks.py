"""Runs real CPU on-hardware checks for NCM frontier capabilities and exports demo progression JSON."""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch  # noqa: E402

from ncm_bench.inference import load_bundle, summarize_texts  # noqa: E402
from ncm_bench.progressive import progressive_decode  # noqa: E402
from ncm_bench.triage import extract_structured_triage, generate_coarse_codes  # noqa: E402
from ncm_train import (  # noqa: E402
    DistillConfig,
    GRPOConfig,
    Stage1Config,
    Stage2Config,
    distill_teacher_into_ncm,
    train_stage1_vqvae,
    train_stage2_grpo,
    train_stage2_ncm,
)


def main() -> None:
    bundle = load_bundle("xsum", device="cpu", quantize="int8", verify_checksums=True, root_dir=ROOT_DIR)

    with open(ROOT_DIR / "data" / "xsum" / "passages.pickle", "rb") as f:
        passages = pickle.load(f)  # nosec B301 - verified by SHA256SUMS
    with open(ROOT_DIR / "data" / "xsum" / "reference_summaries.pickle", "rb") as f:
        refs = pickle.load(f)  # nosec B301 - verified by SHA256SUMS

    sample_docs = passages[:8]
    _ = generate_coarse_codes(bundle, [sample_docs[0]], max_level=2)

    t0 = time.perf_counter()
    coarse_res = generate_coarse_codes(bundle, sample_docs, max_level=2)
    coarse_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    _ = summarize_texts(bundle, sample_docs)
    full_s = time.perf_counter() - t1

    early_exit_metrics = {
        "n": len(sample_docs),
        "max_level": 2,
        "passes_early_exit": coarse_res[0]["passes_executed"],
        "passes_full": coarse_res[0]["total_levels"],
        "early_exit_total_s": round(coarse_s, 4),
        "early_exit_per_doc_ms": round(coarse_s / len(sample_docs) * 1000.0, 2),
        "full_decode_total_s": round(full_s, 4),
        "full_decode_per_doc_ms": round(full_s / len(sample_docs) * 1000.0, 2),
        "speedup": round(full_s / max(coarse_s, 1e-6), 2),
    }

    triage_sample = extract_structured_triage(
        bundle,
        "Urgent security breach detected on auth gateway involving alice@enterprise.org and $480,000 invoice on 2026-11-12.",
        labels=["finance", "security", "legal", "engineering"],
        max_level=2,
        best_of_k=2,
        redact=True,
        restore=True,
    )

    demo_examples = []
    for idx in range(3):
        prog = progressive_decode(bundle, passages[idx], temperature=0.0, use_penalties=True)
        demo_examples.append(
            {
                "id": f"xsum-{idx}",
                "title": f"XSum Article #{idx + 1}",
                "passage": passages[idx][:1200],
                "reference": refs[idx],
                "final_summary": prog["final_summary"],
                "levels": prog["levels"],
            }
        )

    torch.manual_seed(42)
    vocab, seq_len, samples = 16, 16, 48
    sequences = torch.randint(0, vocab, (samples, seq_len), dtype=torch.long)
    prefixes = torch.randint(0, vocab, (samples, seq_len), dtype=torch.long)
    s1 = Stage1Config(
        in_vocab_size=vocab,
        context_length=seq_len,
        embed_dim=32,
        levels=(2, 4, 8, 16),
        vq_vocab_size=16,
        n_layers=1,
        batch_size=16,
        steps=20,
        device="cpu",
    )
    vqvae = train_stage1_vqvae(sequences, s1)["model"]
    s2 = Stage2Config(
        prefix_len=seq_len,
        n_layer=1,
        n_head=2,
        n_embd=32,
        batch_size=16,
        steps=12,
        query_level0=True,
        device="cpu",
    )
    teacher = train_stage2_ncm(prefixes, sequences, vqvae, s1, s2)["model"]
    distill_res = distill_teacher_into_ncm(
        prefixes,
        sequences,
        vqvae,
        s1,
        s2,
        DistillConfig(batch_size=16, steps=20, kl_weight=0.5, device="cpu"),
        teacher_net=teacher,
    )

    def _reward_fn(decoded_logits: torch.Tensor, prompt_rep: torch.Tensor) -> torch.Tensor:
        del prompt_rep
        preds = decoded_logits.argmax(dim=-1)
        return (preds < (vocab // 2)).float().mean(dim=-1)

    grpo_res = train_stage2_grpo(
        prefixes,
        _reward_fn,
        vqvae,
        distill_res["model"],
        GRPOConfig(group_size=4, batch_size=8, steps=24, lr=3e-4, temperature=0.8, kl_beta=0.02, device="cpu"),
    )

    receipt = {
        "hardware": "Ryzen AI 9 HX 370 CPU (INT8 .safetensors)",
        "early_exit_vs_full": early_exit_metrics,
        "triage_sample": triage_sample,
        "distill_smoke": {
            "steps": 20,
            "initial_loss": round(distill_res["initial_loss"], 4),
            "final_loss": round(distill_res["final_loss"], 4),
        },
        "grpo_rlvr_concept_space": {
            "steps": 24,
            "group_size": 4,
            "initial_reward": round(grpo_res["initial_reward"], 4),
            "final_reward": round(grpo_res["final_reward"], 4),
            "delta_reward": round(grpo_res["final_reward"] - grpo_res["initial_reward"], 4),
        },
    }

    out_results = ROOT_DIR / "results" / "hx370-cpu" / "frontier_checks.json"
    out_results.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    out_demo = ROOT_DIR / "docs" / "demo_data.json"
    out_demo.write_text(
        json.dumps({"examples": demo_examples, "early_exit": early_exit_metrics}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
