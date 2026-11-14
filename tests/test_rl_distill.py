"""Tests for Discrete Concept GRPO/RLVR and Teacher-Student Distillation trainers."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

from ncm_bench.cli import main
from ncm_train import (
    DistillConfig,
    DPOConfig,
    GRPOConfig,
    Stage1Config,
    Stage2Config,
    distill_teacher_into_ncm,
    train_stage1_vqvae,
    train_stage2_dpo,
    train_stage2_grpo,
    train_stage2_ncm,
)


def test_grpo_and_teacher_distillation_in_concept_space(tmp_path: Path) -> None:
    torch.manual_seed(7)
    vocab = 16
    seq_len = 16
    sequences = torch.randint(0, vocab, (24, seq_len), dtype=torch.long)
    prefixes = torch.randint(0, vocab, (24, seq_len), dtype=torch.long)

    s1_cfg = Stage1Config(
        in_vocab_size=vocab,
        context_length=seq_len,
        embed_dim=32,
        levels=(2, 4, 8, 16),
        vq_vocab_size=16,
        n_layers=1,
        batch_size=8,
        steps=4,
        device="cpu",
    )
    vqvae = train_stage1_vqvae(sequences, s1_cfg)["model"]

    teacher_cfg = Stage2Config(
        prefix_len=seq_len,
        n_layer=1,
        n_head=4,
        n_embd=64,
        batch_size=8,
        steps=4,
        query_level0=True,
        device="cpu",
    )
    teacher = train_stage2_ncm(prefixes, sequences, vqvae, s1_cfg, teacher_cfg)["model"]

    student_cfg = Stage2Config(
        prefix_len=seq_len,
        n_layer=1,
        n_head=2,
        n_embd=32,
        batch_size=8,
        steps=5,
        query_level0=True,
        device="cpu",
    )
    distill_ckpt = tmp_path / "distilled.pt"
    d_out = distill_teacher_into_ncm(
        prefixes,
        sequences,
        vqvae,
        s1_cfg,
        student_cfg,
        DistillConfig(batch_size=8, steps=5, kl_weight=0.5, device="cpu"),
        teacher_net=teacher,
        out_path=distill_ckpt,
    )
    assert distill_ckpt.exists()
    assert len(d_out["history"]) == 5
    assert d_out["final_loss"] > 0.0

    def _reward_fn(decoded_logits: torch.Tensor, prompt_rep: torch.Tensor) -> torch.Tensor:
        del prompt_rep
        preds = decoded_logits.argmax(dim=-1)
        return (preds < (vocab // 2)).float().mean(dim=-1)

    grpo_ckpt = tmp_path / "grpo.pt"
    g_out = train_stage2_grpo(
        prefixes,
        _reward_fn,
        vqvae,
        d_out["model"],
        GRPOConfig(group_size=3, batch_size=4, steps=4, temperature=0.8, device="cpu"),
        out_path=grpo_ckpt,
    )
    assert grpo_ckpt.exists()
    assert len(g_out["history"]) == 4
    assert all(step["kl"] >= 0.0 for step in g_out["history"])


def test_cli_rl_smoke_and_distill_smoke(tmp_path: Path, capsys) -> None:
    rl_dir = tmp_path / "rl_out"
    with patch.object(
        sys,
        "argv",
        ["ncm-bench", "rl-smoke", "--samples", "12", "--steps", "3", "--group-size", "2", "--out-dir", str(rl_dir)],
    ):
        main()
        raw = capsys.readouterr().out
        out = json.loads(raw[raw.index("{") :])
        assert out["steps"] == 3
        assert (rl_dir / "generator_grpo.pt").exists()

    ds_dir = tmp_path / "ds_out"
    with patch.object(
        sys,
        "argv",
        ["ncm-bench", "distill-smoke", "--samples", "12", "--steps", "4", "--out-dir", str(ds_dir)],
    ):
        main()
        raw = capsys.readouterr().out
        out = json.loads(raw[raw.index("{") :])
        assert out["steps"] == 4
        assert (ds_dir / "generator_distilled.pt").exists()


def test_stage2_dpo_preference_optimization() -> None:
    torch.manual_seed(11)
    vocab = 16
    seq_len = 16
    sequences = torch.randint(0, vocab, (16, seq_len), dtype=torch.long)
    prefixes = torch.randint(0, vocab, (16, seq_len), dtype=torch.long)

    s1_cfg = Stage1Config(
        in_vocab_size=vocab,
        context_length=seq_len,
        embed_dim=32,
        levels=(2, 4, 8, 16),
        vq_vocab_size=16,
        n_layers=1,
        batch_size=8,
        steps=3,
        device="cpu",
    )
    vqvae = train_stage1_vqvae(sequences, s1_cfg)["model"]
    s2_cfg = Stage2Config(
        prefix_len=seq_len,
        n_layer=1,
        n_head=2,
        n_embd=32,
        batch_size=8,
        steps=3,
        query_level0=True,
        device="cpu",
    )
    policy = train_stage2_ncm(prefixes, sequences, vqvae, s1_cfg, s2_cfg)["model"]
    with torch.no_grad():
        chosen = policy.generate(None, prefixes, temperature=0.5, top_k=4)
        rejected = policy.generate(None, prefixes, temperature=1.5, top_k=4)

    dpo_out = train_stage2_dpo(
        prefixes,
        chosen,
        rejected,
        vqvae,
        policy,
        DPOConfig(batch_size=4, steps=4, lr=1e-4, beta=0.1, device="cpu", seed=11),
    )
    assert len(dpo_out["losses"]) == 4
    assert isinstance(dpo_out["initial_loss"], float)
    assert isinstance(dpo_out["final_loss"], float)
    assert 0.0 <= dpo_out["preference_accuracy"] <= 1.0
