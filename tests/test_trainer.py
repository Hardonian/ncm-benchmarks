"""Tests for the clean-room Stage-1 VQ-VAE and Stage-2 Next-Concept Transformer trainers."""

from __future__ import annotations

from pathlib import Path

import torch

from ncm_train import (
    Stage1Config,
    Stage2Config,
    train_stage1_vqvae,
    train_stage2_ncm,
)


def test_stage1_and_stage2_training_and_generation(tmp_path: Path) -> None:
    torch.manual_seed(0)
    sequences = torch.randint(0, 16, (24, 16), dtype=torch.long)
    prefixes = torch.randint(0, 16, (24, 16), dtype=torch.long)

    s1_cfg = Stage1Config(
        in_vocab_size=16,
        context_length=16,
        embed_dim=32,
        levels=(2, 4, 8, 16),
        vq_vocab_size=16,
        n_layers=1,
        batch_size=8,
        steps=4,
        device="cpu",
    )
    s1_ckpt = tmp_path / "vqvae.pt"
    s1_out = train_stage1_vqvae(sequences, s1_cfg, out_path=s1_ckpt)
    vqvae = s1_out["model"]
    assert s1_ckpt.exists()
    assert len(s1_out["history"]) == 4
    assert s1_out["final_loss"] > 0.0

    s2_cfg = Stage2Config(
        prefix_len=16,
        n_layer=1,
        n_head=2,
        n_embd=32,
        batch_size=8,
        steps=4,
        query_level0=True,
        device="cpu",
    )
    s2_ckpt = tmp_path / "ncm.pt"
    s2_out = train_stage2_ncm(prefixes, sequences, vqvae, s1_cfg, s2_cfg, out_path=s2_ckpt)
    model = s2_out["model"]
    assert s2_ckpt.exists()
    assert len(s2_out["history"]) == 4
    assert s2_out["final_loss"] > 0.0

    with torch.no_grad():
        codes = model.generate(
            None,
            prefixes[:2],
            top_k=4,
        )
        assert codes.shape == (2, sum(s1_cfg.levels))
        decoded_logits = vqvae.decode(codes)
    assert decoded_logits.shape == (2, 16, 16)
