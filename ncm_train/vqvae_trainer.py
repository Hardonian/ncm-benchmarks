"""Stage-1 clean-room trainer for Multiscale Residual VQ-VAE text/discrete tokenizers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from ncm_bench.inference import REPO_ROOT, _ensure_ncm_on_path


@dataclass
class Stage1Config:
    """Configuration for Stage-1 Multiscale Residual VQ-VAE training."""

    in_vocab_size: int = 16
    context_length: int = 16
    embed_dim: int = 64
    levels: tuple[int, ...] = (2, 4, 8, 16)
    vq_vocab_size: int = 32
    n_layers: int = 2
    batch_size: int = 16
    steps: int = 50
    lr: float = 3e-4
    vq_loss_weight: float = 1.0
    device: str = "cpu"


def build_vqvae_from_stage1_config(cfg: Stage1Config):
    """Instantiates an upstream VQVAE model from Stage1Config."""
    _ensure_ncm_on_path(REPO_ROOT / "ncm")
    from models.vqvae import VQVAE

    vq_cfg = OmegaConf.create(
        {
            "clss": "multiscale_residual_vector_quantizer",
            "decay": 0.99,
            "epsilon": 1e-5,
            "commitment_cost": 0.25,
            "learned_l1_sampling": True,
            "learned_all_sampling": False,
            "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0, "learnable_ratio": False},
            "levels": {"use_manual_levels": True, "manual_levels": list(cfg.levels)},
        }
    )
    return VQVAE(
        in_vocab_size=cfg.in_vocab_size,
        context_length=cfg.context_length,
        embed_dim=cfg.embed_dim,
        vqvae_vocab_size=[cfg.vq_vocab_size] * len(cfg.levels),
        n_layers=cfg.n_layers,
        vector_quantizer_config=vq_cfg,
        compression_factor=1,
        dropout=0.1,
        pre_quant_groupnorm=4,
        pre_quant_norm_type="group",
        use_qk_norm=True,
    )


def train_stage1_vqvae(
    sequences: torch.Tensor,
    cfg: Stage1Config,
    *,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Trains a Stage-1 Multiscale Residual VQ-VAE on integer token tensors (N, L)."""
    if sequences.ndim != 2 or sequences.shape[1] != cfg.context_length:
        raise ValueError(
            f"Expected sequences of shape (N, {cfg.context_length}), got {tuple(sequences.shape)}"
        )

    dev = torch.device(cfg.device)
    model = build_vqvae_from_stage1_config(cfg).to(dev)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    loader = DataLoader(TensorDataset(sequences.long()), batch_size=cfg.batch_size, shuffle=True)

    history: list[dict[str, float]] = []
    step = 0
    while step < cfg.steps:
        for (batch_x,) in loader:
            if step >= cfg.steps:
                break
            batch_x = batch_x.to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits, vq_loss, _, global_util, _ = model(batch_x)
            recon_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch_x.reshape(-1))
            loss = recon_loss + cfg.vq_loss_weight * vq_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.item()),
                    "recon_loss": float(recon_loss.item()),
                    "vq_loss": float(vq_loss.item()),
                    "codebook_util": float(global_util),
                }
            )
            step += 1

    model.eval()
    if out_path is not None:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, out_file)

    return {"model": model, "history": history, "final_loss": history[-1]["loss"]}

