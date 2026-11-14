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
    ema_decay: float = 0.99
    dropout: float = 0.1
    grad_clip: float = 1.0
    lr_schedule: str = "cosine"
    grad_accum_steps: int = 1
    resume_from: Path | str | None = None
    device: str = "cpu"


def _load_checkpoint_state(path: Path, dev: torch.device) -> dict[str, torch.Tensor]:
    """Loads state_dict from .safetensors or weights-only .pt file."""
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file as st_load_file

        return st_load_file(str(path), device=str(dev))
    sd = torch.load(str(path), map_location=dev, weights_only=True)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        return sd["model"]
    return sd


def build_vqvae_from_stage1_config(cfg: Stage1Config):
    """Instantiates an upstream VQVAE model from Stage1Config."""
    _ensure_ncm_on_path(REPO_ROOT / "ncm")
    from models.vqvae import VQVAE

    vq_cfg = OmegaConf.create(
        {
            "clss": "multiscale_residual_vector_quantizer",
            "decay": cfg.ema_decay,
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
        dropout=cfg.dropout,
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
    if cfg.resume_from is not None and Path(cfg.resume_from).exists():
        model.load_state_dict(_load_checkpoint_state(Path(cfg.resume_from), dev), strict=True)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.steps))
        if cfg.lr_schedule == "cosine"
        else None
    )
    loader = DataLoader(TensorDataset(sequences.long()), batch_size=cfg.batch_size, shuffle=True)
    accum = max(1, int(cfg.grad_accum_steps))

    history: list[dict[str, float]] = []
    step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < cfg.steps:
        for (batch_x,) in loader:
            if step >= cfg.steps:
                break
            batch_x = batch_x.to(dev)
            logits, vq_loss, _, global_util, _ = model(batch_x)
            recon_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch_x.reshape(-1))
            loss = recon_loss + cfg.vq_loss_weight * vq_loss
            (loss / accum).backward()
            micro_step += 1

            if micro_step % accum == 0:
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

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
        if out_file.suffix.lower() == ".safetensors":
            from safetensors.torch import save_file as st_save_file

            st_save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out_file))
        else:
            torch.save({"model": model.state_dict()}, out_file)

    return {"model": model, "history": history, "final_loss": history[-1]["loss"]}

