"""Stage-2 clean-room trainer for Next-Concept Transformer conditional generators."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from ncm_bench.inference import REPO_ROOT, _ensure_ncm_on_path
from ncm_train.vqvae_trainer import Stage1Config


@dataclass
class Stage2Config:
    """Configuration for Stage-2 Next-Concept Transformer training."""

    prefix_len: int = 16
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.1
    batch_size: int = 16
    steps: int = 60
    lr: float = 5e-4
    corrupt_prob: float = 0.1
    query_level0: bool = True
    device: str = "cpu"


def build_ncm_from_configs(s1: Stage1Config, s2: Stage2Config, vqvae):
    """Instantiates upstream NextConceptTransformer wrapped around a frozen VQVAE."""
    _ensure_ncm_on_path(REPO_ROOT / "ncm")
    from models.next_concept import NextConceptTransformer

    levels = list(s1.levels)
    n_heads = len(levels) if s2.query_level0 else len(levels) - 1
    model_cfg = OmegaConf.create(
        {
            "n_layer": s2.n_layer,
            "n_head": s2.n_head,
            "bias": True,
            "dropout": s2.dropout,
            "n_embd": s2.n_embd,
            "context_length": s2.prefix_len + sum(levels),
            "prefix_len": s2.prefix_len,
            "raw_token_prefix": True,
            "vocab_size": [s1.vq_vocab_size] * n_heads,
            "attn_pattern": "block_diagonal",
            "use_positional_encoding": True,
            "use_level_encoding": True,
            "use_rope": True,
            "rope_base": 10000,
            "use_qk_norm": True,
            "query_level0": s2.query_level0,
            "predict_level0": not s2.query_level0,
            "tie_heads_to_codebook": False,
        }
    )
    return NextConceptTransformer(vqvae_local=vqvae, config=model_cfg)


def _corrupt_codes(idx: torch.Tensor, vocab_size: int, prob: float) -> torch.Tensor:
    if prob <= 0.0:
        return idx
    mask = torch.rand_like(idx.float()) < prob
    noise = torch.randint(0, vocab_size, idx.shape, device=idx.device, dtype=idx.dtype)
    return torch.where(mask, noise, idx)


def train_stage2_ncm(
    prefixes: torch.Tensor,
    targets: torch.Tensor,
    vqvae,
    s1_cfg: Stage1Config,
    s2_cfg: Stage2Config,
    *,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Trains NextConceptTransformer given (prefixes, targets) paired integer tensors."""
    dev = torch.device(s2_cfg.device)
    vqvae = vqvae.to(dev).eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    net = build_ncm_from_configs(s1_cfg, s2_cfg, vqvae).to(dev)
    net.train()

    optimizer = torch.optim.AdamW(net.parameters(), lr=s2_cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, s2_cfg.steps))
    loader = DataLoader(
        TensorDataset(prefixes.long(), targets.long()),
        batch_size=s2_cfg.batch_size,
        shuffle=True,
    )

    history: list[dict[str, float]] = []
    step = 0
    while step < s2_cfg.steps:
        for pfx_b, tgt_b in loader:
            if step >= s2_cfg.steps:
                break
            pfx_b = pfx_b.to(dev)
            tgt_b = tgt_b.to(dev)

            with torch.no_grad():
                clean_codes = vqvae.encode(tgt_b)
                noisy_codes = _corrupt_codes(clean_codes, s1_cfg.vq_vocab_size, s2_cfg.corrupt_prob)
                x_ncm_target = vqvae.vq.idx_to_ncm_input(noisy_codes)
                if net.query_level0:
                    q0 = net.build_l0_query_input(pfx_b.shape[0], dtype=x_ncm_target.dtype)
                    x_target_all = torch.cat([q0, x_ncm_target], dim=1)
                    target_labels = clean_codes
                else:
                    x_target_all = x_ncm_target
                    target_labels = clean_codes

                z_prefix = vqvae.encoder.wte(pfx_b)[:, : s2_cfg.prefix_len, :]
                x_in = torch.cat([z_prefix, x_target_all], dim=1)

            optimizer.zero_grad(set_to_none=True)
            logits = net(x_in)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            history.append({"step": float(step), "loss": float(loss.item())})
            step += 1

    net.eval()
    if out_path is not None:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        unigram = torch.bincount(targets.reshape(-1), minlength=s1_cfg.in_vocab_size).float()
        unigram = unigram / unigram.sum().clamp_min(1.0)
        torch.save(
            {"model": net.state_dict(), "inference": {"token_unigram": unigram}},
            out_file,
        )

    return {"model": net, "history": history, "final_loss": history[-1]["loss"]}

