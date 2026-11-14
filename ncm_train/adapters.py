"""Clean-room Concept-Space LoRA adapters, VQ codebook health monitors, DDP wrappers & RLVR reward plugins."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass
class DDPConfig:
    """Configuration for single/multi-GPU DDP or FSDP concept model training."""

    backend: str = "gloo"
    world_size: int = 1
    rank: int = 0
    use_fsdp: bool = False


def wrap_model_distributed(model: nn.Module, cfg: DDPConfig) -> dict[str, Any]:
    """Wraps a Stage-1/Stage-2 model with DDP or FSDP when process group is initialized."""
    if torch.distributed.is_available() and torch.distributed.is_initialized() and cfg.world_size > 1:
        if cfg.use_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            return {"model": FSDP(model), "mode": "fsdp", "world_size": cfg.world_size}
        from torch.nn.parallel import DistributedDataParallel as DDP

        return {"model": DDP(model), "mode": "ddp", "world_size": cfg.world_size}
    return {"model": model, "mode": "single_process", "world_size": cfg.world_size}


class CodebookMonitor:
    """Tracks EMA utilization and perplexity across discrete VQ codebooks and revives dead codes."""

    def __init__(self, vocab_size: int, ema_decay: float = 0.9):
        self.vocab_size = int(vocab_size)
        self.ema_decay = float(ema_decay)
        self.ema_counts = torch.ones(self.vocab_size, dtype=torch.float32) / max(1, self.vocab_size)
        self.steps = 0

    def update(self, indices: torch.Tensor) -> dict[str, float]:
        flat = indices.reshape(-1).detach().cpu()
        hist = torch.bincount(flat.long().clamp(0, self.vocab_size - 1), minlength=self.vocab_size).float()
        probs = hist / hist.sum().clamp_min(1.0)
        self.ema_counts = self.ema_decay * self.ema_counts + (1.0 - self.ema_decay) * probs
        self.steps += 1

        norm_p = self.ema_counts / self.ema_counts.sum().clamp_min(1e-9)
        active = int((norm_p > (0.1 / max(1, self.vocab_size))).sum().item())
        entropy = float(-(norm_p.clamp_min(1e-9) * norm_p.clamp_min(1e-9).log()).sum().item())
        perplexity = float(torch.exp(torch.tensor(entropy)).item())
        return {
            "utilization_ratio": round(active / max(1, self.vocab_size), 4),
            "perplexity": round(perplexity, 4),
            "steps": float(self.steps),
        }

    def reset_dead_codes(
        self,
        codebook_weight: torch.Tensor,
        encoder_latents: torch.Tensor,
        *,
        threshold: float = 0.005,
    ) -> int:
        norm_p = self.ema_counts / self.ema_counts.sum().clamp_min(1e-9)
        dead_mask = norm_p < threshold
        dead_idx = torch.nonzero(dead_mask, as_tuple=False).reshape(-1)
        if dead_idx.numel() == 0 or encoder_latents.numel() == 0:
            return 0
        flat_latents = encoder_latents.detach().reshape(-1, codebook_weight.shape[-1])
        with torch.no_grad():
            for i, code_id in enumerate(dead_idx.tolist()):
                src_row = flat_latents[i % flat_latents.shape[0]].to(
                    device=codebook_weight.device, dtype=codebook_weight.dtype
                )
                codebook_weight[code_id].copy_(src_row)
                self.ema_counts[code_id] = 1.0 / max(1, self.vocab_size)
        return int(dead_idx.numel())


class ConceptLoRALinear(nn.Module):
    """Low-rank adaptation wrapper for discrete Stage-2 Next-Concept projection layers."""

    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = copy.deepcopy(base_layer)
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = max(1, int(rank))
        self.scaling = float(alpha) / self.rank
        dev = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_a = nn.Parameter(torch.randn(base_layer.in_features, self.rank, device=dev, dtype=dtype) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(self.rank, base_layer.out_features, device=dev, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_a @ self.lora_b) * self.scaling


def inject_concept_lora(
    model: nn.Module,
    *,
    rank: int = 4,
    alpha: float = 8.0,
) -> dict[str, Any]:
    """Freezes backbone weights and injects ConceptLoRALinear adapters into top-level Linear submodules."""
    for p in model.parameters():
        p.requires_grad = False

    replaced = 0
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            setattr(model, name, ConceptLoRALinear(module, rank=rank, alpha=alpha))
            replaced += 1
        elif isinstance(module, nn.ModuleList):
            for idx, sub in enumerate(module):
                if isinstance(sub, nn.Linear):
                    module[idx] = ConceptLoRALinear(sub, rank=rank, alpha=alpha)
                    replaced += 1

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "model": model,
        "adapters_injected": replaced,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }


class RegexRewardPlugin:
    """Deterministic regex compliance reward plugin for discrete Concept-RLVR."""

    def __init__(self, required_patterns: list[str], forbidden_patterns: list[str] | None = None):
        self.required = [re.compile(p) for p in required_patterns]
        self.forbidden = [re.compile(p) for p in (forbidden_patterns or [])]

    def score_texts(self, texts: list[str]) -> torch.Tensor:
        rewards: list[float] = []
        denom = max(1, len(self.required))
        for txt in texts:
            hits = sum(1 for pat in self.required if pat.search(txt))
            penalties = sum(1 for pat in self.forbidden if pat.search(txt))
            score = max(0.0, (hits / denom) - (0.5 * penalties))
            rewards.append(round(min(1.0, score), 4))
        return torch.tensor(rewards, dtype=torch.float32)


class CallableRewardPlugin:
    """Wraps a deterministic unit-test / constraint verifier callable into a tensor reward for GRPO."""

    def __init__(self, check_fn: Callable[[list[int]], float]):
        self.check_fn = check_fn

    def __call__(self, decoded_logits: torch.Tensor, prompt_rep: torch.Tensor) -> torch.Tensor:
        del prompt_rep
        token_ids = decoded_logits.argmax(dim=-1).detach().cpu().tolist()
        scores = [float(self.check_fn([int(t) for t in row])) for row in token_ids]
        return torch.tensor(scores, device=decoded_logits.device, dtype=torch.float32)
