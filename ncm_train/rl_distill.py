"""Discrete Concept GRPO/RLVR & Teacher-Student Distillation for Next-Concept Modeling."""
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ncm_train.ncm_trainer import Stage2Config, build_ncm_from_configs
from ncm_train.vqvae_trainer import Stage1Config


@dataclass
class GRPOConfig:
    """Configuration for Group Relative Policy Optimization in discrete concept space."""

    group_size: int = 4
    batch_size: int = 4
    steps: int = 12
    lr: float = 1e-4
    temperature: float = 0.8
    top_k: int = 16
    clip_eps: float = 0.2
    kl_beta: float = 0.04
    device: str = "cpu"


@dataclass
class DistillConfig:
    """Configuration for Teacher-Student distillation into Stage-2 NCM."""

    batch_size: int = 16
    steps: int = 40
    lr: float = 5e-4
    label_smoothing: float = 0.05
    kl_weight: float = 0.5
    temperature: float = 1.5
    device: str = "cpu"


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization in discrete concept space."""

    batch_size: int = 8
    steps: int = 20
    lr: float = 3e-4
    beta: float = 0.1
    device: str = "cpu"
    seed: int = 42


def compute_concept_logprobs(
    net,
    vqvae,
    prefixes: torch.Tensor,
    codes: torch.Tensor,
    *,
    prefix_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes per-code conditional log-probabilities and raw logits in ONE parallel pass."""
    p_len = int(prefix_len or getattr(net, "prefix_len", prefixes.shape[1]))
    x_ncm_target = vqvae.vq.idx_to_ncm_input(codes)
    if getattr(net, "query_level0", True):
        q0 = net.build_l0_query_input(prefixes.shape[0], dtype=x_ncm_target.dtype)
        x_target_all = torch.cat([q0, x_ncm_target], dim=1)
    else:
        x_target_all = x_ncm_target

    z_prefix = vqvae.encoder.wte(prefixes.long())[:, :p_len, :]
    x_in = torch.cat([z_prefix, x_target_all], dim=1)
    logits = net(x_in)
    log_probs_all = F.log_softmax(logits, dim=-1)
    code_logprobs = log_probs_all.gather(-1, codes.unsqueeze(-1)).squeeze(-1)
    return code_logprobs, logits


def train_stage2_grpo(
    prefixes: torch.Tensor,
    reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    vqvae,
    net,
    cfg: GRPOConfig,
    *,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Runs Group Relative Policy Optimization (RLVR) directly over NCM discrete concept trajectories."""
    dev = torch.device(cfg.device)
    vqvae = vqvae.to(dev).eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    net = net.to(dev)
    ref_net = copy.deepcopy(net).to(dev).eval()
    for p in ref_net.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=0.01)
    num_prompts = prefixes.shape[0]
    prefix_len = int(getattr(net, "prefix_len", prefixes.shape[1]))
    history: list[dict[str, float]] = []

    for step in range(cfg.steps):
        idx = torch.randint(0, num_prompts, (min(cfg.batch_size, num_prompts),), device=prefixes.device)
        prompt_b = prefixes[idx].to(dev).long()
        bs = prompt_b.shape[0]
        g = cfg.group_size
        prompt_rep = prompt_b.repeat_interleave(g, dim=0)

        net.eval()
        with torch.no_grad():
            sampled_codes = net.generate(
                None,
                prompt_rep,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
            )
            decoded_logits = vqvae.decode(sampled_codes)
            rewards_flat = reward_fn(decoded_logits, prompt_rep).to(dev).float().reshape(bs, g)

            old_lp, _ = compute_concept_logprobs(net, vqvae, prompt_rep, sampled_codes, prefix_len=prefix_len)
            ref_lp, _ = compute_concept_logprobs(ref_net, vqvae, prompt_rep, sampled_codes, prefix_len=prefix_len)

        mean_r = rewards_flat.mean(dim=1, keepdim=True)
        std_r = rewards_flat.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
        advantages = ((rewards_flat - mean_r) / std_r).reshape(-1).detach()

        net.train()
        cur_lp, _ = compute_concept_logprobs(net, vqvae, prompt_rep, sampled_codes, prefix_len=prefix_len)

        # Clipped GRPO policy objective over average concept trajectory log-prob
        seq_ratio = torch.exp((cur_lp.mean(dim=1) - old_lp.mean(dim=1)).clamp(-5.0, 5.0))
        clipped = torch.clamp(seq_ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
        pg_loss = -torch.minimum(seq_ratio * advantages, clipped * advantages).mean()

        # Non-negative Schulman KL divergence penalty against frozen reference policy
        delta = (ref_lp - cur_lp).clamp(-10.0, 10.0)
        kl_div = (torch.exp(delta) - delta - 1.0).mean()
        loss = pg_loss + cfg.kl_beta * kl_div

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        history.append(
            {
                "step": float(step),
                "mean_reward": float(rewards_flat.mean().item()),
                "max_reward": float(rewards_flat.max().item()),
                "kl": float(kl_div.item()),
                "loss": float(loss.item()),
            }
        )

    net.eval()
    if out_path is not None:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": net.state_dict(), "grpo_history": history}, out_file)

    return {
        "model": net,
        "history": history,
        "initial_reward": history[0]["mean_reward"] if history else 0.0,
        "final_reward": history[-1]["mean_reward"] if history else 0.0,
    }


def distill_teacher_into_ncm(
    prefixes: torch.Tensor,
    teacher_targets: torch.Tensor,
    vqvae,
    s1_cfg: Stage1Config,
    s2_cfg: Stage2Config,
    distill_cfg: DistillConfig | None = None,
    *,
    teacher_net=None,
    student_net=None,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Distills teacher completions (plus optional teacher concept logits) into a compact Stage-2 NCM."""
    d_cfg = distill_cfg or DistillConfig(device=s2_cfg.device)
    dev = torch.device(d_cfg.device)
    vqvae = vqvae.to(dev).eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    student = (student_net or build_ncm_from_configs(s1_cfg, s2_cfg, vqvae)).to(dev)
    student.train()
    if teacher_net is not None:
        teacher_net = teacher_net.to(dev).eval()
        for p in teacher_net.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=d_cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, d_cfg.steps))
    loader = DataLoader(
        TensorDataset(prefixes.long(), teacher_targets.long()),
        batch_size=d_cfg.batch_size,
        shuffle=True,
    )

    history: list[dict[str, float]] = []
    step = 0
    while step < d_cfg.steps:
        for pfx_b, tgt_b in loader:
            if step >= d_cfg.steps:
                break
            pfx_b = pfx_b.to(dev)
            tgt_b = tgt_b.to(dev)

            with torch.no_grad():
                target_codes = vqvae.encode(tgt_b)
                teacher_logits = None
                if teacher_net is not None:
                    _, teacher_logits = compute_concept_logprobs(
                        teacher_net, vqvae, pfx_b, target_codes, prefix_len=s2_cfg.prefix_len
                    )

            _, student_logits = compute_concept_logprobs(
                student, vqvae, pfx_b, target_codes, prefix_len=s2_cfg.prefix_len
            )
            ce_loss = F.cross_entropy(
                student_logits.reshape(-1, student_logits.shape[-1]),
                target_codes.reshape(-1),
                label_smoothing=d_cfg.label_smoothing,
            )
            loss = ce_loss
            if teacher_logits is not None and d_cfg.kl_weight > 0.0:
                t_scale = d_cfg.temperature
                s_logp = F.log_softmax(student_logits / t_scale, dim=-1)
                t_prob = F.softmax(teacher_logits / t_scale, dim=-1)
                kd_loss = F.kl_div(s_logp, t_prob, reduction="batchmean") * (t_scale * t_scale)
                loss = ce_loss + d_cfg.kl_weight * kd_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            history.append({"step": float(step), "loss": float(loss.item())})
            step += 1

    student.eval()
    if out_path is not None:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        unigram = torch.bincount(teacher_targets.reshape(-1), minlength=s1_cfg.in_vocab_size).float()
        unigram = unigram / unigram.sum().clamp_min(1.0)
        torch.save({"model": student.state_dict(), "inference": {"token_unigram": unigram}}, out_file)

    return {
        "model": student,
        "history": history,
        "initial_loss": history[0]["loss"] if history else 0.0,
        "final_loss": history[-1]["loss"] if history else 0.0,
    }


def train_stage2_dpo(
    prefixes: torch.Tensor,
    chosen_codes: torch.Tensor,
    rejected_codes: torch.Tensor,
    vqvae,
    policy_net,
    cfg: DPOConfig,
    *,
    ref_net=None,
) -> dict[str, Any]:
    """Direct Preference Optimization over winning vs losing discrete NCM code sequences."""
    dev = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)

    vqvae = vqvae.to(dev).eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    policy_net = policy_net.to(dev)
    if ref_net is None:
        ref_net = copy.deepcopy(policy_net)
    ref_net = ref_net.to(dev).eval()
    for p in ref_net.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=cfg.lr, weight_decay=0.01)
    num_prompts = prefixes.shape[0]
    prefix_len = int(getattr(policy_net, "prefix_len", prefixes.shape[1]))
    losses: list[float] = []
    accuracies: list[float] = []

    for _step in range(cfg.steps):
        idx = torch.randint(0, num_prompts, (min(cfg.batch_size, num_prompts),), device=prefixes.device)
        x_b = prefixes[idx].to(dev).long()
        c_win = chosen_codes[idx].to(dev).long()
        c_lose = rejected_codes[idx].to(dev).long()

        policy_net.train()
        lp_pi_win, _ = compute_concept_logprobs(policy_net, vqvae, x_b, c_win, prefix_len=prefix_len)
        lp_pi_lose, _ = compute_concept_logprobs(policy_net, vqvae, x_b, c_lose, prefix_len=prefix_len)
        with torch.no_grad():
            lp_ref_win, _ = compute_concept_logprobs(ref_net, vqvae, x_b, c_win, prefix_len=prefix_len)
            lp_ref_lose, _ = compute_concept_logprobs(ref_net, vqvae, x_b, c_lose, prefix_len=prefix_len)

        pi_diff = lp_pi_win.sum(dim=-1) - lp_pi_lose.sum(dim=-1)
        ref_diff = lp_ref_win.sum(dim=-1) - lp_ref_lose.sum(dim=-1)
        logits = cfg.beta * (pi_diff - ref_diff)
        loss = -F.logsigmoid(logits).mean()
        acc = float((logits > 0).float().mean().item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()

        losses.append(float(loss.item()))
        accuracies.append(acc)

    policy_net.eval()
    return {
        "model": policy_net,
        "losses": losses,
        "initial_loss": losses[0] if losses else 0.0,
        "final_loss": losses[-1] if losses else 0.0,
        "preference_accuracy": accuracies[-1] if accuracies else 0.0,
    }
