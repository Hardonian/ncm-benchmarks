"""Clean-room Stage-1 (Multiscale Residual VQ-VAE) and Stage-2 (Next-Concept Transformer) trainers.

Upstream HazyResearch/ncm released inference/evaluation code only without training scripts.
This package implements reproducible Stage-1 and Stage-2 training loops compatible with
upstream NCM checkpoints and NCM inference/evaluation pipelines.
"""
from ncm_train.adapters import (
    CallableRewardPlugin,
    CodebookMonitor,
    ConceptLoRALinear,
    DDPConfig,
    RegexRewardPlugin,
    inject_concept_lora,
    wrap_model_distributed,
)
from ncm_train.ncm_trainer import Stage2Config, train_stage2_ncm
from ncm_train.rl_distill import (
    DistillConfig,
    DPOConfig,
    GRPOConfig,
    compute_concept_logprobs,
    distill_teacher_into_ncm,
    train_stage2_dpo,
    train_stage2_grpo,
)
from ncm_train.vqvae_trainer import Stage1Config, train_stage1_vqvae

__all__ = [
    "CallableRewardPlugin",
    "CodebookMonitor",
    "ConceptLoRALinear",
    "DDPConfig",
    "DPOConfig",
    "DistillConfig",
    "GRPOConfig",
    "RegexRewardPlugin",
    "Stage1Config",
    "Stage2Config",
    "compute_concept_logprobs",
    "distill_teacher_into_ncm",
    "inject_concept_lora",
    "train_stage1_vqvae",
    "train_stage2_dpo",
    "train_stage2_grpo",
    "train_stage2_ncm",
    "wrap_model_distributed",
]

