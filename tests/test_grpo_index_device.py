"""Regression coverage for GRPO prompt sampling across tensor/model devices."""
import math
from unittest.mock import patch

import pytest
import torch

from ncm_train import GRPOConfig, Stage1Config, Stage2Config, train_stage2_grpo
from ncm_train.ncm_trainer import build_ncm_from_configs
from ncm_train.vqvae_trainer import build_vqvae_from_stage1_config


@pytest.mark.parametrize(
    ("prefix_device", "train_device"),
    [("cpu", "cpu"), ("cpu", "cuda"), ("cuda", "cuda"), ("cuda", "cpu")],
)
def test_grpo_samples_on_prefix_device(prefix_device, train_device):
    if "cuda" in (prefix_device, train_device) and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(7)
    s1 = Stage1Config(embed_dim=32, n_layers=1, vq_vocab_size=16)
    s2 = Stage2Config(n_layer=1, n_head=2, n_embd=32)
    vqvae = build_vqvae_from_stage1_config(s1)
    net = build_ncm_from_configs(s1, s2, vqvae)
    prefixes = torch.randint(0, 16, (4, 16), device=prefix_device)
    original_randint = torch.randint
    sampled_devices = []

    def check_randint(*args, **kwargs):
        if not sampled_devices:
            sampled_devices.append(torch.device(kwargs["device"]))
            assert sampled_devices[-1] == prefixes.device
        return original_randint(*args, **kwargs)

    def reward(logits, prompts):
        assert prompts.device.type == train_device
        return logits.argmax(dim=-1).float().mean(dim=-1)

    with patch("ncm_train.rl_distill.torch.randint", side_effect=check_randint):
        result = train_stage2_grpo(prefixes, reward, vqvae, net,
                                   GRPOConfig(group_size=2, batch_size=2, steps=1, device=train_device))
    assert sampled_devices == [prefixes.device]
    assert len(result["history"]) == 1
    assert math.isfinite(result["history"][0]["loss"])
