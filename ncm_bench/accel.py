"""Hardware acceleration primitives: INT4 groupwise packing, ONNX manifests, compile wrappers & fused VQ lookup."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch


def quantize_weights_int4_groupwise(weight: torch.Tensor, group_size: int = 32) -> dict[str, Any]:
    """Packs FP32/FP16 2D weight tensor into signed 4-bit nibbles (2 values per uint8 byte) + per-group scales."""
    w = weight.detach().float().contiguous()
    orig_shape = tuple(w.shape)
    flat = w.reshape(-1)
    pad = (group_size - (flat.numel() % group_size)) % group_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])
    groups = flat.reshape(-1, group_size)
    scales = groups.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) / 7.0
    q_signed = torch.round(groups / scales).clamp(-7, 7).to(torch.int8)
    # Shift [-7, 7] by +7 to [0, 14] and pack pairs of nibbles into uint8
    u4 = (q_signed + 7).to(torch.uint8).reshape(-1)
    if u4.numel() % 2 == 1:
        u4 = torch.cat([u4, torch.zeros(1, dtype=torch.uint8, device=u4.device)])
    hi = u4[0::2] << 4
    lo = u4[1::2] & 0x0F
    packed = (hi | lo).contiguous()
    fp32_bytes = max(1, weight.numel() * 4)
    compressed_bytes = packed.numel() + scales.numel() * 4
    return {
        "packed": packed,
        "scales": scales.squeeze(1).contiguous(),
        "shape": orig_shape,
        "group_size": int(group_size),
        "padded_numel": int(flat.numel()),
        "compression_ratio": round(fp32_bytes / max(1, compressed_bytes), 3),
    }


def dequantize_weights_int4_groupwise(pack: dict[str, Any]) -> torch.Tensor:
    """Unpacks 4-bit nibble tensor and rescales back to FP32 weights."""
    packed = pack["packed"]
    scales = pack["scales"].unsqueeze(1)
    group_size = int(pack["group_size"])
    padded_numel = int(pack["padded_numel"])
    hi = (packed >> 4).to(torch.int8) - 7
    lo = (packed & 0x0F).to(torch.int8) - 7
    unpacked = torch.stack([hi, lo], dim=1).reshape(-1)[:padded_numel]
    groups = unpacked.reshape(-1, group_size).float() * scales
    target_shape = tuple(pack["shape"])
    total = 1
    for dim in target_shape:
        total *= dim
    return groups.reshape(-1)[:total].reshape(target_shape)


def fused_residual_vq_lookup(
    z_rest: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes fused L2 nearest-codebook lookup via torch.addmm without full pairwise allocation."""
    z_flat = z_rest.reshape(-1, codebook.shape[1]).float()
    cb = codebook.float()
    z_sq = (z_flat * z_flat).sum(dim=1, keepdim=True)
    cb_sq = (cb * cb).sum(dim=1).unsqueeze(0)
    dists = torch.addmm(z_sq + cb_sq, z_flat, cb.t(), beta=1.0, alpha=-2.0)
    indices = dists.argmin(dim=-1)
    quantized = cb[indices].reshape(z_rest.shape)
    return indices.reshape(z_rest.shape[:-1]), quantized


def export_ncm_onnx_manifest(
    model: torch.nn.Module,
    out_path: Path | str,
    *,
    seq_len: int = 16,
    input_dim: int | None = None,
) -> dict[str, Any]:
    """Exports an ONNX execution-provider manifest (and ONNX graph when `onnx` is installed)."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    param_count = sum(p.numel() for p in model.parameters())
    providers = ["CPUExecutionProvider"]
    if torch.cuda.is_available():
        providers.append("CUDAExecutionProvider")
    if os.environ.get("OPENVINO_DIR"):
        providers.append("OpenVINOExecutionProvider")

    exported_onnx = False
    if input_dim is not None:
        try:
            import onnx  # noqa: F401

            dummy = torch.randn(1, seq_len, input_dim)
            onnx_file = target.with_suffix(".onnx")
            torch.onnx.export(model, dummy, str(onnx_file), dynamo=False)
            exported_onnx = onnx_file.exists()
        except Exception:
            exported_onnx = False

    manifest = {
        "schema": "ncm-onnx-manifest-v1",
        "param_count": param_count,
        "seq_len": int(seq_len),
        "providers_detected": providers,
        "onnx_graph_exported": exported_onnx,
    }
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def compile_static_concept_decoder(
    model: torch.nn.Module,
    *,
    mode: str = "reduce-overhead",
) -> dict[str, Any]:
    """Compiles a static-shape concept decoder via torch.compile when supported on host."""
    if not hasattr(torch, "compile"):
        return {"model": model, "compiled": False, "mode": mode}
    try:
        compiled_model = torch.compile(model, mode=mode)
        return {"model": compiled_model, "compiled": True, "mode": mode}
    except Exception:
        return {"model": model, "compiled": False, "mode": mode}


def probe_accelerator_capabilities() -> dict[str, Any]:
    """Probes CPU, CUDA, Apple MPS, AMD ROCm/HIP and runs a lightweight fused VQ micro-benchmark."""
    cuda_ok = bool(torch.cuda.is_available())
    mps_ok = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    rocm_ver = getattr(torch.version, "hip", None)
    device = "cuda" if cuda_ok else ("mps" if mps_ok else "cpu")

    z = torch.randn(16, 64, device=device)
    cb = torch.randn(256, 64, device=device)
    t0 = time.perf_counter()
    _, q = fused_residual_vq_lookup(z, cb)
    _ = float(q.sum().item())
    vq_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return {
        "active_device": device,
        "cuda_available": cuda_ok,
        "mps_available": mps_ok,
        "rocm_hip": rocm_ver,
        "device_name": torch.cuda.get_device_name(0) if cuda_ok else f"{os.name}-cpu",
        "vq_lookup_16x256_ms": vq_ms,
    }
