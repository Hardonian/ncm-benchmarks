"""Hardware acceleration primitives: INT4 groupwise packing, ONNX manifests, compile wrappers & fused VQ lookup."""
from __future__ import annotations

import importlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_nibble_lut() -> torch.Tensor:
    """Returns a (256, 2) int8 lookup table decoding packed uint8 nibble pairs into [-7, 8]."""
    bytes_t = torch.arange(256, dtype=torch.int16)
    hi = (bytes_t >> 4) - 7
    lo = (bytes_t & 0x0F) - 7
    return torch.stack([hi, lo], dim=1).to(torch.int8)


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


def _dequantize_from_buffers(
    packed: torch.Tensor,
    scales: torch.Tensor,
    weight_shape: tuple[int, ...],
    group_size: int,
    padded_numel: int,
    *,
    lut: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unpacks uint8 nibble pairs via lookup table and rescales to target tensor shape."""
    unpacked = lut[packed.to(torch.long)].reshape(-1)[:padded_numel]
    groups = unpacked.reshape(-1, group_size).to(out_dtype) * scales.unsqueeze(1).to(out_dtype)
    total = math.prod(weight_shape)
    return groups.reshape(-1)[:total].reshape(weight_shape)


def dequantize_weights_int4_groupwise(pack: dict[str, Any]) -> torch.Tensor:
    """Unpacks 4-bit nibble tensor via nibble LUT and rescales back to FP32 weights."""
    packed = pack["packed"]
    lut = _build_nibble_lut().to(device=packed.device)
    return _dequantize_from_buffers(
        packed,
        pack["scales"],
        tuple(pack["shape"]),
        int(pack["group_size"]),
        int(pack["padded_numel"]),
        lut=lut,
        out_dtype=torch.float32,
    )


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
            importlib.import_module("onnx")

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
    fallback_reason: str | None = None

    def _run_vq_bench(target_dev: str) -> float:
        z = torch.randn(16, 64, device=target_dev)
        cb = torch.randn(256, 64, device=target_dev)
        t0 = time.perf_counter()
        _, q = fused_residual_vq_lookup(z, cb)
        _ = float(q.sum().item())
        return round((time.perf_counter() - t0) * 1000.0, 3)

    try:
        vq_ms = _run_vq_bench(device)
    except Exception as exc:
        if device == "cpu":
            raise
        fallback_reason = str(exc)
        device = "cpu"
        vq_ms = _run_vq_bench("cpu")

    result: dict[str, Any] = {
        "active_device": device,
        "cuda_available": cuda_ok,
        "mps_available": mps_ok,
        "rocm_hip": rocm_ver,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else f"{os.name}-cpu",
        "vq_lookup_16x256_ms": vq_ms,
    }
    if fallback_reason is not None:
        result["accelerator_fallback_reason"] = fallback_reason
    return result


class Int4Linear(nn.Module):
    """Drop-in groupwise 4-bit nibble-packed Linear replacement with LUT-accelerated dequantized GEMM."""

    def __init__(
        self,
        linear: nn.Linear,
        group_size: int = 32,
        *,
        cache_dequantized: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        pack = quantize_weights_int4_groupwise(linear.weight.data, group_size=group_size)
        self.group_size = int(pack["group_size"])
        self.padded_numel = int(pack["padded_numel"])
        self.weight_shape = tuple(pack["shape"])
        self.compression_ratio = float(pack["compression_ratio"])
        self.cache_dequantized = bool(cache_dequantized)
        self._cached_weight: torch.Tensor | None = None
        self.register_buffer("packed", pack["packed"])
        self.register_buffer("scales", pack["scales"])
        self.register_buffer("_nibble_lut", _build_nibble_lut().to(device=pack["packed"].device), persistent=False)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 32,
        *,
        cache_dequantized: bool = False,
    ) -> "Int4Linear":
        """Constructs an Int4Linear module from an existing nn.Linear layer."""
        return cls(linear, group_size=group_size, cache_dequantized=cache_dequantized)

    def _unpack_weights(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.cache_dequantized and self._cached_weight is not None:
            if self._cached_weight.device == device and self._cached_weight.dtype == dtype:
                return self._cached_weight
            self._cached_weight = self._cached_weight.to(device=device, dtype=dtype)
            return self._cached_weight

        lut = self._nibble_lut if self._nibble_lut.device == self.packed.device else self._nibble_lut.to(self.packed.device)
        weight = _dequantize_from_buffers(
            self.packed,
            self.scales,
            self.weight_shape,
            self.group_size,
            self.padded_numel,
            lut=lut,
            out_dtype=dtype,
        ).to(device=device, dtype=dtype)
        if self.cache_dequantized:
            self._cached_weight = weight
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._unpack_weights(device=x.device, dtype=x.dtype)
        bias = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


def quantize_model_int4(
    model: nn.Module,
    *,
    group_size: int = 32,
    cache_dequantized: bool = False,
) -> dict[str, Any]:
    """Recursively replaces nn.Linear submodules with weight-only groupwise Int4Linear modules."""
    layers_quantized = 0
    total_compression = 0.0

    def _recurse(parent: nn.Module) -> None:
        nonlocal layers_quantized, total_compression
        for name, child in list(parent.named_children()):
            if isinstance(child, Int4Linear):
                continue
            if isinstance(child, nn.Linear):
                q_mod = Int4Linear.from_linear(
                    child,
                    group_size=group_size,
                    cache_dequantized=cache_dequantized,
                )
                setattr(parent, name, q_mod)
                layers_quantized += 1
                total_compression += q_mod.compression_ratio
            else:
                _recurse(child)

    _recurse(model)
    avg_ratio = round(total_compression / max(1, layers_quantized), 3) if layers_quantized else 1.0
    return {
        "model": model,
        "layers_quantized": layers_quantized,
        "group_size": int(group_size),
        "cache_dequantized": bool(cache_dequantized),
        "avg_compression_ratio": avg_ratio,
    }
