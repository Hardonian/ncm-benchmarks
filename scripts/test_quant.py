import os
import sys
import time
import torch

os.environ["NCM_HOME"] = os.getcwd()
os.environ["NCM_ROOT"] = os.path.join(os.getcwd(), "ncm")
os.environ["NCM_DEVICE"] = "cpu"
sys.path.insert(0, os.getcwd())

from service.summarize_server import load, summarize

print("[Quant Test] Loading fp32 model...")
S = load("artifacts/xsum/config.yaml", "artifacts/xsum", "cpu")

sample_text = (
    "The UK government has announced a major investment into renewable energy infrastructure, "
    "aiming to double wind and solar capacity by 2030. Ministers said the move will lower bills "
    "and cut emissions across England and Wales."
)

# Benchmark fp32
t0 = time.perf_counter()
fp32_summary = summarize(S, sample_text)
fp32_time = time.perf_counter() - t0
print(f"[Quant Test] FP32 summary ({fp32_time:.2f}s): {fp32_summary}")

# Apply PyTorch dynamic quantization to transformer linear layers
print("[Quant Test] Applying torch.ao.quantization.quantize_dynamic to Linear layers...")
q_net = torch.ao.quantization.quantize_dynamic(
    S["net"],
    {torch.nn.Linear},
    dtype=torch.qint8
)
S_quant = dict(S)
S_quant["net"] = q_net

# Benchmark quantized int8
t1 = time.perf_counter()
int8_summary = summarize(S_quant, sample_text)
int8_time = time.perf_counter() - t1
print(f"[Quant Test] INT8 summary ({int8_time:.2f}s): {int8_summary}")
print(f"[Quant Test] Latency delta: fp32={fp32_time:.2f}s vs int8={int8_time:.2f}s (speedup: {fp32_time/int8_time:.2f}x)")

