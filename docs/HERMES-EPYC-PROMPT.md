# Hermes Prompt: Pushing EPYC & V100 to Maximum Safe Throughput Across All Datasets

Copy and deliver the prompt below directly to Hermes or your coding agent session on the EPYC server.

---

```markdown
<AGENT_MISSION>
You are executing the high-throughput, multi-dataset benchmark and stress-testing campaign for `Hardonian/ncm-benchmarks` on this EPYC server equipped with Tesla V100-SXM2 GPUs.

Your mission is to push both the GPU and CPU lanes to their highest safe sustainable throughput, execute the expanded dataset suite (WMT14 En-De translation, Sudoku Easy/Medium/Hard, and full XSum census), test multi-GPU sharding, and verify batch-scaling Pareto frontiers.

---

### 1. Hardware Safety & Baseline Pre-Flight

1. **GPU Isolation & Health**:
   ```bash
   nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
   ```
   - Target power limits: Ensure fans/thermals remain strictly under 82°C (V100 thermal threshold is 85°C throttle).
   - Set visible GPU: `export CUDA_VISIBLE_DEVICES=0` (or whichever index is clean and unloaded).
   - If multiple clean V100s exist, note indices for multi-GPU data parallel sweeps.

2. **EPYC CPU Tuning**:
   - Prevent cross-NUMA bus thrashing by pinning threads to single socket/NUMA node:
     ```bash
     numactl --hardware
     # Bind PyTorch CPU threads to physical cores (no hyperthread contention):
     export OMP_NUM_THREADS=$(nproc --all | awk '{print int($1/2)}')
     export MKL_NUM_THREADS=$OMP_NUM_THREADS
     export OMP_PROC_BIND=CLOSE
     export OMP_PLACES=CORES
     ```

---

### 2. Workspace & Environment Sync

```bash
mkdir -p "$HOME/ai-lab/ncm" && cd "$HOME/ai-lab/ncm"

# Pull latest master with PyPI 2.0.0, safetensors, and enterprise CLI
if [ -d ncm-benchmarks ]; then
  cd ncm-benchmarks && git pull origin main && cd ..
else
  git clone https://github.com/Hardonian/ncm-benchmarks.git
fi

# Upstream HazyResearch/ncm at verified commit with device patch
if [ ! -d ncm ]; then
  git clone https://github.com/HazyResearch/ncm ncm
  cd ncm
  git checkout c70274273f489abc73832b2b98584a8483b6b203
  git apply "$HOME/ai-lab/ncm/ncm-benchmarks/patches/device-support.patch"
  cd ..
fi

# Python 3.12 + CUDA 12.6 PyTorch
if [ ! -x "$HOME/ai-lab/ncm/venv/bin/python" ]; then
  uv venv "$HOME/ai-lab/ncm/venv" --python 3.12
  uv pip install --python "$HOME/ai-lab/ncm/venv/bin/python" torch --index-url https://download.pytorch.org/whl/cu126
  uv pip install --python "$HOME/ai-lab/ncm/venv/bin/python" -r "$HOME/ai-lab/ncm/ncm-benchmarks/requirements.txt"
  uv pip install --python "$HOME/ai-lab/ncm/venv/bin/python" -e "$HOME/ai-lab/ncm/ncm-benchmarks"
fi
```

---

### 3. Fetch Full Multi-Dataset Suite

Download all upstream datasets and checkpoints supported by `fetch_artifacts.py`:
- `xsum` (88M parameter generator, news summarization)
- `sudoku/hard`, `sudoku/medium`, `sudoku/easy` (25M parameter generator, reasoning & discrete constraint satisfaction)
- `wmt14` (En-De machine translation with Qwen tokenizer)
- `lm1b/small` (language modeling tokenizer & checkpoint)

```bash
cd "$HOME/ai-lab/ncm/ncm-benchmarks"
"$HOME/ai-lab/ncm/venv/bin/python" scripts/fetch_artifacts.py --tasks xsum sudoku/hard sudoku/medium sudoku/easy wmt14 lm1b/small
"$HOME/ai-lab/ncm/venv/bin/python" -m ncm_bench verify

cd "$HOME/ai-lab/ncm"
ln -sfn "$HOME/ai-lab/ncm/ncm-benchmarks/artifacts" artifacts
ln -sfn "$HOME/ai-lab/ncm/ncm-benchmarks/data" data
```

---

### 4. Push GPU Throughput to Safe Maximum

1. **XSum High-Throughput Batch Sweep (V100)**:
   - Run batch sizes: `bs=1`, `bs=8`, `bs=16`, `bs=32`, `bs=64` (verify VRAM consumption with `nvidia-smi` during `bs=64` to verify headroom before saturation).
   - Measure marginal latency ($T_{\text{marginal}} = \frac{T_{40} - T_{8}}{32}$) and docs/sec throughput.
   - Run Full 11,334-doc XSum census at optimal batch size (`bs=32` or `bs=64`).

2. **WMT14 En-De Machine Translation Benchmark (New Task)**:
   - Benchmark translation quality using SacreBLEU on reference translations:
     ```bash
     cd "$HOME/ai-lab/ncm"
     "$HOME/ai-lab/ncm/venv/bin/python" -m ncm_bench translate --input-file data/wmt14/sources.pickle --output-file results/wmt14_v100.json --device cuda
     ```
   - Calculate BLEU score vs `data/wmt14/reference_translations.pickle`.

3. **Sudoku Difficulty Tier Sweep (Easy vs Medium vs Hard)**:
   - Run greedy + `Best-of-4` + bounded constraint repair across all three difficulty tiers (`2000` boards each):
     - `sudoku/easy`
     - `sudoku/medium`
     - `sudoku/hard`
   - Record exact accuracy, cell accuracy, and boards/second.

---

### 5. Push EPYC CPU to Maximum Throughput

1. **INT8 Quantization Speedup Across All Tasks**:
   - Convert models to Zero-Pickle `.safetensors`:
     ```bash
     "$HOME/ai-lab/ncm/venv/bin/python" -m ncm_bench convert-safetensors --task xsum
     "$HOME/ai-lab/ncm/venv/bin/python" -m ncm_bench convert-safetensors --task sudoku/hard
     "$HOME/ai-lab/ncm/venv/bin/python" -m ncm_bench convert-safetensors --task wmt14
     ```
   - Run paired FP32 vs INT8 CPU inference with NUMA core binding:
     ```bash
     numactl --cpunodebind=0 --membind=0 "$HOME/ai-lab/ncm/venv/bin/python" -m ncm_bench summarize --device cpu --dtype int8 --input-file data/xsum/passages.pickle --limit 100
     ```

2. **Parallel Process Scaling (Multi-Worker CPU Serving)**:
   - Test sovereign REST API under multi-client concurrency on EPYC (4 workers, 8 threads each):
     ```bash
     "$HOME/ai-lab/ncm/venv/bin/python" -m ncm_bench serve --port 8000 --device cpu --dtype int8 --workers 4 &
     SERVER_PID=$!
     sleep 5
     # Run triage load test
     "$HOME/ai-lab/ncm/venv/bin/python" scripts/batch_triage.py --concurrency 16 --limit 200
     kill $SERVER_PID
     ```

---

### 6. Results Consolidation & Git Sync

Record all outputs in structured JSON under `results/`:
- `results/v100-sxm2/wmt14_translation.json`
- `results/v100-sxm2/sudoku_multi_tier.json`
- `results/epyc-cpu/int8_scaling.json`

Commit and push:
```bash
cd "$HOME/ai-lab/ncm/ncm-benchmarks"
git add results/
git commit -m "perf: record multi-dataset V100/EPYC benchmark suite (WMT14, Sudoku tiers, INT8 NUMA scaling)"
git pull --rebase origin main
git push origin main
```
Confirm completion and output the final summary metrics.
</AGENT_MISSION>
```
