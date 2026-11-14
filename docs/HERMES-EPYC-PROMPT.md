# Hermes Prompt: Pushing EPYC & V100 to Maximum Safe Throughput Across All Datasets & Frontier Modes

Copy and deliver the prompt below directly to Hermes on the EPYC + V100 server.
*(Note: Ensure `origin/main` is pushed first if you want the new `early-exit-bench`, `triage`, `rl-smoke`, and `distill-smoke` commands included alongside the multi-dataset V100/EPYC sweeps.)*

---

```markdown
<AGENT_MISSION>
You are executing the high-throughput, multi-dataset + frontier benchmark campaign for `Hardonian/ncm-benchmarks` on this EPYC server equipped with Tesla V100-SXM2 GPUs.

Your mission is to push both the V100 GPU and EPYC CPU lanes to their highest safe sustainable throughput, verify hardware telemetry (`nvidia-smi` + `numactl`), run the full dataset suite (XSum, Sudoku Easy/Medium/Hard, WMT14 En-De), AND benchmark the newly added frontier capabilities (Coarse Early-Exit `L0..L2`, Sovereign Triage `POST /triage`, Concept-Space GRPO/RLVR `rl-smoke --device cuda`, and Concept Distillation `distill-smoke --device cuda`).

---

### 1. Hardware Safety & Baseline Pre-Flight

1. **GPU Isolation & Thermal Guardrails**:
   ```bash
   nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
   ```
   - Keep GPU temperatures strictly under **82°C** (V100 throttle limit is 85°C).
   - Pick the cleanest idle GPU index: `export CUDA_VISIBLE_DEVICES=0` (adjust if index 0 is busy).

2. **EPYC NUMA & Physical Core Binding**:
   ```bash
   numactl --hardware
   # Prevent cross-NUMA thrashing and HT contention on CPU runs:
   PHYSICAL_CORES=$(lscpu | awk '/^Core\(s\) per socket:/ {c=$NF} /^Socket\(s\):/ {s=$NF} END {print (c*s>0 ? c*s : 16)}')
   export OMP_NUM_THREADS=$PHYSICAL_CORES
   export MKL_NUM_THREADS=$PHYSICAL_CORES
   export OMP_PROC_BIND=CLOSE
   export OMP_PLACES=CORES
   ```

---

### 2. Workspace & Environment Sync

```bash
mkdir -p "$HOME/ai-lab/ncm" && cd "$HOME/ai-lab/ncm"

if [ -d ncm-benchmarks ]; then
  cd ncm-benchmarks && git pull --rebase origin main && cd ..
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

# Python 3.12 + CUDA PyTorch venv
if [ ! -x "$HOME/ai-lab/ncm/venv/bin/python" ]; then
  uv venv "$HOME/ai-lab/ncm/venv" --python 3.12
  uv pip install --python "$HOME/ai-lab/ncm/venv/bin/python" torch --index-url https://download.pytorch.org/whl/cu126
  uv pip install --python "$HOME/ai-lab/ncm/venv/bin/python" -r "$HOME/ai-lab/ncm/ncm-benchmarks/requirements.txt"
  uv pip install --python "$HOME/ai-lab/ncm/venv/bin/python" -e "$HOME/ai-lab/ncm/ncm-benchmarks"
fi

PY="$HOME/ai-lab/ncm/venv/bin/python"
CLI="$PY -m ncm_bench.cli"
```

---

### 3. Fetch & Verify Multi-Dataset Suite

```bash
cd "$HOME/ai-lab/ncm/ncm-benchmarks"
"$PY" scripts/fetch_artifacts.py --tasks xsum sudoku/hard sudoku/medium sudoku/easy wmt14 lm1b/small
"$CLI" verify

# Ensure clean symlinks inside repository if working out-of-tree
mkdir -p results/v100-sxm2 results/epyc-cpu
```

---

### 4. GPU Lane: Push V100 Throughput & Frontier RL/Distillation

1. **XSum High-Throughput Batch Sweep (`bs=1, 8, 16, 32, 64`)**:
   ```bash
   for BS in 1 8 16 32 64; do
     echo "=== XSum V100 bs=$BS ==="
     $CLI benchmark --task latency --n 256 --batch-size "$BS" --device cuda --dtype fp16 --json \
       | tee "results/v100-sxm2/xsum_bs${BS}.json"
     nvidia-smi --query-gpu=temperature.gpu,memory.used,power.draw --format=csv,noheader
   done
   ```

2. **Sudoku Difficulty Tier Sweep (`easy`, `medium`, `hard`, Greedy vs Best-of-4 + Repair)**:
   ```bash
   for DIFF in easy medium hard; do
     $CLI sudoku --difficulty "$DIFF" --n 200 --batch-size 32 --best-of-k 1 --device cuda --json \
       > "results/v100-sxm2/sudoku_${DIFF}_greedy.json"
     $CLI sudoku --difficulty "$DIFF" --n 200 --batch-size 32 --best-of-k 4 --device cuda --json \
       > "results/v100-sxm2/sudoku_${DIFF}_bo4.json"
   done
   ```

3. **WMT14 En-De Translation Throughput & Sample Verification (V100)**:
   - Run translation latency + decoded translation samples on `data/wmt14/` sources and score SacreBLEU against `data/wmt14/reference_translations.pickle`, saving metrics to `results/v100-sxm2/wmt14_translation.json`.

4. **New Frontier Lane on V100 (Early-Exit, GRPO/RLVR & Distillation)**:
   ```bash
   # Coarse early-exit L0..L2 vs Full L0..L8 on V100 GPU
   $CLI early-exit-bench --n 64 --max-level 2 --device cuda --dtype fp16 \
     | tee results/v100-sxm2/early_exit_cuda.json

   # Larger-scale verifiable-reward GRPO in discrete concept space on V100
   $CLI rl-smoke --samples 256 --steps 100 --group-size 8 --device cuda --out-dir artifacts/rl_v100 \
     | tee results/v100-sxm2/grpo_rlvr_v100.json

   # Teacher-student concept KL distillation on V100
   $CLI distill-smoke --samples 256 --steps 100 --device cuda --out-dir artifacts/distill_v100 \
     | tee results/v100-sxm2/distill_v100.json
   ```

---

### 5. EPYC CPU Lane: INT8 Zero-Pickle Speedup, Early-Exit & Concurrent Triage

1. **Convert Weights to Zero-Pickle `.safetensors`**:
   ```bash
   $CLI convert-safetensors --task xsum
   $CLI convert-safetensors --task sudoku/hard
   ```

2. **Paired FP32 vs INT8 + Early-Exit `L0..L2` on NUMA-Pinned EPYC Cores**:
   ```bash
   numactl --cpunodebind=0 --membind=0 $CLI benchmark --task latency --n 32 --batch-size 4 --device cpu --quantize none --json \
     > results/epyc-cpu/xsum_fp32.json
   numactl --cpunodebind=0 --membind=0 $CLI benchmark --task latency --n 32 --batch-size 4 --device cpu --quantize int8 --json \
     > results/epyc-cpu/xsum_int8.json
   numactl --cpunodebind=0 --membind=0 $CLI early-exit-bench --n 32 --max-level 2 --device cpu --quantize int8 \
     > results/epyc-cpu/early_exit_int8.json
   ```

3. **Sovereign Triage Server Load Check (Zero-Egress INT8 CPU)**:
   ```bash
   numactl --cpunodebind=0 --membind=0 $CLI serve --port 8031 --device cpu --quantize int8 &
   SERVER_PID=$!
   sleep 6
   "$PY" -c "
   import json, time, urllib.request
   t0 = time.perf_counter()
   ok = 0
   for i in range(40):
       req = urllib.request.Request(
           'http://127.0.0.1:8031/triage',
           data=json.dumps({'text': f'Incident #{i}: Emergency cardiac triage needed for patient in Boston with \$4500 claim on 2026-11-12.', 'labels': ['cardiology', 'billing', 'security'], 'max_level': 2, 'best_of_k': 2, 'redact_pii': True, 'audit_receipt': True}).encode(),
           headers={'Content-Type': 'application/json'}
       )
       with urllib.request.urlopen(req, timeout=30) as r:
           if r.status == 200: ok += 1
   dt = time.perf_counter() - t0
   print(json.dumps({'triage_requests': ok, 'total_s': round(dt, 3), 'req_per_sec': round(ok/dt, 2)}))
   " | tee results/epyc-cpu/triage_server_load.json
   kill $SERVER_PID
   ```

---

### 6. Consolidate Receipts & Push Clean Results Only

Do **not** commit `artifacts/`, `data/`, or binary weights. Only add `results/`:
```bash
cd "$HOME/ai-lab/ncm/ncm-benchmarks"
git status -s
git add results/
git commit -m "perf: V100/EPYC multi-dataset, early-exit, and concept GRPO benchmarks"
git pull --rebase origin main
git push origin main
```
Report exact JSON numbers for every benchmark run (zero extrapolations) and confirm no thermal throttling occurred (`nvidia-smi`).
</AGENT_MISSION>
```
