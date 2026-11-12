# Hermes Prompt: Dedicated EPYC / V100 GPU Benchmark Run

Copy and deliver the prompt below directly to Hermes or your coding agent session on the EPYC server.

---

```markdown
<AGENT_MISSION>
You are executing the dedicated GPU benchmark lane for `Hardonian/ncm-benchmarks` on this EPYC machine with Tesla V100-SXM2 GPUs.

Your goal is to perform a clean, unencumbered GPU benchmark run that eliminates the co-tenant memory pressure caveat from our preliminary spike, validates exact throughput scaling, and (optionally) runs the full 11,334-doc XSum test set.

### 1. Pre-Flight Check (GPU Isolation)
Run `nvidia-smi` to inspect the available GPUs.
Ensure you select an index with 0 MB resident memory (or minimal resident usage).
Set `export CUDA_VISIBLE_DEVICES=<clean_gpu_index>` (default is 0 if empty).

### 2. Workspace & Environment Sync
Working directory: `$HOME/ai-lab/ncm`
```bash
mkdir -p "$HOME/ai-lab/ncm" && cd "$HOME/ai-lab/ncm"

# Pull latest ncm-benchmarks
if [ -d ncm-benchmarks ]; then
  cd ncm-benchmarks && git pull origin main && cd ..
else
  git clone https://github.com/Hardonian/ncm-benchmarks.git
fi

# Ensure upstream HazyResearch/ncm is cloned and patched
if [ ! -d ncm ]; then
  git clone https://github.com/HazyResearch/ncm ncm
  cd ncm
  git checkout c70274273f489abc73832b2b98584a8483b6b203
  git apply "$HOME/ai-lab/ncm/ncm-benchmarks/patches/device-support.patch"
  cd ..
fi

# Setup Python 3.12 environment with CUDA PyTorch (cu126)
if [ ! -x "$HOME/ai-lab/ncm/venv/bin/python" ]; then
  uv venv "$HOME/ai-lab/ncm/venv" --python 3.12
  uv pip install --python "$HOME/ai-lab/ncm/venv/bin/python" torch --index-url https://download.pytorch.org/whl/cu126
  uv pip install --python "$HOME/ai-lab/ncm/venv/bin/python" -r "$HOME/ai-lab/ncm/ncm-benchmarks/requirements.txt"
fi

# Verify CUDA visibility and capability
"$HOME/ai-lab/ncm/venv/bin/python" -c "import torch; assert torch.cuda.is_available(); print('Active GPU:', torch.cuda.get_device_name(0), '| Memory:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 2), 'GB')"
```

### 3. Fetch & Verify Artifacts
```bash
cd "$HOME/ai-lab/ncm/ncm-benchmarks"
"$HOME/ai-lab/ncm/venv/bin/python" scripts/fetch_artifacts.py
cd "$HOME/ai-lab/ncm"
ln -sfn "$HOME/ai-lab/ncm/ncm-benchmarks/artifacts" artifacts
ln -sfn "$HOME/ai-lab/ncm/ncm-benchmarks/data" data
```

### 4. Execute the Clean GPU Runbook
Execute the runner:
```bash
chmod +x "$HOME/ai-lab/ncm/ncm-benchmarks/scripts/as-run/epyc_runbook.sh"
# To include the full 11,334-doc XSum pass (~16 min), set RUN_FULL_XSUM=1:
RUN_FULL_XSUM=1 "$HOME/ai-lab/ncm/ncm-benchmarks/scripts/as-run/epyc_runbook.sh"
```

### 5. Inspect & Update Results
Read the generated logs:
1. `results/sudoku_full/score.txt`: verify solved percentage (target: ~86.3% on 2,000 boards).
2. `results/xsum_bs32_n1000/rouge.txt`: verify R-1, R-2, and R-L scores.
3. Compute clean marginal latency from `results/xsum_bs1_n40/time.txt` and `results/xsum_bs1_n8/time.txt`.
4. Update `ncm-benchmarks/results/v100-sxm2/results.json`:
   - Change `"sharing"` to `"dedicated GPU (clean VRAM, zero co-tenant allocator warnings)"`.
   - Update marginal latency and throughput figures.
   - If `RUN_FULL_XSUM=1` was executed, add the `xsum_v100_full_11334` entry with exact ROUGE numbers.

### 6. Commit & Push
```bash
cd "$HOME/ai-lab/ncm/ncm-benchmarks"
git add results/v100-sxm2/results.json
git commit -m "perf: record clean dedicated V100 GPU benchmark results (co-tenant pressure removed)"
git pull --rebase origin main
git push origin main
```

### 7. Sync Storefront Blog Draft
Update the blog post draft located at:
`$HOME/ai-lab/reports/content/drafts/2026-11-12-ncm-*`
Drop the co-tenant footnote and insert the clean V100 latency, throughput, and full census numbers.
Confirm completion with the summary output.
</AGENT_MISSION>
```

