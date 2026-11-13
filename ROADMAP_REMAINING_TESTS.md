# NCM Benchmarks — Remaining Tests & Evaluation Roadmap
**Generated:** 2026-11-12 | **Current State:** v0.2.0 + multi-dataset suite committed (5ddfdda)

---

## ✅ COMPLETED (This Session)

| Task | Device | Status | Artifacts |
|------|--------|--------|-----------|
| XSum batch sweep (bs=1,8,16,32,64) | V100 | ✅ | `xsum_batch_sweep.json` |
| XSum full census (11,334 docs, bs=32) | V100 | ✅ | `xsum_full_census.json` |
| WMT14 En-De single-sample | V100 | ⚠️ Partial | `wmt14_translation.json` |
| Sudoku Easy/Medium/Hard (BoK=4, repair) | V100 | ✅ | `sudoku_multi_tier.json` |
| INT8 CPU XSum latency (NUMA node 0) | EPYC | ✅ | `int8_scaling.json` |
| INT8 CPU Sudoku Hard (BoK=4, repair) | EPYC | ✅ | `sudoku_hard_int8/fp32.json` |
| Safetensors conversion (xsum, sudoku/hard, wmt14) | — | ✅ | artifacts/*.safetensors |

---

## 🔴 HIGH PRIORITY — Core Gaps

### 1. Full WMT14 Evaluation (3,003 sentences)
- **Why:** Single-sample BLEU=0.79 is meaningless; paper reports corpus BLEU
- **Command:**
  ```bash
  cd ~/ai-lab/ncm/ncm-benchmarks
  HF_HOME=... NCM_ROOT=... python -c "
  from ncm_bench.inference import load_bundle, summarize_texts
  import pickle, json
  bundle = load_bundle('wmt14', device='cuda', dtype='fp32')
  with open('data/wmt14/sources.pickle','rb') as f: sources = pickle.load(f)
  with open('data/wmt14/reference_translations.pickle','rb') as f: refs = pickle.load(f)
  # Batch translate all 3003
  translations = []
  for i in range(0, len(sources), 32):
      batch = sources[i:i+32]
      translations.extend(summarize_texts(bundle, batch))
  from sacrebleu import corpus_bleu
  bleu = corpus_bleu(translations, [refs])
  print(f'BLEU: {bleu.score:.2f}')
  " 2>&1 | tee results/v100-sxm2/wmt14_full_bleu.json
  ```

### 2. lm1b/small Language Modeling Benchmark
- **Why:** Fetched but never run; measures perplexity on 1B-word LM
- **Command:**
  ```bash
  python -m ncm_bench benchmark --task lm1b/small --device cuda --dtype fp32 --json
  ```

### 3. Multi-GPU Data Parallel (if 2nd clean V100 available)
- **Why:** Mission explicitly asks for multi-GPU sharding sweeps
- **Check:** `nvidia-smi -L` — if 2x V100, run DDP shards
- **Target:** 2x V100 → 2x throughput on XSum/Sudoku

---

## 🟡 MEDIUM PRIORITY — Quality/Throughput Frontiers

### 4. FP16/BF16 Mixed Precision on V100
- **Why:** V100 supports FP16 tensor cores; 2x throughput potential
- **Configs:**
  ```bash
  # FP16 (native V100 tensor cores)
  --dtype fp16 --batch-size 32
  # BF16 (not native on V100, emulated)
  --dtype bf16 --batch-size 32
  ```
- **Measure:** marginal_s_per_doc, VRAM, numerical parity vs FP32

### 5. Batch Scaling Pareto Frontier (Granular)
- **Current:** bs=1,8,16,32,64 (64 OOM)
- **Needed:** bs=2,4,24,40,48 — find exact VRAM cliff
- **Script:** `scripts/benchmark_mbr.py` adaptation

### 6. MBR Decoding Quality vs Latency
- **Why:** MBR (candidates=4,8,16) trades latency for quality
- **Command:**
  ```bash
  python -m ncm_bench mbr --candidates 4 --text "..." --device cuda --json
  python -m ncm_bench mbr --candidates 8 --text "..." --device cuda --json
  ```
- **Measure:** ROUGE-L vs latency_ms

### 7. Long Document Summarization (XSum map-reduce)
- **Why:** `--long-doc` flag enables chunked summarization
- **Test:** passages >1800 chars → num_chunks, quality, latency

### 8. Sudoku Greedy Baseline (BoK=1, no repair)
- **Why:** Paper reports 86.1% greedy; we have BoK=4+repair=90.7%
- **Command:**
  ```bash
  python -m ncm_bench sudoku --difficulty hard --best-of-k 1 --no-repair --n 2000 --device cuda --json
  ```

---

## 🟢 LOWER PRIORITY — Extended Ecosystem

### 9. vLLM Serve Benchmark (OpenAI-compatible)
- **Why:** Production serving path; compare against Python API
- **Setup:** Export safetensors → vLLM engine → `/v1/completions` load test
- **Target:** concurrent requests, p50/p99 latency, throughput

### 10. P40 24GB Lane — Large Model Benchmarks
- **Models that fit P40 only:**
  - `qwen3.5:27b` (~16GB Q4) — 27B reasoning
  - `codellama:34b` (~20GB Q4) — code generation
  - `deepseek-r1:14b` — reasoning (fits V100 too)
- **Router:** `curl :11438/route-info?model=qwen3.5:27b` → should return `p40`

### 11. Ollama Router Throughput (Port 11438)
- **Why:** All local consumers route through router; measure proxy overhead
- **Test:** `ab -n 1000 -c 16 http://localhost:11438/api/generate`

### 12. Progressive Decoding Inspection (Concept Hierarchy)
- **Why:** NCM unique feature — 9-level coarse-to-fine
- **Command:** `python -m ncm_bench inspect --html out.html --device cuda`

### 13. ComfyUI / Diffusion Benchmarks (P40)
- **If relevant:** SDXL 1024×1024 steps=20/35, Flux FP8
- **Metrics:** sec/image, VRAM peak, quality (manual eval)

---

## 📊 DATASET EXTENSIONS (Fetch + Eval)

| Dataset | Status | Eval Needed |
|---------|--------|-------------|
| `xsum` | ✅ Fetched, ✅ Benchmarked | MBR, long-doc, FP16 |
| `sudoku/easy` | ✅ Fetched, ✅ Benchmarked | Greedy baseline |
| `sudoku/medium` | ✅ Fetched, ✅ Benchmarked | Greedy baseline |
| `sudoku/hard` | ✅ Fetched, ✅ Benchmarked | Greedy baseline, INT8 |
| `wmt14` | ✅ Fetched, ⚠️ Single-sample | **Full 3003 corpus BLEU** |
| `lm1b/small` | ✅ Fetched, ❌ Not run | **Perplexity benchmark** |
| `cnn_dailymail` | ❌ Not fetched | Summarization (longer docs) |
| `squad` | ❌ Not fetched | QA/extractive |
| `gsm8k` | ❌ Not fetched | Math reasoning |
| `humaneval` | ❌ Not fetched | Code generation |

**Add to `fetch_artifacts.py`:**
```python
# In scripts/fetch_artifacts.py --tasks add:
# cnn_dailymail squad gsm8k humaneval
```

---

## 🛠 INFRASTRUCTURE FIXES

| Issue | Fix |
|-------|-----|
| `batch_triage.py` missing `requests` | `uv pip install requests` in venv |
| Serve command `-m ncm_bench` fails | Use CLI: `python -m ncm_bench.cli serve ...` |
| HF offline mode breaks tokenizer loads | Pre-cache all tokenizers online once |
| bs=64 OOM on V100 | Evict Ollama models first: `curl :11434/api/generate -d '{"model":"qwen3.5:9b","keep_alive":0}'` |

---

## 🎯 NEXT SESSION EXECUTION ORDER

```bash
# 1. Full WMT14 corpus BLEU (highest impact)
# 2. lm1b/small perplexity
# 3. FP16 batch sweep (bs=32,48)
# 4. Sudoku greedy baseline (BoK=1)
# 5. MBR candidates=4,8 quality sweep
# 6. If 2nd V100: DDP data parallel
# 7. P40 large model routing verification
# 8. vLLM export + serve benchmark
```

---

## 📁 RESULT FILE NAMING CONVENTION

```
results/<device>/<task>_<config>.json
# Examples:
results/v100-sxm2/wmt14_full_bleu.json
results/v100-sxm2/xsum_mbr_k8.json
results/v100-sxm2/sudoku_hard_greedy.json
results/epyc-cpu/xsum_fp16_bs32.json
results/p40/qwen3.5-27b_routing.json
```
