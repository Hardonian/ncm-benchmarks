# NCM vs. Autoregressive LLMs, Symbolic Solvers & Extractive Baselines

This report documents head-to-head benchmarks comparing **Next-Concept Modeling (NCM)** against matched **Autoregressive Token Transformers (`CausalTokenLM`)**, **classical symbolic solvers**, and **zero-pass extractive/statistical baselines** across scaling laws, constraint-satisfaction games, real news summarization, and sovereign enterprise pipelines.

All figures below were measured directly on consumer CPU hardware (**AMD Ryzen AI 9 HX 370**, Windows 11, PyTorch `2.15.0+cpu`, zero GPU, offline datasets verified via `artifacts/SHA256SUMS`) and recorded verbatim in [`results/hx370-cpu/ncm_vs_baselines.json`](../results/hx370-cpu/ncm_vs_baselines.json). Reproduce any time with:

```bash
ncm-bench compare --n-sudoku 50 --n-xsum 24 --output-file results/hx370-cpu/ncm_vs_baselines.json
```

---

## 1. Executive Summary: Where NCM Wins vs. Where It Trails

| Dimension | Winner | Measured Evidence | Why |
| :--- | :--- | :--- | :--- |
| **Sequence Decoding Complexity** | **NCM ($O(L)$)** | **`13.04x` speedup** at `T=128` (`4.70 ms` vs `61.23 ms`, `4` vs `128` passes) | Hierarchical block-diagonal generation holds neural passes constant regardless of sequence length. |
| **Pretrained Hard Constraint Satisfaction** (`data/sudoku/hard`, `n=50`) | **NCM 149M + Repair** | **`86.0%` solved** (`88.0%` valid, `97.68%` cell acc) vs **`64.0%`** cold 64-node backtrack, **`28.0%`** naked singles, **`0.0%`** 81-step AR | Global discrete concept refinement resolves coupled 81-cell constraints that defeat sequential row decoding. |
| **Abstractive Compression** (`data/xsum`, `n=24`, zero-leak `P=512`) | **NCM 149M INT8** | **`28.74` ROUGE-L** (`36.52` R-1 / `12.02` R-2) vs **`12.97`** TF-IDF Centroid & **`11.83`** Lead-1 | Single-sentence BBC XSum summaries paraphrase aggressively; extractive baselines hit a hard ceiling (`<13` R-L). |
| **Sovereign Privacy, Restoration & Audit** | **NCM Stack** | **`100.0%`** PII/Key/IBAN redacted, **`100.0%`** restored, **`100.0%`** Ed25519 verified, **`$10.00`** FinQA mismatch caught | Unprotected cloud egress leaks `100%` of spans (`0.0%` protected); destructive regex cannot restore entities (`0.0%`). |
| **Micro-Scale Cold Training** (`512` samples, `<0.3M` params, `450` steps) | **CausalTokenLM (AR)** | **`70.83%` vs `35.42%`** on 4x4 Mini-Sudoku; **`68.75%` vs `22.92%`** valid Dyck-2 | Direct token cross-entropy converges immediately on tiny cold datasets; two-stage VQ+NCM needs larger scale or pretraining. |
| **4-Peg Mastermind** (`512` train, `48` eval) | **NCM Best-of-8** | **`20.83%` consistent / `4.17%` exact** vs **`6.25%` / `2.08%`** CausalTokenLM & **`2.08%` / `0.0%`** Random | Parallel concept sampling produces diverse candidates for exact symbolic clue filtering in `4` passes. |
| **Hybrid Zero-Shot Triage** (`L0..L2` frozen XSum codes + lexical prior) | **Keyword Rules** (top-1) / **Hybrid Conformal NCM** (coverage) | **`91.67%`** Keyword rules vs **`58.33%`** Hybrid NCM Top-1 (**`83.33%`** conformal coverage) | Adding `v3.0` lexical prior boosts unfine-tuned NCM top-1 from `41.67%` to `58.33%` while conformal prediction flags uncertainty (`3.08` avg set). |

---

## 2. Scaling Laws: Constant $O(L)$ Concept Passes vs. Linear $O(T)$ Autoregressive

Both models run a matched 64-dimensional, 2-layer, 4-head Transformer backbone (`Stage2NCM` with 4 hierarchical levels vs. `CausalTokenLM` with sequential next-token causal masking) on CPU (`batch=4`):

| Target Tokens ($T$) | NCM Passes ($L$) | AR Passes ($T$) | Pass Reduction | NCM Latency (ms/sample) | AR Latency (ms/sample) | Measured CPU Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `16` | `4` | `16` | `4.0x` | `2.69 ms` | `4.12 ms` | **`1.53x`** |
| `32` | `4` | `32` | `8.0x` | `3.36 ms` | `8.22 ms` | **`2.44x`** |
| `64` | `4` | `64` | `16.0x` | `3.27 ms` | `22.88 ms` | **`7.00x`** |
| `128` | `4` | `128` | `32.0x` | `4.70 ms` | `61.23 ms` | **`13.04x`** |

As sequence length increases `8x` from `16` to `128` tokens, autoregressive CPU latency balloons **`14.86x`** (`4.12 ms -> 61.23 ms`), whereas NCM latency remains flat at **`2.69 ms -> 4.70 ms`**.

---

## 3. Games & Constraint Reasoning Arena

### 3.1 Real 9x9 Sudoku-Hard (`data/sudoku/hard`, `n=50` boards)

| Method | Neural Passes | Solved (%) | Solved Count | 95% Wilson CI | Cell Acc (%) | Valid Board (%) | CPU Wall (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `autoregressive_81step_row_greedy` | `81` | `0.0%` | `0 / 50` | `[0.00, 7.13]` | `59.80%` | `0.0%` | `0.010s` |
| `naked_singles_only` | `0` | `28.0%` | `14 / 50` | `[17.47, 41.67]` | `61.19%` | `28.0%` | `0.034s` |
| `cold_backtrack_64_nodes` | `0` | `64.0%` | `32 / 50` | `[50.14, 75.86]` | `78.91%` | `64.0%` | `0.256s` |
| `ncm_149m_greedy_no_repair` | `9` | `80.0%` | `40 / 50` | `[66.96, 88.76]` | `97.31%` | `80.0%` | `10.115s` |
| `ncm_149m_greedy_repair` | `9` | `84.0%` | `42 / 50` | `[71.49, 91.66]` | `97.41%` | `84.0%` | `9.547s` |
| **`ncm_149m_best_of_4_repair`** | **`9`** | **`86.0%`** | **`43 / 50`** | **`[73.81, 93.05]`** | **`97.68%`** | **`88.0%`** | `18.407s` |

- Pure neural greedy NCM (`80.0%`) outperforms cold 64-node symbolic backtracking (`64.0%`) by **`+16.0 pp`** and best-of-4 + 64-node repair extends the lead to **`+22.0 pp` (`86.0%`)** in only `9` sequential neural passes (`9x` fewer neural passes than 81-step AR).

### 3.2 Clean-Room Cold-Start Games (`n_train=512`, `n_eval=48`, `450` CPU training steps)

To test behavior **without** large-scale pretraining, we trained tiny clean-room models from scratch (`embed_dim=48` / `vq_vocab_size=32` Stage-1 VQ-VAE + `64`-dim 2-layer Stage-2 NCM vs. `64`-dim 2-layer `CausalTokenLM`):

| Game | Method | Passes | Primary Metric | Secondary Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Mini-Sudoku 4x4** (10/16 masked) | `naked_singles_heuristic` | `0` | `31.25%` solved | `78.26%` cell acc |
| **Mini-Sudoku 4x4** (10/16 masked) | **`causal_token_lm_16steps`** | `16` | **`70.83%` solved** | **`87.24%` cell acc** (`0.97 ms/board`) |
| **Mini-Sudoku 4x4** (10/16 masked) | `ncm_concept_4levels_raw` | `4` | `4.17%` solved | `68.49%` cell acc (`1.11 ms/board`) |
| **Mini-Sudoku 4x4** (10/16 masked) | `ncm_concept_4levels_plus_repair` | `4` | `35.42%` solved | `78.52%` cell acc (`1.11 ms/board`) |
| **Mastermind 4-Peg** (3 clues) | `uniform_random` | `0` | `2.08%` consistent | `0.00%` exact recovery |
| **Mastermind 4-Peg** (3 clues) | `causal_token_lm` | `4` | `6.25%` consistent | `2.08%` exact recovery |
| **Mastermind 4-Peg** (3 clues) | `ncm_concept_4levels_greedy` | `4` | `8.33%` consistent | `0.00%` exact recovery |
| **Mastermind 4-Peg** (3 clues) | **`ncm_concept_4levels_best_of_8`** | `4` | **`20.83%` consistent** | **`4.17%` exact recovery** |
| **Dyck-2 Bracket Closure** (16 tok) | **`causal_token_lm_8steps`** | `8` | **`68.75%` valid Dyck-2** | **`8.33%` exact match** |
| **Dyck-2 Bracket Closure** (16 tok) | `ncm_concept_4levels_greedy` | `4` | `2.08%` valid Dyck-2 | `0.00%` exact match |
| **Dyck-2 Bracket Closure** (16 tok) | `ncm_concept_4levels_best_of_8` | `4` | `22.92%` valid Dyck-2 | `2.08%` exact match |

**Honest Scientific Takeaway**: Two-stage discrete concept modeling (`VQ-VAE + Stage-2 NCM`) has an optimization tax on tiny cold-start datasets (`512` examples / `450` steps): `CausalTokenLM` learns direct token-to-token mappings faster in low-data regimes (`70.83%` vs `35.42%` on 4x4 Sudoku; `68.75%` vs `22.92%` on Dyck-2). However, once a VQ codebook and concept hierarchy are properly pretrained (`149M` Sudoku-Hard) or paired with symbolic candidate verification (`Best-of-8` Mastermind `20.83%` vs `6.25%` AR), NCM flips the outcome decisively in fewer sequential passes.

---

## 4. Real BBC XSum Summarization (`data/xsum`, `n=24`, Clean `P=512` Prefix Slice)

Every baseline receives the identical `<=512`-token input article slice (`bundle["P"] = 512`, strictly excluding target summary tokens stored after position `512` in `passages.pickle`):

| Summarization Approach | Neural Passes | ROUGE-1 | ROUGE-2 | ROUGE-L | CPU Wall (`n=24`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `lead_1_extractive` (First Sentence) | `0` | `14.99` | `1.37` | `11.83` | `0.0018s` |
| `tfidf_centroid_extractive` | `0` | `19.12` | `3.84` | `12.97` | `0.0182s` |
| **`ncm_149m_int8_safetensors`** | **`9`** | **`36.52`** | **`12.02`** | **`28.74`** | `47.792s` (`1.99s/doc`) |

- Full `11,334`-document V100 reference (`results/v100-sxm2/`): **`27.93` ROUGE-L** at **`9.47 docs/sec`**.
- NCM INT8 delivers **`+15.77` ROUGE-L** over TF-IDF Centroid and **`+16.91` ROUGE-L** over Lead-1 in `9` parallel passes on CPU with zero GPU.

---

## 5. Enterprise Triage, Uncertainty & Sovereign Compliance

### 5.1 Paraphrased / Synonym-Shifted Incident Triage (`n=12`)

- **Keyword Rules**: **`91.67%`** top-1 accuracy (strong when domain keywords remain intact).
- **Bag-of-Words Cosine**: `58.33%` top-1 accuracy.
- **Hybrid NCM `149M` INT8 `L0..L2` (`alpha=0.1` Conformal + Lexical Prior)**: **`58.33%`** top-1 accuracy -> **`83.33%` conformal prediction-set coverage** (average set size `3.08`), demonstrating how `v3.0.0` hybrid keyword+concept routing matches Bag-of-Words top-1 while conformal calibration flags borderline items.

### 5.2 Sovereign PII Protection, Entity Restoration, FinQA & Signatures

| Capability | Unprotected Cloud LLM | Destructive Regex Mask | Sovereign NCM Pipeline |
| :--- | :--- | :--- | :--- |
| PII, API Key & IBAN Protected Before Egress | `0.0%` | `100.0%` | **`100.0%`** |
| Reversible Post-Inference Entity Restoration | `0.0%` | `0.0%` | **`100.0%`** |
| Detached Ed25519 Audit Signature Verified | `0.0%` | `0.0%` | **`100.0%`** |
| Deterministic FinQA Table Check (`$400.00` vs `$390.00`) | Unverified | Unverified | **`$10.00` Discrepancy Caught** |
