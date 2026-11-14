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
| **Sequence Decoding Complexity** | **NCM ($O(L)$)** | **`12.24x` speedup** at `T=128` (`3.88 ms` vs `47.52 ms`, `4` vs `128` passes) | Hierarchical block-diagonal generation holds neural passes constant regardless of sequence length. |
| **Pretrained Hard Constraint Satisfaction** (`data/sudoku/hard`, `n=50`) | **NCM 149M + Repair** | **`86.0%` solved** (`88.0%` valid, `97.68%` cell acc) vs **`64.0%`** cold 64-node backtrack, **`28.0%`** naked singles, **`0.0%`** 81-step AR | Global discrete concept refinement resolves coupled 81-cell constraints that defeat sequential row decoding. |
| **Abstractive Compression** (`data/xsum`, `n=24`, zero-leak `P=512`) | **NCM 149M INT8** | **`28.74` ROUGE-L** (`36.52` R-1 / `12.02` R-2) vs **`12.97`** TF-IDF Centroid & **`11.83`** Lead-1 | Single-sentence BBC XSum summaries paraphrase aggressively; extractive baselines hit a hard ceiling (`<13` R-L). |
| **Sovereign Privacy, Restoration & Audit** | **NCM Stack** | **`100.0%`** PII/Key/IBAN redacted, **`100.0%`** restored, **`100.0%`** Ed25519 verified, **`$10.00`** FinQA mismatch caught | Unprotected cloud egress leaks `100%` of spans (`0.0%` protected); destructive regex cannot restore entities (`0.0%`). |
| **Micro-Scale Cold Training** (`512` samples, `<0.3M` params, `450` steps) | **CausalTokenLM (AR)** | **`70.83%` vs `25.0%`** on 4x4 Mini-Sudoku; **`68.75%` vs `25.0%`** valid Dyck-2 | Direct token cross-entropy converges immediately on tiny cold datasets; two-stage VQ+NCM needs larger scale or pretraining. |
| **4-Peg Mastermind** (`512` train, `48` eval) | **NCM Best-of-8** | **`16.67%` consistent / `6.25%` exact** vs **`6.25%` / `2.08%`** CausalTokenLM & **`2.08%` / `0.0%`** Random | Parallel concept sampling produces diverse candidates for exact symbolic clue filtering in `4` passes. |
| **Zero-Shot Unfine-tuned Triage** (`L0..L2` frozen XSum codes) | **Keyword Rules** (top-1) / **Conformal NCM** (coverage) | **`91.67%`** Keyword rules vs **`41.67%`** NCM Top-1 (**`83.33%`** conformal coverage) | Pretrained XSum news codes are not news/security fine-tuned; conformal prediction flags uncertainty cleanly (`3.08` avg set). |

---

## 2. Scaling Laws: Constant $O(L)$ Concept Passes vs. Linear $O(T)$ Autoregressive

Both models run a matched 64-dimensional, 2-layer, 4-head Transformer backbone (`Stage2NCM` with 4 hierarchical levels vs. `CausalTokenLM` with sequential next-token causal masking) on CPU (`batch=4`):

| Target Tokens ($T$) | NCM Passes ($L$) | AR Passes ($T$) | Pass Reduction | NCM Latency (ms/sample) | AR Latency (ms/sample) | Measured CPU Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `16` | `4` | `16` | `4.0x` | `3.59 ms` | `4.75 ms` | **`1.33x`** |
| `32` | `4` | `32` | `8.0x` | `3.07 ms` | `8.21 ms` | **`2.67x`** |
| `64` | `4` | `64` | `16.0x` | `3.88 ms` | `15.42 ms` | **`3.98x`** |
| `128` | `4` | `128` | `32.0x` | `3.88 ms` | `47.52 ms` | **`12.24x`** |

As sequence length increases `8x` from `16` to `128` tokens, autoregressive CPU latency balloons **`10.0x`** (`4.75 ms -> 47.52 ms`), whereas NCM latency remains flat at **`3.59 ms -> 3.88 ms`**.

---

## 3. Games & Constraint Reasoning Arena

### 3.1 Real 9x9 Sudoku-Hard (`data/sudoku/hard`, `n=50` boards)

| Method | Neural Passes | Solved (%) | Solved Count | 95% Wilson CI | Cell Acc (%) | Valid Board (%) | CPU Wall (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `autoregressive_81step_row_greedy` | `81` | `0.0%` | `0 / 50` | `[0.00, 7.13]` | `59.80%` | `0.0%` | `0.011s` |
| `naked_singles_only` | `0` | `28.0%` | `14 / 50` | `[17.47, 41.67]` | `61.19%` | `28.0%` | `0.038s` |
| `cold_backtrack_64_nodes` | `0` | `64.0%` | `32 / 50` | `[50.14, 75.86]` | `78.91%` | `64.0%` | `0.273s` |
| `ncm_149m_greedy_no_repair` | `9` | `80.0%` | `40 / 50` | `[66.96, 88.76]` | `97.31%` | `80.0%` | `11.280s` |
| `ncm_149m_greedy_repair` | `9` | `84.0%` | `42 / 50` | `[71.49, 91.66]` | `97.41%` | `84.0%` | `11.219s` |
| **`ncm_149m_best_of_4_repair`** | **`9`** | **`86.0%`** | **`43 / 50`** | **`[73.81, 93.05]`** | **`97.68%`** | **`88.0%`** | `18.173s` |

- Pure neural greedy NCM (`80.0%`) outperforms cold 64-node symbolic backtracking (`64.0%`) by **`+16.0 pp`** and best-of-4 + 64-node repair extends the lead to **`+22.0 pp` (`86.0%`)** in only `9` sequential neural passes (`9x` fewer neural passes than 81-step AR).

### 3.2 Clean-Room Cold-Start Games (`n_train=512`, `n_eval=48`, `450` CPU training steps)

To test behavior **without** large-scale pretraining, we trained tiny clean-room models from scratch (`embed_dim=48` / `vq_vocab_size=32` Stage-1 VQ-VAE + `64`-dim 2-layer Stage-2 NCM vs. `64`-dim 2-layer `CausalTokenLM`):

| Game | Method | Passes | Primary Metric | Secondary Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Mini-Sudoku 4x4** (10/16 masked) | `naked_singles_heuristic` | `0` | `31.25%` solved | `78.26%` cell acc |
| **Mini-Sudoku 4x4** (10/16 masked) | **`causal_token_lm_16steps`** | `16` | **`70.83%` solved** | **`87.24%` cell acc** (`0.79 ms/board`) |
| **Mini-Sudoku 4x4** (10/16 masked) | `ncm_concept_4levels_raw` | `4` | `6.25%` solved | `56.25%` cell acc (`0.62 ms/board`) |
| **Mini-Sudoku 4x4** (10/16 masked) | `ncm_concept_4levels_plus_repair` | `4` | `25.00%` solved | `75.52%` cell acc (`0.62 ms/board`) |
| **Mastermind 4-Peg** (3 clues) | `uniform_random` | `0` | `2.08%` consistent | `0.00%` exact recovery |
| **Mastermind 4-Peg** (3 clues) | `causal_token_lm` | `4` | `6.25%` consistent | `2.08%` exact recovery |
| **Mastermind 4-Peg** (3 clues) | `ncm_concept_4levels_greedy` | `4` | `4.17%` consistent | `0.00%` exact recovery |
| **Mastermind 4-Peg** (3 clues) | **`ncm_concept_4levels_best_of_8`** | `4` | **`16.67%` consistent** | **`6.25%` exact recovery** |
| **Dyck-2 Bracket Closure** (16 tok) | **`causal_token_lm_8steps`** | `8` | **`68.75%` valid Dyck-2** | **`8.33%` exact match** |
| **Dyck-2 Bracket Closure** (16 tok) | `ncm_concept_4levels_greedy` | `4` | `4.17%` valid Dyck-2 | `0.00%` exact match |
| **Dyck-2 Bracket Closure** (16 tok) | `ncm_concept_4levels_best_of_8` | `4` | `25.00%` valid Dyck-2 | `2.08%` exact match |

**Honest Scientific Takeaway**: Two-stage discrete concept modeling (`VQ-VAE + Stage-2 NCM`) has an optimization tax on tiny cold-start datasets (`512` examples / `450` steps): `CausalTokenLM` learns direct token-to-token mappings faster in low-data regimes (`70.83%` vs `25.0%` on 4x4 Sudoku; `68.75%` vs `25.0%` on Dyck-2). However, once a VQ codebook and concept hierarchy are properly pretrained (`149M` Sudoku-Hard) or paired with symbolic candidate verification (`Best-of-8` Mastermind `16.67%` vs `6.25%` AR), NCM flips the outcome decisively in fewer sequential passes.

---

## 4. Real BBC XSum Summarization (`data/xsum`, `n=24`, Clean `P=512` Prefix Slice)

Every baseline receives the identical `<=512`-token input article slice (`bundle["P"] = 512`, strictly excluding target summary tokens stored after position `512` in `passages.pickle`):

| Summarization Approach | Neural Passes | ROUGE-1 | ROUGE-2 | ROUGE-L | CPU Wall (`n=24`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `lead_1_extractive` (First Sentence) | `0` | `14.99` | `1.37` | `11.83` | `0.0011s` |
| `tfidf_centroid_extractive` | `0` | `19.12` | `3.84` | `12.97` | `0.0232s` |
| **`ncm_149m_int8_safetensors`** | **`9`** | **`36.52`** | **`12.02`** | **`28.74`** | `31.725s` (`1.32s/doc`) |

- Full `11,334`-document V100 reference (`results/v100-sxm2/`): **`27.93` ROUGE-L** at **`9.47 docs/sec`**.
- NCM INT8 delivers **`+15.77` ROUGE-L** over TF-IDF Centroid and **`+16.91` ROUGE-L** over Lead-1 in `9` parallel passes on CPU with zero GPU.

---

## 5. Enterprise Triage, Uncertainty & Sovereign Compliance

### 5.1 Paraphrased / Synonym-Shifted Incident Triage (`n=12`)

- **Keyword Rules**: **`91.67%`** top-1 accuracy (strong when domain keywords remain intact).
- **Bag-of-Words Cosine**: `58.33%` top-1 accuracy.
- **Zero-Shot NCM `149M` INT8 `L0..L2` (`alpha=0.1` Conformal)**: `41.67%` top-1 accuracy -> **`83.33%` conformal prediction-set coverage** (average set size `3.08`), demonstrating why fine-tuning task heads or hybrid keyword+concept routing matters while conformal calibration honestly flags low-confidence borderline items.

### 5.2 Sovereign PII Protection, Entity Restoration, FinQA & Signatures

| Capability | Unprotected Cloud LLM | Destructive Regex Mask | Sovereign NCM Pipeline |
| :--- | :--- | :--- | :--- |
| PII, API Key & IBAN Protected Before Egress | `0.0%` | `100.0%` | **`100.0%`** |
| Reversible Post-Inference Entity Restoration | `0.0%` | `0.0%` | **`100.0%`** |
| Detached Ed25519 Audit Signature Verified | `0.0%` | `0.0%` | **`100.0%`** |
| Deterministic FinQA Table Check (`$400.00` vs `$390.00`) | Unverified | Unverified | **`$10.00` Discrepancy Caught** |
