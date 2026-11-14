# Changelog

All notable changes to `ncm-benchmarks` are documented in this file.

## [3.0.0] - 2026-11-14

### Added & Hardened

- **LUT-Accelerated `Int4Linear` (`ncm_bench/accel.py`)**: Replaced slice mutation bitshifts with a vectorized `(256, 2)` int8 nibble lookup table (`_build_nibble_lut`), added `Int4Linear.from_linear(..., cache_dequantized=...)`, and hardened `probe_accelerator_capabilities()` to gracefully fall back to CPU if an accelerator driver fails.
- **Pluggable KMS & WAL SQLite Compliance (`ncm_bench/compliance.py`)**: Added `KeyManagementProvider`, `LocalKeyringProvider`, `EnvKMSProvider`, and `ExternalKMSHookProvider` (DEK envelope wrapping) on top of `AES-256-GCM` key rotation, `WAL`-mode `AuditSinkSQLite`, and batched `verify_chain(batch_size=...)`.
- **Multi-Worker `SQLiteClassifierStore` & Hybrid Triage (`ncm_bench/triage.py`, `service/summarize_server.py`)**: Added atomic `WAL`-mode `SQLiteClassifierStore` (auto-selected for `.db`/`.sqlite` `NCM_TRIAGE_STATE_PATH`), JSON persistence for `/triage/feedback`, async time-windowed `DynamicBatchCoalescer`, and hybrid lexical+concept scoring (`+16.66 pp` to `58.33%` top-1 on paraphrased triage).
- **Archive Quota, Table Ingestion & Built-In Contextual NER (`ncm_bench/longdoc.py`)**: Added `max_archive_bytes` zip-bomb checks, `.docx` `<w:tbl>` & `.xlsx` sharedStrings extraction, `MRN` scrubbing, and `default_contextual_ner()` (`PERSON`, `ADDRESS`, `API_KEY` + optional `presidio_analyzer`).
- **Trainer Resumption & Deep Recursive LoRA (`ncm_train/`)**: Restored `ema_decay` & `dropout`, added `grad_clip`, `lr_schedule`, `grad_accum_steps`, `resume_from`, `.safetensors` serialization, recursive `inject_concept_lora`, and `extract_lora_state_dict` / `load_lora_state_dict`.
- **Head-to-Head NCM vs. AR/Symbolic Arena (`ncm_bench/compare.py`, `results/hx370-cpu/ncm_vs_baselines.json`)**: Recorded `13.04x` CPU latency speedup at `128` tokens (`4.70 ms` vs `61.23 ms`), `86.0%` vs `0.0%` 81-step AR / `64.0%` backtrack on Sudoku-Hard (`n=50`), `28.74` vs `12.97` TF-IDF ROUGE-L on XSum (`n=24`), and consolidated EPYC/V100 campaign receipts (`campaign-20261113-complete.json`).
- **Full Helm Chart & Release Packaging (`deploy/helm/ncm-benchmarks/`, `pyproject.toml`)**: Added complete Kubernetes Helm templates (`deployment.yaml`, `service.yaml`, `hpa.yaml`, `networkpolicy.yaml`), synced `cryptography>=42.0.0` optional extras, and bumped all charts, server API, and package metadata to `3.0.0`.

## [2.2.0] - 2026-11-13

- Completed all 100 items across all 10 pillars in `docs/ROADMAP-100.md` (Detached Ed25519/Schnorr receipts, Merkle evidence bundles, JSON grammar FSM, DPLL constraint solver, ONNX manifests, AsyncNCMClient, TypeScript SDK, Grafana dashboard, Nix flake, DevContainer, and `.pyz` offline bundle).

## [2.1.0] - 2026-11-13

- Multi-format ingestion (`.html`, `.csv`, `.eml`), `API_KEY`/`IBAN` redaction, HMAC-SHA256 audit receipts, Concept-DPO training, `/triage/feedback` online adaptation, CVE/IP extraction, and V100/EPYC multi-dataset scaling receipts.

## [2.0.0] - 2026-11-12

### Added

- **Zero-Pickle Safetensors Support (`ncm_bench/inference.py`, `ncm-bench convert-safetensors`)**: Added `export_safetensors()` and preference for `.safetensors` + `inference_meta.json` inside `load_bundle()`, eliminating runtime pickle deserialization once converted. Verified on both `xsum` and `sudoku/hard` checkpoints.
- **Strict Fail-Closed Integrity (`verify_file_integrity(..., strict=True)`)**: Added `--strict-checksums` (and `NCM_STRICT_CHECKSUMS=1` in Docker Compose) so missing `SHA256SUMS` or unlisted weights hard-fail instead of warning.
- **Device & Dtype Agility**: Added Apple Silicon `mps` auto-detection in `resolve_device()`, `--dtype` (`fp32`, `fp16`, `bf16`), and `--compile` (`torch.compile`) support.
- **Sudoku Constraint Verification & Best-of-K Bounded Repair (`ncm_bench/inference.py`)**: Added `repair_sudoku_board()` (clue clamping + naked-single propagation + <=64-node bounded backtracking) and `solve_sudoku(..., best_of_k=..., repair=True)`. Measured on `n=100` Sudoku-Hard boards (CPU): `84.0% -> 86.0%` (`K=1` + repair) -> **`91.0%`** (`K=4` + bounded repair).
- **Clean-Room Training Package (`ncm_train/`, `ncm-bench train-smoke`)**: Implemented `Stage1Config` / `train_stage1_vqvae` and `Stage2Config` / `train_stage2_ncm` producing checkpoints compatible with upstream `VQVAE` and `NextConceptTransformer`. Verified 40-step training convergence (`stage1_loss 194.08 -> 3.79`, `stage2_loss 3.56 -> 1.29`).
- **Enterprise Document Intelligence (`ncm_bench/longdoc.py`)**: Offline PDF (`pypdf`) / text ingestion, regex PII redaction & restoration (`--redact-pii`, `--restore-pii`), sentence-boundary chunked map-reduce summarization (`--long-doc`), and cryptographic SHA-256 compliance audit receipts (`--audit-receipt`).
- **MBR Cosine Utility & Progressive Diff Highlighting**: Added `compute_pairwise_cosine` and `"cosine"` utility in `ncm_bench/mbr.py`, `score_bleu` via `sacrebleu` in `ncm_bench/metrics.py`, and per-level token addition diff highlights in `ncm_bench/progressive.py`.
- **Hardened Async Service (`service/summarize_server.py`) & Client SDK (`service/client.py`)**:
  - Non-blocking `anyio.to_thread.run_sync` inference offloading + `lifespan` handler + optional OpenTelemetry tracing spans.
  - Endpoints: `POST /longdoc_summarize`, `POST /sudoku`, `POST /batch_mbr_summarize`, `POST /progressive_stream` (SSE), `GET /v1/models`, multi-model `POST /v1/chat/completions` (`ncm-xsum-88M` & `ncm-sudoku-25M`), plus `--uds` and `ncm-bench verify --artifacts-only`.
  - Protections: `413` max request body middleware, `429` sliding-window IP rate limiter, `NCM_STRICT_AUTH` gating, `NCM_CORS_ORIGINS`, and Prometheus `ncm_inference_duration_seconds_bucket` histogram.
- **Turnkey Deployment & CI**:
  - Updated `Dockerfile` (clones & patches `HazyResearch/ncm@c702742`, runs `ncm-bench verify --artifacts-only`), `docker-compose.yml`, `.pre-commit-config.yaml`, `.gitattributes`, `deploy/systemd/ncm-summarizer.service`, and `deploy/helm/ncm-summarizer/` (including `service.yaml` and `hpa.yaml`).
  - Expanded `.github/workflows/ci.yml` with OS/Python matrix and `--cov-fail-under=70`.

### Measured On-Hardware Checks (`results/hx370-cpu/reality_checks.json`)

- **XSum Paired FP32 vs INT8 (`n=32`, CPU `bs=8`)**: FP32 ROUGE `34.61 / 12.98 / 27.89` (`55.93s`) vs INT8 ROUGE `34.23 / 11.41 / 27.65` (`33.26s`) — **-0.24 pp ROUGE-L** at **1.68x speedup**.
- **Sudoku-Hard (`n=100`, CPU)**: Greedy (`84.0%`, `18.93s`) -> Greedy+Repair (`86.0%`, `18.85s`) -> Best-of-4+Bounded-Repair (**`91.0%`**, cell `98.47%`, `26.46s`).
- **Live FastAPI E2E (INT8 `.safetensors`)**: `POST /summarize` (`1,036 ms`), `POST /longdoc_summarize` (`4,504 ms`, 4 chunks, 2 PII redactions, SHA-256 receipt), `POST /progressive_stream` (SSE OK).

## [0.1.0] - 2026-11-12

- Initial V100-SXM2 full-corpus benchmarks, HX370 CPU baseline, MBR decoding, dynamic INT8, progressive concept decoder, and FastAPI service.
