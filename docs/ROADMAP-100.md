# 100-Item Frontier Roadmap (`ncm-benchmarks`) — 100 / 100 Complete (`v2.2.0`)

Comprehensive 10-pillar engineering roadmap tracking all 100 completed capabilities across `v2.0.0`, `v2.1.0`, and `v2.2.0`.

---

## Pillar 1: Core Sovereign Engine & Multi-Format Ingestion (1-10)

1. [x] Pure PyTorch clean-room Next-Concept Modeling (NCM) & Residual VQ-VAE implementation (`ncm_bench/`)
2. [x] Zero-pickle `.safetensors` serialization & weight integrity SHA-256 verification (`artifacts/SHA256SUMS`)
3. [x] Dynamic `int8` PyTorch quantization on x86/ARM CPUs (`1.68x` speedup at `-0.24 pp` ROUGE-L)
4. [x] Air-gapped `HF_HUB_OFFLINE=1` enforcement mode with zero outbound telemetry
5. [x] Sentence-aware long-document map-reduce chunking and hierarchical merge (`ncm_bench/longdoc.py`)
6. [x] PDF text extraction (`pypdf` / `pdftotext` fallback) for enterprise document ingestion
7. [x] Native `.html` / `.htm`, `.csv`, and `.eml` (RFC 822 / multipart email) ingestion (`v2.1.0`)
8. [x] Client-side PII redaction and placeholder restoration (`EMAIL`, `SSN`, `CARD`, `PHONE`, `IPV4`)
9. [x] Secret & financial account scrubbing (`API_KEY`, `IBAN`) prior to model inference (`v2.1.0`)
10. [x] `.docx` / `.xlsx` XML archive extraction & scanned-PDF `ocr_fallback` hook (`ncm_bench/longdoc.py`, `v2.2.0`)

## Pillar 2: Cryptographic Audit, Governance & Compliance (11-20)

11. [x] Deterministic SHA-256 input/output audit receipt generation per document
12. [x] HMAC-SHA256 keyed cryptographic audit receipt signing (`NCM_AUDIT_HMAC_KEY`) (`v2.1.0`)
13. [x] Cryptographic audit receipt tamper-verification API (`verify_audit_receipt`) (`v2.1.0`)
14. [x] Sovereign compliance report generator (`ncm-bench report`) supporting JSON & standalone HTML (`v2.1.0`)
15. [x] Fail-closed bearer token authentication and CORS origin allowlisting in FastAPI service
16. [x] Bandits (`-ll`) static security scanning integrated into CI/release gates
17. [x] Detached asymmetric receipt signatures (`Ed25519` / `Schnorr-2048` in `ncm_bench/compliance.py`, `v2.2.0`)
18. [x] Immutable append-only hash-chained SQLite audit sink (`AuditSinkSQLite`, `v2.2.0`)
19. [x] Field-level authenticated queue encryption-at-rest (`encrypt_field_envelope` / `decrypt_field_envelope`, `v2.2.0`)
20. [x] SOC2 / HIPAA / GDPR Merkle evidence bundle exporter & verifier CLI (`export_evidence_bundle`, `ncm-bench validate-receipt`, `v2.2.0`)

## Pillar 3: Structured Triage, Early-Exit & Online Learning (21-30)

21. [x] Coarse Level `0..2` early-exit decoding achieving `3.38x` CPU speedup (`356.02 ms` vs `1205.13 ms/doc`)
22. [x] `ConceptIndex` deterministic SHA-256 / concept-hash semantic deduplication
23. [x] Bounded LRU coarse signature cache (`512` entries, `clear_coarse_cache`) (`v2.1.0`)
24. [x] Zero-shot + centroid `ConceptClassifier` with escalation confidence thresholding
25. [x] Online incremental class adaptation via `ConceptClassifier.partial_fit()` (`v2.1.0`)
26. [x] Active-learning HTTP feedback endpoint (`POST /triage/feedback`) and `NCMClient.triage_feedback()` (`v2.1.0`)
27. [x] Regex structured entity extraction (`money`, `dates`, `emails`, `urls`)
28. [x] Security telemetry entity extraction (`cves`, `ips`) with character `entity_spans` provenance (`v2.1.0`)
29. [x] Schema validation and Best-of-K repair with `required_fields` and `custom_enums` enforcement (`v2.1.0`)
30. [x] Multi-label hierarchical taxonomy routing with split conformal calibration (`calibrate_conformal`, `predict_conformal_set`, `v2.2.0`)

## Pillar 4: Reasoning, Decoding & Verification (31-40)

31. [x] Parallel Minimum Bayes Risk (MBR) consensus decoding (`+1.82` ROUGE-L boost at $K=4$)
32. [x] Deterministic Sudoku clue-clamping + bounded 64-node constraint propagation repair (`repair_sudoku_board`)
33. [x] Multi-tier Sudoku evaluation harness (Easy `99.8%`, Medium `98.55%`, Hard `90.7%` on V100)
34. [x] Wilson 95% score confidence interval reporting on discrete accuracy suites
35. [x] Configurable `temperature` / `top_k` / `seed` stochastic sampling across hierarchy levels
36. [x] Speculative coarse-to-fine rejection verification (`speculative_coarse_to_fine_verify`, `ncm_bench/reasoning.py`, `v2.2.0`)
37. [x] Grammar-constrained JSON / allowed-token logit masking (`apply_json_grammar_mask`, `ncm_bench/reasoning.py`, `v2.2.0`)
38. [x] Formal symbolic DPLL SAT + Sudoku constraint verifier (`solve_symbolic_constraints`, `ncm_bench/reasoning.py`, `v2.2.0`)
39. [x] Beam-diverse MBR cluster deduplication (`select_diverse_mbr_centroid`, `ncm_bench/mbr.py`, `v2.2.0`)
40. [x] Entropy-triggered adaptive per-level early exit (`adaptive_entropy_early_exit`, `ncm_bench/reasoning.py`, `v2.2.0`)

## Pillar 5: Clean-Room Concept Training & Alignment (`ncm_train/`) (41-50)

41. [x] Clean-room Stage-1 Residual VQ-VAE pretraining & commitment loss smoke suite
42. [x] Clean-room Stage-2 Next-Concept Transformer cross-entropy smoke trainer
43. [x] Discrete Concept-Space GRPO / RLVR (`+0.0938` reward delta in 24 CPU steps)
44. [x] KL-guided Teacher-Student Concept Distillation (`9.398 -> 6.3534` loss in 20 CPU steps)
45. [x] Discrete Concept-Space Direct Preference Optimization (`DPOConfig`, `train_stage2_dpo`) (`v2.1.0`)
46. [x] Tensor device-alignment fix for CUDA / V100 GRPO rollouts (`e17fa93`)
47. [x] Multi-GPU DDP / FSDP distributed training wrapper (`DDPConfig`, `wrap_model_distributed`, `ncm_train/adapters.py`, `v2.2.0`)
48. [x] Learned Codebook utilization monitor & dead-codebook EMA reset (`CodebookMonitor`, `ncm_train/adapters.py`, `v2.2.0`)
49. [x] Low-rank Stage-2 adapter injection (`ConceptLoRALinear`, `inject_concept_lora`, `ncm_train/adapters.py`, `v2.2.0`)
50. [x] Verifiable regex & unit-test callable RLVR reward plugins (`RegexRewardPlugin`, `CallableRewardPlugin`, `v2.2.0`)

## Pillar 6: Hardware Acceleration & Runtime Kernels (51-60)

51. [x] Verified AMD Ryzen AI 9 HX 370 CPU INT8 + FP32 benchmark suite (`results/hx370-cpu/`)
52. [x] Verified NVIDIA Tesla V100-SXM2 16GB FP32 full census suite (`results/v100-sxm2/`)
53. [x] Batched XSum V100 sweep up to `bs=64` achieving `2,728.5 docs/min` (`0.022 s/doc` marginal)
54. [x] Full WMT14 En-De 3,003-sentence translation evaluation (`21.72` BLEU at `35.16 sent/sec`)
55. [x] ONNX ExecutionProvider compatibility manifest & graph exporter (`export_ncm_onnx_manifest`, `ncm_bench/accel.py`, `v2.2.0`)
56. [x] Weight-only `INT4` nibble groupwise packing/dequantization (`quantize_weights_int4_groupwise`, `ncm_bench/accel.py`, `v2.2.0`)
57. [x] Static-shape `torch.compile` concept decoder wrapper (`compile_static_concept_decoder`, `ncm_bench/accel.py`, `v2.2.0`)
58. [x] Cross-accelerator Apple MPS / CUDA / CPU probe & micro-benchmark (`probe_accelerator_capabilities`, `ncm-bench hardware-probe`, `v2.2.0`)
59. [x] AMD ROCm / OpenVINO / CPU capability reporting in `ncm-bench hardware-probe` (`v2.2.0`)
60. [x] Fused Residual-VQ squared L2 lookup primitive via `torch.addmm` (`fused_residual_vq_lookup`, `ncm_bench/accel.py`, `v2.2.0`)

## Pillar 7: Service Architecture, SDK & Observability (61-70)

61. [x] Async FastAPI microservice (`ncm-bench serve`) with OpenAI `/v1/chat/completions` shim
62. [x] Streaming Server-Sent Events (`/progressive_summarize`) across Concept Levels `0..8`
63. [x] Synchronous & batch triage endpoints (`/triage`, `/batch_triage`) + `/triage/feedback` (`v2.1.0`)
64. [x] Prometheus `/metrics` histogram & counters (`latency_seconds`, requests, errors)
65. [x] Resilient Python SDK (`service.client.NCMClient`) with exponential backoff on `429`/`503`
66. [x] Async Python `AsyncNCMClient` (`httpx`) + Pydantic response models (`service/transport.py`, `v2.2.0`)
67. [x] Distributed W3C `traceparent` & `X-Trace-ID` propagation middleware (`service/summarize_server.py`, `v2.2.0`)
68. [x] Length-prefixed compressed binary VPC RPC framing (`encode_binary_frame` / `decode_binary_frame`, `v2.2.0`)
69. [x] Dynamic micro-batch request coalescer (`DynamicBatchCoalescer`, `service/transport.py`, `v2.2.0`)
70. [x] Circuit-breaker escalation client for hybrid local-NCM -> frontier routing (`HybridCircuitBreakerClient`, `v2.2.0`)

## Pillar 8: Interactive Explorer, TCO Modeling & Developer UX (71-80)

71. [x] Zero-backend GitHub Pages progressive concept visualizer (`docs/index.html`)
72. [x] Live V100/CPU hardware receipt matrix & interactive TCO volume calculator tab (`v2.1.0`)
73. [x] Strict CSS externalization (`no-inline-styles`) & WCAG/axe accessible UI controls (`v2.1.0`)
74. [x] Automated GitHub Pages deployment workflow (`.github/workflows/pages.yml`) (`v2.1.0`)
75. [x] Interactive Sudoku constraint propagation step-by-step stepper in [`docs/index.html`](index.html) (`v2.2.0`)
76. [x] Concept-level progression diff view & sandbox output staging in [`docs/index.html`](index.html) (`v2.2.0`)
77. [x] Client-side PII, API Key & IBAN scrubbing sandbox in [`docs/index.html`](index.html) (`v2.2.0`)
78. [x] Grafana observability dashboard template ([`docs/grafana-dashboard.json`](grafana-dashboard.json), `v2.2.0`)
79. [x] Concept-DPO & RLVR walkthrough notebook ([`docs/notebooks/concept_dpo_rlvr_walkthrough.ipynb`](notebooks/concept_dpo_rlvr_walkthrough.ipynb), `v2.2.0`)
80. [x] TypeScript SDK client ([`sdk/typescript/ncm_client.ts`](../sdk/typescript/ncm_client.ts), `v2.2.0`)

## Pillar 9: Packaging, Containers & CI/CD Release Engineering (81-90)

81. [x] Official PyPI distribution (`ncm-benchmarks`) with OIDC Trusted Publishing
82. [x] Full Python `3.10`–`3.12` matrix CI with `ruff` and `bandit` static gates
83. [x] Pre-release artifact/test integrity verifier (`ncm-bench verify`)
84. [x] Reproducible `Dockerfile` & container quickstart documentation
85. [x] Multi-arch (`TARGETPLATFORM` `amd64`/`arm64`) `Dockerfile` support (`v2.2.0`)
86. [x] Production Kubernetes Helm chart (`deploy/helm/ncm-benchmarks/Chart.yaml` & `values.yaml`, `v2.2.0`)
87. [x] Air-gapped offline tarball bundler (`ncm-bench bundle-offline`, `v2.2.0`)
88. [x] Hardware probe & regression baseline command (`ncm-bench hardware-probe`, `v2.2.0`)
89. [x] Nix flake (`flake.nix`) & VS Code DevContainer (`.devcontainer/devcontainer.json`) (`v2.2.0`)
90. [x] Standalone cross-platform executable `.pyz` archive builder inside `ncm-bench bundle-offline` (`v2.2.0`)

## Pillar 10: Frontier Multi-Modal & Next-Gen Benchmarks (91-100)

91. [x] EPYC + V100 automated multi-dataset stress campaign runbook (`docs/HERMES-EPYC-PROMPT.md`)
92. [x] Long-context summarization harness (`run_longcontext_summarization_harness`, `ncm_bench/frontier.py`, `v2.2.0`)
93. [x] Legal clause redaction & exact restoration benchmark (`evaluate_legal_clause_redaction`, `ncm_bench/frontier.py`, `v2.2.0`)
94. [x] FinQA / TAT-QA CSV table arithmetic verification & delta repair (`verify_finqa_table_arithmetic`, `ncm_bench/frontier.py`, `v2.2.0`)
95. [x] Python AST code-concept histogram extractor (`extract_python_ast_concepts`, `ncm_bench/frontier.py`, `v2.2.0`)
96. [x] Cross-lingual coarse concept alignment scorer (`score_cross_lingual_concept_alignment`, `ncm_bench/frontier.py`, `v2.2.0`)
97. [x] Synthetic Concept-DPO preference pair generator (`generate_synthetic_dpo_preferences`, `ncm_bench/frontier.py`, `v2.2.0`)
98. [x] Sovereign energy telemetry estimator (`estimate_energy_telemetry`, `ncm_bench/frontier.py`, `v2.2.0`)
99. [x] Multimodal 2D layout bounding-box + token joint concept encoder (`encode_layout_text_joint_concepts`, `ncm_bench/frontier.py`, `v2.2.0`)
100. [x] Community hardware leaderboard receipt schema validator (`validate_receipt_submission`, `ncm-bench validate-receipt`, `v2.2.0`)
