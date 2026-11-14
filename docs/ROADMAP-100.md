# 100-Item Frontier Roadmap (`ncm-benchmarks`)

Comprehensive 10-pillar engineering roadmap tracking completed milestones (`v2.0.0` and `v2.1.0`) alongside upcoming frontier research and enterprise deployment items.

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
10. [ ] Streaming `.docx` / `.xlsx` / OCR-backed scanned PDF ingestion adapters

## Pillar 2: Cryptographic Audit, Governance & Compliance (11-20)

11. [x] Deterministic SHA-256 input/output audit receipt generation per document
12. [x] HMAC-SHA256 keyed cryptographic audit receipt signing (`NCM_AUDIT_HMAC_KEY`) (`v2.1.0`)
13. [x] Cryptographic audit receipt tamper-verification API (`verify_audit_receipt`) (`v2.1.0`)
14. [x] Sovereign compliance report generator (`ncm-bench report`) supporting JSON & standalone HTML (`v2.1.0`)
15. [x] Fail-closed bearer token authentication and CORS origin allowlisting in FastAPI service
16. [x] Bandits (`-ll`) static security scanning integrated into CI/release gates
17. [ ] Ed25519 / X.509 detached signature support for multi-tenant enterprise KMS integration
18. [ ] Immutable append-only SQLite / PostgreSQL audit sink for `ncm-bench serve`
19. [ ] Field-level AES-GCM encryption-at-rest for buffered document queues
20. [ ] SOC2 / HIPAA / GDPR automated evidence bundle exporter with external verifier CLI

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
30. [ ] Multi-label hierarchical taxonomy routing with conformal calibration guarantees

## Pillar 4: Reasoning, Decoding & Verification (31-40)

31. [x] Parallel Minimum Bayes Risk (MBR) consensus decoding (`+1.82` ROUGE-L boost at $K=4$)
32. [x] Deterministic Sudoku clue-clamping + bounded 64-node constraint propagation repair (`repair_sudoku_board`)
33. [x] Multi-tier Sudoku evaluation harness (Easy `99.8%`, Medium `98.55%`, Hard `90.7%` on V100)
34. [x] Wilson 95% score confidence interval reporting on discrete accuracy suites
35. [x] Configurable `temperature` / `top_k` / `seed` stochastic sampling across hierarchy levels
36. [ ] Speculative coarse-to-fine tree verification with rejection sampling at fine levels
37. [ ] Grammar-constrained (JSON Schema / CFG) discrete logit masking at surface decoding levels
38. [ ] Formal SMT / Z3 solver hook for symbolic verification tasks beyond Sudoku
39. [ ] Beam-diverse MBR deduplication over coarse concept clusters
40. [ ] Entropy-triggered adaptive per-token early exit (`L_k -> stop`) without fixed `max_level`

## Pillar 5: Clean-Room Concept Training & Alignment (`ncm_train/`) (41-50)

41. [x] Clean-room Stage-1 Residual VQ-VAE pretraining & commitment loss smoke suite
42. [x] Clean-room Stage-2 Next-Concept Transformer cross-entropy smoke trainer
43. [x] Discrete Concept-Space GRPO / RLVR (`+0.0938` reward delta in 24 CPU steps)
44. [x] KL-guided Teacher-Student Concept Distillation (`9.398 -> 6.3534` loss in 20 CPU steps)
45. [x] Discrete Concept-Space Direct Preference Optimization (`DPOConfig`, `train_stage2_dpo`) (`v2.1.0`)
46. [x] Tensor device-alignment fix for CUDA / V100 GRPO rollouts (`e17fa93`)
47. [ ] Multi-GPU DistributedDataParallel (DDP) / FSDP Stage-1 and Stage-2 training recipes
48. [ ] Learned Codebook utilization monitor & dead-codebook EMA reset scheduler
49. [ ] LoRA / prefix-adapter fine-tuning inside discrete Stage-2 concept embedding layers
50. [ ] Verifiable unit-test / regex reward plugins for multi-domain Concept-RLVR

## Pillar 6: Hardware Acceleration & Runtime Kernels (51-60)

51. [x] Verified AMD Ryzen AI 9 HX 370 CPU INT8 + FP32 benchmark suite (`results/hx370-cpu/`)
52. [x] Verified NVIDIA Tesla V100-SXM2 16GB FP32 full census suite (`results/v100-sxm2/`)
53. [x] Batched XSum V100 sweep up to `bs=64` achieving `2,728.5 docs/min` (`0.022 s/doc` marginal)
54. [x] Full WMT14 En-De 3,003-sentence translation evaluation (`21.72` BLEU at `35.16 sent/sec`)
55. [ ] ONNX Runtime EP (`CUDAExecutionProvider` / `OpenVINO` / `QNN`) graph export
56. [ ] Weight-only `INT4` / `AWQ` / `GPTQ` quantized concept decoder kernels
57. [ ] Torch `compile` (`inductor` / CUDA Graphs) static-shape concept level execution
58. [ ] Apple Silicon (`MPS` / CoreML / MLX) native hardware receipt suite
59. [ ] AMD ROCm (`MI210` / `RDNA3`) + Ryzen NPU (`XDNA2`) hardware verification receipts
60. [ ] Custom Triton fused Residual-VQ lookup + cross-attention kernel

## Pillar 7: Service Architecture, SDK & Observability (61-70)

61. [x] Async FastAPI microservice (`ncm-bench serve`) with OpenAI `/v1/chat/completions` shim
62. [x] Streaming Server-Sent Events (`/progressive_summarize`) across Concept Levels `0..8`
63. [x] Synchronous & batch triage endpoints (`/triage`, `/batch_triage`) + `/triage/feedback` (`v2.1.0`)
64. [x] Prometheus `/metrics` histogram & counters (`latency_seconds`, requests, errors)
65. [x] Resilient Python SDK (`service.client.NCMClient`) with exponential backoff on `429`/`503`
66. [ ] Async Python `AsyncNCMClient` (`httpx`) + typed Pydantic response models
67. [ ] OpenTelemetry distributed trace propagation (`traceparent`, per-level spans)
68. [ ] gRPC high-throughput binary microservice transport for internal VPC deployments
69. [ ] Multi-worker dynamic batching queue coalescing concurrent `/triage` requests
70. [ ] Circuit-breaker escalation client for hybrid local-NCM -> frontier LLM routing

## Pillar 8: Interactive Explorer, TCO Modeling & Developer UX (71-80)

71. [x] Zero-backend GitHub Pages progressive concept visualizer (`docs/index.html`)
72. [x] Live V100/CPU hardware receipt matrix & interactive TCO volume calculator tab (`v2.1.0`)
73. [x] Strict CSS externalization (`no-inline-styles`) & WCAG/axe accessible UI controls (`v2.1.0`)
74. [x] Automated GitHub Pages deployment workflow (`.github/workflows/pages.yml`) (`v2.1.0`)
75. [ ] Interactive Sudoku constraint propagation step-by-step board visualizer in `docs/`
76. [ ] Side-by-side XSum / WMT14 concept-level diff highlighter in the browser UI
77. [ ] Live upload-and-redact sandbox (`HTML` / `CSV` / `EML`) connected to local `ncm-bench serve`
78. [ ] Grafana JSON dashboard template matching `/metrics` Prometheus series
79. [ ] Pre-configured Jupyter notebook gallery for Concept-DPO and Concept-RLVR experiments
80. [ ] TypeScript / Node.js thin SDK client for Edge & Cloudflare/Browser relay gateways

## Pillar 9: Packaging, Containers & CI/CD Release Engineering (81-90)

81. [x] Official PyPI distribution (`ncm-benchmarks`) with OIDC Trusted Publishing
82. [x] Full Python `3.10`–`3.12` matrix CI with `ruff` and `bandit` static gates
83. [x] Pre-release artifact/test integrity verifier (`ncm-bench verify`)
84. [x] Reproducible `Dockerfile` & container quickstart documentation
85. [ ] Multi-arch (`linux/amd64`, `linux/arm64`) minimal distroless CPU & CUDA Docker images
86. [ ] Production Kubernetes Helm chart with HPA autoscaling and `/healthz` probes
87. [ ] Air-gapped offline tarball bundler (`ncm-bench bundle-offline`) packaging wheel + weights
88. [ ] Automated hardware regression bot running smoke timing checks against golden baselines
89. [ ] Nix flake & DevContainer reproducible development environments
90. [ ] Windows / macOS signed standalone CLI binaries (`PyInstaller` / `shiv`)

## Pillar 10: Frontier Multi-Modal & Next-Gen Benchmarks (91-100)

91. [x] EPYC + V100 automated multi-dataset stress campaign runbook (`docs/HERMES-EPYC-PROMPT.md`)
92. [ ] Long-context GovReport / BillSum / PubMed summarization sovereign receipts
93. [ ] LegalBench / CUAD clause extraction & redaction accuracy benchmarks
94. [ ] FinQA / TAT-QA numerical table triage verification with arithmetic repair
95. [ ] Code-concept coarse-to-fine AST synthesis & repository triage prototypes
96. [ ] Cross-lingual zero-shot coarse concept transfer (`En -> De/Fr/Es/Ja`)
97. [ ] Synthetic preference dataset generator (`chosen` vs `rejected`) for Concept-DPO at scale
98. [ ] Formal energy-per-summary (`Wh / 1k docs`) telemetry harness across CPU/V100/RTX GPUs
99. [ ] Multimodal document-layout (PDF bounding box + text concept) joint VQ tokenizer
100. [ ] Community reproducible hardware leaderboard accepting signed PR receipt JSONs
