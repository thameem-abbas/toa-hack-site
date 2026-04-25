---
title: Activity Log
type: log
---

# Activity Log

## 2026-04-25 gap-fill | Critical Missing Concept Pages

- **Task:** Created two missing concept pages heavily referenced across the wiki
- **Pages created:** [[Chunked Prefill]], [[Continuous Batching]]
- **Context:** These concepts were listed as stubs in `concepts/index.md` but had no actual pages despite 30+ cross-references each
- **Chunked Prefill:** Breaking long prompt prefill into smaller chunks interleaved with decode steps; enabled by default in V1; solves prefill-decode interference for better ITL fairness; token budget allocation via `--max-num-batched-tokens`; integrates naturally with prefix caching and unified scheduler; 0-5% throughput impact (negligible), 2-5× TTFT/ITL variance reduction
- **Continuous Batching:** Iteration-level scheduling where requests added/removed from batch after each forward pass (vs. batch-level in static batching); 2-10× throughput gain over static batching (depending on output length variance); 90-100% GPU utilization vs. 60-70%; enabled by PagedAttention's efficient memory management; Orca paper (OSDI 2022) introduced iteration-level scheduling; V1 unified scheduler treats prefill/decode identically as token budget allocation
- **Key insight:** Both concepts are foundational to vLLM's performance advantages — continuous batching enables high throughput via dynamic request scheduling, while chunked prefill extends iteration-level scheduling to the prefill phase for latency fairness. Together they enable concurrent prefill+decode batching with bounded ITL variance.

## 2026-04-25 ingest | vLLM Observability & Training (Batch 13 of 14)
- Sources: `.raw/articles/vllm-metrics-2026-04-25.md`, `.raw/articles/vllm-reasoning-outputs-2026-04-25.md`, `.raw/articles/vllm-sleep-mode-2026-04-25.md`, `.raw/articles/vllm-rlhf-2026-04-25.md`, `.raw/articles/vllm-async-rl-2026-04-25.md`
- Summary: [[vllm-observability-training]]
- Pages created: [[vLLM Metrics]], [[Reasoning Models]], [[Sleep Mode]], [[RLHF with vLLM]], [[vllm-observability-training]] (source)
- Pages updated: [[architectures/index|Architectures]], [[concepts/index|Concepts]], [[techniques/index|Techniques]]
- Key insight: vLLM provides production-ready observability via Prometheus metrics (server-level Gauges/Counters for state, request-level Histograms for SLOs, KV cache residency tracking, event timeline QUEUED→SCHEDULED→NEW_TOKENS) and training integration for RLHF (sleep mode frees 90% GPU memory for GPU sharing between inference and training, async RL pipelining overlaps generation and training via pause/resume API, weight synchronization via NCCL/IPC backends, integration with 11+ RL frameworks); reasoning models (DeepSeek-R1, Qwen3, etc.) produce internal thinking tokens with optional thinking budget control; fine-grained wake (weights or KV cache separately) avoids OOM during weight updates.

## 2026-04-25 ingest | vLLM Deployment & Benchmarking (Batch 14 of 14, FINAL)
- Sources: `.raw/articles/vllm-openai-server-2026-04-25.md`, `.raw/articles/vllm-benchmark-cli-2026-04-25.md`, `.raw/articles/vllm-optimization-config-2026-04-25.md`, `.raw/articles/vllm-llm-d-2026-04-25.md`, `.raw/articles/vllm-conserving-memory-2026-04-25.md`
- Summary: [[vllm-deployment]]
- Pages created: [[OpenAI-Compatible Server]], [[vLLM Benchmarking]], [[llm-d]], [[vllm-deployment]] (source)
- Pages updated: [[architectures/index|Architectures]], [[benchmarks/index|Benchmarks]], [[tools/index|Tools]]
- Key insight: vLLM production deployment combines (1) OpenAI-compatible HTTP server (drop-in API compatibility + 10+ vLLM-specific endpoints including /generative_scoring, /rerank, /v1/realtime WebSocket ASR), (2) built-in CLI benchmarking (vllm bench serve/throughput/mm-processor with 15+ datasets, load pattern control via request-rate/burstiness/max-concurrency, visualization via timeline/dataset-stats), (3) optimization strategies (chunked prefill by default, -O0 to -O3 levels, NUMA binding, batch-level DP for multimodal encoders, CPU resource planning 2+N minimum physical cores), (4) llm-d Kubernetes-native orchestration (disaggregated prefill/decode pools, per-pool autoscaling, KV cache routing, KServe integration), (5) memory conservation (TP/quantization/context limits/CUDA graph reduction/multimodal input limits with configurable size hints); maximum throughput load pattern (--request-rate=inf --max-concurrency=<limit>) most common for production capacity planning.

## 2026-04-25 ingest | vLLM Internal Architectures (Batch 12 of 14)
- Sources: `.raw/articles/vllm-attention-backends-2026-04-25.md`, `.raw/articles/vllm-model-runner-v2-2026-04-25.md`, `.raw/articles/vllm-logits-processors-2026-04-25.md`, `.raw/articles/vllm-plugin-system-2026-04-25.md`
- Summary: [[vllm-internals]]
- Pages created: [[Attention Backends]], [[Model Runner V2]], [[Logits Processors]], [[Plugin System]], [[vllm-internals]] (source)
- Pages updated: [[architectures/index|Architectures]]
- Key insight: vLLM's internal architecture provides pluggable attention backends (13+ standard, 15+ MLA) with automatic selection via priority-ordered lists; Model Runner V2 redesigns execution from first principles with persistent batch decoupling, async-first design, and explicit CUDA graph management; logits processors operate at batch granularity with stateful BatchUpdate synchronization; plugin system enables OOT hardware support (Ascend NPU, Intel Gaudi, AWS Neuron) via standard Python entry_points with five-component platform plugins (Platform, Worker, AttentionBackend, DeviceCommunicator, CustomOps).

## 2026-04-25 ingest | vLLM Parallelism Strategies (Batch 10 of 14)
- Sources: `.raw/articles/vllm-parallelism-scaling-2026-04-25.md`, `.raw/articles/vllm-data-parallel-deployment-2026-04-25.md`, `.raw/articles/vllm-context-parallel-deployment-2026-04-25.md`, `.raw/articles/vllm-multiprocessing-design-2026-04-25.md`
- Summary: [[vllm-parallelism]]
- Pages created: [[Tensor Parallelism]], [[Pipeline Parallelism]], [[Data Parallelism]], [[Context Parallelism]], [[vllm-parallelism]] (source)
- Pages updated: [[concepts/index|Concepts]]
- Key insight: vLLM supports four parallelism strategies with different memory/communication tradeoffs: (1) Tensor Parallelism shards individual layers via Megatron-LM algorithm (all-reduce per layer, TP=GPUs per node, 5-15% overhead on InfiniBand); (2) Pipeline Parallelism splits layers sequentially across nodes (micro-batching, 10-20% bubble overhead, lower communication than TP for PCIe/Ethernet); (3) Data Parallelism replicates model across GPU groups (ZMQ load balancing, 90-95% linear throughput scaling, DP coordinator for MoE synchronization); (4) Context Parallelism shards long sequences (prefill CP via ring attention under development, decode CP shards KV cache along sequence dimension to eliminate duplication when TP_size > num_kv_heads, 1-10% overhead); combine strategies as DP×TP×PP for multi-node large models, add decode CP to reduce KV cache duplication on MLA/GQA models (DeepSeek-R1, Kimi-K2 with 1 KV head benefit most).

## 2026-04-25 ingest | vLLM Serving Features: LoRA, Structured Outputs (Batch 11 of 14)
- Sources: `.raw/articles/vllm-lora-2026-04-25.md`, `.raw/articles/vllm-lora-resolver-plugins-2026-04-25.md`, `.raw/articles/vllm-structured-outputs-2026-04-25.md`, `.raw/articles/vllm-tool-calling-2026-04-25.md`
- Summary: [[vllm-serving-features]]
- Pages created: [[LoRA]], [[Structured Outputs]], [[vllm-serving-features]] (source)
- Pages updated: [[techniques/index|Techniques]]
- Key insight: vLLM enables production multi-tenant serving via (1) LoRA: per-request adapter selection with multi-LoRA batching using Punica/BGMV kernels, dynamic loading via API endpoints or resolver plugins (filesystem/S3/HF Hub), ~5-10% compute overhead for rank 64; (2) Structured Outputs: logits masking via compiled FSM (xgrammar/guidance backends) to guarantee JSON/regex/grammar conformance, 1-10s first-request compilation overhead with cached reuse, 5-15% per-token slowdown; both compose efficiently with quantization/prefix caching/speculative decoding and enable sophisticated agent workflows (LoRA per tenant + structured outputs for API validity + tool calling for agentic execution).

## 2026-04-25 ingest | vLLM Quantization (Batch 7 of 14)
- Sources: `.raw/articles/vllm-quantization-readme-2026-04-25.md`, `.raw/articles/vllm-fp8-2026-04-25.md`, `.raw/articles/vllm-int8-2026-04-25.md`, `.raw/articles/vllm-int4-2026-04-25.md`, `.raw/articles/vllm-quantized-kvcache-2026-04-25.md`
- Summary: [[vllm-quantization]]
- Pages created: [[Quantization]], [[FP8 Quantization]], [[INT8 W8A8]], [[INT4 W4A16]], [[Quantized KV Cache]], [[vllm-quantization]] (source)
- Pages updated: [[concepts/index|Concepts]], [[techniques/index|Techniques]]
- Key insight: vLLM supports 13+ quantization methods with hardware-specific support matrices (FP8 on Ada/Hopper/MI300, INT8 on Turing/Ampere, INT4 on Ampere+); FP8 W8A8 offers best accuracy-performance tradeoff (2× memory, 1.6× throughput) via static per-channel weights + dynamic per-token activations; INT8 requires SmoothQuant calibration (512+ samples); INT4 uses GPTQ with group-wise quantization (group_size=128); KV cache quantization to FP8 provides orthogonal ~50% memory savings and can be combined with weight quantization for compound 2.5-4.5× total reduction.

## 2026-04-25 ingest | vLLM Speculative Decoding (Batch 9 of 14)
- Sources: `.raw/articles/vllm-speculative-decoding-readme-2026-04-25.md`, `.raw/articles/vllm-speculative-decoding-eagle-2026-04-25.md`, `.raw/articles/vllm-speculative-decoding-draft-model-2026-04-25.md`, `.raw/articles/vllm-speculative-decoding-mtp-2026-04-25.md`, `.raw/articles/vllm-speculative-decoding-ngram-2026-04-25.md`
- Summary: [[vllm-speculative-decoding]]
- Pages created: [[Speculative Decoding]], [[EAGLE]], [[Draft Model Speculation]], [[Multi-Token Prediction]], [[N-gram Speculation]], [[vllm-speculative-decoding]] (source)
- Pages updated: [[concepts/index|Concepts]], [[techniques/index|Techniques]]
- Key insight: vLLM supports seven speculative decoding methods with different accuracy/cost tradeoffs: model-based methods (EAGLE, MTP, draft model, PARD, MLP) provide 2-3× latency reduction at low QPS but consume GPU capacity, while heuristic methods (n-gram, suffix) provide 1.2-1.8× speedup with zero overhead, making them peak-friendly for high-throughput workloads; method selection depends on QPS profile (low QPS → EAGLE/MTP for max speedup, high QPS → n-gram for peak-friendly gains).

## 2026-04-25 ingest | vLLM Quantization Formats: AWQ, GPTQ, GGUF (Batch 8 of 14)
- Sources: `.raw/articles/vllm-auto-awq-2026-04-25.md`, `.raw/articles/vllm-gptqmodel-2026-04-25.md`, `.raw/articles/vllm-gguf-2026-04-25.md`, `.raw/articles/vllm-quark-2026-04-25.md`, `.raw/articles/vllm-modelopt-2026-04-25.md`
- Summary: [[vllm-quantization-formats]]
- Pages created: [[AWQ]], [[GPTQ]], [[GGUF]], [[llm-compressor]], [[vllm-quantization-formats]] (source)
- Pages updated: [[techniques/index|Techniques]], [[tools/index|Tools]]
- Key insight: vLLM supports five quantization ecosystems (AWQ, GPTQ, GGUF, AMD Quark, NVIDIA ModelOpt) with unified execution via Marlin/Machete backends; AutoAWQ and AutoGPTQ are deprecated in favor of llm-compressor, the vLLM-native unified toolkit; GGUF is experimental but enables cross-platform llama.cpp workflows; AMD Quark provides MXFP4/MXFP6 for MI-series GPUs; NVIDIA ModelOpt supports VLMs and QAT.

## 2026-04-25 ingest | vLLM MoE Architecture & Expert Parallelism (Batch 5 of 14)
- Sources: `.raw/articles/vllm-fused-moe-modular-kernel-2026-04-25.md`, `.raw/articles/vllm-moe-kernel-features-2026-04-25.md`, `.raw/articles/vllm-dbo-2026-04-25.md`, `.raw/articles/vllm-expert-parallel-deployment-2026-04-25.md`
- Summary: [[vllm-moe-design]]
- Pages created: [[Mixture of Experts]], [[Expert Parallelism]], [[FusedMoE Modular Kernel]], [[Dual Batch Overlap]], [[vllm-moe-design]] (source)
- Pages updated: [[concepts/index|Concepts]], [[architectures/index|Architectures]], [[techniques/index|Techniques]]
- Key insight: vLLM's modular MoE architecture splits inference into three pluggable components (Prepare/Finalize for all-to-all, Experts for computation, Weight/Reduce for aggregation), with five all-to-all backends (naive, deepep_high_throughput, deepep_low_latency, flashinfer_nvlink_one_sided/two_sided) and 10+ expert kernels (Triton, DeepGemm, CUTLASS, FlashInfer, Marlin); Expert Parallelism shards experts across GPUs (EP_SIZE = TP_SIZE × DP_SIZE) with dynamic load balancing (EPLB: 20-40% throughput gain) and Dual Batch Overlap overlaps all-to-all communication with compute via microbatching (1.3-1.8× decode speedup).

## 2026-04-25 ingest | vLLM Disaggregated Serving (Batch 6 of 14)
- Sources: `.raw/articles/vllm-disagg-prefill-2026-04-25.md`, `.raw/articles/vllm-p2p-nccl-connector-2026-04-25.md`, `.raw/articles/vllm-nixl-connector-2026-04-25.md`, `.raw/articles/vllm-disagg-encoder-2026-04-25.md`
- Summary: [[vllm-disagg-serving]]
- Pages created: [[Disaggregated Serving]], [[KV Cache Transfer]], [[vllm-disagg-serving]] (source)
- Pages updated: [[architectures/index|Architectures]]
- Key insight: vLLM's disaggregated serving separates prefill (compute-bound, TTFT-optimized) and decode (memory-bound, ITL-optimized) onto different GPU pools, with six connector types (P2pNcclConnector, NixlConnector, LMCache, Mooncake, FlexKV, OffloadingConnector) for KV cache transfer; xPyD deployments scale independently without throughput improvement, while disaggregated encoder extends the pattern to multimodal models (E→P→D).

## 2026-04-25 ingest | vLLM Kernel Fusions & CustomOp System (Batch 4 of 14)
- Sources: `.raw/articles/vllm-fusions-2026-04-25.md`, `.raw/articles/vllm-custom-op-2026-04-25.md`, `.raw/articles/vllm-debug-compile-2026-04-25.md`
- Summary: [[vllm-fusions-design]]
- Pages created: [[Kernel Fusions]], [[CustomOp System]], [[vllm-fusions-design]] (source)
- Pages updated: [[techniques/index|Techniques]], [[architectures/index|Architectures]]
- Key insight: vLLM implements 12+ kernel fusions via custom Inductor passes (AllReduce+RMSNorm 5-20% speedup, AsyncTP 7-10% speedup) with hardware-specific enablement (SM100/SM90 for AllReduce, ROCm/AITER for RoPE+KV), while CustomOp system enables OOT hardware plugins to register platform-optimized kernels (Ascend NPU, Intel Gaudi, AWS Neuron) without modifying vLLM core.

## 2026-04-25 ingest | vLLM Compilation & CUDA Graphs (Batch 3 of 14)
- Sources: `.raw/articles/vllm-torch-compile-2026-04-25.md`, `.raw/articles/vllm-cuda-graphs-2026-04-25.md`, `.raw/articles/vllm-optimization-levels-2026-04-25.md`
- Summary: [[vllm-compilation-design]]
- Pages created: [[torch.compile Integration]], [[CUDA Graphs]], [[Optimization Levels]], [[vllm-compilation-design]] (source)
- Pages updated: [[techniques/index|Techniques]], [[concepts/index|Concepts]], [[architectures/index|Architectures]]
- Key insight: vLLM V1 uses torch.compile for kernel generation and graph splitting, with five CUDA graph modes (NONE, PIECEWISE, FULL, FULL_DECODE_ONLY, FULL_AND_PIECEWISE) dispatched dynamically based on batch composition and attention backend compatibility.

## 2026-04-25 ingest | vLLM KV Cache & Prefix Caching (Batch 2 of 14)
- Sources: `.raw/articles/vllm-prefix-caching-2026-04-25.md`, `.raw/articles/vllm-hybrid-kv-cache-2026-04-25.md`, `.raw/articles/vllm-apc-feature-2026-04-25.md`
- Summary: [[vllm-kv-cache-design]]
- Pages created: [[KV Cache]], [[Prefix Caching]], [[Hybrid KV Cache Manager]], [[vllm-kv-cache-design]] (source)
- Pages updated: [[architectures/index|Architectures]]
- Key insight: vLLM's hash-based prefix caching enables 2-10x TTFT reduction for shared-prefix workloads, while hybrid KV cache manager extends block-based allocation to models with mixed attention types (full, sliding window, Mamba) using unified page size and per-layer allocation.

## 2026-04-25 ingest | vLLM Core Architecture
- Sources: `.raw/articles/vllm-arch-overview-2026-04-25.md`, `.raw/articles/vllm-paged-attention-2026-04-25.md`, `.raw/articles/vllm-v1-guide-2026-04-25.md`
- Summary: [[vllm-arch-overview]]
- Pages created: [[vLLM Engine]], [[PagedAttention]], [[V1 Architecture]], [[vLLM]], [[Efficient Memory Management for Large Language Model Serving with PagedAttention]]
- Pages updated: [[architectures/index|Architectures]]
- Key insight: vLLM V1's multi-process architecture separates API serving, scheduling, and GPU execution into distinct processes, with a unified scheduler that treats prompt and output tokens identically.

## 2026-04-25 ingest | vLLM NVTX PyTorch Hooks
- Source: `.raw/articles/nvtx-pytorch-hooks-2026-04-25.md`
- Summary: [[nvtx-pytorch-hooks]]
- Pages created: [[NVTX Profiling]], [[nvtx-pytorch-hooks]] (source)
- Pages updated: [[concepts/index|Concepts]], [[tools/index|Tools]], [[index]]
- Key insight: vLLM has built-in NVTX hook registration for layer-level GPU profiling via `vllm.utils.nvtx_pytorch_hooks`.

---

| Date | Action | Source | Pages Created/Updated |
|------|--------|--------|-----------------------|
| 2026-04-25 | Wiki scaffolded | — | All index pages |
