---
title: Hot Cache
type: meta
updated: 2026-04-25
---

# Hot Cache

Last ingest context for fast session resumption.

## Latest Activity: Gap-Fill Missing Concept Pages

- **Task:** Created critical missing pages for concepts referenced 30+ times each across the wiki
- **Pages created:** [[Chunked Prefill]] (210 lines), [[Continuous Batching]] (240 lines)
- **Source:** Synthesized from existing wiki knowledge (no raw source files)
- **Key context:**
  - **[[Chunked Prefill]]:** Breaking long prompt prefill into chunks (e.g., 512 tokens) that interleave with decode tokens in same batch; solves prefill-decode interference (long prefills blocking decodes); enabled by default in V1 via `--max-num-batched-tokens`; V1 unified scheduler makes implementation trivial (token budget dictionary); works naturally with prefix caching (cache blocks as chunks complete); alternative to disaggregated serving (time-multiplex vs. space-separate); typical gains: 2-5× TTFT/ITL variance reduction, 0-5% throughput impact; common misconception: "chunking slows down prefill" (reality: other requests don't wait, total prefill time nearly unchanged)
  - **[[Continuous Batching]]:** Iteration-level scheduling (add/remove requests after each forward pass) vs. batch-level (wait for entire batch to finish); eliminates padding waste (variable-length sequences, no pre-allocation); 2-10× throughput over static batching (high variance workloads), 90-100% GPU utilization; Orca paper (OSDI 2022) introduced iteration-level scheduling; vLLM V0 had two-queue design (prefill/decode separate), V1 unified scheduler (single queue, token budget allocation); enabled by PagedAttention (block-based KV cache, fast allocation/deallocation); request lifecycle: WAITING → RUNNING (prefill) → RUNNING (decode) → FINISHED; works with chunked prefill, prefix caching, spec decode, data parallelism; primary benefit is throughput, not latency (reduces queue latency but may increase ITL variance slightly)
- **Cross-references added:**
  - Both pages heavily cross-reference: [[V1 Architecture]], [[vLLM Engine]], [[PagedAttention]], [[KV Cache]], [[Prefix Caching]], [[Disaggregated Serving]]
  - [[Chunked Prefill]] also references: [[Speculative Decoding]], [[LoRA]], [[Data Parallelism]]
  - [[Continuous Batching]] also references: [[CUDA Graphs]], [[Optimization Levels]]
- **Format:** Full Obsidian Flavored Markdown with frontmatter, wikilinks, comparison tables, code examples, cross-references
- **Insight:** These are foundational concepts underlying vLLM's 2-4× performance advantage — continuous batching for high throughput, chunked prefill for latency fairness

## Previous Ingest: vLLM Observability & Training (Batch 13 of 14)

- **Sources:** `docs/design/metrics.md`, `features/reasoning_outputs.md`, `features/sleep_mode.md`, `training/rlhf.md`, `training/async_rl.md`
- **Key entities:** Prometheus Metrics, Reasoning Models, Sleep Mode, RLHF Integration, Async RL
- **New concepts:**
  - [[vLLM Metrics]] — Prometheus-compatible metrics system: (1) server-level (Gauges/Counters: num_requests_running/waiting, kv_cache_usage_perc, prompt/generation_tokens_total), (2) request-level (Histograms: TTFT, inter-token latency, E2E latency, prefill/decode time, queue time), (3) event timeline (QUEUED→SCHEDULED→PREEMPTED→NEW_TOKENS), (4) KV cache residency (kv_block_lifetime/idle_before_evict/reuse_gap histograms via sampling), (5) prefix cache (queries/hits counters → derive hit rate), (6) speculative decoding (acceptance rate as counters, not gauges); design: metrics collected in API server (not engine core), monotonic timestamps for intervals (immune to NTP skew), same-process comparison requirement; /metrics endpoint (vllm: prefix, model_name label), LoggingStatLogger (INFO logs every 5s); Grafana dashboard with critical metrics; deprecated V0 metrics (swapped requests, CPU cache usage); future: parallel sampling, OpenTelemetry integration, autoscaling saturation detection
  - [[Reasoning Models]] — Models that produce internal reasoning tokens (<think>...</think>) before final answer; 11+ families (DeepSeek-R1, Qwen3, QwQ, Granite 3.2, ERNIE-4.5, GLM-4.5, Holo2, Hunyuan A13B, MiniMax-M2); thinking budget control (thinking_token_budget limits reasoning tokens, vLLM forces reasoning_end_str when budget exhausted); server-level defaults (--default-chat-template-kwargs), request-level override (extra_body); streaming (delta.reasoning field), tool calling (reasoning available, tools parsed from content only), structured outputs (reasoning skipped by constraint engines); ReasoningParser (extract_reasoning/extract_reasoning_streaming methods), Reasoner class (is_reasoning_end for xgrammar integration); performance: 2-10× TTFT increase, thinking budget controls overhead; migration: reasoning_content → reasoning
  - [[Sleep Mode]] — Temporarily release GPU memory without stopping server; Level 1 (offload weights to CPU RAM, discard KV cache, fast wake) for same-model resume, Level 2 (discard weights + KV cache, retain buffers) for weight updates; benefits: ~90% GPU memory freed, distributed support (TP/PP/DP), fine-grained wake (tags=["weights"] or ["kv_cache"] separately); RLHF weight update pattern (sleep L2 → wake weights → reload_weights → wake KV cache) avoids OOM during weight reload; platforms: CUDA, ROCm (chunked allocation via VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE); API: llm.sleep(level), llm.wake_up(tags), llm.collective_rpc("reload_weights"); HTTP endpoints (requires VLLM_SERVER_DEV_MODE=1): /sleep, /wake_up, /collective_rpc, /is_sleeping; use cases: RLHF GPU sharing, cost optimization, model switching
  - [[RLHF with vLLM]] — vLLM as inference backend for RLHF training; integration with 11+ RL frameworks (TRL, OpenRLHF, verl, NeMo-RL, Unsloth, Prime-RL, Cosmos-RL, Open Instruct, SkyRL, PipelineRL, ms-swift); architecture patterns: (1) sequential (generate → train → generate), (2) async pipelined (overlap generation and training), (3) GPU sharing (sleep mode); weight synchronization via NCCL backend (multi-GPU, GPU-to-GPU transfer) or IPC backend (same-GPU, shared memory); weight update with sleep mode (L2 → wake weights → reload → wake KV cache); GRPO example with TRL; optimization: speculative decoding (EAGLE/n-gram), quantization (FP8/INT4), prefix caching; multi-GPU: TP (sharded NCCL transfer), DP (broadcast to all ranks); monitoring via vLLM metrics (E2E latency, TTFT, throughput, KV cache usage); limitations: 5-30s weight sync overhead, stale KV cache, async complexity
  - Async RL — Async RL pipelining: separate generation and training into parallel coroutines to overlap computation; pause/resume API for mid-flight weight updates: pause_generation(mode="keep|wait|abort", clear_cache=True) freezes requests, resume_generation() continues with new weights; HTTP endpoints (/pause, /resume) require VLLM_SERVER_DEV_MODE=1; data parallelism: internal LB (single pause/resume auto-coordinates), external LB (must pause/resume each instance); typical flow: generate → pause (mode="keep") → sleep (L2) → train → wake → reload → resume; key insight: requests paused with mode="keep" produce tokens from old weights before pause, new weights after resume; clear_cache=True discards KV cache (all tokens after resume use new weights)
- **Cross-refs:**
  - [[vLLM Metrics]] connects to [[vLLM Engine]], [[V1 Architecture]], [[Prefix Caching]], [[Speculative Decoding]], [[KV Cache]], [[LoRA]], [[Data Parallelism]], [[NVTX Profiling]]
  - [[Reasoning Models]] connects to [[Structured Outputs]], [[vLLM Engine]], [[Speculative Decoding]], [[Multi-Token Prediction]], [[Logits Processors]], [[Quantization]]
  - [[Sleep Mode]] connects to [[RLHF with vLLM]], Async RL, [[vLLM Engine]], [[Tensor Parallelism]], [[Prefix Caching]]
  - [[RLHF with vLLM]] connects to [[Sleep Mode]], Async RL, [[Data Parallelism]], [[Tensor Parallelism]], [[Speculative Decoding]], [[vLLM Metrics]]
  - Async RL connects to [[RLHF with vLLM]], [[Sleep Mode]], [[vLLM Engine]], [[Data Parallelism]]
- **Key metrics:**
  - vLLM Metrics: server-level (running/waiting requests, KV cache usage %), request-level (TTFT, ITL, E2E, prefill/decode time, queue time), KV cache residency (lifetime, idle before evict, reuse gaps), prefix cache (queries/hits counters), spec decode (accepted/draft tokens counters)
  - Reasoning Models: 11+ model families, thinking budget (per-request token limit), 2-10× TTFT increase, structured outputs + tool calling compatible
  - Sleep Mode: L1 (20-40 GB saved), L2 (~90% GPU freed, 160-180 GB for 70B FP16), 5-30s sleep/wake latency, ROCm chunking (default 256 MB)
  - RLHF: 11+ RL frameworks, 5-30s weight sync overhead, 2 backends (NCCL multi-GPU, IPC same-GPU), 3 architecture patterns (sequential, async, GPU sharing)
  - Async RL: 3 pause modes (abort/wait/keep), clear_cache parameter for KV cache invalidation, internal LB auto-coordinates DP ranks

## Previous Ingest: vLLM Deployment & Benchmarking (Batch 14 of 14, FINAL)

- **Sources:** `docs/serving/openai_compatible_server.md`, `benchmarking/cli.md`, `configuration/optimization.md`, `deployment/integrations/llm-d.md`, `configuration/conserving_memory.md`
- **Key pages:** [[OpenAI-Compatible Server]], [[vLLM Benchmarking]], [[llm-d]]
- **Key insight:** vLLM production deployment combines OpenAI-compatible HTTP server (drop-in API + 10+ vLLM-specific endpoints), built-in CLI benchmarking (15+ datasets, load pattern control, visualization), optimization strategies (chunked prefill, -O0 to -O3, NUMA binding, batch-level DP for multimodal), llm-d Kubernetes orchestration (disaggregated prefill/decode, per-pool autoscaling), memory conservation (TP/quantization/context limits); maximum throughput load pattern (--request-rate=inf --max-concurrency=<limit>) most common for capacity planning

## Previous Ingest: vLLM Internal Architectures (Batch 12 of 14)

- **Sources:** `docs/design/attention_backends.md`, `model_runner_v2.md`, `logits_processors.md`, `plugin_system.md`
- **Key entities:** Attention Backends, Model Runner V2, Logits Processors, Plugin System
- **Key insight:** vLLM's internal architecture provides pluggable attention backends (13+ standard, 15+ MLA) with automatic selection via priority-ordered lists; Model Runner V2 redesigns execution from first principles with persistent batch decoupling, async-first design, and explicit CUDA graph management; logits processors operate at batch granularity with stateful BatchUpdate synchronization; plugin system enables OOT hardware support (Ascend NPU, Intel Gaudi, AWS Neuron) via standard Python entry_points with five-component platform plugins

## Previous Ingest: vLLM Parallelism Strategies (Batch 10 of 14)

- **Sources:** `docs/serving/parallelism_scaling.md`, `data_parallel_deployment.md`, `context_parallel_deployment.md`, `design/multiprocessing.md`
- **Key entities:** Tensor Parallelism, Pipeline Parallelism, Data Parallelism, Context Parallelism
- **Key insight:** vLLM supports four parallelism strategies: (1) TP (Megatron-LM, all-reduce per layer, 5-15% overhead InfiniBand), (2) PP (sequential splits, micro-batching, 10-20% bubble), (3) DP (model replication, ZMQ LB, 90-95% linear scaling), (4) CP (prefill ring attention under dev, decode KV cache sharding to eliminate duplication when TP > num_kv_heads); combine as DP×TP×PP for multi-node

## Previous Ingest: vLLM Serving Features: LoRA, Structured Outputs (Batch 11 of 14)

- **Sources:** `docs/features/lora.md`, `lora_resolver_plugins.md`, `structured_outputs.md`, `tool_calling.md`
- **Key insight:** Multi-tenant serving via (1) LoRA (per-request adapters, Punica/BGMV kernels, ~5-10% overhead), (2) Structured Outputs (logits masking, xgrammar/guidance backends, 1-10s compilation, 5-15% per-token slowdown); both compose with quantization/prefix caching/speculative decoding

## Vault State

- 9 index pages (concepts, architectures, techniques, papers, benchmarks, tools, open-questions, sources, main)
- 13 batches ingested (vLLM core architecture, KV cache, compilation, kernels, MoE, disagg serving, quantization, quantization formats, speculative decoding, parallelism, serving features, internal architectures, observability & training)
- 60+ content pages (concepts, architectures, techniques, tools, papers, sources)
