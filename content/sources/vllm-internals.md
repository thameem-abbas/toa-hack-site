---
title: vLLM Internal Architecture Designs
type: source-summary
created: 2026-04-25
tags: [vllm, attention, model-runner, logits-processors, plugins, design]
---

# vLLM Internal Architecture Designs

Source summary for vLLM internal design documents covering attention backends, model runner V2, logits processors, and the plugin system.

## Sources

1. **Attention Backend Feature Support** (`.raw/articles/vllm-attention-backends-2026-04-25.md`)
   - Auto-generated feature support matrix for attention backends
   - Backend selection logic (manual vs automatic)
   - Priority-ordered backend lists for CUDA (Blackwell vs Ampere/Hopper)
   - Standard attention backends (MHA/MQA/GQA): FlashAttention-2/3/4, FlashInfer, TRTLLM, Triton, flex_attention, ROCm AITER, CPU
   - MLA backends (DeepSeek-style): prefill (TRT-LLM Ragged, FlashInfer, cuDNN, FlashAttention) and decode (CUTLASS_MLA, FLASHINFER_MLA, FLASHMLA, etc.)
   - Feature support: dtypes (fp16/bf16/fp32), KV dtypes (auto/fp8/int8), block sizes, head sizes, attention sink, sparse attention, multimodal prefix, decode context parallelism
   - Validation pattern: `AttentionBackend.validate_configuration()`
   - CUDA graph compatibility via `AttentionCGSupport` enum

2. **Model Runner V2 Design** (`.raw/articles/vllm-model-runner-v2-2026-04-25.md`)
   - Redesign from first principles to address V1 technical debt
   - Persistent batch decoupling: separate persistent state tensors from per-step input tensors
   - Async-first design: CUDA stream-based execution with no CPU synchronization points
   - Eliminating async barrier: temporary pinned copies instead of shared buffers
   - `StagedWriteTensor` for incremental GPU tensor updates (block tables)
   - GPU-native input preparation via Triton kernels
   - Universal Virtual Addressing (UVA) for CPU-resident tensor access
   - Triton-native sampler: Gumbel sampling, efficient top-k logprobs, memory-efficient prompt logprobs
   - Modularity: dedicated files for features, `InputBatch` class
   - Explicit CUDA graph management via `CUDAGraphManager`

3. **Logits Processors** (`.raw/articles/vllm-logits-processors-2026-04-25.md`)
   - Batch-granularity logits transformation: `(num_requests) x (vocab_size)` tensor
   - Stateful processors: maintain per-request metadata
   - Two-phase execution: `update_state()` for batch changes, `apply()` for transformation
   - Argmax-invariant vs non-argmax-invariant processors
   - `LogitsProcessor` base class: `__init__`, `apply`, `is_argmax_invariant`, `update_state`, `validate_params`
   - `BatchUpdate` data structure: removed/added/moved requests (UNIDIRECTIONAL vs SWAP)
   - Batch update processing order: removes → adds → moves
   - Built-in processors: Min-P, Logit bias, Min-tokens
   - Legacy hard-coded processors (to be refactored): temperature, top-k, top-p, repetition penalty, frequency penalty, presence penalty, allowed token IDs, bad words

4. **Plugin System** (`.raw/articles/vllm-plugin-system-2026-04-25.md`)
   - Standard Python `entry_points` mechanism for extensibility
   - Four plugin types: general (models), platform (OOT hardware), IO processor (multimodal), stat logger
   - Plugin structure: group name, plugin name, plugin value (fully qualified name)
   - Platform plugin workflow: register function → Platform class → Worker class → AttentionBackend → DeviceCommunicator → CustomOps
   - OOT hardware plugins: vllm-ascend (Ascend NPU), vllm-gaudi (Intel Gaudi), vllm-neuron (AWS Neuron), vllm-kunlun
   - Plugin discovery: `vllm.plugins.load_plugins_by_group()`
   - Filtering: `VLLM_PLUGINS` environment variable
   - Compatibility guarantee: documented interfaces (e.g., `ModelRegistry.register_model`) always available

## Key Claims

### Attention Backends

1. **Backend selection is automatic unless explicitly specified**: vLLM iterates through priority-ordered backends and selects the first compatible one based on configuration validation
2. **Priority order differs by compute capability**: Blackwell (SM 10.x) prefers FlashInfer → FlashAttention → Triton, while Ampere/Hopper (SM 8.x-9.x) prefers FlashAttention → FlashInfer → Triton
3. **MLA requires separate prefill and decode backends**: prefill uses TRT-LLM Ragged (SM100 default) or FlashAttention varlen, decode uses CUTLASS_MLA/FLASHINFER_MLA/FLASHMLA
4. **FlashInfer uses TRTLLM on Blackwell**: can be disabled via `--attention-config.use_trtllm_attention=0`
5. **FlashAttention version selection is hardware-dependent**: FA4 on SM100+ (Blackwell), FA3 on SM90 (Hopper), FA2 otherwise

### Model Runner V2

1. **V1's persistent batch couples state with inputs**: requires complex tensor reordering and `CachedRequestState` backup copies
2. **MRV2 decouples state from inputs**: fixed-size pre-allocated tensors (1024 rows default) with permanent row assignment per request lifetime
3. **V1 async barrier is bug-prone and limits CPU-GPU overlap**: requires explicit synchronization around shared pinned buffers
4. **MRV2 eliminates async barrier**: temporary pinned copies prevent CPU-GPU race conditions without synchronization
5. **StagedWriteTensor reduces CPU-GPU transfer overhead**: stage diffs on CPU, pack into contiguous buffers, copy to GPU, apply via single kernel
6. **GPU-native input preparation improves async behavior**: Triton kernels derive input metadata without CPU bottlenecks
7. **Gumbel sampling avoids softmax materialization**: stateless in-kernel RNG from seed input
8. **V1 dummy_run handles too many responsibilities**: profiling, CUDA graph capture, warmup, empty DP forward passes
9. **MRV2 explicit CUDA graph management**: `CUDAGraphManager` captures and launches full graphs via standard PyTorch APIs

### Logits Processors

1. **Logits processors operate at batch granularity**: consume entire `(num_requests) x (vocab_size)` tensor per engine step
2. **Processors are stateful**: maintain per-request metadata synchronized with persistent batch state
3. **Argmax-invariant processors can be skipped for greedy sampling**: e.g., Min-P doesn't change the max-logit token ID
4. **Batch update processing order is removes → adds → moves**: ensures correct state transitions
5. **Move operations are UNIDIRECTIONAL (one-way) or SWAP (two-way)**: UNIDIRECTIONAL leaves empty slots, SWAP exchanges requests
6. **Built-in processors are always loaded**: Min-P, Logit bias, Min-tokens use the new programming model
7. **Legacy processors are hard-coded in sampler**: temperature, top-k, top-p, penalties awaiting refactor

### Plugin System

1. **Plugins use standard Python entry_points**: `vllm.general_plugins`, `vllm.platform_plugins`, `vllm.io_processor_plugins`, `vllm.stat_logger_plugins`
2. **Platform plugins enable OOT hardware support**: Ascend NPU, Intel Gaudi, AWS Neuron, Kunlun without modifying vLLM core
3. **Platform plugins must implement Platform → Worker → AttentionBackend chain**: minimal implementation requires `check_and_update_config`, `get_attn_backend_cls`, `get_device_communicator_cls`
4. **Worker class must implement execute_model**: basic inference requires `init_device`, `initialize_cache`, `load_model`, `get_kv_cache_spec`, `determine_available_memory`, `initialize_from_config`, `execute_model`
5. **vLLM guarantees documented plugin interface stability**: e.g., `ModelRegistry.register_model` always available
6. **Plugin functions must be re-entrant**: may be called multiple times across processes

## Cross-References

### Attention Backends connects to
- [[vLLM Engine]] — backend validation in engine initialization
- [[V1 Architecture]] — backend selection in V1 model runner
- [[CUDA Graphs]] — `AttentionCGSupport` enum for graph compatibility
- [[Hybrid KV Cache Manager]] — KV cache block size constraints
- [[PagedAttention]] — block-based KV cache attention
- [[Quantized KV Cache]] — FP8 KV cache support in backends
- [[Plugin System]] — OOT hardware backends via platform plugins

### Model Runner V2 connects to
- [[vLLM Engine]] — MRV2 as alternative model runner implementation
- [[V1 Architecture]] — comparison with V1 design choices
- [[CUDA Graphs]] — explicit graph management via `CUDAGraphManager`
- [[Speculative Decoding]] — better compatibility via `idx_mapping` in sampler
- [[Logits Processors]] — Triton-native sampler integration

### Logits Processors connects to
- [[vLLM Engine]] — persistent batch state synchronization
- [[V1 Architecture]] — logits processor lifecycle in V1 engine
- [[Structured Outputs]] — logits processors for guided generation
- [[Speculative Decoding]] — logits processors in multi-token verification

### Plugin System connects to
- [[CustomOp System]] — custom ops for OOT hardware (communicator, common, csrc)
- [[vLLM Engine]] — plugin loading in engine initialization
- [[Attention Backends]] — OOT attention backend registration
- [[Multi-Modal Models]] — IO processor plugins for pre/post-processing

## Metrics

### Attention Backends
- **Standard backends**: 13 backends (CPU_ATTN, FLASHINFER, FLASH_ATTN, FLASH_ATTN_DIFFKV, FLEX_ATTENTION, ROCM_AITER_FA, ROCM_AITER_UNIFIED_ATTN, ROCM_ATTN, TREE_ATTN, TRITON_ATTN, TURBOQUANT)
- **MLA prefill backends**: 4 backends (TRT-LLM Ragged, FlashInfer, cuDNN, FlashAttention)
- **MLA decode backends**: 11 backends (CUTLASS_MLA, FLASHINFER_MLA, FLASHINFER_MLA_SPARSE, FLASHMLA, FLASHMLA_SPARSE, FLASH_ATTN_MLA, ROCM_AITER_MLA, ROCM_AITER_MLA_SPARSE, ROCM_AITER_TRITON_MLA, TRITON_MLA, XPU_MLA_SPARSE)
- **Supported dtypes**: fp16, bf16, fp32 (backend-dependent)
- **KV cache dtypes**: auto, float16, bfloat16, fp8, fp8_e4m3, fp8_e5m2, int8_per_token_head, fp8_per_token_head, turboquant variants
- **Block sizes**: 16, 32, 64, 128 (fixed), %16 (multiples of 16), Any (backend-dependent)
- **Head sizes**: 32-512 (backend-dependent), Any for FLASH_ATTN/TRITON_ATTN/FLEX_ATTENTION

### Model Runner V2
- **Persistent batch size**: 1024 rows (default max_num_reqs)
- **Async race condition elimination**: zero synchronization overhead via temporary pinned copies
- **StagedWriteTensor**: single kernel launch for batched diffs (vs full CPU-GPU copy)

### Logits Processors
- **Built-in processors**: 3 (Min-P, Logit bias, Min-tokens) using new programming model
- **Legacy processors**: 8 (allowed token IDs, bad words, repetition penalty, frequency penalty, presence penalty, temperature, top-k, top-p) awaiting refactor
- **Batch update operations**: 3 types (remove, add, move)
- **Move directionality**: 2 types (UNIDIRECTIONAL, SWAP)

### Plugin System
- **Plugin types**: 4 (general, platform, IO processor, stat logger)
- **OOT hardware plugins**: 4+ official (vllm-ascend, vllm-gaudi, vllm-neuron, vllm-kunlun)
- **Platform plugin components**: 5 (Platform class, Worker class, AttentionBackend, DeviceCommunicator, CustomOps)

## Open Questions

1. **MRV2 feature completeness**: Which V1 features are still missing in MRV2? (document states "not yet feature-complete")
2. **Logits processor refactor timeline**: When will legacy hard-coded processors (temperature, top-k, top-p, penalties) be migrated to the new programming model?
3. **Attention backend auto-selection tuning**: How does vLLM measure and update backend priority orders for new hardware?
4. **Plugin compatibility versioning**: How does vLLM manage breaking changes to plugin interfaces across releases?
5. **MRV2 performance comparison**: What are the measured speedups/memory savings of MRV2 vs V1 across different workloads?
