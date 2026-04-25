---
title: vLLM Compilation & CUDA Graphs Design
type: source
tags: [compilation, cuda-graphs, optimization, torch-compile]
source_urls:
  - file:///tmp/vllm/docs/design/torch_compile.md
  - file:///tmp/vllm/docs/design/cuda_graphs.md
  - file:///tmp/vllm/docs/design/optimization_levels.md
ingested: 2026-04-25
---

# vLLM Compilation & CUDA Graphs Design

This source summary covers three design documents from vLLM that explain how the framework achieves high performance through compilation and CUDA graph optimization.

## Key Takeaways

### torch.compile Integration

vLLM V1 uses `torch.compile` as a critical component enabled by default:

1. **Compilation Cache** — Stores artifacts at `~/.cache/vllm/torch_compile_cache/` based on hash of configs, PyTorch settings, and model code. Cache is portable across deployments and eliminates cold-start compilation.

2. **Pre-compilation Guarantee** — All compilation finishes before serving requests, preventing unexpected latency spikes during inference.

3. **Dynamic Shapes Strategy** — Three modes for handling variable batch sizes:
   - `BACKED` (default) — Allows PyTorch to add guards for maximum performance, but may unsoundly drop some
   - `UNBACKED` — Strongest guarantee against guards, but may miss optimizations
   - `BACKED_SIZE_OBLIVIOUS` (experimental) — Balance between safety and performance

4. **Graph Splitting** — Computation graph split by attention operations into submodules:
   - First layer before attention
   - Middle layers (attention-to-attention)
   - Final layer after attention
   - Allows piecewise compilation and CUDA graph capture

5. **Inductor Compilation** — Each subgraph compiled to Triton kernels with optional auto-tuning for specific batch sizes. Auto-tuning disabled by default due to time cost but can provide significant speedups.

### CUDA Graphs

CUDA graphs reduce CPU launch overhead by capturing GPU kernel sequences and replaying them:

1. **Five Modes** via `cudagraph_mode`:
   - `NONE` — Disable CUDA graphs (debugging)
   - `PIECEWISE` — Exclude attention from graphs (most compatible)
   - `FULL` — Full graphs for all batches
   - `FULL_DECODE_ONLY` — Full graphs only for uniform decode (saves memory in P/D setups)
   - `FULL_AND_PIECEWISE` (default) — Full for uniform decode, piecewise for prefill/mixed

2. **Nested Wrapper Design** — FULL wrapper outside model, PIECEWISE wrappers inside piecewise backends. Dispatcher selects runtime mode based on batch composition.

3. **Batch Descriptor** — Unique key for CUDA graph dispatch: `(num_tokens, num_reqs, uniform, has_lora)`

4. **Attention Backend Compatibility** — Tracked via `AttentionCGSupport` enum:
   - `ALWAYS` — FlashAttention v3, Triton Attention
   - `UNIFORM_BATCH` — FlashAttention v2, FlashMLA, FlashInfer (with TRTLLM on Blackwell)
   - `UNIFORM_SINGLE_TOKEN_DECODE` — FlashInfer, Mamba, AITER MLA, CUTLASS MLA
   - `NEVER` — All unlisted backends

5. **Memory vs Latency** — V1 uses more memory for CUDA graphs than V0. `FULL_AND_PIECEWISE` requires the most memory but provides best latency for small models and MoEs.

### Optimization Levels

Four levels trading startup time for runtime performance:

1. **-O0** — No optimization. Fastest startup, lowest performance. For development/debugging.
   - `cudagraph_mode=NONE`, `mode=NONE`, all fusions disabled

2. **-O1** — Fast optimization. Basic compilation and piecewise CUDA graphs.
   - `cudagraph_mode=PIECEWISE`, `mode=VLLM_COMPILE`
   - Fusions: norm_quant, act_quant, act_padding, mla_dual_rms_norm

3. **-O2** (default) — Full optimization. Production-ready.
   - `cudagraph_mode=FULL_AND_PIECEWISE`
   - Additional fusions: allreduce_rms, rope_kvcache

4. **-O3** — Same as -O2, reserved for future experimental optimizations.

User-specified flags override optimization level defaults.

## Architecture Components

### Compilation Pipeline

```
Model Forward → Dynamo Graph Capture → Graph Splitting by Attention Ops 
→ Subgraph Compilation (Inductor) → Optional Auto-tuning → Cached Kernels
```

### CUDA Graph Dispatcher

```
Batch → BatchDescriptor → CudagraphDispatcher.dispatch() 
→ (runtime_mode, batch_descriptor) → CUDAGraphWrapper (FULL or PIECEWISE)
→ Capture or Replay
```

### Key Classes

- `CompilationConfig` — Controls torch.compile behavior, dynamic shapes, CUDA graph modes
- `CudagraphDispatcher` — Central controller, single source of truth for available graphs
- `CUDAGraphWrapper` — Wraps callable with capture/replay logic
- `BatchDescriptor` — Dispatch key for CUDA graph selection
- `AttentionCGSupport` — Enum tracking backend CUDA graph capabilities

## Cross-References

- [[torch.compile Integration]] — Full technique page on vLLM's compilation strategy
- [[CUDA Graphs]] — Concept page on CUDA graph mechanics and modes
- [[Optimization Levels]] — Architecture page on the four optimization levels
- [[vLLM Engine]] — Core engine that uses compilation and CUDA graphs
- [[V1 Architecture]] — V1's multi-process design enables piecewise compilation
- [[PagedAttention]] — Attention backend compatibility varies for CUDA graphs
- [[Chunked Prefill]] — Interacts with CUDA graph mode selection

## Open Questions

1. **Inductor Graph Partitioning** — Experimental feature (`use_inductor_graph_partition=True`) for torch>=2.9 that allows custom passes like attention fusion to work with piecewise compilation. Will this become the default?

2. **Auto-tuning Tradeoff** — Auto-tuning disabled by default due to time cost. For long-running production deployments, is there tooling to pre-warm the cache with auto-tuned kernels?

3. **CUDA Graph Memory Cost in V1** — V1 uses more memory for CUDA graphs than V0. What are the specific memory multipliers for each mode?

4. **BatchDescriptor Evolution** — The descriptor may need `uniform_query_len` for multiple uniform decode lengths. How will this affect dispatch logic?

5. **Cascade Attention** — Never CUDA graph compatible, always falls back to PIECEWISE or NONE. Is there a path to make it graph-compatible?

## Related Papers

- [[Efficient Memory Management for Large Language Model Serving with PagedAttention]] — vLLM's foundational paper on KV cache management
