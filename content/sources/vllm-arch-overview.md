---
title: vLLM Architecture Overview
type: source
created: 2026-04-25
source_files:
  - .raw/articles/vllm-arch-overview-2026-04-25.md
  - .raw/articles/vllm-paged-attention-2026-04-25.md
  - .raw/articles/vllm-v1-guide-2026-04-25.md
source_urls:
  - /tmp/vllm/docs/design/arch_overview.md
  - /tmp/vllm/docs/design/paged_attention.md
  - /tmp/vllm/docs/usage/v1_guide.md
fetched: 2026-04-25
---

# vLLM Architecture Overview

Source summary covering vLLM's core architecture, PagedAttention kernel, and V1 design.

## Sources

1. **Architecture Overview** (`docs/design/arch_overview.md`)
   - V1 multi-process architecture (API Server, Engine Core, GPU Workers, DP Coordinator)
   - Process count formula and examples
   - LLMEngine, AsyncLLMEngine, Worker, Model Runner, Model
   - Class hierarchy and VllmConfig pattern

2. **Paged Attention** (`docs/design/paged_attention.md`)
   - Historical document describing original 2023 PagedAttention kernel
   - Kernel flow: Query load, Key iteration, QK dot product, Softmax, Value dot product, LV reduction, Output write
   - Memory layout and access patterns (coalesced reads, shared vs. register memory)
   - Block, warp, thread group, thread block concepts

3. **V1 Guide** (`docs/usage/v1_guide.md`)
   - V0 fully deprecated; V1 goals: simple/modular, near-zero CPU overhead, unified optimizations, zero configs
   - Unified scheduler: treats prompt and output tokens identically via token budget dictionary
   - Differences from V0: chunked prefill default, CUDA graph memory, logprobs semantics, prompt logprobs not cached
   - Feature support matrix: hardware (NVIDIA, AMD, Intel, TPU, CPU), models (decoder-only, Whisper, Mamba, multimodal), features (prefix caching, LoRA, FP8 KV, spec decode)
   - Removed features: best_of, per-request logits processors, KV swapping, request-level structured output backend

## Key Takeaways

1. **V1 Process Architecture:** vLLM V1 separates API serving (HTTP, tokenization), scheduling (request queue, KV cache allocation), and GPU execution (model forward pass) into distinct processes communicating via ZMQ. Total process count = A + DP + N + (1 if DP > 1), where N = DP × PP × TP.

2. **Unified Scheduler:** V1's scheduler treats prompt and output tokens identically, allocating token budgets as `{request_id: num_tokens}`. This eliminates prefill/decode mode switching and enables clean integration of chunked prefill, prefix caching, and speculative decoding.

3. **PagedAttention Kernel:** The original kernel (described in historical docs) uses carefully designed memory access patterns (coalesced reads, shared memory for queries, register memory for keys/values) to achieve high performance despite non-contiguous KV cache blocks.

4. **VllmConfig Pattern:** All components accept a single `VllmConfig` object containing engine-level global state. This enables extensibility (adding features without changing deep constructor signatures) and uniformity (all models share `def __init__(self, *, vllm_config: VllmConfig, prefix: str = "")`).

5. **Sharding at Initialization:** Model weight transformations (tensor parallelism sharding, quantization) happen during initialization, not after. This is critical for memory efficiency with large models (e.g., 405B model on 16×H100: each GPU loads only its 50GB shard, not full 810GB).

6. **V1 Design Goals:** Simple/modular codebase, near-zero CPU overhead (busy loop in engine core), unified optimizations (no prefill/decode separation), zero configs (chunked prefill and prefix caching enabled by default).

7. **Feature Removals in V1:** best_of sampling (low usage), per-request logits processors (replaced by global processors), GPU↔CPU KV swapping (no longer needed with better scheduling), request-level structured output backend (global backend only).

## Related Pages

**Architectures:**
- [[vLLM Engine]] — V1 process architecture, LLMEngine, workers
- [[V1 Architecture]] — V1 design, unified scheduler, differences from V0

**Concepts:**
- [[PagedAttention]] — KV cache block management
- [[Continuous Batching]] — dynamic request scheduling
- [[Chunked Prefill]] — breaking long prefills into chunks
- [[Prefix Caching]] — KV cache reuse for shared prefixes
- [[Tensor Parallelism]] — model layer sharding
- [[Pipeline Parallelism]] — model stage pipelining
- [[Data Parallelism]] — request-level parallelism

**Papers:**
- [[Efficient Memory Management for Large Language Model Serving with PagedAttention]] — foundational SOSP 2023 paper

**Tools:**
- [[vLLM]] — serving engine tool page

## Notes

- The PagedAttention document is explicitly marked as historical. The kernel has evolved significantly since the 2023 paper (FlashAttention integration, FP8 KV cache, etc.).
- V1 architecture diagrams referenced but not included in text sources (see `docs/assets/design/arch_overview/v1_process_architecture_*.png`).
- Process count formula critical for CPU resource sizing in production deployments.
- VllmConfig pattern applies to all vLLM models, including out-of-tree registered models (shim pattern provided for backwards compatibility).
