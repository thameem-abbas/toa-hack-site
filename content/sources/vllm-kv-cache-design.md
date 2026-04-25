---
title: vLLM KV Cache Design Documentation
type: source-summary
sources:
  - file:///tmp/vllm/docs/design/prefix_caching.md
  - file:///tmp/vllm/docs/design/hybrid_kv_cache_manager.md
  - file:///tmp/vllm/docs/features/automatic_prefix_caching.md
ingested: 2026-04-25
tags: [vllm, kv-cache, prefix-caching, hybrid-models]
---

# vLLM KV Cache Design Documentation

## Overview

This source bundle covers vLLM's KV cache management system, including:
1. **Prefix caching** — Hash-based block deduplication for shared prefixes
2. **Hybrid KV cache manager** — Per-layer allocation for models with mixed attention types
3. **Automatic Prefix Caching (APC)** — User-facing API and usage examples

These documents describe the V1 architecture's approach to memory management, which is foundational to vLLM's performance advantages over naive implementations.

## Key Takeaways

### Prefix Caching (Hash-Based)

**Core idea**: Cache KV cache blocks by content hash, allowing requests with shared prefixes to skip redundant computation.

**Hash composition**:
```
block_hash = H(
    parent_hash,      # Hash of previous block (ensures context sensitivity)
    block_tokens,     # Exact token IDs in this block
    extra_hashes      # LoRA ID, multi-modal hashes, cache_salt
)
```

**Multi-modal support**: Image/audio embeddings that replace placeholder tokens are hashed and included in `extra_hashes`, ensuring different media produces different hashes.

**Security**: `cache_salt` parameter enables per-tenant cache isolation in multi-tenant environments, preventing timing-based side-channel attacks.

**Data structures**:
- Block pool (pre-allocated, fixed size)
- Free block queue (doubly-linked list for O(1) LRU eviction)
- Cache blocks hash map (`block_hash → block_id`)
- Request blocks map (`request_id → [block_ids]`)

**Eviction policy**: LRU with reverse-order insertion (last block of request added first, as it has longest prefix hash and is least likely to be reused).

**V1 difference from V0**: Block tables are append-only, causing temporary duplicate cached blocks. Cleaned up when request completes.

### Hybrid KV Cache Manager

**Problem**: Hybrid models (Gemma 2/3, Llama 4, Jamba) mix attention types with different memory needs:
- Full attention: KV cache for all tokens
- Sliding window: KV cache for last N tokens only
- Mamba: State vectors (different size than KV cache)

**Solution**: Single memory pool with unified page size, but allocate different numbers of blocks to different layers.

**KV cache groups**: Layers grouped by:
1. Identical attention type (same allocation logic)
2. Identical page size (share memory pool)

**Grouping strategy**: Use `min(num_layers_type_A, num_layers_type_B)` as group size to minimize the number of groups.

**Example**: Gemma-3-27b (10 full, 52 sliding window)
- `group_size = 10`
- 7 groups total (1 full, 6 sliding window)
- Last group has padding overhead (2 layers + 8 padding)

**Prefix caching intersection**: For full + sliding window models:
1. Find longest cache hit for full attention (scan left-to-right)
2. Find longest cache hit for sliding window within that length (scan right-to-left)
3. Return the intersection

**Memory layout**: `m` buffers (one per layer in a group), each shared by `n` layers (one from each group). One logical block maps to `m` physical chunks.

**Coordinator selection**:
- No prefix caching: `KVCacheCoordinatorNoPrefixCache`
- Single group: `UnitaryKVCacheCoordinator`
- Two groups (full + efficient): `HybridKVCacheCoordinator`
- Other configurations: Not supported (disable prefix caching)

**Mamba handling**: Increase `block_size` until attention block size matches Mamba state size, then pad. Can cause very large block sizes (400+).

### Automatic Prefix Caching (User API)

**Enabling**:
```python
llm = LLM(model="...", enable_prefix_caching=True)
```

**Best workloads**:
- Multi-turn chat (shared conversation history)
- RAG (shared document context)
- Few-shot prompting (shared instruction + examples)
- Agent loops (shared system prompt + tools)

**Limitations**:
- No benefit for unique prompts with no prefix sharing
- No benefit for short prompts (<16 tokens, less than one block)
- Only reduces prefill time, not decode time
- Decode-heavy workloads see minimal gain

**Typical gains**:
- 2-10x TTFT reduction for long shared prefixes (1000+ tokens)
- 2-5x throughput increase when most requests share prefixes

## Related Wiki Pages

### Concepts
- [[KV Cache]] — Block-based memory management, eviction strategies, FP8 quantization
- [[Prefix Caching]] — Hash-based deduplication, cache isolation, multi-modal support
- [[PagedAttention]] — Virtual memory analogy, block allocation, attention kernels
- [[Continuous Batching]] — Dynamic batching enabled by block-based allocation
- [[Chunked Prefill]] — Interleaving prefill and decode, partial block allocation

### Architectures
- [[Hybrid KV Cache Manager]] — Per-layer allocation for mixed attention types
- [[vLLM Engine]] — Scheduler integration, block manager orchestration
- [[V1 Architecture]] — Multi-process design, unified scheduler, append-only block tables

### Tools
- [[vLLM]] — LLM inference engine (parent tool)

## Open Questions

1. **Padding overhead optimization**: Can we reduce the ~20% overhead in Case 3 (irregular layer counts) with better grouping heuristics?

2. **Mamba prefix caching**: How to extend hash-based caching to Mamba state vectors? State depends on entire history, not just block content.

3. **3+ attention types**: How to generalize the intersection algorithm beyond full + efficient (e.g., full + sliding + local chunked)?

4. **Cross-request block sharing limits**: V1's append-only block tables create duplicate cached blocks for identical requests. Is this acceptable long-term, or should we reintroduce block table mutations with careful locking?

5. **Quantized prefix caching**: Does FP8 KV cache affect hash stability? Do we need separate cache entries for FP16 vs FP8?

6. **Distributed prefix caching**: How to coordinate prefix caching across disaggregated prefill/decode pools? Cache replication? Centralized hash map?

## Implementation Checklist

If extending or modifying the hybrid KV cache manager:

- [ ] Verify page size uniformity across all groups
- [ ] Ensure coordinators handle your attention type (or disable prefix caching)
- [ ] Add cache hit tests for new attention types
- [ ] Update memory layout diagrams in docs
- [ ] Benchmark padding overhead for new grouping strategies
- [ ] Test with multi-modal inputs if applicable
- [ ] Validate cache_salt isolation if multi-tenant

## Performance Notes

### Hash Computation Overhead

- SHA-256: ~50 µs per block (negligible vs. prefill time)
- xxHash: ~10 µs per block (faster but non-cryptographic)
- Hash map lookups: ~1 µs per block

### Memory Overhead

- Block metadata: ~64 bytes per block
- Hash maps: ~1-2% of total memory
- Padding (hybrid models): 5-20% depending on layer count ratios

### Cache Hit Rates (Observed)

- Multi-turn chat: 70-90% of prefill tokens cached
- RAG with fixed documents: 90-99% cached
- Few-shot prompting: 50-80% cached (depends on example reuse)
- Unique prompts: 0% cached (but no overhead)

## Code Pointers

```
vllm/v1/core/kv_cache_manager.py
  - KVCacheManager (scheduler-facing API)
  
vllm/v1/core/kv_cache_coordinator.py
  - KVCacheCoordinatorNoPrefixCache
  - UnitaryKVCacheCoordinator
  - HybridKVCacheCoordinator
  
vllm/v1/core/single_type_kv_cache_manager.py
  - SingleTypeKVCacheManager (base class)
  - FullAttentionManager
  - SlidingWindowManager
  - MambaManager (WIP)
  
vllm/v1/core/kv_cache_spec.py
  - KVCacheSpec (per-layer config)
  - KVCacheGroup (grouping metadata)
  
vllm/attention/backends/
  - Attention kernels that consume block tables
```

## Version Notes

- **v0.11+**: SHA-256 default (collision-free), CBOR serialization available
- **V1 architecture**: Append-only block tables, unified scheduler
- **Commit 458e74eb**: Initial hybrid KV cache manager implementation (early stage)

## External References

- vLLM docs: `docs/design/prefix_caching.md`
- vLLM docs: `docs/design/hybrid_kv_cache_manager.md`
- vLLM docs: `docs/features/automatic_prefix_caching.md`
- SGLang (similar hash-based approach): [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)
- Gemma 2 paper (sliding window): [arxiv.org/abs/2408.00118](https://arxiv.org/abs/2408.00118)
- Jamba paper (Mamba + attention): [arxiv.org/abs/2403.19887](https://arxiv.org/abs/2403.19887)
