---
title: KV Cache
type: concept
tags: [memory-management, attention, optimization]
related: [PagedAttention, Prefix Caching, Continuous Batching, Chunked Prefill]
---

# KV Cache

## Overview

The KV (Key-Value) cache is a fundamental memory optimization in autoregressive transformer inference that stores previously computed key and value tensors from attention layers. This avoids redundant computation during token-by-token generation, trading memory for speed.

## Mechanics

### What Gets Cached

During transformer attention computation, each token produces three vectors:
- **Query (Q)**: Used only for the current step
- **Key (K)**: Cached for all subsequent steps
- **Value (V)**: Cached for all subsequent steps

For autoregressive generation at step `t`, the attention mechanism needs:
- Current query: `Q_t`
- All keys up to current step: `K_0, K_1, ..., K_t`
- All values up to current step: `V_0, V_1, ..., V_t`

Without caching, we would recompute `K_0...K_{t-1}` and `V_0...V_{t-1}` at every step. The KV cache stores these tensors so they're computed exactly once per token.

### Memory Footprint

The memory required for KV cache is proportional to:

```
memory_bytes = batch_size × seq_len × num_layers × num_kv_heads × head_dim × 2 (K+V) × dtype_size
```

**Example calculation** for a 7B model (LLaMA-like):
- `num_layers = 32`
- `num_kv_heads = 32` (or 8 with GQA)
- `head_dim = 128`
- `dtype_size = 2` bytes (FP16)
- `batch_size = 1`, `seq_len = 2048`

```
memory = 1 × 2048 × 32 × 32 × 128 × 2 × 2 = 1,073,741,824 bytes ≈ 1 GB per request
```

For a 70B model with batch size 32 and 4K context, this can reach **hundreds of gigabytes**, making KV cache the dominant memory consumer in LLM inference.

## Block-Based Management in vLLM

### Virtual Memory Analogy

vLLM treats KV cache like virtual memory in operating systems:
- **Blocks**: Fixed-size chunks of KV cache (e.g., 16 tokens per block)
- **Block Table**: Maps logical token positions to physical memory blocks (like a page table)
- **Memory Pool**: Pre-allocated GPU memory divided into fixed-size blocks
- **Block Manager**: Allocates, frees, and evicts blocks as needed

This design enables:
- **Non-contiguous allocation**: Blocks don't need to be adjacent in physical memory
- **Efficient memory reuse**: Freed blocks immediately available for new requests
- **Sharing**: Multiple requests can reference the same physical block (see [[Prefix Caching]])

### Block Allocation Example

For a request with 35 tokens and block size 16:

```
Logical sequence: [token_0, token_1, ..., token_34]

Block table:
  Block 0: tokens 0-15   → Physical block 7
  Block 1: tokens 16-31  → Physical block 3
  Block 2: tokens 32-34  → Physical block 12 (partial, 3/16 slots filled)
```

Physical memory is allocated non-contiguously (blocks 7, 3, 12 might be scattered in GPU RAM), but appears sequential to the attention kernel via the block table indirection.

### Block Lifecycle

1. **Allocation**: Scheduler requests blocks from the free pool
2. **Population**: Model execution fills blocks with computed K/V tensors
3. **Reference counting**: Blocks track how many requests use them (for [[Prefix Caching]])
4. **Eviction**: LRU eviction when memory pressure requires freeing cached blocks
5. **Free**: Blocks with ref_count=0 return to the free pool when request completes

## Eviction Strategies

### Least Recently Used (LRU)

vLLM's default eviction policy uses an LRU queue:
- Blocks are added to the **tail** when freed
- Blocks are evicted from the **head** when memory is needed
- Cached blocks that are "touched" (reused) are moved to the tail

**Reverse-order freeing**: When a request completes, its blocks are added to the free queue in reverse order (last block first). Rationale: later blocks have longer prefix hashes and are less likely to match future requests, so should be evicted sooner.

### Sliding Window Eviction

For models with sliding window attention (e.g., Mistral, Gemma 2):
- Blocks outside the sliding window are automatically freed
- Only the most recent `sliding_window_size` tokens are retained
- Prefix caching still applies within the sliding window

## Quantized KV Cache

To reduce memory footprint, vLLM supports **FP8 KV cache**:
- K and V tensors stored in 8-bit floating point (1 byte per element)
- Reduces memory by ~50% compared to FP16/BF16
- Slight quality degradation (typically <1% perplexity increase)
- Enables 2x larger batch sizes or 2x longer sequences

**Activation**: Set `kv_cache_dtype="fp8"` when initializing the engine.

**Memory savings example** (7B model, batch=1, seq=2048):
- FP16: 1 GB per request
- FP8: 512 MB per request

## Data Structures (vLLM V1)

### KVCacheBlock

```python
class KVCacheBlock:
    block_id: int                           # Immutable physical block ID
    block_hash: BlockHash                   # Content-based hash (for prefix caching)
    ref_cnt: int                            # Number of requests using this block
    prev_free_block: KVCacheBlock | None    # Doubly-linked list pointer
    next_free_block: KVCacheBlock | None    # Doubly-linked list pointer
```

**Design rationale**:
- All blocks pre-allocated at initialization to avoid Python object creation overhead
- Intrusive doubly-linked list allows O(1) removal from the middle of the free queue
- Block hash enables content-based deduplication (see [[Prefix Caching]])

### Block Pool Components

At initialization, the KV cache manager creates:
- **Block Pool**: Array of all `KVCacheBlock` objects (fixed size)
- **Free Block Queue**: Head and tail pointers for LRU eviction
- **Cache Blocks**: Hash map `block_hash → block_id` for prefix caching
- **Request Blocks**: Hash map `request_id → [block_ids]` for tracking allocations

## Performance Characteristics

### Memory Efficiency Gains

Compared to naive pre-allocation for maximum batch size:
- **PagedAttention**: ~80% reduction in memory waste (from internal fragmentation)
- **Prefix Caching**: Additional 2-10x effective capacity when prompts share prefixes

### Throughput Impact

- **Without KV cache**: Quadratic complexity in sequence length (recompute all keys/values)
- **With KV cache**: Linear complexity (compute only new token's K/V)
- **Bottleneck shift**: From compute-bound to memory-bandwidth-bound for long sequences

## Interaction with Other Features

### [[Prefix Caching]]

Blocks can be shared across requests via content-based hashing:
- Hash includes: parent block hash, token IDs, LoRA ID, multi-modal hashes, cache salt
- Full blocks are cached when completed
- Cache hits skip prefill computation entirely

### [[Continuous Batching]]

Block-based allocation enables dynamic batching:
- Requests can be added/removed mid-flight without memory reallocation
- Preemption: pause a request by keeping its blocks, resume later
- Priority scheduling: evict low-priority blocks first

### [[Chunked Prefill]]

Long prompts are processed in chunks to avoid blocking decode:
- Each chunk fills some blocks partially
- Blocks are finalized and potentially cached as chunks complete
- Interleaves prefill and decode for better GPU utilization

### [[Disaggregated Serving]]

Prefill and decode pools share a distributed KV cache:
- Prefill workers compute and write blocks to shared storage
- Decode workers read blocks and append new tokens
- Requires coordination to avoid consistency issues

## Hybrid Models

For models with mixed attention types (e.g., Gemma 2/3, Llama 4):
- **Full attention layers**: Allocate blocks for all tokens
- **Sliding window layers**: Allocate blocks only for recent `sliding_window_size` tokens
- **Shared memory pool**: All layer types use blocks from the same pool with unified page size

See [[Hybrid KV Cache Manager]] for details on per-layer allocation strategies.

## Implementation Notes

### Physical Memory Layout

vLLM allocates KV cache as a contiguous GPU tensor:

```python
# Shape: [num_blocks, block_size, num_kv_heads, head_dim]
k_cache = torch.empty(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")
v_cache = torch.empty(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")
```

Attention kernels use the block table to gather non-contiguous blocks into a logically contiguous sequence.

### CUDA Kernel Integration

PagedAttention kernels take:
- **Query**: Current step's query tensor
- **Key cache**: GPU pointer to the entire K cache
- **Value cache**: GPU pointer to the entire V cache
- **Block table**: Maps logical positions to physical block IDs
- **Block size**: Number of tokens per block

The kernel performs gather operations to assemble keys/values from scattered blocks.

## Limitations and Tradeoffs

### Memory Overhead

Block-based management introduces overhead:
- **Block table storage**: ~1 MB per 10K requests (negligible)
- **Internal fragmentation**: Last block of each request is often partially filled
- **Metadata**: Block objects, hash maps, free queue pointers

Net effect: ~5-10% overhead, far outweighed by elimination of external fragmentation.

### Prefill vs Decode

KV cache only benefits the decode phase. During prefill:
- All keys and values are computed from scratch
- KV cache is populated but not yet reused
- Prefill is still compute-bound, not memory-bound

### Context Length Limits

Maximum context length is constrained by:
```
max_tokens = num_gpu_blocks × block_size
```

With 80 GB A100 and 70B model:
- Available memory for KV cache: ~40 GB (after model weights)
- Block size: 16
- Max tokens per GPU: ~10K-20K depending on batch size

## See Also

- [[PagedAttention]] — The attention kernel that enables block-based KV cache
- [[Prefix Caching]] — Content-based block sharing across requests
- [[Hybrid KV Cache Manager]] — Per-layer allocation for mixed attention types
- [[V1 Architecture]] — Unified scheduler and block management in vLLM V1
- [[vLLM Engine]] — High-level orchestration of block allocation and request scheduling

## References

- vLLM design doc: `docs/design/prefix_caching.md`
- vLLM design doc: `docs/design/hybrid_kv_cache_manager.md`
- PagedAttention paper: [[Efficient Memory Management for Large Language Model Serving with PagedAttention]]
