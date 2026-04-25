---
title: Hybrid KV Cache Manager
type: architecture
tags: [vllm, kv-cache, hybrid-models, sliding-window, mamba]
related: [KV Cache, Prefix Caching, vLLM Engine, V1 Architecture]
status: early-stage
---

# Hybrid KV Cache Manager

## Overview

The Hybrid KV Cache Manager extends vLLM's block-based KV cache to support **hybrid models** — architectures that mix multiple attention types within a single model. This enables efficient serving of models like Gemma 2/3, Llama 4, Jamba, and other architectures that combine full attention, sliding window attention, Mamba layers, or local chunked attention.

**Key challenge**: Different attention types have different memory requirements:
- **Full attention**: Needs KV cache for **all** tokens
- **Sliding window**: Needs KV cache only for the most recent `sliding_window_size` tokens
- **Mamba**: Needs state vectors (different size than attention KV cache)

The hybrid manager allocates memory from a **unified pool** while respecting per-layer requirements and enabling prefix caching with layer-specific rules.

## Problem: Why Hybrid Models Need Special Handling

### Naive Approach Fails

Suppose a model has 20 sliding window layers and 10 full attention layers. A naive implementation might:

1. Allocate full KV cache for all 30 layers (wasteful for sliding window)
2. Use separate memory pools per attention type (prevents memory reuse)
3. Disable prefix caching (too complex to implement per-layer)

All of these are unacceptable for production serving.

### vLLM's Solution

Use a **single memory pool** with **unified page size**, but allocate **different numbers of blocks** to different layers:

```
Request with 112 tokens, block_size=16, sliding_window_size=32

Full attention layers (need all 112 tokens):
  Allocate 7 blocks: [0, 1, 2, 3, 4, 5, 6]

Sliding window layers (need only last 32 tokens):
  Allocate 2 blocks: [7, 8]  (tokens 96-112, within sliding window)
```

Memory is reused across attention types, and prefix caching works with layer-specific cache hit rules.

## Architecture Layers

### Component Hierarchy

```
KVCacheManager (top-level interface)
    ├─ KVCacheCoordinator (chooses allocation strategy)
    │   ├─ KVCacheCoordinatorNoPrefixCache (prefix caching disabled)
    │   ├─ UnitaryKVCacheCoordinator (single KV cache group)
    │   └─ HybridKVCacheCoordinator (2 groups: full + efficient attention)
    └─ SingleTypeKVCacheManager[] (one per KV cache group)
        ├─ FullAttentionManager (allocates blocks for all tokens)
        ├─ SlidingWindowManager (allocates blocks for sliding window)
        └─ MambaManager (allocates blocks for Mamba state, WIP)
```

**KVCacheManager**: Scheduler-facing API, dispatches to coordinator.

**KVCacheCoordinator**: Orchestrates multiple SingleTypeKVCacheManagers, computes cache hit intersections.

**SingleTypeKVCacheManager**: Manages one "KV cache group" with homogeneous attention type.

## KV Cache Groups

### Definition

A **KV cache group** is a set of layers with:
1. **Identical attention type** (same allocation logic)
2. **Identical page size** (share the same memory pool)

Layers in the same group can share block IDs without memory waste.

### Grouping Strategy

Given a model with `n_full` full attention layers and `n_sw` sliding window layers:

1. Find the minimum layer count: `group_size = min(n_full, n_sw)`
2. Create groups of size `group_size`
3. Pad the last group if necessary

**Example**: Gemma-3-27b (10 full, 52 sliding window)

```
group_size = min(10, 52) = 10

Groups:
  Group 0: 10 full attention layers (full.0 - full.9)
  Group 1: 10 sliding window layers (sw.0 - sw.9)
  Group 2: 10 sliding window layers (sw.10 - sw.19)
  Group 3: 10 sliding window layers (sw.20 - sw.29)
  Group 4: 10 sliding window layers (sw.30 - sw.39)
  Group 5: 10 sliding window layers (sw.40 - sw.49)
  Group 6: 2 sliding window layers (sw.50 - sw.51) + 8 padding layers
```

Each group has the same page size:

```
page_size = 10 × kv_hidden_size × block_size
```

**Tradeoff**: Group 6 has 80% padding overhead (8 layers of wasted space). Acceptable for now; alternative grouping strategies under consideration.

## Memory Allocation

### Unified Page Size

All KV cache groups must use the same physical page size to share the memory pool. For models with the same `kv_hidden_size` across all layers:

```
page_size = layers_per_group × kv_hidden_size × block_size
```

**Example**: 10 layers per group, `kv_hidden_size = 256 bytes`, `block_size = 16`:

```
page_size = 10 × 256 × 16 = 40,960 bytes
```

### Per-Group Allocation

For a request with 112 tokens (example from above):

```
Full attention group (Group 0):
  Needs 112 tokens → 7 blocks (ceil(112 / 16))
  Block IDs: [0, 1, 2, 3, 4, 5, 6]

Sliding window group 1 (Group 1):
  Needs min(112, 32) = 32 tokens → 2 blocks
  Block IDs: [7, 8]

Sliding window group 2 (Group 2):
  Needs min(112, 32) = 32 tokens → 2 blocks
  Block IDs: [9, 10]
```

Total blocks allocated: 11 (7 + 2 + 2).

### Memory Layout

Physical memory is divided into **m buffers** (one per layer in a group). Each buffer is shared by **n layers** (one from each group).

**Example**: Model with 3 groups (1 full, 2 sliding window), each with 10 layers:

```
Physical memory:
  KVCacheTensor 0: shared by [full.0, sw.0, sw.10]
  KVCacheTensor 1: shared by [full.1, sw.1, sw.11]
  ...
  KVCacheTensor 9: shared by [full.9, sw.9, sw.19]
```

Each tensor is divided into **block_size × kv_hidden_size** chunks. Block ID `i` maps to chunk `i` in each tensor.

For a request with block IDs `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`:

```
KVCacheTensor 0:
  Blocks 0, 7, 9 → used by [full.0, sw.0, sw.10]

KVCacheTensor 1:
  Blocks 1, 8, 10 → used by [full.1, sw.1, sw.11]

KVCacheTensor 2:
  Blocks 2 → used by [full.2]
  (sw.2 and sw.12 don't need early tokens)

...and so on
```

**Consequence**: One logical "block" is physically scattered across `m` tensors (one piece per tensor).

## Prefix Caching for Hybrid Models

### Cache Hit Rules

Different attention types have different cache hit requirements:

| Attention Type | Cache Hit Condition |
|----------------|---------------------|
| Full attention | **All** prefix tokens must be cached |
| Sliding window | Only the last `sliding_window_size - 1` tokens must be cached |
| Mamba | (Work in progress) |

### Cache Hit Intersection (Full + Sliding Window)

For a model with both full and sliding window layers:

1. **Full attention**: Scan blocks left-to-right, find longest cached prefix
2. **Sliding window**: Scan blocks right-to-left (within full attention's cache hit length), find longest cached suffix
3. **Intersection**: The result is the longest prefix that satisfies both

**Example**: Request with 15 tokens (block_size = 1, sliding_window_size = 4)

```
Blocks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
Cached: [0, 1, 11, 12, 13]  (blue)

Full attention (left-to-right):
  0: cached → 1: cached → 2: NOT cached → STOP
  Cache hit length: 2 tokens

Sliding window (right-to-left, within length 2):
  1: cached → 0: cached
  Cache hit length: 2 tokens

Final cache hit: 2 tokens
```

If blocks 0-14 were all cached:

```
Full attention: 15 tokens cached

Sliding window (right-to-left):
  14: cached → 13: cached → 12: cached → 11: cached
  Last 4 tokens cached → can use them to compute token 14
  Cache hit length: 14 tokens (must prefill [11, 12, 13] → [14])
```

**Why right-to-left for sliding window?** We want the **longest suffix** within the sliding window. Scanning right-to-left finds it efficiently.

**Why stop at full attention's cache hit?** A cache hit for sliding window requires that all earlier layers (full attention) also have their cache hit. The intersection ensures both are satisfied.

### Cache Keying

Blocks are cached with a compound key:

```python
cache_key = (block_hash, group_id)
```

This allows the **same token sequence** to be cached independently for different groups. For example:
- Block with tokens `[A, B, C, D]` in the full attention group
- Block with tokens `[A, B, C, D]` in the sliding window group

Both are cached separately because they represent different layers' KV caches.

### LRU Eviction

All groups share a **single LRU free queue**. When a block is freed:
- If the request completed: block goes to the tail of the free queue
- If the block is outside the sliding window: block goes to the tail of the free queue

**Implication**: Blocks from all groups compete equally for memory. Frequently reused blocks (any group) stay in cache longer.

## Handling Different `kv_hidden_size` (Mamba Models)

### Problem

Mamba layers have state vectors with **different sizes** than attention KV caches. For example:
- Attention layer: `kv_hidden_size = 256 bytes`
- Mamba layer: `state_size = 4096 bytes`

If we allocate blocks with `page_size = block_size × 256`, Mamba layers need 16 blocks to store the same information as 1 attention block.

### Solution: Padding

Increase `block_size` for attention layers until:

```
block_size × kv_hidden_size_attn ≥ state_size_mamba
```

Then pad the Mamba state to `block_size × kv_hidden_size_attn`.

**Example**:

```
kv_hidden_size_attn = 256 bytes
state_size_mamba = 4096 bytes

Required block_size:
  block_size ≥ 4096 / 256 = 16 tokens

Set block_size = 16, page_size = 16 × 256 = 4096 bytes

Attention layers: 16 tokens per block (no padding)
Mamba layers: 1 state vector per block (4096 bytes, no padding)
```

**Tradeoff**: Very large block sizes (e.g., 400+) for attention layers when Mamba state is huge. This increases internal fragmentation. Alternative padding strategies under investigation.

## Coordinator Selection Logic

vLLM automatically selects the appropriate coordinator based on model configuration:

```python
if not enable_prefix_caching:
    coordinator = KVCacheCoordinatorNoPrefixCache(...)
elif len(kv_cache_groups) == 1:
    coordinator = UnitaryKVCacheCoordinator(...)
elif len(kv_cache_groups) == 2 and has_full_attention_group:
    coordinator = HybridKVCacheCoordinator(...)
else:
    raise NotImplementedError("Unsupported KV cache group configuration")
```

**Current limitations**:
- Exactly 2 groups required for `HybridKVCacheCoordinator`
- One group must be full attention
- Models with 3+ attention types not supported (disable prefix caching)

## Example: Gemma-2-9B (Simplified)

### Model Configuration

- 10 full attention layers
- 20 sliding window layers (window size = 4096)
- `kv_hidden_size = 256 bytes` (same for all)
- `block_size = 16`

### KV Cache Groups

```
Group 0: 10 full attention layers
Group 1: 10 sliding window layers (sw.0 - sw.9)
Group 2: 10 sliding window layers (sw.10 - sw.19)

page_size = 10 × 256 × 16 = 40,960 bytes
```

### Allocation for 8192-Token Request

```
Full attention group (Group 0):
  Needs 8192 tokens → 512 blocks

Sliding window group 1 (Group 1):
  Needs min(8192, 4096) = 4096 tokens → 256 blocks

Sliding window group 2 (Group 2):
  Needs min(8192, 4096) = 4096 tokens → 256 blocks

Total: 1024 blocks
```

**Memory savings vs. naive approach**:
- Naive (all full): 3 groups × 512 blocks = 1536 blocks
- Hybrid: 1024 blocks
- Savings: 33%

### Prefix Caching Example

Request 1 (8192 tokens, all blocks cached):

```
Group 0 (full): blocks 0-511 cached
Group 1 (sw): blocks 512-767 cached
Group 2 (sw): blocks 768-1023 cached
```

Request 2 (same 8192-token prefix):

```
Full attention cache hit (left-to-right):
  All blocks 0-511 cached → 8192 tokens

Sliding window cache hit (right-to-left, within 8192):
  Last 4096 tokens (blocks 512-767, 768-1023) cached
  → 8192 tokens (but only need to compute last 4096 with sliding window)
  
Actually: sliding window only needs last 4096 tokens cached
  → Cache hit length: 8192 - 4096 + (sliding_window - 1) = 4096 + 4095 = 8191
  → Prefill only token 8191

Effective cache hit: 8191 tokens
Prefill computation: 1 token (massive savings)
```

## Limitations and Future Work

### Supported Configurations

Currently supports:
- Full attention only (1 group)
- Sliding window only (1 group)
- Full + sliding window (2 groups)
- Full + Mamba (2 groups, Mamba prefix caching WIP)

**Not supported**:
- 3+ attention types (e.g., full + sliding + local)
- Multiple efficient attention types without full attention (e.g., sliding + Mamba only)

### Padding Overhead

- Case 3 (irregular layer counts): Up to ~20% padding overhead in last group
- Case 4 (Mamba): Can require block_size > 400, causing internal fragmentation

### Prefix Caching Limitations

- Right-to-left scanning for sliding window can iterate entire prompt on cache miss (overhead vs. full attention)
- Mamba prefix caching not yet implemented

### Performance Unknowns

- Memory fragmentation with highly variable request lengths
- Allocation overhead with many groups (e.g., 10+ groups)

## Implementation Files

Key components in vLLM codebase:

```
vllm/v1/core/kv_cache_manager.py          # KVCacheManager (top-level)
vllm/v1/core/kv_cache_coordinator.py       # Coordinators (no-cache, unitary, hybrid)
vllm/v1/core/single_type_kv_cache_manager.py  # Per-group managers
vllm/v1/core/kv_cache_spec.py              # KVCacheSpec, KVCacheGroup
```

## See Also

- [[KV Cache]] — Core block-based KV cache concept
- [[Prefix Caching]] — Hash-based prefix caching algorithm
- [[vLLM Engine]] — Scheduler integration
- [[V1 Architecture]] — Multi-process design and unified scheduler
- [[PagedAttention]] — Block-based attention kernel

## References

- vLLM design doc: `docs/design/hybrid_kv_cache_manager.md`
- vLLM PR introducing hybrid support: (check commit 458e74eb)
- Gemma 2 paper: [arxiv.org/abs/2408.00118](https://arxiv.org/abs/2408.00118) (sliding window)
- Jamba paper: [arxiv.org/abs/2403.19887](https://arxiv.org/abs/2403.19887) (Mamba + attention)
