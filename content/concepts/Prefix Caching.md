---
title: Prefix Caching
type: concept
tags: [caching, optimization, memory-management]
related: [KV Cache, PagedAttention, vLLM Engine, LoRA, Disaggregated Serving]
---

# Prefix Caching

## Overview

Prefix caching (also called **Automatic Prefix Caching** or **APC** in vLLM) is a memory optimization that caches KV cache blocks of previously processed requests, allowing new requests to skip redundant computation if they share a common prefix with cached requests.

**Key insight**: Many inference workloads have natural prefix reuse:
- **Multi-turn chat**: conversation history is a shared prefix across turns
- **RAG (Retrieval-Augmented Generation)**: document context is prepended to many queries
- **Few-shot prompting**: instruction + examples form a shared prefix
- **Agent loops**: system prompts + tool definitions repeated across invocations

Instead of recomputing the KV cache for these shared prefixes every time, vLLM caches them and detects reuse via content-based hashing.

## Hash-Based Approach

vLLM uses **content-based hashing** to identify identical KV cache blocks. Each block is hashed based on:

1. **Parent hash**: The hash of the previous block (creates a hash chain)
2. **Block tokens**: The exact token IDs in this block
3. **Extra hashes**: Additional context to disambiguate:
   - **LoRA ID**: Different adapters produce different KV caches
   - **Multi-modal hashes**: Image/audio embeddings that replace placeholder tokens
   - **Cache salt**: Per-request isolation for multi-tenant security

### Hash Chaining Example

Given a sequence tokenized as `[A, B, C, D, E, F]` with block size 4:

```
Block 0: [A, B, C, D]
  hash = H(parent=None, tokens=[A, B, C, D], extras=[])

Block 1: [E, F, _, _]  (partially filled)
  hash = H(parent=hash(Block 0), tokens=[E, F], extras=[])
```

The parent hash ensures that **identical tokens in different contexts produce different hashes**. For example, the token sequence `[E, F]` after `[A, B, C, D]` has a different hash than `[E, F]` after `[X, Y, Z, W]`.

### Hash Algorithms

vLLM supports multiple hashing algorithms (configured via `--prefix-caching-hash-algo`):

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| `sha256` (default) | Cryptographically secure, uses Python pickle | General use, secure multi-tenancy |
| `sha256_cbor` | Secure + canonical serialization | Cross-version/cross-language reproducibility |
| `xxhash` | Fast non-cryptographic hash (128-bit) | Single-tenant, performance-critical |
| `xxhash_cbor` | Fast + canonical serialization | Single-tenant with reproducibility needs |

**Security note**: Non-cryptographic hashes (xxHash) have higher collision risk, which could leak information in multi-tenant environments via timing side-channels. Use SHA-256 variants for production multi-tenant deployments.

## Block Caching Rules

### Full Blocks Only

Only **completely filled blocks** are cached. Partially filled blocks are not cached because:
- They are mutable (more tokens may be appended)
- Their hash would change as they fill up
- They are typically at the end of a request (least likely to be reused)

### Cache Entry Example

With block size 4, a request with tokens `[A, B, C, D, E, F]` generates:

```
Time 0 (prefill):
  Block 0: [A, B, C, D] — FULL → cached immediately
  Block 1: [E, F]       — partial, not cached

Time 1 (decode, generate G):
  Block 1: [E, F, G]    — still partial, not cached

Time 2 (decode, generate H):
  Block 1: [E, F, G, H] — NOW FULL → cached
```

**Implication**: Prefixes must be at least one full block (e.g., ≥16 tokens) to benefit from caching.

## Multi-Modal Support

For requests with images, audio, or other embeddings:

### Placeholder Token Replacement

Multi-modal processors convert media to embeddings, replacing placeholder tokens:

```python
messages = [
    {"role": "user",
     "content": [
         {"type": "text", "text": "What's in this image?"},
         {"type": "image_url", "image_url": {"url": image_url}},
    ]},
]
```

Becomes tokenized as:

```
Text tokens: [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, 10, 4]
                    ↓ [IMG] token is replaced by 41 placeholders
With placeholders: [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <P>, <P>, ..., <P>, 4]
```

During prefill, placeholders are replaced with image embeddings. The KV cache from these embeddings must be distinguished from other placeholders.

### Multi-Modal Hash Injection

vLLM computes a hash of the image (or other media) and includes it in the `extra_hashes` for all blocks containing placeholders:

```
Block 0: tokens=[..., <P>, <P>, ...], extra_hashes=[image_hash_xyz]
Block 1: tokens=[<P>, <P>, ...],     extra_hashes=[image_hash_xyz]
Block 2: tokens=[<P>, ..., 4],       extra_hashes=[image_hash_xyz]
```

This ensures:
- Different images produce different hashes (even with identical placeholder patterns)
- Same image + same text reuses cached KV cache across requests

## Cache Isolation and Security

### Problem: Timing Side-Channels

In multi-tenant environments, prefix caching can leak information:
- Request A caches a sensitive prefix
- Request B (from different user) shares the same prefix
- Request B completes faster due to cache hit
- Attacker (user B) infers user A accessed the same content

### Solution: Cache Salt

vLLM supports per-request `cache_salt` to partition the cache:

```json
{
  "messages": [...],
  "cache_salt": "tenant-12345"
}
```

The salt is injected into the hash of the **first block**, ensuring:
- Only requests with the same salt can reuse blocks
- Different tenants/users get isolated cache partitions
- Performance impact: minimal (salt only hashed once per request)

**Use cases**:
- Multi-tenant SaaS: one salt per tenant
- Privacy-sensitive apps: one salt per user session
- Collaborative environments: shared salt for teams

## Workflow: Cache Hit Detection

### New Request Arrives

1. Scheduler tokenizes the prompt and chunks it into blocks
2. For each block (in order), scheduler computes the hash
3. Scheduler queries `cache_blocks` hash map: `block_hash → block_id`
4. If hit: increment `ref_cnt`, mark block as "computed"
5. If miss: stop checking (no further blocks can hit)
6. Return the **longest contiguous prefix** of cached blocks

### Allocation with Cache Hits

Given a request with 7 blocks, where blocks 0-4 are cached:

```
Blocks: [0-cached, 1-cached, 2-cached, 3-cached, 4-cached, 5-new, 6-new]

Allocation:
  - Touch blocks 0-4: increment ref_cnt, remove from free queue
  - Allocate blocks 5-6 from free queue
  - Prefill only tokens in blocks 5-6
```

**Memory savings**: Avoided prefill of 5 blocks (e.g., 5 × 16 = 80 tokens).

**Latency savings**: Prefill is typically 10-100x slower than decode per token, so skipping 80 tokens can reduce TTFT by 100-1000ms.

## Workflow: Block Eviction (LRU)

When memory pressure requires freeing a cached block:

1. Pop the **head** of the free queue (least recently used block)
2. If the block is cached (has a `block_hash`):
   - Remove `block_hash → block_id` mapping from `cache_blocks`
   - Clear the `block_hash` field
3. Reuse the physical block for a new allocation

**LRU invariant**: The free queue is ordered by recency. Recently used blocks are moved to the tail when "touched" (cache hit).

## Workflow: Request Completion

When a request finishes:

1. Decrement `ref_cnt` for all its blocks
2. For blocks with `ref_cnt == 0`:
   - Add them to the free queue (tail)
   - If they are full, keep their `block_hash` (stay cached)
   - If they are partial, clear their content (not cached)
3. **Reverse order insertion**: Add blocks to the tail in reverse (last block first)

**Rationale for reverse order**: Later blocks have longer prefix hashes (hash more tokens). They are less likely to match future requests, so should be evicted sooner.

### Example

Request 1 completes with blocks `[2, 3, 4, 8]`:

```
Before:
  Free queue: [head=5] → [6] → [7] → [tail=9]
  
After (reverse order):
  Free queue: [head=5] → [6] → [7] → [9] → [8] → [4] → [3] → [tail=2]
```

Blocks 2, 3, 4 remain cached. Block 8 (partial, end of request) is not cached.

## Duplicated Blocks (V1 Append-Only Block Tables)

In vLLM V0, duplicate cached blocks were deduplicated immediately. In V1, block tables are **append-only**, creating temporary duplicates.

### Example

Request 1 generates `[A, B, C, D, E, F, G, H, I]` (block size 4):

```
Time 0: Prefill [A-F], decode G
  Blocks: [0: ABCD (cached), 1: EFGH (cached)]

Time 1: Decode H
  Blocks: [0: ABCD, 1: EFGH (cached)]

Time 2: Decode I
  Blocks: [0: ABCD, 1: EFGH, 2: I (partial)]
```

Request 2 (same prompt, greedy sampling → same outputs):

```
Time 0: Prefill [A-F], decode G
  Blocks: [0: ABCD (reuse), 3: EFGH (NEW, duplicate of block 1)]
  Cache: {hash(ABCD): 0, hash(EFGH): 1, hash(EFGH): 3}  ← duplicate!

Time 1: Decode H
  Blocks: [0: ABCD, 3: EFGH (cached again)]
```

Block 3 is a duplicate of block 1 (both hash to `EFGH` with the same prefix). In V0, we'd change Request 2's block table from `[0, 3]` to `[0, 1]`. In V1, we **keep the duplicate** and clean it up when the request finishes.

**Impact**: Slight memory waste during overlapping identical requests. Acceptable because:
- Duplicates are rare (requires greedy sampling or identical random seeds)
- Cleanup happens automatically on request completion
- Avoids block table mutations (simplifies V1 scheduler)

## Interaction with Sliding Window Attention

For models with sliding window attention (e.g., Mistral 7B, Gemma 2):

### Cache Hit Strategy

- **Full attention layers**: Cache hit requires all tokens in the prefix to be cached
- **Sliding window layers**: Cache hit requires only the last `sliding_window_size - 1` tokens to be cached

### Intersection Algorithm

For hybrid models (full + sliding window):

1. Find longest cache hit for **full attention** (scan left to right)
2. Find longest cache hit for **sliding window** within that length (scan right to left)
3. Return the intersection

**Example** (sliding window size = 4, block size = 1):

```
Prompt: 15 tokens, blocks [0-14]
Cached blocks: [0, 1, 11, 12, 13] (blue)

Full attention cache hit:
  Check 0: cached → continue
  Check 1: cached → continue
  Check 2: NOT cached → STOP
  Result: 2 tokens cached

Sliding window cache hit (within length 2):
  Check from right: token 1 is cached
  Result: 2 tokens cached

Final cache hit length: 2 tokens
```

If all blocks 0-14 were cached:

```
Full attention: 15 tokens cached
Sliding window (scan from right):
  Check 14: cached → check 13: cached → check 12: cached → check 11: cached
  Last 4 tokens cached (11-14) → can skip to token 14
  Result: 14 tokens cached (need to compute prefill with tokens [11, 12, 13] → [14])
```

See [[Hybrid KV Cache Manager]] for details.

## Performance Characteristics

### When Prefix Caching Helps

**High benefit**:
- Multi-turn chat (shared conversation history)
- RAG with fixed documents (shared context)
- Few-shot prompting (shared examples)
- Agent loops (shared system prompt + tools)
- Batch jobs with common prefix (e.g., instruction tuning)

**Typical gains**:
- **TTFT reduction**: 2-10x for long shared prefixes (e.g., 1000+ token documents)
- **Throughput increase**: 2-5x when most requests share prefixes
- **Effective capacity**: Can serve more requests with same GPU memory

### When Prefix Caching Doesn't Help

**No benefit**:
- Unique prompts with no sharing
- Short prompts (<16 tokens, less than one block)
- Decode-heavy workloads (long completions, little prefill)
- Highly variable multi-modal content (different images every request)

**Overhead**:
- Hash computation: ~10-50 µs per block (negligible)
- Hash map lookups: ~1 µs per block
- Memory overhead: ~1-2% for hash maps and metadata

## Implementation Details

### Data Structures

```python
# Block pool (pre-allocated at init)
block_pool: list[KVCacheBlock]

# Free queue (LRU eviction)
free_queue_head: KVCacheBlock | None
free_queue_tail: KVCacheBlock | None

# Cache lookup
cache_blocks: dict[BlockHash, int]  # block_hash → block_id

# Request tracking
request_blocks: dict[str, list[int]]  # request_id → [block_ids]
```

### Block State Machine

```
┌─────────┐
│  FREE   │ ← Initial state (in free queue)
└────┬────┘
     │ allocate()
     ↓
┌─────────┐
│ALLOCATED│ ← ref_cnt > 0, not cached
└────┬────┘
     │ block fills up
     ↓
┌─────────┐
│ CACHED  │ ← ref_cnt > 0, block_hash set, in cache_blocks
└────┬────┘
     │ ref_cnt → 0
     ↓
┌─────────┐
│CACHED   │ ← ref_cnt == 0, in free queue + cache_blocks
│+ FREE   │    (can be reused OR evicted)
└────┬────┘
     │ evict() or reuse()
     ↓
┌─────────┐
│  FREE   │ ← block_hash cleared, only in free queue
└─────────┘
```

## API Usage

### Enable Prefix Caching

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    enable_prefix_caching=True,
    block_size=16,
)
```

### Set Cache Salt (Multi-Tenant)

```python
from vllm import SamplingParams

# Tenant A
response_a = llm.generate(
    "Summarize this document: ...",
    SamplingParams(cache_salt="tenant-A")
)

# Tenant B (isolated, cannot reuse tenant A's cache)
response_b = llm.generate(
    "Summarize this document: ...",  # same prompt!
    SamplingParams(cache_salt="tenant-B")
)
```

### Configure Hash Algorithm

```bash
# Server mode
vllm serve meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256_cbor  # reproducible hashing
```

## Limitations

### V1 Append-Only Block Tables

In vLLM V1, block tables cannot be mutated after creation. This causes:
- **Duplicate cached blocks** for identical requests with greedy sampling
- Cleaned up on request completion
- Slight temporary memory waste

### Partial Blocks Not Cached

The last block of a request is often partial and not cached. This means:
- Very short prompts (<16 tokens) get no caching benefit
- Decode-phase tokens (generated one at a time) are not cached until the block fills

### Hash Collisions

While extremely rare with SHA-256, hash collisions could cause:
- Wrong KV cache reused (incorrect outputs)
- Mitigated by including exact token IDs in hash (not just a hash of tokens)

## See Also

- [[KV Cache]] — The underlying memory structure that prefix caching optimizes
- [[PagedAttention]] — Block-based attention kernel that enables efficient caching
- [[Hybrid KV Cache Manager]] — Per-layer caching for models with mixed attention types
- [[vLLM Engine]] — Scheduler integration with prefix caching
- [[Disaggregated Serving]] — Prefix caching across prefill/decode pools

## References

- vLLM design doc: `docs/design/prefix_caching.md`
- vLLM feature doc: `docs/features/automatic_prefix_caching.md`
- SGLang (also uses hash-based prefix caching): [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)
