---
title: PagedAttention
type: concept
created: 2026-04-25
tags: [attention, memory-management, kv-cache, cuda-kernel]
---

# PagedAttention

Virtual memory-inspired approach to KV cache management for LLM serving, enabling non-contiguous memory allocation and near-zero fragmentation.

## Core Idea

Traditional LLM serving pre-allocates large contiguous memory buffers for each request's KV cache. This leads to:
- **Internal fragmentation:** Unused space within allocated buffers (e.g., padding for max sequence length)
- **External fragmentation:** Memory holes between allocations
- **Memory waste:** 60-80% of GPU memory can be wasted on padding/fragmentation

PagedAttention borrows virtual memory concepts from operating systems:
- KV cache divided into **blocks** (analogous to OS pages)
- Each block stores fixed number of tokens (`BLOCK_SIZE`, typically 16)
- Blocks allocated non-contiguously as needed
- Block table maps logical KV cache positions to physical memory blocks

**Benefits:**
- Near-zero memory fragmentation
- 2-4× throughput improvement (from memory efficiency → more concurrent requests)
- Enables [[Prefix Caching]] (shared blocks for common prefixes)

## Block Structure

**Block:** Fixed-size unit storing KV cache for one head.

```
Block size = BLOCK_SIZE tokens × HEAD_SIZE elements
Example: 16 tokens × 128 elements = 2048 elements per block per head
```

**Key cache layout:** `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
- `x` = elements processed by one thread group (e.g., 8)

**Value cache layout:** `[num_blocks, num_kv_heads, head_size, block_size]`

**Memory management:**
- Requests receive blocks from free pool as tokens generated
- Blocks returned to pool when request completes
- Block table per request maps logical token index → physical block

## Kernel Implementation (Historical)

> [!warning] Historical Document
> The description below is from the original 2023 paper. The actual kernel has evolved significantly since then. See `csrc/attention/attention_kernels.cu` for current implementation.

The PagedAttention kernel (`paged_attention_kernel`) performs multi-head query attention with paged KV caches.

### Template Parameters

```cpp
template<typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS, int PARTITION_SIZE = 0>
__device__ void paged_attention_kernel(...)
```

- `scalar_t`: data type (FP16, BF16, etc.)
- `HEAD_SIZE`: elements per attention head (e.g., 128)
- `BLOCK_SIZE`: tokens per block (e.g., 16)
- `NUM_THREADS`: threads per thread block
- `PARTITION_SIZE`: tensor parallel partitions (0 = disabled)

### Key Concepts

- **Sequence:** A client request (one query token per sequence in decode phase)
- **Context:** Previously generated tokens (KV cache)
- **Vec:** Elements fetched together (16 bytes at a time for coalesced memory access)
- **Thread group:** Small group of threads processing one Q-K pair (e.g., 2 threads)
- **Warp:** 32 threads executing simultaneously (processes Q vs. one block of K tokens)
- **Thread block:** `NUM_THREADS` threads sharing shared memory (processes Q vs. all context)
- **Grid:** `(num_heads, num_seqs, max_num_partitions)` thread blocks

### Execution Flow

1. **Query Load (Shared Memory)**
   - Each thread group loads its portion of query token
   - Stored in shared memory for reuse
   - Coalesced memory access (neighboring threads read neighboring memory)

2. **Key Iteration (Register Memory)**
   - Outer loop: iterate over blocks
   - Inner loop: iterate over tokens in block
   - Each thread loads key vecs into registers
   - `k_ptr` iterates across physical blocks via block table

3. **QK Dot Product**
   - Dot product between query and key vecs
   - Cross-thread-group reduction yields full QK score
   - Result: attention logit for one Q-K pair

4. **Softmax (Thread Block Reduction)**
   - Compute `qk_max` across all context tokens (warp-level then block-level reduction)
   - Compute `exp_sum = sum(exp(qk - qk_max))` (block-level reduction)
   - Normalize: `logits[i] = exp(qk[i] - qk_max) / exp_sum`

5. **Value Dot Product**
   - Outer loop: iterate over blocks
   - Inner loop: iterate over rows (head positions)
   - Each thread fetches `V_VEC_SIZE` elements from same tokens
   - Accumulate `accs[i] += dot(logits_vec, v_vec)`

6. **LV Reduction**
   - Warp-level reduction of `accs`
   - Thread-block-level reduction via shared memory
   - Final output: attention output for current head/sequence

7. **Output Write**
   - Write accumulated results from registers to global memory
   - One output per head per sequence

### Memory Access Patterns

**Coalesced access:** Neighboring threads read neighboring 16-byte chunks.
- Query: shared memory (reused by multiple threads)
- Key: register memory (one-time use per thread)
- Value: register memory (column-wise access, different rows per iteration)

**Optimization:** Vec-based fetching ensures 16-byte aligned reads for maximum memory bandwidth.

## Use Cases

### Continuous Batching

PagedAttention enables dynamic request batching by allowing:
- Requests to be added/removed from batch mid-flight
- Blocks freed when request completes → immediately reusable
- No need to preempt entire request (can preempt at token granularity)

### Prefix Caching

Shared prefixes (system prompts, few-shot examples) can share physical blocks:
- Block table for request A: `[0, 1, 2, 5, 7]` (blocks 0-2 are prefix)
- Block table for request B: `[0, 1, 2, 9, 11]` (blocks 0-2 shared)
- Copy-on-write semantics when prefix diverges

See [[Prefix Caching]] for details.

### Beam Search / Parallel Sampling

Multiple sequences can share prefix blocks (e.g., prompt blocks shared across beam candidates).

## Trade-offs

**Pros:**
- Near-zero memory waste
- Enables high request concurrency
- Dynamic memory allocation (no need to know max sequence length upfront)

**Cons:**
- Block table lookup overhead (mitigated by block-level parallelism)
- Non-contiguous memory access (mitigated by careful kernel design)

## Evolution Since 2023

The kernel described above is from the original paper. Current vLLM includes:
- **FlashAttention integration:** Uses FlashAttention-2 kernel for many workloads
- **xFormers backend:** Optional xFormers attention backend
- **Multi-query attention (MQA) / Grouped-query attention (GQA):** Efficient handling of models with fewer KV heads than Q heads
- **FP8 KV cache:** Quantized KV cache (see [[V1 Architecture#Features]])
- **Block size tuning:** Configurable via `--block-size` flag

## Cross-References

**Concepts:**
- [[KV Cache]] — attention cache fundamentals
- [[Prefix Caching]] — sharing blocks for common prefixes
- [[Continuous Batching]] — dynamic batching enabled by paged memory
- [[Chunked Prefill]] — breaking long prefills into chunks

**Architectures:**
- [[vLLM Engine]] — engine using PagedAttention
- [[V1 Architecture]] — V1 KV cache manager

**Papers:**
- [[Efficient Memory Management for Large Language Model Serving with PagedAttention]] — Kwon et al., SOSP 2023

**Tools:**
- [[vLLM]] — serving engine implementing PagedAttention

## See Also

- [vLLM Paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)
- [Paged Attention Kernel Source](https://github.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cu)
- [SOSP 2023 Presentation](https://www.youtube.com/watch?v=80bIUggRJf4)
