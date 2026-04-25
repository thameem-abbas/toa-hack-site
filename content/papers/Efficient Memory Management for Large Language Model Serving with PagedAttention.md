---
title: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
type: paper
created: 2026-04-25
tags: [paged-attention, kv-cache, memory-management, vllm]
authors: ["Woosuk Kwon", "Zhuohan Li", "Siyuan Zhuang", "Ying Sheng", "Lianmin Zheng", "Cody Hao Yu", "Joseph E. Gonzalez", "Hao Zhang", "Ion Stoica"]
venue: SOSP 2023
year: 2023
url: https://arxiv.org/abs/2309.06180
---

# Efficient Memory Management for Large Language Model Serving with PagedAttention

Foundational paper introducing PagedAttention, a virtual memory-inspired approach to KV cache management that reduces memory waste by 60-80% and achieves 2-4× throughput improvement.

## Metadata

**Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica

**Affiliations:** UC Berkeley (Sky Computing Lab, RISELab)

**Venue:** ACM SIGOPS 29th Symposium on Operating Systems Principles (SOSP 2023)

**Published:** September 2023 (arXiv:2309.06180)

**Code:** [vLLM GitHub](https://github.com/vllm-project/vllm)

## Problem Statement

LLM serving systems waste 60-80% of GPU memory on KV cache due to:

1. **Fragmentation:**
   - **Internal:** Pre-allocate fixed buffers per request (e.g., max 2048 tokens), but actual sequences vary (e.g., 100-500 tokens) → wasted padding
   - **External:** Memory holes between allocations when requests complete

2. **Reserved memory:** Pre-reserve space for unknown future sequence lengths → conservative allocation → low utilization

3. **Existing solutions:**
   - Static batching: Pad all sequences to max length → high waste
   - Offloading: Swap KV cache to CPU → high latency (100ms+ for swap-in)

**Impact:** Low batch sizes (e.g., 8-16 sequences) → low GPU utilization → low throughput

## Key Insight

KV cache access patterns resemble OS virtual memory:
- KV cache for one sequence = large contiguous logical array
- Not all of this array is needed at once (generation is token-by-token)
- Can split into blocks and allocate non-contiguously (like OS pages)

**Analogy:**

| OS Virtual Memory | PagedAttention |
|-------------------|----------------|
| Page | Block (fixed # tokens) |
| Page table | Block table (logical → physical block mapping) |
| Page fault | Not applicable (pre-allocate blocks as tokens generated) |
| Shared pages | Shared blocks (for prefix caching, beam search) |

## PagedAttention Mechanism

### Block-Based Allocation

**Block:** Fixed-size memory unit storing KV cache for `BLOCK_SIZE` tokens (typically 16) at one attention head.

```
Block size = BLOCK_SIZE × HEAD_SIZE elements
Example: 16 tokens × 128 head_size = 2048 FP16 elements = 4KB per block per head
```

**Block table:** Per-request mapping from logical token index to physical block.

```python
# Example: 100-token sequence with BLOCK_SIZE=16
block_table = [5, 12, 3, 9, 17, 8, 22]  # 7 blocks (7 × 16 = 112 ≥ 100)
# Logical tokens 0-15 → physical block 5
# Logical tokens 16-31 → physical block 12
# ...
```

**Allocation:**
- New request: allocate 1 block (for first tokens)
- As tokens generated: allocate additional blocks from free pool
- Request completes: return all blocks to free pool (immediately reusable)

### Attention Kernel Modification

Standard attention requires contiguous KV cache:
```python
K = kv_cache[seq_id, :context_len, :]  # Requires contiguous array
attention_output = softmax(Q @ K.T) @ V
```

PagedAttention kernel accesses KV cache via block table:
```cpp
for block_idx in block_table:
    k_block = k_cache[block_idx]  # Non-contiguous access
    qk += Q @ k_block.T
attention_output = softmax(qk) @ V  # Iterate over V blocks similarly
```

**Key optimization:** Kernel designed for coalesced memory access despite non-contiguity. See [[PagedAttention#Kernel Implementation]] for details.

## Advanced Features

### Copy-on-Write for Shared Prefixes

Multiple sequences can share blocks for common prefixes:

```python
# Two requests with same system prompt
request_A_blocks = [1, 2, 3, 10, 15]  # Blocks 1-3 are shared prefix
request_B_blocks = [1, 2, 3, 12, 18]  # Blocks 1-3 are shared prefix
```

When a shared block needs modification (divergence point), copy block before writing (copy-on-write).

**Use cases:**
- System prompts (same for all requests)
- Few-shot examples (shared across similar tasks)
- Beam search (beam candidates share prefix)
- Parallel sampling (multiple samples share prompt)

### Memory Pool Management

Free blocks organized as free list. Allocation strategy:
- **Best-fit:** Allocate smallest block that fits (not applicable; all blocks same size)
- **First-fit:** Allocate first free block (what vLLM uses)

Preemption: When memory full, evict lowest-priority request, return blocks to free pool.

## Evaluation (Paper Results)

**Setup:**
- Models: OPT (13B, 66B, 175B), LLaMA (13B)
- Dataset: ShareGPT conversations (real user prompts + LLM responses)
- Baseline: Orca (FasterTransformer-based serving system)

**Metrics:**
- Throughput: requests/second
- Latency: time from request arrival to completion
- Memory utilization: fraction of GPU memory used for KV cache

### Throughput Gains

| Model | Baseline (req/s) | vLLM (req/s) | Speedup |
|-------|------------------|--------------|---------|
| OPT-13B | 0.8 | 2.3 | 2.9× |
| OPT-66B | 0.3 | 0.7 | 2.3× |
| OPT-175B | 0.1 | 0.3 | 3.0× |

**Why:** Higher batch sizes (32-64 vs. 8-16) due to memory efficiency.

### Memory Utilization

| Scenario | Baseline Waste | vLLM Waste |
|----------|----------------|------------|
| Variable seq lengths | 76% | 4% |
| Max seq length = 512 | 60% | 3.5% |
| Beam search (k=4) | 85% | 5% |

**Takeaway:** PagedAttention reduces waste from 60-85% to 3-5%.

### Prefix Caching Impact

Workload: 100 requests with 80% shared prefix (e.g., system prompt).

- **Without prefix caching:** 100 full prefill computations
- **With prefix caching:** 1 prefill computation (cached) + 99 reuses

**Result:** 3.5× throughput improvement for prefix-heavy workloads.

## Implementation (vLLM)

**Language:** Python (frontend), CUDA (kernels)

**Components:**
- Scheduler: Allocates blocks, manages request queue (see [[Continuous Batching]])
- Block allocator: Free list, allocation/deallocation
- Attention kernel: `paged_attention_kernel` in `csrc/attention/attention_kernels.cu`
- Tokenizer/detokenizer: HuggingFace transformers integration

**Evolution since paper:**
- V1 architecture: Multi-process design (see [[V1 Architecture]])
- FlashAttention integration: Uses FlashAttention-2 for many workloads
- FP8 KV cache: Quantized KV cache (8-bit storage)
- Chunked prefill: Break long prefills into chunks (see [[Chunked Prefill]])

## Limitations & Trade-offs

### Block Size Selection

- **Too small (e.g., 1 token/block):** High block table lookup overhead, more blocks to manage
- **Too large (e.g., 128 tokens/block):** Internal fragmentation (last block often partially filled)
- **Sweet spot:** 16-32 tokens/block (vLLM default: 16)

### Non-Contiguous Access Overhead

Block table lookup adds small overhead vs. contiguous memory. Mitigated by:
- CUDA kernel optimizations (coalesced access)
- Block-level parallelism (warps process entire blocks)

**Measured impact:** <5% latency increase vs. contiguous baseline (far outweighed by batching gains).

### Prefix Caching Overhead

- Hash computation for prefix matching
- Reference counting for shared blocks
- Copy-on-write when divergence occurs

**Measured impact:** <2% overhead when prefix caching not beneficial. 3-10× speedup when beneficial.

## Key Claims

1. **60-80% memory waste reduction:** Internal + external fragmentation eliminated via paging
2. **2-4× throughput improvement:** Higher batch sizes enabled by memory efficiency
3. **Near-zero additional latency:** Paging overhead <5% due to careful kernel design
4. **Production-ready:** vLLM serves models at scale (used by Anthropic, Meta, OpenAI, etc.)

## Impact & Follow-on Work

**Adoption:**
- vLLM: 2000+ contributors, de facto standard for open-source LLM serving
- Inspired TensorRT-LLM, SGLang, MLC-LLM, llama.cpp

**Extensions:**
- Disaggregated serving: Separate prefill/decode pools (see [[Splitwise]], [[DistServe]])
- Quantized KV cache: FP8, INT8 KV cache (vLLM V1 feature)
- Multi-tier caching: GPU → CPU → SSD cache hierarchy

**Criticisms/Limitations:**
- Block table lookup overhead (acknowledged but small)
- Complex kernel implementation (harder to extend than contiguous kernels)
- Prefix caching assumes hash-based matching (approximate matching harder)

## Cross-References

**Concepts:**
- [[PagedAttention]] — mechanism details, kernel implementation
- [[KV Cache]] — attention cache fundamentals
- [[Continuous Batching]] — enabled by dynamic block allocation
- [[Prefix Caching]] — copy-on-write shared blocks

**Architectures:**
- [[vLLM Engine]] — system implementing PagedAttention
- [[V1 Architecture]] — V1 refinements to original design

**Tools:**
- [[vLLM]] — implementation of paper's ideas

## Citation

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## See Also

- [Paper (arXiv)](https://arxiv.org/abs/2309.06180)
- [SOSP 2023 Talk](https://www.youtube.com/watch?v=80bIUggRJf4)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM Blog](https://blog.vllm.ai/)
