---
title: Context Parallelism
type: concept
created: 2026-04-25
tags: [parallelism, long-context, kv-cache, ring-attention, prefill, decode]
---

# Context Parallelism

**Context Parallelism (CP)** shards long input sequences across multiple GPUs to enable efficient serving of requests that exceed the KV cache capacity of a single GPU. Unlike [[Tensor Parallelism]] (which shards layers) or [[Pipeline Parallelism]] (which splits stages), CP partitions the **sequence dimension** of the KV cache and query/key/value tensors.

## Problem Statement

As context lengths grow (8k → 32k → 128k+ tokens), two bottlenecks emerge:

1. **Prefill latency (TTFT):** Computing attention for a 100k-token prompt is slow. Amortizing computation across GPUs reduces TTFT.
2. **Decode throughput:** Storing KV cache for 100k tokens per request limits batch size. Sharding KV cache across GPUs increases capacity.

CP addresses both problems, but with different strategies for prefill vs decode.

## Prefill Context Parallel

### Objective

Reduce **Time to First Token (TTFT)** by parallelizing attention computation across query tokens.

### Strategies

For a long request with `T` new tokens and `N` GPUs:

#### 1. Partial Query, Full KV (Ring Attention - under development)

**Scenario:** Moderately long context (10k-50k tokens). Goal is to accelerate prefill.

**Approach:**

1. Split query into `N` chunks: Each GPU computes query/key/value for `T/N` tokens.
2. **Gather full KV:** All GPUs gather key/value tensors from all other GPUs (all-gather).
3. **Compute attention:** Each GPU computes attention output for its query chunk against the full KV.

**Communication:** One all-gather per layer (to gather full KV).

**Memory:** Full KV tensors must fit on each GPU.

**Speedup:** ~`N × TTFT` (ideal). Practical: 0.7-0.9N due to all-gather overhead.

**Status:** Under active development. Not yet available in stable vLLM.

#### 2. Partial Query, Partial KV (Ring Attention)

**Scenario:** Very long context (100k+ tokens). Cannot afford holding full KV on each GPU.

**Approach:**

1. Split query/key/value into `N` chunks: Each GPU computes `T/N` tokens.
2. **Ring communication:** Use [ring attention](http://arxiv.org/abs/2310.01889) to send/receive KV chunks between GPUs in a ring pattern.
3. **Incremental attention:** Each GPU computes attention incrementally as KV chunks arrive.

**Communication:** Multiple send/recv rounds (ring-style). Higher overhead than strategy 1.

**Memory:** Each GPU stores only `1/N` of the KV cache. Enables arbitrarily long contexts.

**Speedup:** Lower than strategy 1 due to communication overhead. Typical: 0.5-0.7N.

**Status:** Under active development. Not yet available in stable vLLM.

### Prefill CP Status

> [!note] Development Status
> Both prefill CP strategies are **under active development** (as of April 2025). Not yet available in stable vLLM releases. Track progress in the `#sig-context-parallel` channel of [vLLM Slack](https://slack.vllm.ai/).

## Decode Context Parallel

### Objective

Increase **KV cache capacity** to serve more requests or longer contexts during decode.

### Core Idea

Shard the KV cache along the **sequence dimension** (T) across GPUs. Each GPU stores `1/N` of the KV cache for each request.

### KV Cache Sharding

For a model with `H` KV heads and a request with `T` tokens:

- **KV cache size:** `H × T` key/value tensors.
- **Tensor parallelism alone:** Shard along KV head dimension (`H`). Each GPU stores `H/TP_size × T` tensors.

**Problem:** If `TP_size > H`, KV cache is **duplicated** across GPUs.

**Example:** DeepSeek-R1 with MLA (1 KV head), TP=8 → 8× KV cache duplication. Each GPU stores the full `1 × T` KV cache.

### Solution: Decode Context Parallel (DCP)

DCP further shards the KV cache along the **sequence dimension** (`T`):

- **DCP size:** `1 ≤ DCP_size ≤ TP_size / H`.
- **Per-GPU KV cache:** `(H / TP_size) × (T / DCP_size)` tensors.

**Example:** DeepSeek-R1, TP=8, DCP=8:

- KV heads per GPU: `1 / 8` (duplicated 8× without DCP).
- With DCP=8: `(1 / 8) × (T / 8) = 1/64` of total KV cache per GPU.
- **Duplication factor:** `TP_size / (H × DCP_size) = 8 / (1 × 8) = 1` (no duplication).

### Interleaving Strategy

KV cache grows during decoding (new tokens are generated). To ensure future tokens are naturally sharded, vLLM uses an **interleaving strategy** along the `T` dimension:

- **GPU 0:** Stores tokens 0, N, 2N, 3N, ...
- **GPU 1:** Stores tokens 1, N+1, 2N+1, 3N+1, ...
- **GPU N-1:** Stores tokens N-1, 2N-1, 3N-1, ...

**Proposed by:** [Chao Hong from Moonshot](https://github.com/youzhedian). Detailed in [this paper](http://arxiv.org/abs/2507.07120).

### Configuration

```bash
# DeepSeek-R1: 1 KV head, TP=8 → 8× duplication
# Add DCP=8 to eliminate duplication
vllm serve deepseek-ai/deepseek-r1 \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 8

# Kimi-K2: 1 KV head, TP=16 → 16× duplication
# DCP=16: no duplication (higher communication overhead)
vllm serve kimi/kimi-k2 \
  --tensor-parallel-size 16 \
  --decode-context-parallel-size 16

# DCP=8: 2× duplication (lower communication overhead, intra-node only)
vllm serve kimi/kimi-k2 \
  --tensor-parallel-size 16 \
  --decode-context-parallel-size 8

# Qwen3-235B-A22B: 4 KV heads, TP=8 → 2× duplication
# DCP=2 eliminates duplication
vllm serve Qwen/Qwen3-235B-A22B \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 2
```

**CLI shorthand:** `-tp <size>` for `--tensor-parallel-size`, `-dcp <size>` for `--decode-context-parallel-size`.

### DCP Size Bounds

- **Minimum:** `DCP_size = 1` (no CP, same as standard TP).
- **Maximum:** `DCP_size = TP_size / H` (eliminates all duplication).

**Why upper bound?** For `DCP_size > TP_size / H`, the KV cache is fully sharded, but non-attention layers (e.g., MLPs) have idle GPUs. For simplicity, vLLM caps DCP at `TP_size / H`.

**To accelerate decode further:** Increase `TP_size` first, then increase `DCP_size`.

## Case Studies

### DeepSeek-R1 (1 KV Head)

**Architecture:** MLA (Multi-head Latent Attention) with 1 KV head.

**Deployment:** TP=8 (single node, 8 GPUs).

**Without DCP:**

- KV cache duplication: 8× (each GPU stores the full KV cache).
- Wasted memory: 7/8 of KV cache memory is redundant.

**With DCP=8:**

- KV cache duplication: 1× (no redundancy).
- Communication overhead: All-gather on every decode step (across 8 GPUs).

**Recommendation:** Use DCP=8 for memory-bound workloads (many long requests). Omit DCP for latency-sensitive workloads (communication overhead).

### Kimi-K2 (1 KV Head, Multi-Node)

**Architecture:** Similar to DeepSeek-R1, but larger (requires multi-node).

**Deployment:** TP=16 (2 nodes × 8 GPUs).

**Without DCP:**

- KV cache duplication: 16×.

**With DCP=16:**

- KV cache duplication: 1×.
- Communication: Cross-node all-gather (slower than intra-node).

**With DCP=8:**

- KV cache duplication: 2×.
- Communication: Intra-node only (GPUs 0-7 on node 0, GPUs 8-15 on node 1).
- **Trade-off:** 2× duplication acceptable, but communication is faster.

**Recommendation:** Use DCP=8 for lower latency, DCP=16 for maximum memory efficiency.

### Qwen3-235B-A22B (4 KV Heads)

**Architecture:** GQA (Grouped Query Attention) with 4 KV heads.

**Deployment:** TP=8 (single node, 8 GPUs).

**Without DCP:**

- KV cache duplication: `TP_size / H = 8 / 4 = 2×` (each GPU stores 2 copies).

**With DCP=2:**

- KV cache duplication: 1×.
- Communication overhead: All-gather across 2 GPUs (minimal).

**Recommendation:** Always use DCP=2 for this model. Minimal communication overhead, eliminates duplication.

## When to Use Context Parallelism

Use decode CP when:

1. **KV cache duplication is high.** Models with 1-2 KV heads (MLA, GQA) on large TP groups.
2. **Memory-bound workloads.** Many long requests, or very long context lengths (100k+ tokens).
3. **TP size exceeds KV head count.** `TP_size / H > 1` → duplication. Add DCP to reduce.

**Do not use CP if:**

- `TP_size ≤ H` (no duplication).
- Latency is critical and memory is plentiful (communication overhead outweighs duplication cost).

## Communication Overhead

### Decode CP

Every decode step requires **all-gather** across `DCP_size` GPUs to reconstruct the full KV cache for attention:

- **Intra-node (NVLink):** ~1-2% overhead for DCP=8.
- **Cross-node (InfiniBand):** ~5-10% overhead for DCP=8-16.

**Trade-off:** Communication overhead vs memory savings.

### Prefill CP (Future)

Ring attention (strategy 2) has higher communication overhead:

- **Intra-node:** ~10-20% overhead for DCP=8.
- **Cross-node:** ~20-40% overhead for DCP=8.

## Interaction with Other Parallelism Strategies

### CP + Tensor Parallelism

CP is implemented **on top of TP**. The DCP size is bounded by `TP_size / H`.

```bash
# TP=8, DCP=4 (Qwen3-235B-A22B with 2 KV heads)
vllm serve Qwen/Qwen3-235B-A22B \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 4
```

**KV cache layout:**

- TP shards along KV head dimension: Each GPU stores `2 / 8 = 0.25` KV heads.
- DCP shards along sequence dimension: Each GPU stores `1 / 4 = 0.25` of the sequence.
- Per-GPU KV cache: `0.25 × 0.25 = 0.0625` of total KV cache.

### CP + Data Parallelism

CP is orthogonal to [[Data Parallelism]]. Each DP rank can use CP independently:

```bash
# DP=2, TP=8, DCP=8 on 16 GPUs
vllm serve deepseek-ai/deepseek-r1 \
  --data-parallel-size 2 \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 8
```

**GPU layout:**

- DP rank 0: GPUs 0-7 (TP=8, DCP=8).
- DP rank 1: GPUs 8-15 (TP=8, DCP=8).

### CP + Pipeline Parallelism

CP is orthogonal to [[Pipeline Parallelism]]. Each PP stage can use CP:

```bash
# TP=8, PP=2, DCP=8 on 16 GPUs
vllm serve deepseek-ai/deepseek-r1 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --decode-context-parallel-size 8
```

**GPU layout:**

- PP stage 0: GPUs 0-7 (TP=8, DCP=8, layers 0-N/2).
- PP stage 1: GPUs 8-15 (TP=8, DCP=8, layers N/2-N).

### CP + Multi-Token Prediction (MTP)

Some attention backends support combining decode CP with [[Multi-Token Prediction]] (MTP):

- **MTP:** Model predicts multiple tokens per forward pass (e.g., DeepSeek-V3).
- **CP + MTP:** CP shards KV cache, MTP reduces decode steps.

**Speedup:** CP enables larger batch sizes (more KV cache capacity), MTP reduces latency. Combined: 1.5-2× decode speedup.

## Memory and Performance Characteristics

### Memory Savings

- **Without DCP:** KV cache duplication factor = `TP_size / H`.
- **With DCP:** Duplication factor = `TP_size / (H × DCP_size)`.

**Example:** DeepSeek-R1, TP=8, DCP=8, H=1:

- Without DCP: 8× duplication → 100 GB KV cache.
- With DCP=8: 1× duplication → 12.5 GB KV cache.
- **Savings:** 87.5 GB per GPU.

### Communication Overhead

- **Intra-node (NVLink):** 1-2% for DCP=8.
- **Cross-node (InfiniBand):** 5-10% for DCP=8-16.

### Throughput Impact

- **Positive:** Larger batch sizes (more KV cache capacity) → higher throughput.
- **Negative:** Communication overhead → lower per-request throughput.

**Net impact:** Typically 0-10% throughput reduction at high batch sizes (communication overhead), but 2-4× throughput increase at low batch sizes (can now fit more requests).

## Troubleshooting

### Common Issues

1. **Invalid DCP size:** `DCP_size > TP_size / H`. Reduce DCP or increase TP.
2. **High communication overhead:** DCP is cross-node. Use DCP=`TP_size/nodes` to keep communication intra-node.
3. **No memory savings:** `TP_size ≤ H` (no duplication). DCP has no effect. Remove DCP.

### Debugging CP

```bash
# Enable verbose logs
vllm serve ... --verbose

# Look for:
# - "Decode context parallel size: X"
# - "KV cache duplication factor: Y"
# - "All-gather time: Z ms"
```

## Cross-References

- [[Tensor Parallelism]] — CP is implemented on top of TP (DCP shards beyond TP's KV head sharding)
- [[Data Parallelism]] — Each DP rank can use CP independently
- [[Pipeline Parallelism]] — Each PP stage can use CP
- [[KV Cache]] — CP shards the KV cache along the sequence dimension
- [[Mixture of Experts]] — MLA models (DeepSeek, Kimi) benefit most from CP (1 KV head)
- [[Multi-Token Prediction]] — Some attention backends support CP + MTP for combined speedup
- [[vLLM Engine]] — Scheduler and executor orchestrate CP communication

## Key Metrics

- **Prefill CP:** Under development (no metrics yet).
- **Decode CP:**
  - **Memory savings:** `1 - (1 / DCP_size)` (e.g., 87.5% for DCP=8).
  - **Communication overhead:** 1-2% (intra-node), 5-10% (cross-node).
  - **Typical DCP sizes:** 2, 4, 8, 16 (bounded by `TP_size / H`).
  - **Duplication factor:** `TP_size / (H × DCP_size)`.

## Further Reading

- **Ring Attention paper:** [Liu et al., 2023](http://arxiv.org/abs/2310.01889)
- **Interleaving strategy paper:** [Hong et al., 2025](http://arxiv.org/abs/2507.07120)
- **vLLM docs:** [Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment.html)
- **vLLM Slack:** `#sig-context-parallel` channel for discussions

## Recommended Strategy

1. **First, increase TP size** until you get satisfactory performance.
2. **Then, add DCP** to reduce KV cache duplication (if `TP_size > H`).
3. **Start with DCP = min(TP_size / H, 8)** to balance memory savings and communication overhead.
4. **For cross-node deployments,** use DCP = GPUs per node to keep communication intra-node.
