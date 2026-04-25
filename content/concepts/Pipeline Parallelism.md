---
title: Pipeline Parallelism
type: concept
created: 2026-04-25
tags: [parallelism, distributed-inference, multi-node, micro-batching, bubble-overhead]
---

# Pipeline Parallelism

**Pipeline Parallelism (PP)** splits a model's layers sequentially across multiple GPUs or nodes, with each GPU/node responsible for a contiguous subset of layers. Unlike [[Tensor Parallelism]], which shards individual layers, PP partitions the entire model into stages and pipelines data through them.

## Core Mechanism

### Layer Partitioning

For a model with `L` layers and `pp_size=N`:

- **GPU 0:** Layers 0 to `L/N - 1`
- **GPU 1:** Layers `L/N` to `2L/N - 1`
- ...
- **GPU N-1:** Layers `(N-1)L/N` to `L - 1`

Each GPU executes its assigned layers, then passes activations to the next stage.

### Micro-Batching

To keep all GPUs busy, PP uses **micro-batching**:

1. Split a batch of `B` requests into `M` micro-batches (each size `B/M`).
2. **Forward pass:** GPU 0 processes micro-batch 1, passes to GPU 1; GPU 0 then processes micro-batch 2, etc.
3. **Backward pass (training):** Similar pipeline in reverse. (vLLM is inference-only, so no backward pass.)

**Example:** 4 GPUs, 8 micro-batches:

```
Time 0: GPU0 processes micro-batch 0
Time 1: GPU0 processes micro-batch 1, GPU1 processes micro-batch 0
Time 2: GPU0 processes micro-batch 2, GPU1 processes micro-batch 1, GPU2 processes micro-batch 0
...
```

### Bubble Overhead

**Bubble:** Periods when GPUs are idle, waiting for the pipeline to fill or drain.

- **Fill time:** First `N-1` micro-batches (GPUs 1 to N-1 are idle).
- **Drain time:** Last `N-1` micro-batches (GPU 0 is idle).
- **Bubble fraction:** `(N-1) / M` (ratio of idle to total micro-batches).

**Minimizing bubbles:**

- **Increase micro-batch count `M`:** More micro-batches → smaller bubble fraction. But too many micro-batches → increased scheduling overhead.
- **Larger batch size:** More requests per batch → more micro-batches naturally.

**Typical bubble overhead:** 10-20% for `M=8` and `N=2-4`. Acceptable for latency-tolerant workloads (offline batch inference).

## When to Use Pipeline Parallelism

Use PP when:

1. **Model layers exceed single-node GPU memory.** PP shards layers across nodes, enabling models that don't fit on one node.
2. **Cross-node deployment is required.** PP has lower communication overhead than [[Tensor Parallelism]] for PCIe/Ethernet connections (only activations are transferred, not all-reduce on every layer).
3. **GPUs lack high-bandwidth interconnects (NVLink).** PP transfers activations once per stage, while TP requires all-reduce on every layer. For GPUs without NVLink (e.g., L40S), PP is often faster.
4. **Uneven layer splits.** PP supports uneven partitions (e.g., GPU 0 has 10 layers, GPU 1 has 15 layers). [[Tensor Parallelism]] requires even splits.

**Trade-off:** PP increases latency (pipeline fill/drain) but reduces communication overhead compared to TP.

## Configuration

### Single-Node PP (Uneven GPU Splits)

If a model fits within a single node but requires all GPUs (and the GPU count doesn't evenly divide layers), use PP:

```bash
# 6 GPUs, uneven layer split
vllm serve gpt-j-6b \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 6
```

**Edge case:** GPUs without NVLink (e.g., L40S). Use PP instead of TP for higher throughput:

```bash
vllm serve facebook/opt-13b \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 4
```

### Multi-Node PP + TP

Combine PP (cross-node) with [[Tensor Parallelism]] (within-node):

```bash
# 2 nodes × 8 GPUs = 16 GPUs
vllm serve meta-llama/Llama-3-70b \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray
```

**Rule of thumb:** Set `tp_size` to the number of GPUs per node, and `pp_size` to the number of nodes.

### Multi-Node PP with Multiprocessing

For deployments without Ray, use native Python `multiprocessing`:

**Head node:**

```bash
vllm serve meta-llama/Llama-3-70b \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 0 \
  --master-addr <HEAD_NODE_IP>
```

**Worker node:**

```bash
vllm serve meta-llama/Llama-3-70b \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 1 \
  --master-addr <HEAD_NODE_IP> --headless
```

See Python Multiprocessing for tradeoffs between `fork`, `spawn`, and Ray.

## Communication Patterns

### Activation Transfer

PP transfers activations between stages. For a transformer with `hidden_size=H` and batch size `B`:

- **Forward pass:** Transfer tensor of shape `[B, seq_len, H]` from GPU `i` to GPU `i+1`.
- **Bandwidth requirement:** Lower than [[Tensor Parallelism]] (no all-reduce on every layer).

**Example:** GPT-3 (H=12288, B=32, seq_len=2048) transfers ~100 MB per stage. On Ethernet (10 Gbps), this takes ~80 ms. On InfiniBand (200 Gbps), ~4 ms.

### Communication Overhead

- **TP:** All-reduce on every layer (high frequency, small messages).
- **PP:** Activation transfer once per stage (low frequency, large messages).

**Comparison:**

| Interconnect | TP Overhead | PP Overhead |
|--------------|-------------|-------------|
| NVLink (900 GB/s) | 5% | 10% (bubble) |
| InfiniBand (200 GB/s) | 10-15% | 15-20% (bubble + transfer) |
| Ethernet (10 Gbps) | 50%+ (not viable) | 20-30% (bubble + transfer) |

**Takeaway:** For cross-node deployments with Ethernet or slow InfiniBand, PP is preferred over TP.

## Interaction with Other Parallelism Strategies

### PP + Data Parallelism

[[Data Parallelism]] replicates the model across separate PP pipelines. Total GPUs = `DP_size × PP_size × TP_size`.

Example: `DP=2, PP=2, TP=4` on 16 GPUs:

```bash
vllm serve $MODEL \
  --data-parallel-size 2 \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4
```

Each DP rank uses 8 GPUs (2 nodes × 4 GPUs TP). Requests are load-balanced across 2 DP ranks.

### PP + Expert Parallelism (MoE)

For [[Mixture of Experts]] models, PP splits expert layers across stages just like attention layers:

```bash
vllm serve deepseek-ai/deepseek-v3 \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel
```

Expert layers on GPU 0 (stage 0) communicate with expert layers on GPU 1 (stage 1) via activation transfer.

### PP + Context Parallelism

[[Context Parallelism]] is orthogonal to PP. CP shards sequences, PP shards layers. They can be combined:

```bash
vllm serve meta-llama/Llama-3-70b \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 4
```

## Memory and Performance Characteristics

### Memory Footprint

- **Weights:** Reduced by factor of `pp_size` (e.g., 2× reduction with PP=2). Each GPU stores `1/pp_size` of the model.
- **KV cache:** Stored only on the **last stage** (GPU with output layers). Earlier stages don't need KV cache for inference.
- **Activations:** Each stage stores activations for its layers (typically small compared to weights).

**KV cache implication:** The last GPU in the pipeline often has higher memory usage than earlier stages. Consider this when sizing GPUs or using [[Data Parallelism]].

### Latency Characteristics

- **Prefill (TTFT):** Increased by pipeline bubble overhead (10-20% typical). Acceptable for offline batch inference.
- **Decode (ITL):** Similar bubble overhead. PP is not ideal for low-latency online serving; prefer [[Tensor Parallelism]] or [[Data Parallelism]].

### Throughput

- **Batch size matters:** Larger batches → more micro-batches → smaller bubble fraction → higher throughput.
- **Optimal micro-batch count:** `M = 4 × pp_size` (rule of thumb). For `pp_size=2`, use 8 micro-batches.

## KV Cache Sizing

After deploying with PP, check the KV cache capacity:

```text
INFO 07-23 13:56:04 [kv_cache_utils.py:775] GPU KV cache size: 643,232 tokens
INFO 07-23 13:56:04 [kv_cache_utils.py:779] Maximum concurrency for 40,960 tokens per request: 15.70x
```

**Note:** KV cache is only on the last stage. The reported capacity reflects the last GPU's memory.

## Uneven Layer Splits

PP supports uneven partitions. Example: 40-layer model on 3 GPUs:

- GPU 0: Layers 0-12 (13 layers)
- GPU 1: Layers 13-26 (14 layers)
- GPU 2: Layers 27-39 (13 layers)

vLLM automatically partitions layers as evenly as possible. No manual configuration required.

**Use case:** Heterogeneous GPUs (e.g., A100 + V100). Assign more layers to the faster GPU.

## Multi-Node Deployment with Ray

See [[Tensor Parallelism#Multi-Node Deployment with Ray]] for Ray cluster setup. The steps are identical for PP.

**Deploy vLLM with PP:**

```bash
# 16 GPUs across 2 nodes (8 GPUs per node)
vllm serve meta-llama/Llama-3-70b \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray
```

## Troubleshooting

### Common Issues

1. **High latency (TTFT):** Bubble overhead. Increase batch size or use fewer pipeline stages.
2. **Imbalanced GPU memory (last GPU OOM):** KV cache is on the last stage. Reduce `max_num_seqs` or use [[Data Parallelism]] to spread KV cache.
3. **Slow activation transfer:** Check network bandwidth. Use InfiniBand for cross-node PP.
4. **Uneven layer distribution:** vLLM auto-partitions layers. Manual control is not supported.

### Debugging PP

```bash
# Enable verbose logs
vllm serve ... --verbose

# Look for:
# - "Pipeline stage 0: layers 0-X"
# - "Pipeline stage 1: layers X+1-Y"
```

## Cross-References

- [[Tensor Parallelism]] — Split individual layers across GPUs (higher communication overhead, lower latency)
- [[Data Parallelism]] — Replicate model across separate PP pipelines
- [[Expert Parallelism]] — Distribute MoE experts across stages
- [[Context Parallelism]] — Shard long sequences (orthogonal to PP)
- [[vLLM Engine]] — Scheduler and executor architecture that orchestrates PP stages
- Python Multiprocessing — Distributed executor backend tradeoffs (fork vs spawn vs Ray)
- [[Mixture of Experts]] — MoE architecture and PP interaction

## Key Metrics

- **Memory reduction:** `1 / pp_size` (e.g., 2× reduction with PP=2)
- **Bubble overhead:** 10-20% (typical for `M=8` micro-batches, `N=2-4` stages)
- **Communication overhead:** 15-30% on InfiniBand/Ethernet (lower than TP for slow interconnects)
- **Optimal micro-batch count:** `M = 4 × pp_size`
- **Typical PP sizes:** 2, 4 (multi-node), 6-8 (uneven single-node splits)

## Further Reading

- **Megatron-LM paper (includes PP):** [Shoeybi et al., 2019](https://arxiv.org/pdf/1909.08053.pdf)
- **GPipe (pipeline parallelism for training):** [Huang et al., 2019](https://arxiv.org/abs/1811.06965)
- **vLLM docs:** [Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html)
