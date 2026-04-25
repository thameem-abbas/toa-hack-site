---
title: Tensor Parallelism
type: concept
created: 2026-04-25
tags: [parallelism, distributed-inference, megatron-lm, all-reduce, nccl]
---

# Tensor Parallelism

**Tensor Parallelism (TP)** splits individual model layers (specifically, weight matrices) across multiple GPUs within a single node or across nodes with high-bandwidth interconnects. Each GPU computes only a portion of the layer, then synchronizes intermediate results via collective communication (all-reduce).

## Core Mechanism

vLLM implements the Megatron-LM tensor parallel algorithm ([Shoeybi et al., 2019](https://arxiv.org/pdf/1909.08053.pdf)):

- **Column-parallel linear layers:** Split weight matrix along the column dimension. Each GPU computes a subset of output features. No communication required after the layer.
- **Row-parallel linear layers:** Split weight matrix along the row dimension. Each GPU computes partial sums, then performs an all-reduce to sum results across GPUs.

### Example: Multi-Layer Perceptron (MLP)

For a two-layer MLP with `hidden_size=8192` and `tp_size=4`:

1. **Column-parallel layer 1:** Each GPU computes 2048 output features. No communication.
2. **Activation function:** Each GPU applies activation independently. No communication.
3. **Row-parallel layer 2:** Each GPU computes partial output, then all-reduce combines results.

**Communication pattern:** One all-reduce per transformer block (after row-parallel output projection).

## When to Use Tensor Parallelism

Use TP when:

1. **Model does not fit on a single GPU.** TP shards weights, reducing per-GPU memory footprint.
2. **Single-node, multi-GPU deployment.** TP requires high-bandwidth, low-latency interconnects (NVLink, InfiniBand). Typical configuration: `tp_size = num_gpus_per_node` (4 or 8 GPUs).
3. **Low communication overhead is critical.** TP introduces one all-reduce per layer. High-bandwidth networks (NVLink: 900 GB/s, InfiniBand: 200 GB/s) minimize overhead.

**Edge case:** If GPUs lack NVLink (e.g., L40S), use [[Pipeline Parallelism]] instead. PP has lower communication overhead for PCIe-only connections.

## Configuration

### Single-node TP

```python
from vllm import LLM
llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
output = llm.generate("San Francisco is a")
```

```bash
vllm serve facebook/opt-13b --tensor-parallel-size 4
```

### Multi-node TP + PP

For very large models, combine TP (within-node) with [[Pipeline Parallelism]] (cross-node):

```bash
# 2 nodes × 8 GPUs = 16 GPUs total
vllm serve gpt-j-6b \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray
```

**Rule of thumb:** Set `tp_size` to the number of GPUs per node, and `pp_size` to the number of nodes.

## Communication Optimization

### GPUDirect RDMA

GPUDirect RDMA (Remote Direct Memory Access) allows network adapters to directly access GPU memory, bypassing the CPU. This reduces latency and CPU overhead for cross-node TP.

**Docker setup:**

```bash
docker run --gpus all \
  --ipc=host \
  --shm-size=16G \
  -v /dev/shm:/dev/shm \
  vllm/vllm-openai
```

**Kubernetes setup:**

```yaml
spec:
  containers:
    - name: vllm
      securityContext:
        capabilities:
          add: ["IPC_LOCK"]
      volumeMounts:
        - mountPath: /dev/shm
          name: dshm
  volumes:
    - name: dshm
      emptyDir:
        medium: Memory
```

**Verify GPUDirect RDMA is active:**

```bash
NCCL_DEBUG=TRACE vllm serve ...
```

Look for `[send] via NET/IB/GDRDMA` in logs (efficient). If you see `[send] via NET/Socket`, NCCL is using TCP (inefficient for TP).

### InfiniBand Configuration

For InfiniBand adapters:

```bash
# Add to run_cluster.sh helper script
--privileged -e NCCL_IB_HCA=mlx5
```

Consult your system administrator for the correct `NCCL_IB_HCA` value.

## Distributed Executor Backends

vLLM supports two backends for managing TP workers:

1. **Python `multiprocessing` (default for single-node):** Fastest startup, no external dependencies. Use `--distributed-executor-backend mp`.
2. **Ray (default for multi-node):** Cluster management, fault tolerance, resource scheduling. Use `--distributed-executor-backend ray`.

Override the default:

```bash
vllm serve facebook/opt-13b \
  --tensor-parallel-size 4 \
  --distributed-executor-backend ray
```

See [[Python Multiprocessing]] for tradeoffs between `fork`, `spawn`, and `forkserver` methods.

## Interaction with Other Parallelism Strategies

### TP + Data Parallelism

[[Data Parallelism]] replicates the model across separate TP groups. Total GPUs = `DP_size × TP_size`.

Example: `DP=4, TP=2` on 8 GPUs:

```bash
vllm serve $MODEL --data-parallel-size 4 --tensor-parallel-size 2
```

Each DP rank uses 2 GPUs for TP. Requests are load-balanced across the 4 DP ranks.

### TP + Expert Parallelism (MoE)

For [[Mixture of Experts]] models, TP shards both attention and expert layers. By default, expert layers form a TP group of size `TP_size`. To use [[Expert Parallelism]] (EP) instead:

```bash
vllm serve deepseek-ai/deepseek-v3 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel
```

With EP enabled, expert layers form an EP group of size `DP_size × TP_size`, while attention layers remain TP-sharded.

### TP + Context Parallelism

[[Context Parallelism]] (CP) shards long sequences across GPUs. For models with few KV heads (e.g., DeepSeek-R1 with 1 KV head), TP duplicates the KV cache. Use CP to eliminate duplication:

```bash
# DeepSeek-R1: 1 KV head, TP=8 → 8× KV cache duplication
vllm serve deepseek-ai/deepseek-r1 \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 8  # Removes duplication
```

**Rule:** `dcp_size ≤ tp_size / num_kv_heads`. See [[Context Parallelism]] for details.

## Memory and Performance Characteristics

### Memory Footprint

- **Weights:** Reduced by factor of `tp_size` (e.g., 4× reduction with TP=4).
- **KV cache:** Sharded along KV head dimension (no reduction if `num_kv_heads < tp_size`; see [[Context Parallelism]]).
- **Activations:** Sharded along feature dimension (reduced by `tp_size`).

### Compute Overhead

- **All-reduce latency:** One all-reduce per transformer block. On NVLink (900 GB/s), overhead is <5% for typical models. On InfiniBand (200 GB/s), overhead is 5-15%.
- **Scalability:** TP scales well up to 8 GPUs per node (NVLink limit). Beyond 8 GPUs, use [[Pipeline Parallelism]] or [[Data Parallelism]].

### Throughput vs Latency

- **Prefill (TTFT):** TP improves throughput but does not reduce latency per token (all-reduce is a synchronization point).
- **Decode (ITL):** TP improves throughput for large batch sizes. For small batches, single-GPU may be faster due to communication overhead.

## KV Cache Sizing

After deploying with TP, check the KV cache capacity:

```text
INFO 07-23 13:56:04 [kv_cache_utils.py:775] GPU KV cache size: 643,232 tokens
INFO 07-23 13:56:04 [kv_cache_utils.py:779] Maximum concurrency for 40,960 tokens per request: 15.70x
```

- `GPU KV cache size`: Total tokens that can be stored across all GPUs.
- `Maximum concurrency`: Number of concurrent requests (assuming `max_model_len` tokens per request).

If capacity is insufficient, add more GPUs or reduce `max_num_seqs`.

## Kernel Fusions and TP

[[Kernel Fusions]] can combine TP all-reduce with other operations:

- **AllReduce+RMSNorm:** Fuses all-reduce with RMSNorm, reducing kernel launches. 5-20% speedup on Hopper/Blackwell (requires TP>1).
- **AsyncTP GEMM+Collective:** Overlaps GEMM computation with all-reduce communication. 7-10% speedup for high token counts.

Enable fusions via [[Optimization Levels]]:

```bash
vllm serve $MODEL --tensor-parallel-size 4 --compilation-config '{"level": 2}'
```

See [[Kernel Fusions]] for fusion-specific flags.

## Multi-Node Deployment with Ray

For models that exceed single-node capacity, deploy across multiple nodes using Ray:

### Ray Cluster Setup

1. **Head node:**

```bash
bash run_cluster.sh vllm/vllm-openai <HEAD_NODE_IP> --head \
  /path/to/huggingface/cache -e VLLM_HOST_IP=<HEAD_NODE_IP>
```

2. **Worker nodes:**

```bash
bash run_cluster.sh vllm/vllm-openai <HEAD_NODE_IP> --worker \
  /path/to/huggingface/cache -e VLLM_HOST_IP=<WORKER_NODE_IP>
```

3. **Verify cluster:**

```bash
docker exec -it <container_name> bash
ray status
ray list nodes
```

### Deploy vLLM on Ray Cluster

```bash
# 16 GPUs across 2 nodes (8 GPUs per node)
vllm serve meta-llama/Llama-3-70b \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray
```

Alternatively, set `tensor_parallel_size` to the total GPU count:

```bash
vllm serve meta-llama/Llama-3-70b \
  --tensor-parallel-size 16 \
  --distributed-executor-backend ray
```

vLLM automatically distributes GPUs across nodes.

### Security Warning

> [!warning] Network Security
> Traffic between nodes is **unencrypted** and vulnerable to arbitrary code execution if an adversary gains network access. Use a private network segment and restrict access to trusted parties.

## Troubleshooting

### Common Issues

1. **Out of memory (OOM):** Increase `tp_size` or reduce `max_num_seqs`.
2. **Slow all-reduce:** Check `NCCL_DEBUG=TRACE` logs. Ensure NVLink/InfiniBand is active (not TCP sockets).
3. **Hang on startup:** Ray cluster misconfiguration. Verify `ray status` shows all nodes and GPUs.
4. **Uneven GPU splits:** Use [[Pipeline Parallelism]] instead (supports uneven layer splits).

### Debugging NCCL

```bash
# Enable verbose NCCL logs
NCCL_DEBUG=TRACE vllm serve ...

# Look for:
# - "[send] via NET/IB/GDRDMA" (InfiniBand + GPUDirect, efficient)
# - "[send] via NET/Socket" (TCP fallback, inefficient)
```

See [Troubleshooting distributed deployments](https://docs.vllm.ai/en/latest/serving/distributed_troubleshooting.html) for more.

## Cross-References

- [[Pipeline Parallelism]] — Split model layers across nodes (lower communication overhead than TP for PCIe/Ethernet)
- [[Data Parallelism]] — Replicate model, split batches across TP groups
- [[Expert Parallelism]] — Distribute MoE experts across GPUs (alternative to TP for expert layers)
- [[Context Parallelism]] — Shard long sequences to reduce KV cache duplication
- [[Kernel Fusions]] — AllReduce+RMSNorm, AsyncTP GEMM+Collective fusions
- [[vLLM Engine]] — Scheduler and executor architecture that orchestrates TP workers
- [[Python Multiprocessing]] — Distributed executor backend tradeoffs (fork vs spawn vs Ray)
- [[Mixture of Experts]] — MoE architecture and TP/EP interaction

## Key Metrics

- **Memory reduction:** `1 / tp_size` (e.g., 4× reduction with TP=4)
- **All-reduce overhead:** 5-15% on InfiniBand, <5% on NVLink
- **Typical TP sizes:** 2, 4, 8 (single-node), 16 (multi-node with PP)
- **NVLink bandwidth:** 900 GB/s (H100), 600 GB/s (A100)
- **InfiniBand bandwidth:** 200 GB/s (HDR), 400 GB/s (NDR)

## Further Reading

- **Megatron-LM paper:** [Shoeybi et al., 2019](https://arxiv.org/pdf/1909.08053.pdf)
- **vLLM docs:** [Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html)
- **NCCL docs:** [NVIDIA Collective Communication Library](https://docs.nvidia.com/deeplearning/nccl/)
- **GPUDirect RDMA:** [NVIDIA GPUDirect](https://developer.nvidia.com/gpudirect)
