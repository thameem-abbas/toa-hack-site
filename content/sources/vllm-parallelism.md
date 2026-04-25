---
title: vLLM Parallelism Strategies
type: source
created: 2026-04-25
source_urls:
  - /tmp/vllm/docs/serving/parallelism_scaling.md
  - /tmp/vllm/docs/serving/data_parallel_deployment.md
  - /tmp/vllm/docs/serving/context_parallel_deployment.md
  - /tmp/vllm/docs/design/multiprocessing.md
tags: [parallelism, distributed-inference, vllm-docs]
---

# vLLM Parallelism Strategies

Comprehensive documentation covering vLLM's four parallelism strategies for distributed LLM inference: Tensor Parallelism (TP), Pipeline Parallelism (PP), Data Parallelism (DP), and Context Parallelism (CP).

## Sources

1. **`docs/serving/parallelism_scaling.md`** — Overview of parallelism strategies, decision tree for choosing TP/PP/DP, multi-node deployment with Ray and multiprocessing, GPUDirect RDMA configuration.
2. **`docs/serving/data_parallel_deployment.md`** — Data Parallel deployment modes (internal, hybrid, external load balancing), ZMQ socket topology, DP coordinator for MoE, prefix-aware routing.
3. **`docs/serving/context_parallel_deployment.md`** — Context Parallel for long-context serving, prefill CP (ring attention, under development), decode CP (KV cache sharding, interleaving strategy), case studies (DeepSeek-R1, Kimi-K2, Qwen3).
4. **`docs/design/multiprocessing.md`** — Python multiprocessing tradeoffs (fork/spawn/forkserver), vLLM's best-effort method selection, v1 engine changes, compatibility with CUDA/PyTorch dependencies.

## Key Concepts

### [[Tensor Parallelism]]

**What:** Split individual layers (weight matrices) across GPUs. Megatron-LM algorithm: column-parallel and row-parallel linear layers, all-reduce after each layer.

**When:** Model too large for single GPU, within-node deployment (high bandwidth needed).

**Configuration:** `--tensor-parallel-size <N>` (typically N = GPUs per node).

**Communication:** One all-reduce per transformer block. Requires NVLink (900 GB/s) or InfiniBand (200 GB/s) for efficiency.

**Edge case:** GPUs without NVLink (e.g., L40S) → use [[Pipeline Parallelism]] instead (lower communication overhead).

### [[Pipeline Parallelism]]

**What:** Split model layers sequentially across GPUs/nodes. Each GPU executes a contiguous subset of layers.

**When:** Model exceeds single-node capacity, cross-node deployment, GPUs lack NVLink.

**Configuration:** `--pipeline-parallel-size <N>` (typically N = number of nodes).

**Micro-batching:** Split batch into M micro-batches to keep GPUs busy. Bubble overhead: `(N-1) / M` (10-20% typical).

**Uneven splits:** PP supports uneven layer partitions (e.g., GPU 0 has 10 layers, GPU 1 has 15 layers). TP requires even splits.

**Combined TP+PP:** `tp_size = GPUs per node`, `pp_size = number of nodes`. Example: 2 nodes × 8 GPUs = TP=8, PP=2.

### [[Data Parallelism]]

**What:** Replicate model across separate TP/PP groups. Each DP rank processes independent batches.

**When:** Throughput bottleneck, model fits on single GPU or small TP group.

**Configuration:** `--data-parallel-size <N>`. Total GPUs = `DP_size × TP_size × PP_size`.

**Load balancing modes:**

1. **Internal LB:** Single API server, ZMQ sockets to all DP ranks. Scale API server with `--api-server-count`.
2. **Hybrid LB:** Per-node API servers, external load balancer distributes across nodes. Use `--data-parallel-hybrid-lb`.
3. **External LB:** Each DP rank has its own HTTP endpoint. External load balancer (nginx, HAProxy) distributes requests.

**MoE-specific:** DP coordinator process ensures expert synchronization (5-10% overhead). Dummy forward passes in idle ranks.

**ZMQ topology:** API server → ZMQ REQ/REP → DP ranks (core engine processes).

**Prefix caching:** Each DP rank has independent KV cache. Route similar prompts to same rank for prefix cache hits.

### [[Context Parallelism]]

**What:** Shard long sequences across GPUs to reduce KV cache duplication and improve TTFT.

**Prefill CP (under development):** Two strategies:

1. **Partial query, full KV:** All-gather KV, compute attention per GPU. For moderately long contexts.
2. **Partial query, partial KV:** Ring attention (send/recv KV chunks). For very long contexts (100k+ tokens).

**Decode CP (stable):** Shard KV cache along sequence dimension (T).

- **Problem:** TP shards along KV head dimension (H). If `TP_size > H`, KV cache is duplicated.
- **Solution:** Add `-dcp <size>` to shard along T. DCP size: `1 ≤ DCP_size ≤ TP_size / H`.
- **Interleaving strategy:** GPU 0 stores tokens {0, N, 2N, ...}, GPU 1 stores {1, N+1, 2N+1, ...}.
- **Communication:** All-gather on every decode step (1-2% overhead intra-node, 5-10% cross-node).

**Case studies:**

- DeepSeek-R1 (1 KV head, TP=8): 8× duplication → use DCP=8 to eliminate.
- Kimi-K2 (1 KV head, TP=16): 16× duplication → use DCP=16 (cross-node) or DCP=8 (intra-node, 2× duplication).
- Qwen3-235B-A22B (4 KV heads, TP=8): 2× duplication → use DCP=2.

## Decision Tree

```
Is model too large for single GPU?
  No → Single GPU inference
  Yes ↓
    Fits on single node with multiple GPUs?
      Yes → Tensor Parallelism (TP = num_gpus_per_node)
      No  → TP + Pipeline Parallelism (TP = GPUs/node, PP = num_nodes)

Is throughput the bottleneck?
  Yes → Add Data Parallelism (DP_size × TP_size = total_gpus)

Do you have KV cache duplication (TP_size > num_kv_heads)?
  Yes → Add Decode Context Parallelism (DCP ≤ TP_size / num_kv_heads)

Are you serving very long contexts (100k+ tokens)?
  Yes → Wait for Prefill Context Parallelism (under development)
```

## Multi-Node Deployment

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

3. **Verify:**

```bash
docker exec -it <container> bash
ray status
ray list nodes
```

### Deploy vLLM

```bash
# 16 GPUs across 2 nodes (8 GPUs per node)
vllm serve meta-llama/Llama-3-70b \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray
```

### Multiprocessing (No Ray)

**Head node:**

```bash
vllm serve meta-llama/Llama-3-70b \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 0 \
  --master-addr <HEAD_NODE_IP>
```

**Worker node:**

```bash
vllm serve meta-llama/Llama-3-70b \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 1 \
  --master-addr <HEAD_NODE_IP> --headless
```

## GPUDirect RDMA

GPUDirect RDMA allows network adapters to directly access GPU memory, reducing latency for TP all-reduce.

**Docker:**

```bash
docker run --gpus all --ipc=host --shm-size=16G \
  -v /dev/shm:/dev/shm vllm/vllm-openai
```

**Kubernetes:**

```yaml
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

**Verify:**

```bash
NCCL_DEBUG=TRACE vllm serve ...
# Look for: "[send] via NET/IB/GDRDMA" (efficient)
# Avoid: "[send] via NET/Socket" (inefficient)
```

## Python Multiprocessing

vLLM uses Python multiprocessing for single-node TP and native distributed backends. Three methods:

1. **`fork`** (default for library usage): Fastest, but incompatible with CUDA/PyTorch threads. May crash on macOS.
2. **`spawn`** (default for `vllm` command): Compatible with dependencies, but requires `__main__` guard when used as a library. Slower startup.
3. **`forkserver`** (Python 3.14+ default): Same issues as `spawn` (requires `__main__` guard).

**vLLM's strategy:**

- Default to `fork` for library usage (fastest).
- Use `spawn` when `vllm` command is executed (we control main process).
- If CUDA is already initialized, force `spawn` and emit warning (fork will crash).

**Environment variable:** `VLLM_WORKER_MULTIPROC_METHOD=spawn|fork` to override.

**Known issue:** Using vLLM as a library + initializing CUDA before vLLM + no `__main__` guard → infinite recursion. Solution: Add `if __name__ == "__main__":` guard or disable multiprocessing.

## Configuration Examples

### Example 1: Single-Node TP

```bash
# 4 GPUs, model fits on node
vllm serve facebook/opt-13b --tensor-parallel-size 4
```

### Example 2: Multi-Node TP+PP

```bash
# 2 nodes × 8 GPUs = 16 GPUs
vllm serve meta-llama/Llama-3-70b \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray
```

### Example 3: Data Parallel (Single-Node)

```bash
# DP=4, TP=2 on 8 GPUs
vllm serve facebook/opt-13b \
  --data-parallel-size 4 \
  --tensor-parallel-size 2
```

### Example 4: Data Parallel (Multi-Node, Internal LB)

```bash
# Node 0 (head)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345

# Node 1 (worker)
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 2 \
  --data-parallel-start-rank 2 \
  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

### Example 5: Data Parallel (External LB)

```bash
# Rank 0
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL \
  --data-parallel-size 2 --data-parallel-rank 0 --port 8000

# Rank 1
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL \
  --data-parallel-size 2 --data-parallel-rank 1 --port 8001
```

### Example 6: Context Parallel (Decode)

```bash
# DeepSeek-R1: 1 KV head, TP=8 → 8× duplication
vllm serve deepseek-ai/deepseek-r1 \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 8
```

## Cross-References

- [[Tensor Parallelism]] — TP concept page with Megatron-LM algorithm details
- [[Pipeline Parallelism]] — PP concept page with micro-batching and bubble overhead
- [[Data Parallelism]] — DP concept page with load balancing modes and MoE coordinator
- [[Context Parallelism]] — CP concept page with ring attention and KV cache sharding
- [[Mixture of Experts]] — MoE architecture and EP/DP interaction
- [[Expert Parallelism]] — EP concept page for expert-specific sharding
- [[KV Cache]] — KV cache mechanics and memory footprint
- [[vLLM Engine]] — Scheduler and executor architecture
- Python Multiprocessing — Multiprocessing method tradeoffs (fork/spawn)

## Key Metrics

- **TP memory reduction:** `1 / tp_size` (e.g., 4× reduction with TP=4).
- **TP communication overhead:** 5-15% on InfiniBand, <5% on NVLink.
- **PP bubble overhead:** 10-20% (typical for `M=8` micro-batches).
- **DP throughput scaling:** 90-95% of linear.
- **DP coordinator overhead (MoE):** 5-10% decode time.
- **CP memory savings (decode):** `1 - (1 / DCP_size)` (e.g., 87.5% for DCP=8).
- **CP communication overhead (decode):** 1-2% intra-node, 5-10% cross-node.

## Further Reading

- **Megatron-LM paper:** [Shoeybi et al., 2019](https://arxiv.org/pdf/1909.08053.pdf)
- **Ring Attention paper:** [Liu et al., 2023](http://arxiv.org/abs/2310.01889)
- **CP interleaving strategy:** [Hong et al., 2025](http://arxiv.org/abs/2507.07120)
- **vLLM Slack:** `#sig-context-parallel` channel
- **Ray documentation:** [Ray Docs](https://docs.ray.io/en/latest/index.html)
