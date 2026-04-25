---
title: Data Parallelism
type: concept
created: 2026-04-25
tags: [parallelism, distributed-inference, load-balancing, zmq, api-server, moe]
---

# Data Parallelism

**Data Parallelism (DP)** replicates the entire model across multiple GPU groups, with each group (called a **DP rank**) processing independent batches of requests. Unlike [[Tensor Parallelism]] (which shards layers) or [[Pipeline Parallelism]] (which splits stages), DP creates full model copies to scale throughput.

## Core Mechanism

### Model Replication

For `DP_size=4`:

- **DP rank 0:** Full model on GPU(s) 0-k
- **DP rank 1:** Full model on GPU(s) k+1-2k
- **DP rank 2:** Full model on GPU(s) 2k+1-3k
- **DP rank 3:** Full model on GPU(s) 3k+1-4k

Each DP rank is an independent vLLM engine core (separate process). Requests are load-balanced across ranks.

### DP Coordinator Process

For [[Mixture of Experts]] (MoE) models, DP ranks are **not fully independent**:

- **Forward pass alignment:** Expert layers across all ranks must synchronize during every forward pass, even when some ranks have no requests.
- **Dummy forward passes:** Idle ranks perform empty forward passes to maintain synchronization.
- **DP Coordinator:** Separate process that communicates with all ranks via collective operations. Determines when all ranks become idle and can be paused.

**Dense models:** No coordinator required. Ranks are fully independent.

**MoE models:** Coordinator ensures expert synchronization. Communication overhead: ~5-10% of decode time.

### ZMQ Socket Topology

Each DP rank is a separate "core engine" process. Communication between front-end (API server) and engine cores uses **ZMQ sockets**:

```
┌─────────────────┐
│  API Server(s)  │  ← FastAPI process(es) handling HTTP requests
└────────┬────────┘
         │ ZMQ REQ/REP
    ┌────┴────┬────────┬────────┐
    ▼         ▼        ▼        ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ DP Rank│ │ DP Rank│ │ DP Rank│ │ DP Rank│
│   0    │ │   1    │ │   2    │ │   3    │
└────────┘ └────────┘ └────────┘ └────────┘
  (GPUs     (GPUs     (GPUs     (GPUs
   0-1)      2-3)      4-5)      6-7)
```

**Load balancing:** API server selects DP rank based on queue depth and KV cache state.

## When to Use Data Parallelism

Use DP when:

1. **Throughput is the bottleneck.** DP scales throughput linearly (up to network/CPU limits). Each DP rank processes independent requests.
2. **Model fits on a single GPU or small TP group.** No need to shard layers. Replicate instead.
3. **Prefix caching benefits are high.** Intelligent routing can maximize prefix cache hits by directing similar prompts to the same DP rank.
4. **MoE models with load imbalance.** DP spreads load across replicas. Combine with [[Expert Parallelism]] (EP) for expert-specific sharding.

**Trade-off:** DP does **not** reduce memory per GPU (each rank stores the full model). Use [[Tensor Parallelism]] or [[Pipeline Parallelism]] to reduce per-GPU memory.

## Configuration

### Single-Node DP

```bash
# DP=4 on 4 GPUs (no TP)
vllm serve facebook/opt-13b --data-parallel-size 4

# DP=4, TP=2 on 8 GPUs
vllm serve facebook/opt-13b \
  --data-parallel-size 4 \
  --tensor-parallel-size 2
```

**Memory calculation:** `DP_size × TP_size = total_gpus`. Each DP rank uses `TP_size` GPUs.

**Request capacity:** `max_num_seqs` applies **per DP rank**. Total capacity = `DP_size × max_num_seqs`.

### Multi-Node DP (Internal Load Balancing)

**Node 0 (head):**

```bash
vllm serve $MODEL \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-address 10.99.48.128 \
  --data-parallel-rpc-port 13345
```

**Node 1 (worker):**

```bash
vllm serve $MODEL --headless \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 2 \
  --data-parallel-address 10.99.48.128 \
  --data-parallel-rpc-port 13345
```

**Explanation:**

- `--data-parallel-size 4`: Global DP size (total ranks across all nodes).
- `--data-parallel-size-local 2`: Ranks on this node.
- `--data-parallel-start-rank 2`: Rank IDs on worker node (2, 3).
- `--data-parallel-address`: IP of the head node (API server).
- `--headless`: Worker node has no API server.

**API endpoint:** Single HTTP server on Node 0. Load balancing is internal.

### Multi-Node DP with Ray

Using Ray simplifies multi-node setup:

```bash
vllm serve $MODEL \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-backend=ray
```

**Advantages:**

- **Single launch command:** No need to SSH into each node.
- **Auto-configuration:** Ray determines node IPs and allocates ranks.
- **Fault tolerance:** Ray restarts failed ranks.

**Environment variable:** For multi-node DP groups (single model replica across nodes), set:

```bash
export VLLM_RAY_DP_PACK_STRATEGY="span"
```

This tells Ray to distribute a single DP group across multiple nodes (instead of packing each group onto one node).

### API Server at One Node, Engines at Another

Decouple API server from engine cores:

**Node 0 (API server only):**

```bash
vllm serve $MODEL \
  --data-parallel-size 4 \
  --data-parallel-size-local 0 \
  --data-parallel-address 10.99.48.128 \
  --data-parallel-rpc-port 13345
```

**Node 1 (all engines):**

```bash
vllm serve $MODEL --headless \
  --data-parallel-size 4 \
  --data-parallel-size-local 4 \
  --data-parallel-address 10.99.48.128 \
  --data-parallel-rpc-port 13345
```

**Use case:** GPU-poor head node. API server runs on CPU-only node, engines run on GPU node(s).

## Load Balancing Modes

vLLM supports three DP load balancing modes:

### 1. Internal Load Balancing (Default)

**Architecture:** Single API server process(es) on head node. ZMQ sockets to all DP ranks.

**Load balancing logic:** API server selects DP rank based on:

- Running queue depth (number of in-flight requests)
- Waiting queue depth (number of queued requests)

**Future enhancement:** KV cache-aware routing (maximize prefix cache hits).

**Scaling API server:** For large DP sizes, API server can become a bottleneck. Scale out:

```bash
vllm serve $MODEL \
  --data-parallel-size 16 \
  --api-server-count 4
```

This creates 4 API server processes (still a single HTTP endpoint). All share the same ZMQ pool to DP ranks.

**Diagram (from vLLM docs):**

```
         ┌─────────────────┐
         │  API Server(s)  │  ← Single HTTP endpoint
         └────────┬────────┘
                  │ ZMQ load balancing
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐    ┌────────┐    ┌────────┐
│ DP     │    │ DP     │    │ DP     │
│ Rank 0 │    │ Rank 1 │    │ Rank 2 │
└────────┘    └────────┘    └────────┘
```

### 2. Hybrid Load Balancing

**Architecture:** Each node runs its own API server(s), which only queue to local DP ranks. External load balancer (e.g., Kubernetes Ingress) distributes requests across per-node endpoints.

**Configuration:**

```bash
# Node 0
vllm serve $MODEL \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 0 \
  --data-parallel-hybrid-lb

# Node 1
vllm serve $MODEL \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 2 \
  --data-parallel-hybrid-lb
```

**Key differences from internal LB:**

- `--data-parallel-hybrid-lb`: Enable hybrid mode.
- No `--headless`: Every node exposes an API endpoint.
- External load balancer required (e.g., nginx, HAProxy, Kubernetes Service).

**Advantages:**

- Reduces cross-node traffic (each node's API server only queues to local engines).
- Avoids single-node bottleneck at large DP sizes.

**Use case:** Large DP deployments (DP > 8).

### 3. External Load Balancing

**Architecture:** Each DP rank runs its own API server (separate HTTP endpoint). External load balancer distributes requests across all endpoints.

**Configuration (co-located ranks):**

```bash
# Rank 0
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL \
  --data-parallel-size 2 \
  --data-parallel-rank 0 \
  --port 8000

# Rank 1
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL \
  --data-parallel-size 2 \
  --data-parallel-rank 1 \
  --port 8001
```

**Configuration (multi-node):**

```bash
# Rank 0 (10.99.48.128)
vllm serve $MODEL \
  --data-parallel-size 2 \
  --data-parallel-rank 0 \
  --data-parallel-address 10.99.48.128 \
  --data-parallel-rpc-port 13345

# Rank 1
vllm serve $MODEL \
  --data-parallel-size 2 \
  --data-parallel-rank 1 \
  --data-parallel-address 10.99.48.128 \
  --data-parallel-rpc-port 13345
```

**Use case:**

- **Dense models:** Ranks are fully independent. No DP coordinator. External LB can use standard metrics (latency, queue depth) for routing.
- **MoE models:** DP coordinator still runs (co-located with rank 0). Ranks must synchronize, but external LB handles HTTP routing.

**Advantages:**

- Maximum flexibility for custom load balancing logic.
- Can integrate with Kubernetes HPA (Horizontal Pod Autoscaler), Prometheus metrics, etc.

**Diagram (from vLLM docs):**

```
     External Load Balancer
       (nginx/HAProxy)
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ DP Rank │ │ DP Rank │ │ DP Rank │
│    0    │ │    1    │ │    2    │
│ (API +  │ │ (API +  │ │ (API +  │
│ Engine) │ │ Engine) │ │ Engine) │
└─────────┘ └─────────┘ └─────────┘
  (Port     (Port     (Port
   8000)     8001)     8002)
```

## Interaction with Other Parallelism Strategies

### DP + Tensor Parallelism

Each DP rank uses multiple GPUs via [[Tensor Parallelism]]:

```bash
# DP=4, TP=2 on 8 GPUs
vllm serve $MODEL \
  --data-parallel-size 4 \
  --tensor-parallel-size 2
```

**Memory layout:**

- DP rank 0: GPUs 0-1 (TP group 0)
- DP rank 1: GPUs 2-3 (TP group 1)
- DP rank 2: GPUs 4-5 (TP group 2)
- DP rank 3: GPUs 6-7 (TP group 3)

**Total capacity:** `DP_size × max_num_seqs` concurrent requests.

### DP + Expert Parallelism (MoE)

For [[Mixture of Experts]] models, expert layers can use [[Expert Parallelism]] (EP) while attention layers use DP:

```bash
vllm serve deepseek-ai/deepseek-v3 \
  --data-parallel-size 4 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel
```

**Expert group size:** `DP_size × TP_size = 8` (all GPUs form a single expert group).

**Communication pattern:**

- **Attention layers:** TP all-reduce within each DP rank (no cross-rank communication).
- **Expert layers:** All-to-all dispatch/combine across all 8 GPUs (EP group).

**DP Coordinator:** Required for MoE models to synchronize expert forward passes.

See [[Expert Parallelism]] for details.

### DP + Pipeline Parallelism

Each DP rank is a multi-stage pipeline:

```bash
# DP=2, PP=2, TP=4 on 16 GPUs
vllm serve $MODEL \
  --data-parallel-size 2 \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4
```

**GPU layout:**

- DP rank 0: GPUs 0-7 (2 PP stages × 4 TP GPUs)
- DP rank 1: GPUs 8-15 (2 PP stages × 4 TP GPUs)

## Prefix Caching and DP

Each DP rank has an **independent KV cache** (including [[Prefix Caching]]). To maximize prefix cache hits:

- **Smart routing:** Route similar prompts to the same DP rank.
- **Affinity mapping:** Hash prompt prefix → DP rank ID.

**Example:** For few-shot prompting, route all requests with the same system prompt to the same rank.

**Future enhancement:** vLLM's internal load balancer will incorporate KV cache state for routing decisions.

## Offline Inference with DP

DP is also supported for offline batch inference via the `LLM` class:

```python
# See examples/offline_inference/data_parallel.py
from vllm import LLM

llm = LLM(
    model="facebook/opt-13b",
    data_parallel_size=4,
    tensor_parallel_size=2,
)
outputs = llm.generate(prompts)
```

## MoE-Specific Considerations

### Forward Pass Alignment

MoE expert layers require **all DP ranks** to perform forward passes in lockstep:

- **Active rank:** Processes requests normally.
- **Idle rank:** Performs dummy forward passes (no actual tokens, but maintains expert synchronization).

**Overhead:** 5-10% decode time for coordination.

### DP Coordinator

The coordinator process:

- Runs on the head node (co-located with DP rank 0 engine).
- Communicates with all ranks via NCCL collectives.
- Determines when all ranks are idle (no requests in any rank) and can be paused.

**Configuration:** Automatic for MoE models. No manual setup required.

### Expert Parallelism Toggle

By default, expert layers form a **tensor parallel group** of size `DP_size × TP_size`:

```bash
# Default: DP=4, TP=2 → expert TP group of size 8
vllm serve deepseek-ai/deepseek-v3 \
  --data-parallel-size 4 \
  --tensor-parallel-size 2
```

To use [[Expert Parallelism]] instead:

```bash
# Expert EP group of size 8
vllm serve deepseek-ai/deepseek-v3 \
  --data-parallel-size 4 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel
```

**Difference:**

- **TP mode:** Experts are sharded within each DP rank (standard TP behavior).
- **EP mode:** Experts form a group across all DP ranks and TP GPUs. All-to-all communication across the full DP×TP group.

See [[Expert Parallelism]] for performance tradeoffs.

## Memory and Performance Characteristics

### Memory Footprint

- **Weights:** Not reduced (each DP rank stores the full model). Use TP/PP to reduce per-GPU memory.
- **KV cache:** Each DP rank has independent KV cache. Total KV cache memory = `DP_size × per_rank_KV_cache`.
- **Activations:** Independent per rank.

**Implication:** DP **increases** total memory usage (scales with `DP_size`). But per-GPU memory is unchanged (vs single DP rank).

### Throughput Scaling

- **Ideal scaling:** `DP_size × throughput_per_rank`.
- **Practical scaling:** 90-95% of ideal (due to load balancing overhead, ZMQ latency, coordinator overhead for MoE).

**Bottlenecks:**

- **API server CPU:** Mitigate with `--api-server-count`.
- **Network bandwidth (multi-node):** Mitigate with hybrid/external load balancing (reduces cross-node traffic).
- **DP coordinator (MoE):** 5-10% overhead. Cannot be eliminated (inherent to MoE synchronization).

### Latency

- **TTFT:** Same as single DP rank (no additional latency).
- **ITL:** Same as single DP rank.

**Load balancing latency:** <1 ms (ZMQ socket overhead). Negligible compared to inference time.

## Troubleshooting

### Common Issues

1. **Uneven load distribution:** API server not balancing correctly. Check `--verbose` logs for queue depths. Consider external load balancing for more control.
2. **High DP coordinator overhead (MoE):** Reduce `DP_size` or use larger batches (fewer dummy forward passes).
3. **API server bottleneck:** Increase `--api-server-count` (e.g., `--api-server-count=4`).
4. **Ray allocation issues (multi-node):** Set `VLLM_RAY_DP_PACK_STRATEGY="span"` to distribute DP groups across nodes.

### Debugging DP

```bash
# Enable verbose logs
vllm serve ... --verbose

# Look for:
# - "DP rank X initialized"
# - "ZMQ socket connected to rank X"
# - "DP coordinator started"
```

## Cross-References

- [[Tensor Parallelism]] — Combine DP with TP for large models (DP scales throughput, TP reduces per-GPU memory)
- [[Pipeline Parallelism]] — Combine DP with PP for multi-node large models
- [[Expert Parallelism]] — EP mode for MoE expert layers (alternative to TP within DP groups)
- [[Context Parallelism]] — Orthogonal to DP (can combine for long-context throughput scaling)
- [[Mixture of Experts]] — MoE architecture and DP coordinator requirements
- [[Prefix Caching]] — Each DP rank has independent prefix cache (route similar prompts to same rank)
- [[vLLM Engine]] — Scheduler and executor architecture for DP ranks
- [[V1 Architecture]] — V1 engine uses multiprocessing for DP ranks (separate core processes)

## Key Metrics

- **Throughput scaling:** 90-95% of linear (ideal `DP_size × throughput_per_rank`)
- **Memory scaling:** `DP_size × per_rank_memory` (total memory increases)
- **Load balancing latency:** <1 ms (ZMQ overhead)
- **DP coordinator overhead (MoE):** 5-10% decode time
- **API server bottleneck:** Appears at DP > 8-12 (mitigate with `--api-server-count`)
- **Typical DP sizes:** 2-8 (single-node), 4-16 (multi-node)

## Further Reading

- **vLLM docs:** [Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html)
- **Expert Parallel Deployment:** [vLLM EP docs](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html)
- **Offline DP example:** [examples/offline_inference/data_parallel.py](https://github.com/vllm-project/vllm/tree/main/examples/offline_inference/data_parallel.py)
