---
title: Disaggregated Serving
type: concept
created: 2026-04-25
tags: [serving, architecture, distributed-inference, latency-optimization]
---

# Disaggregated Serving

Disaggregated serving separates the **prefill** and **decode** phases of LLM inference onto different GPU pools, enabling independent optimization and scaling of each phase.

## Core Idea

LLM inference consists of two phases with fundamentally different resource profiles:

1. **Prefill**: Process input prompt → generate first token
   - Compute-bound (high arithmetic intensity)
   - Optimizes for **TTFT** (time-to-first-token)
   - Benefits from high GPU utilization, large batch sizes
   - Short-lived (milliseconds to seconds)

2. **Decode**: Generate output tokens autoregressively
   - Memory-bound (memory bandwidth limited)
   - Optimizes for **ITL** (inter-token latency)
   - Benefits from low latency, small batch sizes
   - Long-lived (seconds to minutes)

Disaggregated serving runs these phases on separate GPU instances, allowing different parallel strategies (TP, PP, batch sizes) for each.

## Why Disaggregated Serving?

### 1. Independent TTFT and ITL Tuning

**Without disaggregation:**
- Single instance must compromise between TTFT and ITL optimization
- Prefill jobs inserted during decode increase tail latency
- Same TP/PP configuration for both phases (suboptimal)

**With disaggregation:**
- Prefill pool: High TP for TTFT, large batches
- Decode pool: Low TP for ITL, small batches, optimized memory access
- Independent scaling: Add prefill capacity without affecting decode latency

### 2. Tail Latency Control

**Problem:** In unified serving, prefill jobs can preempt decode, causing unpredictable ITL spikes.

**Alternative:** [[Chunked Prefill]] breaks prefill into chunks to limit preemption, but requires careful chunk size tuning (workload-dependent).

**Disaggregated solution:** Physical separation eliminates preemption risk entirely. Decode instances never execute prefill work (unless KV load fails with `kv_load_failure_policy="recompute"`).

### 3. Does NOT Improve Throughput

> [!important] Throughput Neutrality
> Disaggregated prefill optimizes **latency** (TTFT, ITL), not throughput. Total compute remains the same; it's redistributed across specialized instances.

## Architecture Components

### Instance Types

**Prefill Instance (P)**
- Role: `kv_producer`
- Executes: Full prefill → generates KV cache
- Sends: KV cache to decode instance via [[KV Cache Transfer]] connector
- Configuration: High GPU utilization (0.9+), large `max_num_seqs`, `max_tokens=1` (from proxy)

**Decode Instance (D)**
- Role: `kv_consumer`
- Executes: Receives KV cache → skips prefill → generates tokens
- Receives: KV cache from prefill instance
- Configuration: Lower GPU utilization (0.7), moderate `max_num_seqs`, full output generation

**Proxy/Router**
- Selects P+D pair for each request (round-robin, load-based, or trie-based prefix matching)
- Generates `request_id` encoding P/D addresses
- Maintains service discovery (`http_addr → zmq_addr` or similar)
- Monitors heartbeats (e.g., every 3s)

### Deployment Patterns

**xPyD:** x prefill instances + y decode instances

- **1P1D:** Simplest, 1:1 mapping
- **1P3D:** Optimize for decode throughput (decode-heavy workload)
- **3P1D:** Optimize for prefill throughput (prefill-heavy workload)
- **96P144D:** Large-scale production (e.g., DeepSeek)

Ratio selection depends on:
- Average prompt length vs output length
- Request rate
- TTFT vs ITL SLO requirements

### Request Flow (P2P NCCL Example)

1. Client → Proxy: `/v1/completions` request
2. Proxy selects 1P1D pair, generates `request_id`, modifies `max_tokens=1`
3. Proxy → P: Prefill-only request
4. Proxy → D: Original request (full output)
5. P executes prefill → sends KV to D via PUT_ASYNC (NCCL)
6. D receives KV in dedicated thread → GPU buffer or memory pool
7. D retrieves KV from buffer → skips prefill → decodes
8. D → Proxy → Client: Streaming response

## KV Cache Transfer

See [[KV Cache Transfer]] for full architecture details. Key connector types:

### P2P NCCL Connector

**Transport:** ZMQ (metadata) + NCCL (data)

**Modes:**
- **PUT_ASYNC** (best): Async send via dedicated thread
- **GET**: Decode pulls from prefill buffer
- **PUT**: Sync send (blocks main process)

**Topology:** Symmetric TP only (asymmetric TP/PP planned)
- Each P rank establishes NCCL group with corresponding D rank
- World size = 2 per group (point-to-point)
- Dynamic scaling: add/remove instances without full restart

**Memory:**
- `kv_buffer_size`: GPU buffer (5-10% GPU memory)
- Tensor memory pool: CPU fallback (buddy allocator, PCIe 4.0 ~21 GB/s)

**Limitations:** NCCL group overhead (52-100 MB each) limits large-scale deployments (96P144D requires RDMA/UCCL)

### NIXL Connector

**Transport:** NIXL library (UCX, RDMA, GPU Direct, LIBFABRIC backends)

**Features:**
- Fully asynchronous send/recv
- Multi-host support via side channel
- ROCm support (RIXL)
- Experimental: Heterogeneous KV layout, cross-layer blocks

**Failure policy:**
- `kv_load_failure_policy="fail"`: Reject request on KV load failure (prevents performance degradation)
- `kv_load_failure_policy="recompute"`: Decode instance recomputes prefill (causes jitter, defeats purpose)

### Other Connectors

- **LMCacheConnectorV1:** Uses NIXL under the hood
- **MooncakeConnector:** Third-party storage backend
- **MultiConnector:** Combines multiple connectors (ordered list)
- **OffloadingConnector:** CPU/disk offload for KV cache
- **FlexKVConnectorV1:** Distributed KV store with multi-level caching

## Disaggregated Encoder (Multimodal Extension)

For multimodal models, vision encoder can be separated from language model:

**E→PD (Encoder → Prefill/Decode):**
- Encoder instance: Lightweight vision encoder
- PD instance: Heavy language model
- ECConnector transfers encoder cache embeddings

**E→P→D (Encoder → Prefill → Decode):**
- Encoder → Prefill: EC transfer
- Prefill → Decode: KV transfer
- Full three-stage disaggregation

**Benefits:**
- Independent scaling (encoder is lightweight, doesn't need large TP)
- TTFT reduction for language-only requests (bypass encoder)
- Cross-process encoder output caching (shared across workers)

## Implementation

**Code locations:**
- `vllm/distributed/kv_transfer/`: Core abstractions
- `vllm/distributed/ec_transfer/`: Encoder cache transfer
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/`: P2P NCCL examples
- `tests/v1/kv_connector/nixl_integration/`: NIXL integration tests

**Core abstractions:**

**Connector**
- Interface for KV consumer to retrieve KV from KV producer
- Scheduler connector: schedules transfer ops
- Worker connector: executes transfer ops

**LookupBuffer**
- `insert(kv_cache)`: Non-blocking insert
- `drop_select(condition)`: Blocking select-and-drop (SQL-like)

**Pipe**
- `send_tensor()` / `recv_tensor()`: FIFO tensor transmission

**Extension patterns:**
1. Fully-customized connector: Implement `Connector` interface
2. Database-like connector: Implement `LookupBuffer` with SQL-like API
3. Distributed P2P connector: Implement `Pipe` with send/recv API

## Performance Characteristics

### P2P NCCL Benchmark (1K input, 200 output)

- **E2E P99 latency:** ~2s
- **Transfer mode ranking:** PUT_ASYNC > GET > PUT
- **kv_buffer_size:** 10% GPU memory (empirical)
- **Memory pool speed:** PCIe 4.0 ~21 GB/s

### Tail Latency Comparison

| Approach | Tail ITL Control | Tuning Complexity | Throughput Impact |
|----------|------------------|-------------------|-------------------|
| Unified serving | Poor | N/A | Baseline |
| [[Chunked Prefill]] | Good | High (chunk size tuning) | None |
| Disaggregated | Excellent | Medium (xPyD ratio) | None |

## Configuration

### Prefill Instance

```bash
vllm serve <MODEL> \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \
  --kv-transfer-config '{
    "kv_connector":"P2pNcclConnector",
    "kv_role":"kv_producer",
    "kv_buffer_size":"1e1",
    "kv_port":"21001",
    "kv_connector_extra_config":{
      "proxy_ip":"10.0.1.1",
      "proxy_port":"30001",
      "http_port":"20001"
    }
  }'
```

### Decode Instance

```bash
vllm serve <MODEL> \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 256 \
  --kv-transfer-config '{
    "kv_connector":"P2pNcclConnector",
    "kv_role":"kv_consumer",
    "kv_buffer_size":"8e9",
    "kv_port":"22001",
    "kv_connector_extra_config":{
      "proxy_ip":"10.0.1.1",
      "proxy_port":"30001",
      "http_port":"20002"
    }
  }'
```

### NIXL Connector (Multi-Host)

```bash
# Prefill on Machine A
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP1} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=all \
vllm serve <MODEL> \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{
    "kv_connector":"NixlConnector",
    "kv_role":"kv_producer",
    "kv_load_failure_policy":"fail"
  }'

# Decode on Machine B
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP2} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=all \
vllm serve <MODEL> \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{
    "kv_connector":"NixlConnector",
    "kv_role":"kv_consumer",
    "kv_load_failure_policy":"fail"
  }'
```

## Trade-offs and Limitations

### Advantages
- Independent TTFT/ITL optimization
- Eliminates prefill-decode interference
- Dynamic scaling without restart (P2P connectors)
- Flexible deployment ratios (xPyD)

### Disadvantages
- Increased operational complexity (multiple instance types)
- KV transfer overhead (network bandwidth, GPU buffer memory)
- NCCL memory overhead limits large-scale deployments
- No throughput improvement (latency-only optimization)

### When to Use

**Use disaggregated serving when:**
- Tail latency SLOs are strict
- TTFT and ITL have different optimization targets
- Workload has variable prompt/output length ratios
- Need independent scaling of prefill vs decode capacity

**Avoid when:**
- Throughput is the only metric
- Workload is uniform (similar prompt/output lengths)
- Operational complexity is a major constraint
- Single-instance optimization is sufficient

## Related Concepts

- [[KV Cache Transfer]] — Connector architectures and transfer mechanisms
- [[KV Cache]] — Fundamentals of KV cache management
- [[Prefix Caching]] — Reusing cached prefixes across requests
- [[Chunked Prefill]] — Alternative tail latency control via chunking
- [[vLLM Engine]] — Scheduler and worker architecture
- [[Tensor Parallelism]] — Intra-instance parallelism (TP configuration affects xPyD design)
- [[Pipeline Parallelism]] — Inter-stage parallelism (not yet supported in disaggregated mode)

## References

- vLLM docs: `docs/features/disagg_prefill.md`
- P2P NCCL design: `docs/design/p2p_nccl_connector.md`
- NIXL usage: `docs/features/nixl_connector_usage.md`
- Encoder disaggregation: `docs/features/disagg_encoder.md`
- [[vllm-disagg-serving]] — Source summary
