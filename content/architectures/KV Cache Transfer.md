---
title: KV Cache Transfer
type: architecture
created: 2026-04-25
tags: [kv-cache, distributed-inference, nccl, nixl, rdma]
---

# KV Cache Transfer

Architecture for transferring KV cache between vLLM instances in [[Disaggregated Serving]] deployments. Enables prefill instances to send computed KV cache to decode instances, allowing decode to skip prefill computation.

## Overview

In disaggregated serving, prefill and decode run in separate vLLM instances on potentially different machines. The prefill instance computes the KV cache for the input prompt, then transfers it to the decode instance via a **connector**. The decode instance loads the KV cache and proceeds directly to token generation.

**Key requirement:** KV cache transfer must be fast enough that total latency (prefill + transfer + decode) < baseline unified serving latency. Otherwise, transfer overhead defeats the purpose.

## Core Abstractions

Located in `vllm/distributed/kv_transfer/`:

### Connector

Interface for KV consumer to retrieve KV cache from KV producer.

**Roles:**
- **Scheduler connector:** Lives in scheduler process, schedules KV transfer operations
- **Worker connector:** Lives in worker processes, executes transfer operations

**Lifecycle:**
1. Prefill worker computes KV cache layer-by-layer
2. Scheduler connector schedules send/insert
3. Worker connector transmits KV via Pipe or LookupBuffer
4. Decode scheduler connector checks availability
5. Decode worker connector retrieves KV cache
6. Decode worker loads KV into attention backend

### LookupBuffer

SQL-like buffer for storing and retrieving KV cache.

**API:**
- `insert(kv_cache)`: **Non-blocking** insert operation
- `drop_select(condition)`: **Blocking** select-and-drop (returns matching KV, removes from buffer)

**Semantics:**
- Prefill side: `insert()` KV cache after computing each layer
- Decode side: `drop_select()` blocks until matching KV available, then removes it

### Pipe

Single-direction FIFO for tensor transmission.

**API:**
- `send_tensor(tensor)`: Async send
- `recv_tensor()`: Async receive

Used for point-to-point tensor transfer in connectors like P2P NCCL.

## Connector Implementations

### P2P NCCL Connector

Point-to-point NCCL communication with ZMQ metadata channel. Designed for xPyD deployments with dynamic scaling.

#### Communication Stack

**ZMQ (Control Plane):**
- Handshake and connection establishment
- Metadata transfer (tensor shapes, dtypes)
- Service discovery (register `http_addr → zmq_addr`)
- Heartbeat monitoring (default: every 3s)

**NCCL (Data Plane):**
- KV cache tensor transfer
- Point-to-point groups (world_size=2)
- GPU-to-GPU direct transfer (via NVLink/IB)

#### Transfer Modes

Three modes specified via `kv_connector_extra_config.send_type`:

**1. PUT_ASYNC (Best Performance)**
- Prefill **actively sends** KV to decode
- Uses dedicated send thread (non-blocking)
- Decode receives in dedicated receive thread
- KV stored in decode GPU buffer or memory pool

**2. GET**
- Prefill stores KV in local buffer
- Decode **actively pulls** KV when ready
- Requires larger `kv_buffer_size` on prefill side
- Slower than PUT_ASYNC

**3. PUT**
- Prefill actively sends KV to decode
- **Synchronous** send (blocks main process)
- Slowest mode

**Performance ranking:** PUT_ASYNC > GET > PUT

#### Memory Management

**GPU Buffer (`kv_buffer_size`):**
- Temporary storage for incoming KV cache
- Prefill: minimal in PUT/PUT_ASYNC modes (set to 1), large in GET mode
- Decode: 5-10% of GPU memory (empirical)
- Too small → overflow to memory pool (higher latency)
- Too large → reduces inference KV capacity → smaller batch size → lower throughput

**Tensor Memory Pool:**
- CPU memory fallback (buddy allocator)
- Acts as "flood diversion" during traffic spikes
- PCIe 4.0 bandwidth: ~21 GB/s (usually faster than prefill rate)
- Large capacity (TB-scale on servers)
- No need for prefix caching or block reuse (unlike GPU KV cache)

#### NCCL Group Topology

**Constraint:** Currently supports symmetric TP only (asymmetric TP/PP planned).

**Example: 1P2D with TP=2:**
- 7 NCCL groups total:
  - 3 intra-instance TP groups (P, D0, D1)
  - 4 inter-instance transfer groups:
    - P_rank0 ↔ D0_rank0
    - P_rank0 ↔ D1_rank0
    - P_rank1 ↔ D0_rank1
    - P_rank1 ↔ D1_rank1

**Memory overhead per NCCL group:**
- `NCCL_MAX_NCHANNELS=8`: ~52 MB
- `NCCL_MAX_NCHANNELS=16`: ~100 MB

**Scalability limit:**
- Large-scale deployments (e.g., 96P144D) create too many NCCL groups
- Memory overhead becomes prohibitive
- Future: Migrate to RDMA or UCCL

#### Dynamic Scaling

**Key feature:** Add/remove P/D instances without full system restart.

**Mechanism:**
- Each instance creates single `P2pNcclEngine`
- ZMQ server listens for connection requests
- First KV transfer: establish ZMQ + NCCL group
- Subsequent transfers: reuse existing connections
- Point-to-point groups (world_size=2) not constrained by global rank

**Service discovery:**
- Proxy maintains `http_addr → zmq_addr` mapping
- Instances send heartbeat to proxy (every 3s)
- Proxy removes timed-out instances (planned feature)

#### Request Flow

1. Client → Proxy: HTTP request
2. Proxy selects 1P1D, generates `request_id` encoding P/D addresses
3. Proxy → P: Prefill request with `max_tokens=1`
4. Proxy → D: Original request (full output)
5. P computes prefill → sends KV via PUT_ASYNC to D's `zmq_addr`
6. D dedicated thread receives KV → stores in GPU buffer or memory pool
7. D main thread retrieves KV from buffer → loads into attention
8. D decodes → streams response to Proxy → Client

#### Configuration Example

```bash
# Prefill instance
vllm serve <MODEL> \
  --gpu-memory-utilization 0.9 \
  --kv-transfer-config '{
    "kv_connector":"P2pNcclConnector",
    "kv_role":"kv_producer",
    "kv_buffer_size":"1e1",
    "kv_port":"21001",
    "kv_connector_extra_config":{
      "proxy_ip":"10.0.1.1",
      "proxy_port":"30001",
      "http_port":"20001",
      "send_type":"PUT_ASYNC"
    }
  }'

# Decode instance
vllm serve <MODEL> \
  --gpu-memory-utilization 0.7 \
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

### NIXL Connector

High-performance connector using NVIDIA Inference eXchange Library (NIXL) for fully asynchronous KV transfer.

#### Transport Backends (Plugins)

NIXL supports multiple transport backends selected via `kv_connector_extra_config.backends`:

**UCX (Default):** Unified Communication X
- Supports RDMA, GPU Direct, shared memory
- Environment variables: `UCX_TLS`, `UCX_NET_DEVICES`
- Example: `UCX_TLS=all UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1`

**LIBFABRIC:** Alternative fabric library
- Requires NIXL compiled with LIBFABRIC support

**GDS:** GPU Direct Storage
- Direct GPU-to-storage path

**RIXL:** ROCm variant for AMD GPUs
- Included in ROCm docker base image
- UCX support via `requirements/kv_connectors_rocm.txt`

**Multi-backend example:**
```bash
--kv-transfer-config '{
  "kv_connector":"NixlConnector",
  "kv_role":"kv_both",
  "kv_buffer_device":"cuda",
  "kv_connector_extra_config":{"backends":["UCX", "GDS"]}
}'
```

#### Fully Asynchronous Operations

Unlike P2P NCCL's thread-based async, NIXL provides native async send/recv:
- Non-blocking by design
- No dedicated send/recv threads needed
- Lower CPU overhead

#### Multi-Host Configuration

**Environment variables:**

`VLLM_NIXL_SIDE_CHANNEL_PORT` (required)
- Port for initial NIXL handshake
- Default: 5600
- Each vLLM worker needs unique port on its host
- Same port number can be used across different hosts
- For TP/DP: port = base_port + dp_rank

`VLLM_NIXL_SIDE_CHANNEL_HOST`
- Host address for multi-machine deployments
- Default: "localhost"
- Connection info passed via KVTransferParams from prefill to decode

`VLLM_NIXL_ABORT_REQUEST_TIMEOUT`
- Timeout for releasing prefill KV cache if decode hasn't read it
- Default: 480 seconds
- Prevents indefinite KV cache holding on aborted requests

**Example (multi-machine):**
```bash
# Prefill on Machine A (IP: ${IP1})
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

# Decode on Machine B (IP: ${IP2})
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

#### KV Load Failure Policy

Controls behavior when decode fails to load KV from prefill:

**`kv_load_failure_policy="fail"` (Default, Recommended)**
- Immediately fail request with error
- Prevents performance degradation
- Avoids decode-side prefill recomputation

**`kv_load_failure_policy="recompute"`**
- Decode instance recomputes prefill locally
- Causes performance jitter (decode config optimized for ITL, not TTFT)
- Interferes with other decode requests (tail latency spike)
- Defeats purpose of disaggregation

> [!warning] Production Recommendation
> Always use `kv_load_failure_policy="fail"` in production. Recompute mode is for debugging only.

#### Experimental Features

**Heterogeneous KV Layout:**
- Prefill with 'HND', decode with 'NHD' (or vice versa)
- Enable: `--kv-transfer-config '{..., "enable_permute_local_kv":"True"}'`

**Cross-layer Blocks:**
- Contiguous memory for logical blocks across layers
- Reduces number of buffers to transfer
- Enable: `--kv-transfer-config '{..., "kv_connector_extra_config": {"enable_cross_layers_blocks": "True"}}'`

#### ROCm Support (RIXL)

- Included in `docker/Dockerfile.rocm_base`
- RIXL is NIXL variant for AMD GPUs
- Dependencies in `requirements/kv_connectors_rocm.txt`
- Future: Pre-compiled binary packages (not in docker image)

### LMCache Connector V1

Uses NIXL as underlying transport.

```bash
--kv-transfer-config '{
  "kv_connector":"LMCacheConnectorV1",
  "kv_role":"kv_both"
}'
```

See: `examples/others/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh`

### Mooncake Connector

Third-party storage backend for KV cache.

```bash
--kv-transfer-config '{
  "kv_connector":"MooncakeConnector",
  "kv_role":"kv_both"
}'
```

See: `examples/online_serving/disaggregated_serving/mooncake_connector/run_mooncake_connector.sh`

### Multi Connector

Combines multiple connectors in an ordered list. Uses `kv_connector_extra_config` to stash connector configurations.

```bash
--kv-transfer-config '{
  "kv_connector":"MultiConnector",
  "kv_role":"kv_both",
  "kv_connector_extra_config":{
    "connectors":[
      {
        "kv_connector":"NixlConnector",
        "kv_role":"kv_both"
      },
      {
        "kv_connector":"ExampleConnector",
        "kv_role":"kv_both",
        "kv_connector_extra_config":{
          "shared_storage_path":"local_storage"
        }
      }
    ]
  }
}'
```

### Offloading Connector

Offloads KV data to CPU memory or disk.

**Configuration:**
```bash
--kv-transfer-config '{
  "kv_connector":"OffloadingConnector",
  "kv_role":"kv_both",
  "kv_connector_extra_config":{
    "block_size": 64,
    "cpu_bytes_to_use": 1000000000
  }
}'
```

**Use case:** When GPU memory is limited, offload to CPU/disk as fallback.

### FlexKV Connector V1

Distributed KV store with multi-level cache management for ultra-large-scale inference.

```bash
--kv-transfer-config '{
  "kv_connector":"FlexKVConnectorV1",
  "kv_role":"kv_both"
}'
```

See: `examples/offline_inference/prefix_caching_flexkv.py`

### Example Connector

Reference implementation for testing and development.

```bash
--kv-transfer-config '{
  "kv_connector":"ExampleConnector",
  "kv_role":"kv_both",
  "kv_connector_extra_config":{
    "shared_storage_path":"local_storage"
  }
}'
```

See: `examples/offline_inference/disaggregated-prefill-v1/run.sh`

## Encoder Cache (EC) Transfer

For multimodal models with disaggregated encoder. See [[Disaggregated Serving#Disaggregated Encoder]].

**Location:** `vllm/distributed/ec_transfer/`

**Connector:** `ECConnector`
- Scheduler role: Checks cache existence, schedules loads
- Worker role: Loads embeddings into memory

**Flow:**
1. Encoder instance processes vision input → EC embeddings
2. ECConnector transfers to PD instance
3. PD injects at required attention layers
4. PD performs language prefill + decode

**E→P→D:**
1. Encoder → Prefill: EC transfer
2. Prefill executes 1 step (prefill → 1 token)
3. Prefill → Decode: KV transfer (via NixlConnector or other)
4. Decode generates remaining tokens

## Connector Extension Patterns

Three recommended implementation approaches:

### 1. Fully-Customized Connector

Implement `Connector` interface directly.

**Pros:**
- Maximum control
- Can edit model input for custom prefilling
- Call arbitrary third-party libraries

**Cons:**
- Fragile (may break with vLLM updates)
- High maintenance burden

### 2. Database-like Connector

Implement `LookupBuffer` with SQL-like `insert` / `drop_select` API.

**Pros:**
- Clean abstraction
- Reuses vLLM's buffer management
- More stable across versions

**Cons:**
- Less flexibility than fully-customized

### 3. Distributed P2P Connector

Implement `Pipe` with `send_tensor` / `recv_tensor` API (like `torch.distributed`).

**Pros:**
- Familiar API for distributed systems developers
- Good for P2P and collective communication

**Cons:**
- Lower-level than LookupBuffer

## Performance Characteristics

### P2P NCCL Benchmark (1K input, 200 output)

- **E2E P99 latency:** ~2s
- **Transfer modes:** PUT_ASYNC (best) > GET > PUT
- **kv_buffer_size:** 10% GPU memory (empirical optimum)
- **Memory pool speed:** PCIe 4.0 ~21 GB/s

### NIXL vs P2P NCCL

| Feature | P2P NCCL | NIXL |
|---------|----------|------|
| Async operations | Thread-based | Native async |
| Multi-host | ZMQ + NCCL | UCX RDMA |
| ROCm support | NCCL only | RIXL |
| NCCL group overhead | 52-100 MB each | None |
| Dynamic scaling | Yes | Yes |
| Symmetric TP only | Yes | No constraint |

### Scalability

**P2P NCCL limitations:**
- Large xPyD (e.g., 96P144D) creates hundreds of NCCL groups
- Memory overhead: 52-100 MB × num_groups
- Future: RDMA/UCCL migration

**NIXL advantages:**
- No NCCL group overhead
- Native RDMA support via UCX
- Better for large-scale deployments

## Integration with vLLM Engine

### Scheduler Integration

Scheduler connector orchestrates transfer:
1. Prefill scheduler: After each layer's attention, schedule `insert(kv_cache)`
2. Decode scheduler: Before attention, schedule `drop_select(request_id)`
3. Worker connectors execute actual transfer

### Worker Integration

Worker connector executes transfer during model execution:
1. Prefill worker: After computing attention layer i, send KV[i] via Pipe
2. Decode worker: Before attention layer i, receive KV[i] via Pipe or LookupBuffer
3. Decode worker: Load KV[i] into attention backend (skip prefill computation)

### Layer-by-Layer Transfer

**Why layer-by-layer?**
- Overlaps computation and communication
- Reduces memory footprint (don't need to store all layers simultaneously)
- Allows pipelining (prefill layer i+1 while transferring layer i)

**Workflow:**
```
Prefill:
  for layer in layers:
    compute_attention(layer)
    connector.send(kv[layer])

Decode:
  for layer in layers:
    kv[layer] = connector.recv(layer)
    load_kv_to_attention(kv[layer])
    compute_attention(layer)  # uses transferred KV, no prefill
```

See: `docs/features/disagg_prefill/workflow.png`

## Compatibility with Prefix Caching

[[Prefix Caching]] and KV cache transfer are complementary:

**Within instance:** Prefix caching reuses KV blocks for shared prefixes
**Across instances:** KV transfer sends computed KV from prefill to decode

**Interaction:**
- Prefill uses prefix cache to avoid recomputing shared prefixes
- Prefill sends only newly computed KV to decode
- Decode can also use prefix caching for its own shared prefixes

**Metadata:**
- Connector must preserve prefix hash information
- Decode instance can rebuild prefix cache from transferred KV

## Cross-References

- [[Disaggregated Serving]] — High-level concept and deployment patterns
- [[KV Cache]] — KV cache fundamentals and memory management
- [[Prefix Caching]] — Reusing cached prefixes
- [[vLLM Engine]] — Scheduler and worker architecture
- [[Tensor Parallelism]] — Symmetric TP required for P2P NCCL
- [[Chunked Prefill]] — Alternative tail latency control
- [[CUDA Graphs]] — Attention backend compatibility with transfer mechanisms
- [[PagedAttention]] — Block-based KV cache allocation

## References

- `docs/features/disagg_prefill.md` — Feature overview
- `docs/design/p2p_nccl_connector.md` — P2P NCCL design
- `docs/features/nixl_connector_usage.md` — NIXL usage guide
- `vllm/distributed/kv_transfer/` — Implementation
- `vllm/distributed/ec_transfer/` — Encoder cache transfer
- [[vllm-disagg-serving]] — Source summary
