---
title: vLLM Disaggregated Serving
type: source
created: 2026-04-25
tags: [vllm, disaggregated-serving, kv-cache-transfer, distributed-inference]
---

# vLLM Disaggregated Serving

Source summary covering vLLM's disaggregated serving architecture, including disaggregated prefill/decode separation and disaggregated encoder for multimodal models.

## Sources

1. `/tmp/vllm/docs/features/disagg_prefill.md` — Disaggregated prefilling feature overview
2. `/tmp/vllm/docs/design/p2p_nccl_connector.md` — P2P NCCL connector design for xPyD deployments
3. `/tmp/vllm/docs/features/nixl_connector_usage.md` — NIXL connector usage guide
4. `/tmp/vllm/docs/features/disagg_encoder.md` — Disaggregated encoder for multimodal models

## Core Concepts

### Disaggregated Prefill/Decode

vLLM supports running prefill and decode phases in separate instances, enabling independent optimization and scaling:

- **Prefill instances**: Compute-bound, optimized for TTFT
- **Decode instances**: Memory-bound, optimized for ITL
- **xPyD deployments**: x prefill instances + y decode instances (e.g., 1P3D, 3P1D, 96P144D)

**Key benefits:**
- Tune TTFT and ITL independently with different parallel strategies (TP, PP)
- Control tail ITL by preventing prefill jobs from interfering with decode
- Does NOT improve throughput (only latency optimization)

**Alternative approach:** [[Chunked Prefill]] achieves similar tail latency control but requires careful chunk size tuning.

### KV Cache Transfer Connectors

Six connector types for transferring KV cache between prefill and decode:

1. **[[KV Cache Transfer#P2P NCCL Connector|P2pNcclConnector]]** — Point-to-point NCCL with ZMQ metadata, supports PUT/GET/PUT_ASYNC modes
2. **[[KV Cache Transfer#NIXL Connector|NixlConnector]]** — NVIDIA Inference eXchange Library, fully async, supports UCX/RDMA/GPU Direct
3. **[[KV Cache Transfer#LMCache Connector|LMCacheConnectorV1]]** — Uses NIXL as underlying transport
4. **[[KV Cache Transfer#Mooncake Connector|MooncakeConnector]]** — Third-party connector
5. **[[KV Cache Transfer#MultiConnector|MultiConnector]]** — Combines multiple connectors
6. **[[KV Cache Transfer#OffloadingConnector|OffloadingConnector]]** — CPU/disk offload
7. **[[KV Cache Transfer#FlexKV Connector|FlexKVConnectorV1]]** — Distributed KV store with multi-level caching

### Disaggregated Encoder (Multimodal)

For multimodal models, vision encoder can run in separate instance from language model:

- **Encoder instance**: Runs vision encoder (lightweight)
- **PD instance**: Runs language prefill + decode (heavy)
- **Benefits**: Independent scaling, lower TTFT for language-only requests, cross-process encoder output caching

Can be combined with disaggregated prefill: E→P→D (encoder → prefill → decode)

## Architecture Abstractions

### Core Components (in `vllm/distributed/kv_transfer`)

**Connector**
- Interface for KV consumer to retrieve KV caches from KV producer
- Scheduler connector: schedules KV transfer ops
- Worker connectors: execute KV transfer ops

**LookupBuffer**
- `insert(kv_cache)`: non-blocking insert operation
- `drop_select(condition)`: blocking select-and-drop (SQL-like semantics)

**Pipe**
- Single-direction FIFO for tensor transmission
- `send_tensor()` and `recv_tensor()` APIs

### P2P NCCL Connector Details

**Communication Stack:**
- **ZMQ**: Control plane for metadata (handshake, tensor shapes/dtypes)
- **NCCL**: Data plane for KV cache transfer

**Key features:**
- Point-to-point communication (not constrained by rank/world size)
- Dynamic scaling without full system restart
- Each NCCL group: world_size=2 (one prefill rank, one decode rank)
- Heartbeat-based service discovery via proxy

**Transfer modes (performance ranking):**
1. **PUT_ASYNC** (best): Async send via dedicated thread, non-blocking
2. **GET**: Decode pulls KV from prefill buffer
3. **PUT**: Sync send, blocks main process

**Memory management:**
- `kv_buffer_size`: GPU buffer size (typically 5-10% of GPU memory)
- Tensor memory pool: CPU memory fallback when GPU buffer overflows (buddy allocator)
- PCIe 4.0 speed: ~21 GB/s for CPU-GPU transfer

**Topology (symmetric TP only):**
- 1P2D with TP=2: 7 NCCL groups total
  - 3 intra-instance TP groups (one per instance)
  - 4 inter-instance groups (P_rank0↔D0_rank0, P_rank0↔D1_rank0, P_rank1↔D0_rank1, P_rank1↔D1_rank1)
- NCCL group memory: 52MB (NCCL_MAX_NCHANNELS=8) to 100MB (NCCL_MAX_NCHANNELS=16)

**Limitations:**
- Asymmetric TP and PP not yet supported
- Large-scale deployments (e.g., 96P144D) require RDMA/UCCL due to NCCL memory overhead

### NIXL Connector Details

**Transport backends (plugins):**
- **UCX** (default): Unified Communication X, supports RDMA, GPU Direct
- **LIBFABRIC**: Alternative fabric library
- **GDS**: GPU Direct Storage
- **RIXL**: ROCm variant for AMD GPUs

**Key features:**
- Fully asynchronous send/recv
- Multi-host support via `VLLM_NIXL_SIDE_CHANNEL_HOST` and `VLLM_NIXL_SIDE_CHANNEL_PORT`
- `kv_load_failure_policy`: "fail" (default) vs "recompute" (degrades performance)
- Experimental: Heterogeneous KV layout (HND vs NHD), cross-layer blocks

**Environment variables:**
- `VLLM_NIXL_SIDE_CHANNEL_PORT`: Handshake port (default 5600, each worker needs unique port)
- `VLLM_NIXL_SIDE_CHANNEL_HOST`: Host for multi-machine setups
- `VLLM_NIXL_ABORT_REQUEST_TIMEOUT`: Timeout for releasing prefill KV cache (default 480s)

**Configuration:**
```bash
--kv-transfer-config '{
  "kv_connector":"NixlConnector",
  "kv_role":"kv_both",
  "kv_buffer_device":"cuda",
  "kv_connector_extra_config":{"backends":["UCX", "GDS"]}
}'
```

## Deployment Patterns

### Proxy/Router Architecture

**Responsibilities:**
- Route requests to selected 1P1D pairs (round-robin or load-based)
- Generate request_id encoding prefill/decode addresses
- Service discovery: maintain `http_addr → zmq_addr` mapping
- Heartbeat monitoring (every 3s)

**Request flow (P2P NCCL):**
1. Client → Proxy `/v1/completions`
2. Proxy selects 1P1D, generates request_id, modifies `max_tokens=1`
3. Proxy forwards to P instance (prefill only)
4. Proxy forwards original request to D instance
5. P performs prefill, sends KV to D via PUT_ASYNC
6. D receives KV in dedicated thread → GPU buffer or memory pool
7. D performs decode, returns result to Proxy → Client

### xPyD Configurations

**1P3D:** One prefill pool, three decode pools (optimize for decode throughput)
**3P1D:** Three prefill pools, one decode pool (optimize for prefill throughput)
**xPyD:** Arbitrary ratio based on workload characteristics

### Disaggregated Encoder Flow

**E→PD (encoder → prefill/decode):**
1. Encoder instance processes vision input → EC embeddings
2. ECConnector transfers embeddings to PD instance
3. PD injects embeddings at required attention layers
4. PD performs language prefill + decode

**E→P→D (encoder → prefill → decode):**
1. Encoder → Prefill: EC transfer via ECConnector
2. Prefill performs 1 step → 1 token output
3. Prefill → Decode: KV transfer via NixlConnector (or other KV connector)
4. Decode generates remaining tokens

## Implementation Details

**Code locations:**
- `vllm/distributed/kv_transfer/`: Disaggregated prefill implementation
- `vllm/distributed/ec_transfer/`: Disaggregated encoder implementation
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/`: P2P NCCL examples
- `tests/v1/kv_connector/nixl_integration/`: NIXL connector tests
- `benchmarks/disagg_benchmarks/`: Performance benchmarks

**Connector extension patterns:**
1. **Fully-customized connector**: Implement `Connector` interface (most control, most fragile)
2. **Database-like connector**: Implement `LookupBuffer` with `insert`/`drop_select` APIs
3. **Distributed P2P connector**: Implement `Pipe` with `send_tensor`/`recv_tensor` APIs

## Performance Characteristics

### P2P NCCL Benchmark (1K input, 200 output tokens)

- **E2E P99 latency**: ~2s
- **Best transfer mode**: PUT_ASYNC (async, non-blocking)
- **kv_buffer_size tuning**: 10% of GPU memory (empirical)
- **Memory pool fallback**: PCIe 4.0 ~21 GB/s

### NIXL Connector

- **kv_load_failure_policy="fail"**: Prevents performance degradation from decode-side prefill recomputation
- **Async operations**: Non-blocking send/recv
- **Multi-host**: Supports cross-node KV transfer via UCX RDMA

## Cross-References

- [[Disaggregated Serving]] — High-level concept page
- [[KV Cache Transfer]] — Architecture page for all connectors
- [[KV Cache]] — KV cache fundamentals
- [[Prefix Caching]] — Reusing cached prefixes
- [[vLLM Engine]] — Scheduler and worker architecture
- [[Chunked Prefill]] — Alternative approach to tail latency control
- [[Tensor Parallelism]] — Symmetric TP required for P2P NCCL

## Open Questions

1. **Asymmetric TP/PP support**: When will P2P NCCL support asymmetric tensor parallelism?
2. **Large-scale NCCL overhead**: What's the RDMA/UCCL migration plan for 96P144D+?
3. **Chunk size vs disaggregation**: What are the quantitative tradeoffs between chunked prefill and full disaggregation?
4. **Encoder cache hit rate**: What hit rates are achievable in production multimodal workloads?
5. **Multi-connector performance**: How does MultiConnector prioritize/combine different backends?
