---
title: Expert Parallelism
type: concept
created: 2026-04-25
related:
  - "Mixture of Experts"
  - "Data Parallelism"
  - "Tensor Parallelism"
  - "vLLM Engine"
  - "FusedMoE Modular Kernel"
  - "Dual Batch Overlap"
---

# Expert Parallelism

Expert Parallelism (EP) is a distributed inference strategy for [[Mixture of Experts]] models where experts are sharded across multiple GPUs, allowing each GPU to store a subset of experts rather than all experts.

## Concept

### Problem: MoE Memory Footprint

MoE models have all expert parameters even though only top-K are used per token:
- DeepSeek-V3: 256 experts, 671B total parameters
- Even with top-8 routing, all 256 experts must be GPU-resident
- Single GPU memory insufficient for large MoE models

### Solution: Expert Sharding

Distribute experts across GPUs:
- **8 GPUs**: Each GPU holds 256/8 = 32 experts
- **16 GPUs**: Each GPU holds 256/16 = 16 experts
- All-to-all communication routes tokens to the GPU holding their selected expert

### EP vs Tensor Parallelism

| Dimension | Tensor Parallelism | Expert Parallelism |
|-----------|-------------------|-------------------|
| **What is sharded** | Weights within each layer (QKV, FFN) | Entire experts across MoE layer |
| **Communication** | All-reduce after matmul | All-to-all dispatch/combine |
| **When used** | Dense layers (attention, non-MoE FFN) | MoE layers only |
| **Compute overlap** | Limited (synchronous all-reduce) | High (async all-to-all) |
| **Load balance** | Always balanced | Can be skewed (data-dependent routing) |

## Architecture in vLLM

### EP Size Calculation

```text
EP_SIZE = TP_SIZE × DP_SIZE
```

Where:
- `TP_SIZE`: Tensor parallel size (for attention layers)
- `DP_SIZE`: Data parallel size (model replicas)
- `EP_SIZE`: Expert parallel size (auto-computed)

**Example**: TP=2, DP=4 → 8 GPUs total
- EP_SIZE = 2 × 4 = 8
- Experts sharded across all 8 GPUs
- Attention uses TP=2 within each of 4 DP groups

### Layer Behavior with EP Enabled

vLLM's `--enable-expert-parallel` flag changes only MoE layers:

| Layer Type | Without EP | With EP |
|-----------|-----------|---------|
| **Attention layers** | TP or replicated | Same (TP or replicated) |
| **Dense FFN layers** | TP or replicated | Same (TP or replicated) |
| **MoE expert layers** | TP or replicated | **EP across all GPUs** |

**Key insight**: EP provides expert-specific parallelism without changing dense layer behavior.

### Single-Node Configuration

Example: DeepSeek-V3-0324 on 8× H200 GPUs

```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel
```

- Attention weights: Replicated 8 times (TP=1, so full DP replication)
- Expert weights: Sharded across 8 GPUs (32 experts per GPU)
- KV cache: Per-DP-rank (8 separate caches)

### Multi-Node Configuration

Example: DeepSeek-V3-0324 on 2 nodes, 16 GPUs total

```bash
# Node 1 (primary)
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --all2all-backend deepep_low_latency \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --data-parallel-size 16 \
  --data-parallel-size-local 8 \
  --data-parallel-address 192.168.1.100 \
  --data-parallel-rpc-port 13345 \
  --api-server-count=8

# Node 2 (headless worker)
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --all2all-backend deepep_low_latency \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --data-parallel-size 16 \
  --data-parallel-size-local 8 \
  --data-parallel-start-rank 8 \
  --data-parallel-address 192.168.1.100 \
  --data-parallel-rpc-port 13345 \
  --headless
```

- `--data-parallel-size`: Total DP size across all nodes
- `--data-parallel-size-local`: DP size on this node
- `--data-parallel-start-rank`: Rank offset for non-primary nodes
- `--headless`: Secondary nodes don't run API server

## All-to-All Communication Backends

vLLM provides multiple communication backends via `--all2all-backend`:

### naive (allgather_reducescatter)

- **Method**: Standard PyTorch distributed primitives
- **Use case**: Default, general purpose
- **Pros**: Works everywhere, no special dependencies
- **Cons**: Not optimized for MoE patterns

### deepep_high_throughput

- **Method**: DeepEP library, continuous layout, grouped GEMM
- **Use case**: Prefill-dominated workloads, multi-node
- **Quantization**: FP8 with block size 128 (G(128), or A/T after dispatch)
- **Activation format**: Standard/contiguous
- **Async support**: Yes (enables [[Dual Batch Overlap]])
- **Pros**: Optimized for large batch prefill
- **Cons**: Requires DeepEP installation, not ideal for mixed workloads

### deepep_low_latency

- **Method**: DeepEP library, masked layout, CUDA graph compatible
- **Use case**: Decode-dominated workloads, low-latency serving
- **Quantization**: FP8 (G(128), A/T after dispatch)
- **Activation format**: Batched (tokens grouped by expert)
- **Async support**: Yes (enables [[Dual Batch Overlap]])
- **Pros**: Ultra-low latency, CUDA graph support
- **Cons**: Requires DeepEP, optimized for decode only

### flashinfer_nvlink_one_sided / two_sided

- **Method**: FlashInfer library, one-sided or two-sided all-to-all
- **Use case**: Multi-node NVLink (MNNVL) systems
- **Quantization**: NVFP4 or FP8
- **Activation format**: Standard
- **Async support**: No
- **Pros**: Optimized for NVLink fabric
- **Cons**: Requires specific hardware topology

## All-to-All Communication Flow

### Dispatch Phase (All-to-All Scatter)

For each token:
1. Router/gating network computes scores for all experts
2. Select top-K expert indices and weights
3. **All-to-all dispatch**: Send each token to GPU holding its selected expert(s)
   - GPU 0 sends tokens routed to experts 32-63 to GPU 1
   - GPU 1 sends tokens routed to experts 0-31 to GPU 0
   - etc.
4. Each GPU receives tokens for its local experts

### Expert Computation

Each GPU processes tokens for its local experts:
1. Group tokens by expert (if not already batched)
2. For each expert: W1 matmul → activation → W2 matmul
3. Optionally apply router weights to outputs

### Combine Phase (All-to-All Gather)

Reverse of dispatch:
1. **All-to-all combine**: Send expert outputs back to token's origin GPU
2. Each GPU receives outputs for its original tokens
3. Apply router weights (if not done in expert computation)
4. Reduce across top-K experts per token

## Performance Challenges

### Load Imbalance

Token routing is data-dependent and often skewed:
- Example: In code generation, "code expert" may receive 60% of tokens
- Most skewed expert can have 10× load of least loaded expert
- Creates GPU utilization imbalance: some GPUs bottleneck, others idle

**Symptom**: Observed throughput << theoretical maximum even with high batch size

### Communication Bottleneck

All-to-all is bandwidth-intensive:
- **Single node**: NVLink bandwidth (600 GB/s per GPU for H100)
- **Multi node**: InfiniBand or Ethernet bandwidth (much lower)
- **Impact**: All-to-all can dominate latency for decode (small batches)

**Mitigation**: [[Dual Batch Overlap]] overlaps communication with compute

### Expert Footprint vs KV Cache Tradeoff

More EP ranks → fewer experts per GPU → more memory for KV cache:
- **8 GPUs**: 32 experts/GPU, more KV cache space
- **16 GPUs**: 16 experts/GPU, even more KV cache space

But: More communication overhead with larger EP size.

## Optimizations

### Dual Batch Overlap (DBO)

Enable via `--enable-dbo` (requires DeepEP backend, DP+EP):

Split batch into 2 microbatches, run on separate CPU threads:
- While microbatch 1 dispatches, microbatch 2 computes
- While microbatch 1 combines, microbatch 2 dispatches
- Overlap achieved via CPU thread ping-pong at yield points

**Benefit**: Hide all-to-all latency behind compute
**Cost**: Requires CUDA graphs, adds complexity

See [[Dual Batch Overlap]] for details.

### Expert Parallel Load Balancer (EPLB)

Enable via `--enable-eplb`:

Dynamic expert reassignment to balance load:
1. Track tokens per expert over sliding window (default: 1000 steps)
2. Compute load imbalance metrics
3. Periodically rebalance expert-to-GPU assignments (default: every 3000 steps)
4. Use redundant experts: Replicate hot experts on multiple GPUs

**Configuration**:
```bash
--eplb-config '{
  "window_size": 1000,
  "step_interval": 3000,
  "num_redundant_experts": 2,
  "log_balancedness": true
}'
```

**Expert distribution**:
- Without redundancy: `num_experts / num_ep_ranks` per GPU
- With redundancy: `(num_experts + num_redundant_experts) / num_ep_ranks` per GPU

**Memory overhead**: `~2.4 GB per redundant expert` for DeepSeek-V3

**Throughput improvement**: 20-40% for skewed workloads

### Quantization

Reduce expert memory footprint:
- **FP8**: 2× reduction, supported by DeepGemm, CUTLASS, Triton
- **NVFP4**: 4× reduction, supported by CUTLASS, FlashInfer
- **MXFP4**: 4× reduction with block-wise scaling, supported by TRT-LLM

**Example**: DeepSeek-V3 with FP8 experts:
- BF16 experts: ~500 GB (256 experts × ~2 GB each)
- FP8 experts: ~250 GB
- **Benefit**: Fit on fewer GPUs or allocate more memory to KV cache

## Disaggregated Serving (Prefill/Decode Split)

Separate prefill and decode instances with EP:

**Prefill Instance**:
```bash
vllm serve model \
  --all2all-backend deepep_high_throughput \
  --enable-expert-parallel \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

**Decode Instance**:
```bash
vllm serve model \
  --all2all-backend deepep_low_latency \
  --enable-expert-parallel \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

**KV Cache Transfer**:
- Prefill computes KV cache, transfers to decode instance via NIXL (GPU-Direct RDMA)
- Decode receives KV cache, generates tokens
- **Benefit**: Independent scaling, optimized backends for each phase

**Requirements**:
- `gdrcopy` for GPU-Direct RDMA (optional but recommended)
- `nixl` library for KV transfer
- Client-side orchestration to link prefill → decode

## Troubleshooting

### InfiniBand Initialization Hang

**Symptom**: vLLM hangs during distributed initialization on IB clusters

**Solution**: Set environment variable
```bash
export GLOO_SOCKET_IFNAME=eth0
```
Forces torch distributed to use Ethernet for initial setup, IB for data.

### NVSHMEM Errors

**Error**: `init failed for transport: IBGDA` or `NVSHMEM API called before initialization`

**Cause**: InfiniBand GDA kernel modules missing

**Solution**: Run `tools/ep_kernels/configure_system_drivers.sh` on each node, reboot

### Memory Registration Error

**Error**: `non-zero status: 7 cannot register cq buf`

**Cause**: Insufficient locked memory limit

**Solution**: Set `ulimit -l unlimited` on host and all containers

### Peer Disconnect

**Symptom**: NVSHMEM peer disconnect errors

**Cause**: Networking misconfiguration

**Solution** (Kubernetes):
- Set `hostNetwork: true` in pod spec
- Set `securityContext.privileged: true`
- Verify InfiniBand devices visible in pod

## Benchmarking

### Simulating Balanced Routing

Use environment variables to test best-case performance:
```bash
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1
```

### Increasing Dispatch Chunk Size

Tune all-to-all batch size:
```bash
export VLLM_MOE_DP_CHUNK_SIZE=2048  # default varies
```

May improve throughput but can hit NVSHMEM queue depth limit.

**If error** `assert self.nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2`:
```bash
export NVSHMEM_QP_DEPTH=8192  # increase from default
```

### Decode-Only Benchmarking

Simulate disaggregated decode instance (skip prefill):
```bash
vllm serve model \
  --kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both"}'
```

Populates KV cache with random values, measures pure decode throughput.

## References

- Concept: [[Mixture of Experts]]
- Implementation: [[FusedMoE Modular Kernel]]
- Optimization: [[Dual Batch Overlap]]
- Related parallelism: [[Data Parallelism]], [[Tensor Parallelism]]
- Source: [[vllm-moe-design]]
- Engine: [[vLLM Engine]]
