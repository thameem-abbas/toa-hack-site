---
title: vLLM MoE Design Documents
type: source
created: 2026-04-25
source_files:
  - /tmp/vllm/docs/design/fused_moe_modular_kernel.md
  - /tmp/vllm/docs/design/moe_kernel_features.md
  - /tmp/vllm/docs/design/dbo.md
  - /tmp/vllm/docs/serving/expert_parallel_deployment.md
pages_created:
  - "Mixture of Experts"
  - "Expert Parallelism"
  - "FusedMoE Modular Kernel"
  - "Dual Batch Overlap"
---

# vLLM MoE Design Documents

Comprehensive design documentation for vLLM's Mixture of Experts inference architecture, covering the modular kernel framework, communication backends, expert parallelism, and dual batch overlap optimization.

## Overview

Four interconnected documents describing vLLM's MoE inference stack:

1. **Fused MoE Modular Kernel**: Three-component architecture (Prepare/Finalize, Experts, Weight/Reduce)
2. **MoE Kernel Features**: Catalog of all2all backends and expert kernel implementations
3. **Dual Batch Overlap**: Communication-computation overlap via microbatching
4. **Expert Parallel Deployment**: Production deployment guide for EP+DP configurations

## Key Concepts

### Modular Kernel Architecture

**Three pluggable components**:
- `FusedMoEPrepareAndFinalizeModular`: Quantization + all-to-all dispatch/combine
- `FusedMoEExpertsModular`: Core expert computation (W1, activation, W2)
- `TopKWeightAndReduce`: Router weight application + reduction across experts

**Benefits**:
- Mix-and-match backends and expert kernels
- Independent development of communication and compute
- Automated compatibility testing

### Activation Formats

**Standard/Contiguous**: `(M, K)` activations, `(M, topk)` ids/weights
- Used by: `deepep_high_throughput`, `flashinfer_nvlink_*`, naive
- Better for prefill (large batches, contiguous memory)

**Batched**: `(num_experts, max_tokens, K)` activations, tokens grouped by expert
- Used by: `deepep_low_latency`
- Better for decode (CUDA graph compatible, natural expert parallelism)

### All-to-All Backends

| Backend | Format | Async | Quant | Use Case |
|---------|--------|-------|-------|----------|
| naive | standard | No | all | General purpose |
| deepep_high_throughput | standard | Yes | FP8 G(128) | Prefill workloads |
| deepep_low_latency | batched | Yes | FP8 G(128) | Decode workloads |
| flashinfer_nvlink_one_sided | standard | No | NVFP4 | MNNVL systems |
| flashinfer_nvlink_two_sided | standard | No | FP8, NVFP4 | MNNVL systems |

### Expert Kernels

| Kernel | Format | Quant | Features |
|--------|--------|-------|----------|
| TritonExperts | standard | all | General purpose |
| BatchedTritonExperts | batched | all | CUDA graph compatible |
| DeepGemmExperts | standard | FP8 G(128) | Highly optimized |
| BatchedDeepGemmExperts | batched | FP8 G(128) | Decode-optimized |
| CutlassExpertsFp8 | standard | FP8 A/T | CUDA-only |
| CutlassExpertsFp4 | standard | NVFP4 A/T | 4-bit quantization |
| FlashInferExperts | standard | FP8, NVFP4 T | NVLink-optimized |
| MarlinExperts | standard | uint4/8, fp4/8 | Low-bit quant |

## Expert Parallelism

### Configuration

```text
EP_SIZE = TP_SIZE × DP_SIZE
```

- **Single node**: `--tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel`
  - Experts sharded across 8 GPUs
  - Attention replicated 8 times
  
- **Multi node**: Additional flags for distributed coordination
  - `--data-parallel-size`: Total DP size across nodes
  - `--data-parallel-size-local`: DP size on this node
  - `--data-parallel-start-rank`: Rank offset for non-primary nodes
  - `--headless`: Secondary nodes run without API server

### Load Balancing (EPLB)

**Problem**: Token routing can be highly skewed (10× imbalance common).

**Solution**: Expert Parallel Load Balancer dynamically redistributes experts.

**Configuration**:
```bash
--enable-eplb \
--eplb-config '{
  "window_size": 1000,
  "step_interval": 3000,
  "num_redundant_experts": 2,
  "log_balancedness": true
}'
```

**Metrics**:
- Track tokens per expert over sliding window (1000 steps)
- Rebalance every 3000 steps
- Replicate hot experts on multiple GPUs (2 redundant experts)
- Memory overhead: ~2.4 GB per redundant expert (DeepSeek-V3)

**Benefit**: 20-40% throughput improvement for skewed workloads.

## Dual Batch Overlap

### Mechanism

Split batch into two microbatches, execute on separate CPU threads with ping-pong synchronization:

```
Thread 0: Dispatch μbatch 0 → Compute μbatch 1 → Combine μbatch 0
Thread 1: Dispatch μbatch 1 → Compute μbatch 0 → Combine μbatch 1
```

**Overlap schedule** (for models with shared experts):
```
Comp: |-A0₀-A1₀-||-MLP₁-||-S₁-MLP₀-||-S₀-A0₁-A1₁-|
Comm: |----D₁---||--D₀--||----C₁---||-----C₀-----|
```

### Configuration

```bash
vllm serve model \
  --enable-dbo \
  --data-parallel-size 2 \              # DP > 1 required
  --enable-expert-parallel \             # EP required
  --all2all-backend deepep_low_latency \ # Async backend required
  --dbo-decode-token-threshold 64 \      # Min tokens for decode batch
  --dbo-prefill-token-threshold 256      # Min tokens for prefill batch
```

**Requirements**:
- DP + EP deployment
- Async backend (DeepEP high-throughput or low-latency)
- Full CUDA graphs (DBO captures both microbatches)

**Performance**: 1.3-1.8× speedup for decode-heavy workloads.

## Quantization

### FP8 Block Quantization (G(128))

DeepEP backends prefer:
- Block size 128 elements
- FP8 E4M3 format (8-bit)
- 2× memory reduction vs BF16
- <1% accuracy loss

### NVFP4 (4-bit)

FlashInfer and CUTLASS kernels:
- 4-bit floating point
- 4× memory reduction
- Per-tensor or per-activation scales
- ~2-3% accuracy loss

### MXFP4 (Microscaling FP4)

TRT-LLM kernels:
- Block-wise scaling with block size 16 or 32
- Better accuracy than plain NVFP4
- Requires TensorRT-LLM library

## Disaggregated Serving

Separate prefill and decode instances with KV cache transfer:

**Prefill instance**:
```bash
vllm serve model \
  --all2all-backend deepep_high_throughput \
  --enable-expert-parallel \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

**Decode instance**:
```bash
vllm serve model \
  --all2all-backend deepep_low_latency \
  --enable-expert-parallel \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

**KV transfer**: GPU-Direct RDMA via NIXL (requires `gdrcopy` for best performance).

## Testing

### Modular Kernel Combinations

Automated testing of all prepare/finalize × experts combinations:
```bash
python3 -m tests.kernels.moe.test_modular_kernel_combinations \
  --pf-type DeepEPLLPrepareAndFinalize \
  --experts-type BatchedTritonExperts
```

### Profiling

Torch trace for single forward pass:
```bash
python3 -m tests.kernels.moe.modular_kernel_tools.profile_modular_kernel \
  --pf-type DeepEPHTPrepareAndFinalize \
  --experts-type TritonExperts
```

## Key Metrics

### Memory Footprint

- **DeepSeek-V3**: 256 experts, ~671B parameters total
  - BF16: ~500 GB expert weights
  - FP8: ~250 GB expert weights
  - Per GPU (16 GPUs): 31 GB (BF16) or 15.6 GB (FP8)

### Communication Overhead

- **Single node (NVLink)**: 600 GB/s per GPU (H100)
- **Multi node (InfiniBand)**: 200-400 GB/s (varies by config)
- **Dispatch + Combine**: ~200-500 μs per layer for typical batch sizes

### Performance Gains

- **DBO**: 1.3-1.8× speedup (decode-heavy workloads)
- **EPLB**: 20-40% throughput improvement (skewed routing)
- **FP8 quantization**: 2× memory reduction, 1.2-1.5× speedup
- **EP vs TP**: 1.2-1.4× throughput improvement for MoE models

## Troubleshooting

### InfiniBand Issues

- **Hang during init**: Set `GLOO_SOCKET_IFNAME=eth0`
- **NVSHMEM errors**: Run `tools/ep_kernels/configure_system_drivers.sh`, reboot
- **Memory registration**: Set `ulimit -l unlimited`
- **Peer disconnect**: Use `hostNetwork: true` and `privileged: true` in Kubernetes

### DBO Not Activating

- Check `--data-parallel-size > 1`
- Check `--enable-expert-parallel`
- Check `--all2all-backend` is DeepEP variant
- Check token count exceeds thresholds

### EPLB Memory Issues

- Reduce `num_redundant_experts`
- Use FP8 quantization to free memory
- Increase GPU memory or reduce KV cache size

## Related Pages

- [[Mixture of Experts]] — Core MoE concept
- [[Expert Parallelism]] — EP parallelism strategy
- [[FusedMoE Modular Kernel]] — Modular kernel architecture
- [[Dual Batch Overlap]] — Communication-compute overlap
- [[Data Parallelism]] — DP strategy
- [[Tensor Parallelism]] — TP strategy
- [[vLLM Engine]] — vLLM architecture
- [[CUDA Graphs]] — Kernel capture/replay
