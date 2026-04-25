---
title: Mixture of Experts
type: concept
created: 2026-04-25
related:
  - "Expert Parallelism"
  - "FusedMoE Modular Kernel"
  - "Data Parallelism"
  - "Tensor Parallelism"
  - "Dual Batch Overlap"
---

# Mixture of Experts

Mixture of Experts (MoE) is a sparse neural network architecture where only a subset of "expert" networks process each token, enabling larger model capacity with approximately constant compute cost per token.

## Core Mechanism

### Expert Selection via Routing

Each token is routed to top-K experts by a gating network:

1. **Gating/Router Network**: Lightweight network computes scores for all experts
2. **Top-K Selection**: Select K experts with highest scores per token (typically K=1-8)
3. **Weight Normalization**: Normalize scores across selected experts (often via softmax)
4. **Token Dispatch**: Route each token to its selected experts
5. **Weighted Aggregation**: Combine expert outputs using normalized routing weights

### Sparse Activation

Key efficiency property: For a model with N total experts and top-K routing:
- **Parameters used per token**: ~K/N fraction of total expert parameters
- **Total capacity**: Can have N experts, each full-size MLP
- **Compute cost**: Approximately K × (single expert cost), not N × (expert cost)

Example: DeepSeek-V3 with 256 experts and top-K=8 uses only 3.1% of expert parameters per token.

## Benefits

### Scaling Capacity Without Proportional Compute

- **Dense model**: 7B parameters → 7B parameters used per token
- **MoE model**: 56B parameters (8 experts × 7B each) → 7B parameters used per token (top-1)
- **Result**: 8× parameter capacity, similar compute cost

### Specialization

Experts can specialize on different domains, tasks, or linguistic patterns:
- Code vs natural language
- Different human languages
- Technical vs conversational text
- Common vs rare tokens

Evidence: Analysis of DeepSeek-V2/V3 shows experts develop distinct activation patterns for different content types.

## MoE Models in Production

### DeepSeek-V2 / DeepSeek-V3

- **Architecture**: 160 experts, top-6 routing (V2); 256 experts, top-8 routing (V3)
- **Expert type**: Routed FFN experts + shared experts for common patterns
- **Inference challenge**: All experts must fit in GPU memory despite sparse activation
- **vLLM support**: Full support via [[Expert Parallelism]] and [[FusedMoE Modular Kernel]]

### Mixtral-8x7B / Mixtral-8x22B

- **Architecture**: 8 experts, top-2 routing
- **Characteristics**: Relatively small number of large experts
- **Inference**: Fits on single GPU or small TP group

### Qwen MoE Series

- **Qwen1.5-MoE-A2.7B**: 64 experts, top-4 routing
- **Qwen3-30B-A3B**: Variable expert counts
- **Inference**: Supported via vLLM's modular MoE kernels

## Inference Challenges

### Memory Footprint

All expert parameters must be resident in GPU memory:
- Cannot page experts to CPU during inference (latency spike)
- Total memory = (num_experts × expert_size) + KV cache + activations
- **Solution**: [[Expert Parallelism]] to shard experts across GPUs

### Load Imbalance

Token routing is data-dependent and often skewed:
- Some experts receive 10×+ more tokens than others
- Creates compute imbalance across GPUs in EP deployment
- **Solution**: vLLM's Expert Parallel Load Balancer (EPLB) dynamically redistributes expert mappings

### Communication Overhead

With [[Expert Parallelism]] (experts split across GPUs):
- All-to-all communication to route tokens to correct GPU
- Each GPU must send/receive tokens based on routing decisions
- Can become bottleneck, especially in multi-node deployment
- **Solutions**:
  - [[Dual Batch Overlap]]: Overlap communication with compute
  - DeepEP kernels: Optimized all-to-all for high-throughput and low-latency modes
  - FlashInfer NVLink backends: Specialized for multi-node NVLink systems

### Kernel Efficiency

Standard dense matrix multiplication is inefficient for MoE:
- Irregular memory access patterns (tokens scattered to different experts)
- Need to gather tokens for same expert, compute, then scatter results
- **Solution**: Fused MoE kernels that combine gather, matmul, activation, scatter

## vLLM MoE Architecture

### Modular Kernel Design

vLLM's [[FusedMoE Modular Kernel]] separates concerns:

1. **Prepare & Finalize** (`FusedMoEPrepareAndFinalizeModular`):
   - Quantization of input activations
   - All-to-all dispatch (scatter tokens to experts)
   - All-to-all combine (gather results back)
   - Multiple backends: DeepEP (high-throughput/low-latency), FlashInfer NVLink, naive

2. **Experts** (`FusedMoEExpertsModular`):
   - Core computation: W1 matmul, activation, W2 matmul
   - Multiple implementations: Triton, CUTLASS, DeepGemm, Marlin, TRT-LLM
   - Support different quantization schemes (FP8, NVFP4, MXFP4)

3. **Weight Application & Reduction** (`TopKWeightAndReduce`):
   - Apply router weights to expert outputs
   - Reduce across selected experts
   - Can happen in Experts or Prepare/Finalize depending on implementation

### Activation Formats

- **Contiguous/Standard**: Activations as (M, K) tensor with separate TopK ids/weights
  - Used by: DeepEP high-throughput, FlashInfer backends
  - Efficient for prefill-heavy workloads

- **Batched**: Activations as (num_experts, max_tokens, K) with expert-grouped tokens
  - Used by: DeepEP low-latency backend
  - Efficient for decode-heavy workloads
  - Enables CUDA graph capture for ultra-low latency

## Parallelism Strategies

### Expert Parallelism (EP)

Distribute experts across GPUs:
- Each GPU holds subset of experts (e.g., GPU0: experts 0-63, GPU1: experts 64-127)
- All-to-all communication routes tokens to expert's GPU
- **Benefit**: Can serve models too large for single GPU memory
- **Challenge**: Communication overhead, load imbalance

See [[Expert Parallelism]] for details.

### EP + Data Parallelism (DP)

Combine expert sharding with model replication:
- Multiple complete model replicas across DP groups
- Within each replica, experts sharded across EP GPUs
- **Benefit**: High throughput via DP + memory efficiency via EP
- **Configuration**: vLLM computes `EP_SIZE = TP_SIZE × DP_SIZE`

### EP + Tensor Parallelism (TP)

Experts sharded across EP dimension, attention layers sharded via TP:
- Attention: TP groups within each DP group
- MoE layers: EP across all GPUs
- **Example**: TP=2, DP=4 → 8 GPUs total, EP_SIZE=8
  - Attention: 4 TP groups of size 2
  - Experts: 1 EP group of size 8

## Performance Optimizations

### Dual Batch Overlap (DBO)

Split batch into two microbatches, run on separate CPU threads:
- Thread 1: Dispatch microbatch 1 → compute microbatch 2 → combine microbatch 1
- Thread 2: Dispatch microbatch 2 → compute microbatch 1 → combine microbatch 2
- **Benefit**: Overlap all-to-all communication with compute
- **Requirement**: DP + EP deployment, DeepEP backend

See [[Dual Batch Overlap]] for details.

### Expert Parallel Load Balancer (EPLB)

Dynamic expert redistribution to balance load:
- Track tokens per expert over sliding window (default: 1000 steps)
- Periodically rebalance expert assignments (default: every 3000 steps)
- Redundant experts: Replicate hot experts across multiple GPUs
- **Benefit**: 20-40% throughput improvement for skewed workloads
- **Cost**: ~2.4 GB GPU memory per redundant expert (DeepSeek-V3)

### Quantization

Reduce memory and compute cost:
- **FP8**: 2× reduction, supported by DeepGemm, CUTLASS, Triton kernels
- **NVFP4**: 4× reduction, supported by CUTLASS, FlashInfer, TRT-LLM
- **MXFP4**: 4× reduction with block-wise scaling, supported by TRT-LLM
- **Grouped quantization**: Block-wise quantization (e.g., block size 128) for better accuracy

Quantization can happen before or after all-to-all dispatch depending on backend.

## Configuration Examples

### Single-Node EP (DeepSeek-V3 on 8× H200)

```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel
```

- Attention: Replicated across 8 GPUs
- Experts: Sharded across 8 GPUs (32 experts per GPU)

### Multi-Node EP + DBO (2 nodes, 16 GPUs)

```bash
# Node 1
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --all2all-backend deepep_low_latency \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --data-parallel-size 16 \
  --data-parallel-size-local 8 \
  --enable-dbo \
  --data-parallel-address 192.168.1.100 \
  --data-parallel-rpc-port 13345

# Node 2
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --all2all-backend deepep_low_latency \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --data-parallel-size 16 \
  --data-parallel-size-local 8 \
  --data-parallel-start-rank 8 \
  --enable-dbo \
  --data-parallel-address 192.168.1.100 \
  --data-parallel-rpc-port 13345 \
  --headless
```

- 16 experts per GPU (256 experts / 16 GPUs)
- DBO overlaps inter-node all-to-all with compute

### Single-Node EP + EPLB (Qwen3-30B-A3B)

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2}'
```

- EPLB tracks load every 1000 steps, rebalances every 3000 steps
- 2 redundant experts per GPU for hot expert replication

## References

- Source: [[vllm-moe-design]]
- Architecture: [[FusedMoE Modular Kernel]]
- Parallelism: [[Expert Parallelism]], [[Data Parallelism]], [[Tensor Parallelism]]
- Optimization: [[Dual Batch Overlap]]
- Related: [[vLLM Engine]], [[Kernel Fusions]], [[CUDA Graphs]]
