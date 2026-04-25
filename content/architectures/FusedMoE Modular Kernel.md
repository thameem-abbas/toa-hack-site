---
title: FusedMoE Modular Kernel
type: architecture
created: 2026-04-25
related:
  - "Mixture of Experts"
  - "Expert Parallelism"
  - "Kernel Fusions"
  - "CUDA Graphs"
  - "Dual Batch Overlap"
---

# FusedMoE Modular Kernel

The FusedMoE Modular Kernel is vLLM's composable architecture for [[Mixture of Experts]] inference, enabling mix-and-match of communication backends, expert computation kernels, and quantization schemes.

## Architecture

### Three-Component Design

`FusedMoEModularKernel` splits MoE operations into three pluggable components:

```python
class FusedMoEModularKernel:
    def __init__(self,
                 prepare_finalize: FusedMoEPrepareAndFinalizeModular,
                 fused_experts: FusedMoEExpertsModular):
        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts

    def forward(self, input_activations):
        # 1. Prepare: Quantize + All2All Dispatch
        Aq, A_scale, ... = self.prepare_finalize.prepare(input_activations, ...)
        
        # 2. Allocate workspaces
        workspace13_shape, workspace2_shape, ... = self.fused_experts.workspace_shapes(...)
        workspace_13 = torch.empty(workspace13_shape, ...)
        workspace_2 = torch.empty(workspace2_shape, ...)
        
        # 3. Execute experts
        fe_out = self.fused_experts.apply(Aq, A_scale, workspace13, workspace2, ...)
        
        # 4. Finalize: Weight application + All2All Combine
        war_impl = self.fused_experts.finalize_weight_and_reduce_impl()
        output = self.prepare_finalize.finalize(fe_out, war_impl, ...)
        
        return output
```

### Component 1: FusedMoEPrepareAndFinalizeModular

Handles communication and quantization:

**Responsibilities**:
- Input activation quantization (FP16/BF16 → FP8/NVFP4/MXFP4)
- All-to-all dispatch: Route tokens to expert's GPU
- All-to-all combine: Gather expert outputs back to origin
- Optionally apply router weights and reduce across experts

**Key methods**:
- `prepare()`: Quantize and dispatch
- `prepare_no_receive()`: Async dispatch without waiting (returns callback)
- `finalize()`: Combine and optionally weight/reduce
- `activation_format()`: Return "standard" or "batched" format
- `topk_indices_dtype()`: Required dtype for TopK indices (if any)

**Implementations** (via `--all2all-backend`):

| Backend | Format | Quant Types | Async | Use Case |
|---------|--------|-------------|-------|----------|
| `naive` | standard | all | No | General purpose |
| `deepep_high_throughput` | standard | FP8 G(128)/A/T | Yes | Prefill workloads |
| `deepep_low_latency` | batched | FP8 G(128)/A/T | Yes | Decode workloads |
| `flashinfer_nvlink_one_sided` | standard | NVFP4 G/A/T | No | MNNVL systems |
| `flashinfer_nvlink_two_sided` | standard | NVFP4, FP8 G/A/T | No | MNNVL systems |

**Format details**:
- **Standard/Contiguous**: Activations as `(M, K)` tensor, TopK ids/weights as `(M, num_topk)`
- **Batched**: Activations as `(num_experts, max_tokens, K)`, tokens grouped by expert

### Component 2: FusedMoEExpertsModular

Handles core expert computation:

**Responsibilities**:
- Permute activations (group by expert if not already batched)
- W1 matmul: `hidden → intermediate`
- Activation function (SiLU, GELU, etc.) + multiply (for SwiGLU variants)
- W2 matmul: `intermediate → hidden`
- Unpermute outputs
- Optionally apply router weights and reduce

**Key methods**:
- `apply()`: Main computation
- `workspace_shapes()`: Declare workspace memory requirements
- `activation_formats()`: Supported input/output formats
- `finalize_weight_and_reduce_impl()`: Return TopKWeightAndReduce strategy
- `supports_expert_map()`: Whether expert remapping is supported

**Implementations**:

| Kernel | Format | Quant | Activations | Modular |
|--------|--------|-------|-------------|---------|
| `TritonExperts` | standard | all | silu, gelu, swiglu | Yes |
| `BatchedTritonExperts` | batched | all | silu, gelu | Yes |
| `DeepGemmExperts` | standard | FP8 G(128)/A/T | silu, gelu | Yes |
| `BatchedDeepGemmExperts` | batched | FP8 G(128)/A/T | silu, gelu | Yes |
| `CutlassExpertsFp8` | standard | FP8 A/T | silu, gelu | Yes |
| `CutlassBatchedExpertsFp8` | batched | FP8 A/T | silu, gelu | Yes |
| `CutlassExpertsFp4` | standard | NVFP4 A/T | silu | Yes |
| `FlashInferExperts` | standard | NVFP4, FP8 T | (SwiGLU) | Yes |
| `MarlinExperts` | standard | uint4/8, fp4/8 | silu, swiglu | Yes |
| `BatchedMarlinExperts` | batched | uint4/8, fp4/8 | silu, swiglu | Yes |
| `TrtLlmMxfp4ExpertsModular` | standard | MXFP4 G(16)/G(32) | (SwiGLU) | Yes |
| `TrtLlmNvfp4ExpertsModular` | standard | NVFP4 G(16)/G(32) | (SwiGLU) | Yes |

### Component 3: TopKWeightAndReduce

Handles router weight application and reduction across experts:

**Why separate?**
- Some implementations fuse this into expert kernel (more efficient)
- Some implementations do it in finalize step (simpler)
- Modular design allows both approaches

**Implementations**:
- `TopKWeightAndReduceNoOp`: Weight/reduce already done in experts
- `TopKWeightAndReduceContiguous`: Standard format weight/reduce
- `TopKWeightAndReduceNaiveBatched`: Batched format weight/reduce
- `TopKWeightAndReduceDelegate`: Delegate to another implementation

**Flow**:
1. `FusedMoEExpertsModular::apply()` decides whether to do weight/reduce internally
2. `finalize_weight_and_reduce_impl()` returns appropriate `TopKWeightAndReduce` object
3. `FusedMoEPrepareAndFinalizeModular::finalize()` invokes it (possibly as no-op)

## Activation Flow

### Standard (Contiguous) Format

Used by: `deepep_high_throughput`, `flashinfer_nvlink_*`, naive

```
Input: (M, K) activations, (M, num_topk) topk_ids, (M, num_topk) topk_weights

Prepare:
  → Quantize: (M, K) @ dtype → (M, K) @ quant_dtype
  → All2All Dispatch: (M, K) → (M', K) where M' = tokens after redistribution

Experts:
  → Permute by expert: Group tokens by expert_id
  → W1: (M', K) @ (E, K, I) → (M', I)
  → Activation + Mul
  → W2: (M', I) @ (E, I, K) → (M', K)
  → Unpermute: Restore token order
  → Maybe apply topk_weights and reduce

Finalize:
  → All2All Combine: (M', K) → (M, K) (back to original ranks)
  → Maybe apply topk_weights and reduce
  → Return: (M, K) final output
```

### Batched Format

Used by: `deepep_low_latency`

```
Input: (M, K) activations, (M, num_topk) topk_ids, (M, num_topk) topk_weights

Prepare:
  → Quantize: (M, K) @ dtype → (M, K) @ quant_dtype
  → All2All Dispatch: (M, K) → (E, T, K) where E=num_local_experts, T=max_tokens
  → Also return expert_num_tokens: (E,) with valid token counts per expert

Experts:
  → Already batched by expert (no permute needed)
  → For each expert e:
      W1: (expert_num_tokens[e], K) @ (K, I) → (expert_num_tokens[e], I)
      Activation + Mul
      W2: (expert_num_tokens[e], I) @ (I, K) → (expert_num_tokens[e], K)
  → Maybe apply topk_weights and reduce

Finalize:
  → All2All Combine: (E, T, K) → (M, K)
  → Maybe apply topk_weights and reduce
  → Return: (M, K) final output
```

**Batched benefit**: Expert computation naturally parallelizes, CUDA graph compatible.

## Modular Kernel Families

Compatible combinations (tested):

### DeepEP High-Throughput

**Backend**: `deepep_high_throughput` (standard format, async)

**Compatible Experts**:
- `DeepGemmExperts` (FP8, optimized grouped GEMM)
- `TritonExperts` (all quant types)
- `TritonOrDeepGemmExperts` (dispatcher: picks best based on shape/quant)
- `CutlassExpertsFp8` (FP8)
- `MarlinExperts` (uint4/8, fp4/8)

**Use case**: Prefill-heavy workloads, large batches

### DeepEP Low-Latency

**Backend**: `deepep_low_latency` (batched format, async, CUDA graph support)

**Compatible Experts**:
- `BatchedDeepGemmExperts` (FP8)
- `BatchedTritonExperts` (all quant types)
- `CutlassBatchedExpertsFp8` (FP8)
- `BatchedMarlinExperts` (uint4/8, fp4/8)

**Use case**: Decode-heavy workloads, low latency, CUDA graphs

### FlashInfer NVLink

**Backend**: `flashinfer_nvlink_one_sided` or `flashinfer_nvlink_two_sided` (standard format)

**Compatible Experts**:
- `FlashInferExperts` (NVFP4, FP8)

**Use case**: Multi-node NVLink (MNNVL) systems

## Integration with Dual Batch Overlap

[[Dual Batch Overlap]] requires async backends:
- `deepep_high_throughput` or `deepep_low_latency`
- `prepare_no_receive()` must be supported
- Experts must support batched execution

**DBO Yield Points** (in `FusedMoEModularKernel::forward`):
1. After dispatch send, before dispatch receive
2. After expert compute
3. After combine send, before combine receive

**Thread ping-pong**:
- UBatch 0 dispatches, UBatch 1 computes
- UBatch 0 computes, UBatch 1 combines
- Overlap achieved via CPU thread synchronization

## Quantization Support

### Per-Tensor (T)

Single scale factor for entire activation tensor:
- Simplest, least accurate
- Minimal metadata overhead
- Supported by all kernels

### Per-Activation-Token (A)

Scale factor per token:
- Better accuracy than per-tensor
- Moderate metadata overhead
- Supported by most kernels

### Grouped / Block-wise (G, G(N))

Scale factor per block of N values:
- Best accuracy for given bit width
- Higher metadata overhead
- `G(128)`: DeepEP backends prefer block size 128
- `G(16)`, `G(32)`: TRT-LLM kernels

**Example**: FP8 G(128) for DeepSeek-V3
- Block size 128 elements
- Quantize to FP8 E4M3 (8-bit)
- 2× memory reduction vs BF16
- <1% accuracy loss vs BF16

## Workspace Management

Experts allocate large intermediate buffers, but all use same memory:

**Problem**: Allocating separate tensors per operation is inefficient.

**Solution**: `workspace_shapes()` pre-declares requirements.

**Flow**:
```python
# Query workspace requirements
workspace13_shape, workspace2_shape, dtype, output_shape = 
    fused_experts.workspace_shapes(batch_size, hidden_dim, ...)

# Allocate once
workspace_13 = torch.empty(workspace13_shape, dtype=dtype, device='cuda')
workspace_2 = torch.empty(workspace2_shape, dtype=dtype, device='cuda')
output = torch.empty(output_shape, dtype=dtype, device='cuda')

# Pass to apply() for reuse
fused_experts.apply(..., workspace_13, workspace_2, output)
```

**Benefit**: Single large allocation, reused across forward passes.

## Initialization

### FusedMoEMethodBase Class Hierarchy

vLLM's quantization methods extend `FusedMoEMethodBase`:
- `UnquantizedFusedMoEMethod`
- `Fp8MoEMethod`
- `ModelOptFp8MoEMethod`
- `CompressedTensorsW8A8Fp8MoEMethod`
- `CompressedTensorsW4A4Nvfp4MoEMethod`
- `GptOssMxfp4MoEMethod`

### Three-Step Initialization

1. **`maybe_make_prepare_finalize()`**: Construct `FusedMoEPrepareAndFinalizeModular`
   - Base class: Handles EP+DP case (DeepEP, FlashInfer NVLink)
   - Derived class: Can override for custom scenarios (e.g., EP+TP)

2. **`select_gemm_impl()`**: Construct `FusedMoEExpertsModular`
   - Undefined in base class
   - Each derived class implements based on quantization scheme

3. **`init_prepare_finalize()`**: Build `FusedMoEModularKernel`
   - Calls `maybe_make_prepare_finalize()` → get prepare/finalize
   - Calls `select_gemm_impl()` → get experts
   - Construct `FusedMoEModularKernel(prepare_finalize, fused_experts)`
   - Override `self.fused_experts` with modular kernel (transparent to caller)

**Key insight**: Once modular kernel is constructed, calling code doesn't know whether it's modular or monolithic.

## Testing

### Unit Tests

`tests/kernels/moe/test_modular_kernel_combinations.py`:
- Iterates all combinations of prepare/finalize × experts
- Checks compatibility (format, quantization, etc.)
- Runs correctness tests for compatible combinations

**To add new implementation**:
1. Add type to `MK_ALL_PREPARE_FINALIZE_TYPES` or `MK_FUSED_EXPERT_TYPES`
2. Update `Config::is_*` methods for format/quant support
3. Tests auto-include new implementation

### Compatibility Checking

Run as script to test specific combination:
```bash
python3 -m tests.kernels.moe.test_modular_kernel_combinations \
  --pf-type DeepEPLLPrepareAndFinalize \
  --experts-type BatchedTritonExperts
```

Errors if incompatible (e.g., standard format backend with batched-only experts).

### Profiling

Generate torch trace for performance analysis:
```bash
python3 -m tests.kernels.moe.modular_kernel_tools.profile_modular_kernel \
  --pf-type DeepEPHTPrepareAndFinalize \
  --experts-type TritonExperts
```

Outputs trace of single `FusedMoEModularKernel::forward()` call.

## Design Benefits

### Decoupled Development

- Communication backends (all2all) developed independently of expert kernels
- New quantization scheme: Implement new expert kernel, works with all backends
- New hardware: Implement new backend, works with all expert kernels

### Code Reuse

- Single `FusedMoEModularKernel::forward()` for all combinations
- Abstract classes enforce interface contracts
- No combinatorial explosion of implementations

### Testability

- Each component tested independently
- Automated compatibility checking
- Easy to add new implementations

### Flexibility

- Router weight application: In experts or finalize, implementation decides
- Async support: Optional `prepare_no_receive()` for DBO
- Expert mapping: Optional support for EPLB remapping

## Performance Considerations

### Format Choice

- **Standard format**: Better for prefill (large batches, contiguous memory)
- **Batched format**: Better for decode (small batches, CUDA graph compatible)

### Quantization Timing

- **Before dispatch**: Less communication bandwidth (e.g., FP8 vs BF16 is 2× smaller)
- **After dispatch**: Dispatch in high precision, quantize for compute (more flexible)
- DeepEP high-throughput: Prefers pre-quantized FP8 G(128)
- DeepEP low-latency: Quantizes after dispatch

### Kernel Selection

- Triton: General purpose, works everywhere, moderate performance
- DeepGemm: Highly optimized for FP8, requires DeepGEMM library
- CUTLASS: Good FP8/NVFP4 performance, CUDA-only
- FlashInfer: Optimized for NVLink systems
- Marlin: Optimized for low-bit quantization (uint4/8)

## References

- Concept: [[Mixture of Experts]], [[Expert Parallelism]]
- Optimization: [[Dual Batch Overlap]], [[Kernel Fusions]]
- Related: [[CUDA Graphs]], [[vLLM Engine]]
- Source: [[vllm-moe-design]]
