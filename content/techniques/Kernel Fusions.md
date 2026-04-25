---
title: Kernel Fusions
type: technique
created: 2026-04-25
tags: [compilation, optimization, cuda, rocm, performance]
---

# Kernel Fusions

Kernel fusions merge multiple GPU operations into single kernels to reduce memory bandwidth consumption and kernel launch overhead. vLLM implements 12+ production fusions via custom [[torch.compile Integration|torch.compile]] Inductor passes, controlled by `PassConfig` flags and enabled automatically at appropriate [[Optimization Levels]].

## What Are Kernel Fusions?

Traditional GPU execution launches separate kernels for each operation (e.g., AllReduce → RMSNorm → Quantization). Each kernel launch incurs:
- **CPU overhead** (kernel dispatch, argument marshaling)
- **Memory round-trips** (intermediate results written to/read from global memory)

Kernel fusion eliminates these costs by combining operations into a single CUDA/HIP kernel:

```text
# Unfused (3 kernel launches, 2 memory round-trips):
x1 = AllReduce(x0)          # kernel 1, write x1 to memory
x2 = RMSNorm(x1)            # kernel 2, read x1, write x2
x3 = QuantizeFP8(x2)        # kernel 3, read x2, write x3

# Fused (1 kernel launch, 0 intermediate writes):
x3 = AllReduceRMSQuantFused(x0)  # single kernel, direct x0 → x3
```

**Benefits:**
- Reduced global memory bandwidth (major bottleneck on modern GPUs)
- Lower kernel launch overhead (especially important at small batch sizes)
- Better cache utilization (intermediate data stays in registers/shared memory)

**Trade-offs:**
- Increased kernel complexity (harder to debug, longer compile times)
- Not always profitable (depends on batch size, hardware, operation mix)

## vLLM's Fusion Strategy

vLLM separates optimizations from model definitions by applying fusions at **compile time** via custom Inductor passes. Model code remains clean and layer abstractions are preserved.

### Configuration

All fusions are exposed via `PassConfig` (nested in `CompilationConfig`):

```python
from vllm import LLM
from vllm.config import CompilationConfig, PassConfig

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    optimization_level=2,  # Default fusion presets
    compilation_config=CompilationConfig(
        pass_config=PassConfig(
            fuse_allreduce_rms=True,   # Enable AllReduce + RMSNorm
            fuse_attn_quant=False,     # Disable Attention + Quant
            fuse_norm_quant=True,      # Enable RMSNorm + Quant
        )
    ),
)
```

**CLI flags:**

```bash
# Enable O2 defaults, but turn off AllReduce fusion
vllm serve meta-llama/Llama-3.1-8B-Instruct -O2 \
  -cc.pass_config.fuse_allreduce_rms=False

# Equivalent verbose form:
vllm serve meta-llama/Llama-3.1-8B-Instruct -O2 \
  --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'
```

User-set flags **always override** optimization-level defaults.

## Fusion Catalog

### Communication + Normalization Fusions

#### AllReduce + RMSNorm (`fuse_allreduce_rms`)

**What it fuses:** Tensor-parallel all-reduce → residual add → RMSNorm → optional quantization (FP8 static or NVFP4).

**Hardware:** NVIDIA Hopper (SM90) / Blackwell (SM100) only, requires FlashInfer installed.

**Speedup:** 5-20% end-to-end at **low token counts** (fusion only applied in lower compiled range).

**Configuration:** Enabled by default at [[Optimization Levels#-O2|-O2]] when TP > 1.

**Constraints:**
- TP+DP and TP+PP combinations currently broken ([#34458](https://github.com/vllm-project/vllm/issues/34458), [#35426](https://github.com/vllm-project/vllm/issues/35426))
- Maximum tensor size: 64 MB for TP=2 (configurable via `PassConfig.fi_allreduce_fusion_max_size_mb`)

**Patterns:**
- `AllReduce → RMSNorm(+residual_add)` (FP16/BF16)
- `AllReduce → RMSNorm(+residual_add) → FP8 static quant` (SM90+)
- `AllReduce → RMSNorm(+residual_add) → NVFP4 quant` (SM100 only)

**Code:** [`vllm/compilation/passes/fusion/allreduce_rms_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/allreduce_rms_fusion.py), [`vllm/distributed/device_communicators/flashinfer_all_reduce.py`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/flashinfer_all_reduce.py)

**Related:** [[Tensor Parallelism]], [[Quantization]], FlashInfer Backend

---

#### MiniMax QK Norm (`fuse_minimax_qk_norm`)

**What it fuses:** MiniMax M2 Q/K variance all-reduce → Q/K RMSNorm (tensor-parallel normalization path).

**Hardware:** CUDA (SM80+), requires TP > 1 and custom op `minimax_allreduce_rms_qk`.

**Speedup:** 2-3% (MiniMax M2 specific).

**Configuration:** Off by default (model-specific pass).

**Example:**

```bash
vllm serve MiniMaxAI/MiniMax-M2.5 --tensor-parallel-size 4 \
  --compilation-config '{"mode": 3, "pass_config": {"fuse_minimax_qk_norm": true}}'
```

**Code:** [`vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py), [`csrc/minimax_reduce_rms_kernel.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/minimax_reduce_rms_kernel.cu)

**Related:** [[Tensor Parallelism]], QK Normalization

### Attention Fusions

#### Attention + Quantization (`fuse_attn_quant`)

**What it fuses:** Attention output → FP8/NVFP4 quantization (eliminates full-precision memory round-trip).

**Hardware:** CUDA (SM89+) / ROCm, attention backend dependent.

**Speedup:** 3-7% (standard Attention), TBD (MLA Attention — no memory savings yet).

**Configuration:** Off by default (must set explicitly), requires **full model graph visibility** (Inductor partition or `splitting_ops=[]`).

**Supported backends:**

Standard `Attention → FP8 static`:
- `TRITON_ATTN` (CUDA, ROCm)
- `FLASHINFER` (CUDA SM100+)
- `ROCM_ATTN` (ROCm)
- `ROCM_AITER_UNIFIED_ATTN` (ROCm + AITER)

Standard `Attention → NVFP4 dynamic`:
- `FLASHINFER` (CUDA SM100+ only)

`MLAAttention → FP8 static/per-group/NVFP4` (DeepSeek-V2/V3/R1):
- Works with all MLA prefill/decode backend combinations
- Currently writes to intermediate buffer (no direct FP8/FP4 output from kernels yet)

**Note:** MLA fusion is not expected to yield measurable speedup until MLA kernels support direct FP8/FP4 output.

**Code:** [`vllm/compilation/passes/fusion/attn_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/attn_quant_fusion.py) (standard), [`vllm/compilation/passes/fusion/mla_attn_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/mla_attn_quant_fusion.py) (MLA)

**Related:** [[Quantization]], [[Attention Backends]], MLA Attention, DeepSeek V3

---

#### QK Norm + RoPE (`enable_qk_norm_rope_fusion`)

**What it fuses:** Split QKV → reshape → Q/K RMSNorm → reshape → rotary embedding (single `fused_qk_norm_rope` CUDA kernel).

**Hardware:** CUDA (SM80+), tested on SM90/SM100.

**Speedup:** 2-3% at **low token counts**.

**Configuration:** Off by default (perf issues on H100, [#34391](https://github.com/vllm-project/vllm/issues/34391)).

**Applicable models:** Qwen family (and others applying per-head RMSNorm to Q/K before RoPE).

**Pattern:**

```python
# Unfused:
q, k, v = split(qkv)
q_norm = rms_norm(q.view(heads))
k_norm = rms_norm(k.view(kv_heads))
q_rope, k_rope = rotary_embedding(q_norm, k_norm, ...)

# Fused:
fused_qk_norm_rope(qkv, ...)
```

**Code:** [`vllm/compilation/passes/fusion/qk_norm_rope_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/qk_norm_rope_fusion.py), [`csrc/ops.h`](https://github.com/vllm-project/vllm/blob/main/csrc/ops.h)

**Related:** RoPE, QK Normalization, Qwen Models

### Tensor Parallelism Fusions

#### Sequence Parallelism (`enable_sp`)

**What it fuses:** Transforms `AllReduce → RMSNorm` into `ReduceScatter → local RMSNorm → AllGather` (splits sequence dimension across TP ranks).

**Hardware:** NVIDIA CUDA (tested on H100/SM90), possibly ROCm. FP8 all-gather requires SM90+.

**Speedup:** None directly (this is a **prerequisite** for AsyncTP pass `fuse_gemm_comms`).

**Configuration:** Off by default (SM90 autoconfigured for models with `hidden_size >= 8192`, threshold configurable via `PassConfig.sp_min_token_num`).

**Applied range:** Only at **high token counts** (above `sp_min_token_num`).

**Transformation:**

```text
Input → AllReduce → RMSNorm → Output

becomes:

Input → ReduceScatter → local RMSNorm → AllGather → Output
```

**Patterns:**
- First block: `AllReduce → RMSNorm` → `ReduceScatter → RMSNorm → AllGather`
- Middle blocks: `AllReduce → fused_add_RMSNorm` → `ReduceScatter → fused_add_RMSNorm → AllGather`
- Both with optional `→ FP8 static quant` suffix

**Constraints:** Requires `use_inductor_graph_partition=True` OR piecewise compilation with static sizes divisible by `tensor_parallel_size`.

**Code:** [`vllm/compilation/passes/fusion/sequence_parallelism.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/sequence_parallelism.py)

**Related:** [[Tensor Parallelism]], AsyncTP, Distributed Serving

---

#### AsyncTP GEMM + Collective Overlap (`fuse_gemm_comms`)

**What it fuses:** After Sequence Parallelism transforms the graph, fuses GEMM kernels with surrounding reduce-scatter (output projection) and all-gather (input projection) using `torch.ops.symm_mem` symmetric-memory primitives.

**Hardware:** NVIDIA CUDA with symmetric-memory support (`torch.distributed._symmetric_memory`).

**Speedup:** 7-10% at **high token counts** (fusion only applied above `PassConfig.sp_min_token_num`).

**Configuration:** Off by default, requires `enable_sp=True` (automatically enabled if SP is active).

**Patterns:**
- `GEMM → reduce-scatter` → `fused_matmul_reduce_scatter`
- `all-gather → GEMM` → `all_gather_matmul`
- FP8 scaled variants of both

**Overlapping:** Communication and computation run concurrently (hides TP collective latency).

**Constraints:**
- Requires Sequence Parallelism (no-op if SP not applied)
- On B200, must disable FlashInfer FP8 scaled MM: `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel` ([#27893](https://github.com/vllm-project/vllm/issues/27893))

**Code:** [`vllm/compilation/passes/fusion/collective_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/collective_fusion.py)

**Related:** [[Tensor Parallelism]], Sequence Parallelism, Distributed Serving, GEMM Optimization

### Normalization + Quantization Fusions

#### RMSNorm + Quantization (`fuse_norm_quant`)

**What it fuses:** `rms_norm` / `fused_add_rms_norm` → FP8/FP4 quantization (eliminates full-precision activation tensor write).

**Hardware:** CUDA (all SMs) / ROCm (HIP + AITER).

**Speedup:** 1-4%.

**Configuration:** Enabled at [[Optimization Levels#-O1|-O1]] **conditionally** (only if either `rms_norm` or `quant_fp8` uses a custom kernel; Inductor's native fusion is faster on NVIDIA).

**Variants:**
- Plain: `rms_norm(x) → quant_fp8(y)`
- Fused-add: `fused_add_rms_norm(x, residual) → quant_fp8(y)` (also updates residual in-place)

**Quantization schemes:**
- FP8 static per-tensor (CUDA & HIP kernel)
- FP8 dynamic per-token (CUDA & HIP kernel, AITER)
- FP8 dynamic per-token-group (128/64) (CUDA & HIP kernel, AITER)

**Code:** [`vllm/compilation/passes/fusion/rms_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rms_quant_fusion.py) (CUDA/HIP), [`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py) (AITER), [`csrc/layernorm_quant_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_quant_kernels.cu)

**Related:** [[Quantization]], RMSNorm, AITER Backend

---

#### SiLU+Mul + Quantization (`fuse_act_quant`)

**What it fuses:** `silu_and_mul` gate-up projection activation → FP8/NVFP4 quantization (avoids materialization of full-precision post-activation tensor).

**Hardware:** CUDA (SM89+) / ROCm (AITER).

**Speedup:** 1-4%.

**Configuration:** Enabled at [[Optimization Levels#-O1|-O1]] **conditionally** (only if either `silu_and_mul` or `quant_fp8` uses a custom kernel, OR for NVFP4-quantized models where FP4 quant is always a custom op).

**Quantization schemes:**
- FP8 static per-tensor (CUDA & HIP kernel)
- FP8 dynamic per-group (128/64) (CUDA SM89+, not active when DeepGemm used on SM100+)
- NVFP4 dynamic (CUDA SM100+ only with FlashInfer)
- FP8 per-token-group (128) (ROCm AITER only)

**Code:** [`vllm/compilation/passes/fusion/act_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/act_quant_fusion.py) (CUDA/HIP), [`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py) (AITER), [`csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu)

**Related:** [[Quantization]], SiLU Activation, Gate-Up Projection

### ROCm/AITER-Specific Fusions

#### RoPE + KV-Cache Update (`fuse_rope_kvcache`)

**What it fuses:** Rotary positional embedding → KV-cache scatter/write (single kernel, avoids separate key/value tensor reads/writes).

**Hardware:** AMD ROCm with AITER enabled (not available on NVIDIA CUDA or CPU).

**Speedup:** 2-4% at **low token counts** (≤ 256 by default, configurable via `PassConfig.rope_kvcache_fusion_max_token_num`).

**Configuration:** Enabled at [[Optimization Levels#-O1|-O1]] when AITER is active and `kv_cache` update op visible in graph.

**Constraints:** Requires `rotary_embedding` custom op active (automatic) and KV-cache update in graph (Inductor partition or removed from `splitting_ops`).

**Code:** [`vllm/compilation/passes/fusion/rope_kvcache_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rope_kvcache_fusion.py)

**Related:** RoPE, [[KV Cache]], AITER Backend

---

#### RMSNorm + Padding (`fuse_act_padding`)

**What it fuses:** Residual add + RMSNorm → padding (pads hidden dimension to multiple required by downstream AITER Triton GEMM kernels).

**Hardware:** AMD ROCm with AITER RMSNorm enabled.

**Speedup:** TBD (targeted at GPT-OSS models).

**Configuration:** Enabled at [[Optimization Levels#-O1|-O1]] when hidden size is 2880 and AITER Triton GEMMs **not** enabled.

**Code:** [`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py) (`RocmAiterTritonAddRMSNormPadFusionPass`)

**Related:** AITER Backend, RMSNorm, GPT-OSS Models

---

#### MLA Dual RMSNorm (`fuse_mla_dual_rms_norm`)

**What it fuses:** DeepSeek-V3 / Kimi-K2 MLA attention `q_a_layernorm` + `kv_a_layernorm` → single `fused_qk_rmsnorm` HIP kernel (reduces 2 kernel launches to 1 per MLA layer).

**Hardware:** AMD ROCm with AITER enabled.

**Speedup:** ~2%.

**Configuration:** Enabled at [[Optimization Levels#-O1|-O1]] when AITER is available.

**Note:** Inductor's built-in fusion already handles this when native `rms_norm` is used. This explicit pass targets AITER's custom `rms_norm` op (which Inductor cannot fuse).

**Pattern:**

```python
# Unfused:
q_c, kv_lora = split(projected, [q_dim, kv_dim])
kv_c, k_pe   = split(kv_lora,  [kv_c_dim, k_pe_dim])
q_c  = rms_norm(q_c,  q_weight,  eps)
kv_c = rms_norm(kv_c, kv_weight, eps)

# Fused:
q_c, kv_lora = split(projected, [q_dim, kv_dim])
kv_c, k_pe   = split(kv_lora,  [kv_c_dim, k_pe_dim])
q_normed, kv_normed = fused_mla_dual_rms_norm(
    q_c, q_weight, kv_c, kv_weight, eps1, eps2)
```

**Code:** [`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py) (`MLADualRMSNormFusionPass`), [`vllm/_aiter_ops.py`](https://github.com/vllm-project/vllm/blob/main/vllm/_aiter_ops.py), [AITER kernel PR](https://github.com/ROCm/aiter/pull/2442)

**Related:** MLA Attention, DeepSeek V3, Kimi K2, AITER Backend

## Hardware Support Matrix

| Fusion                   | SM100 (Blackwell)                      | SM90 (Hopper)                          | SM89 (Ada)                             | SM80 (Ampere) | ROCm                                   |
| ------------------------ | -------------------------------------- | -------------------------------------- | -------------------------------------- | ------------- | -------------------------------------- |
| `fuse_allreduce_rms`     | FP16/BF16, FP8 static, NVFP4           | FP16/BF16, FP8 static                  | —                                      | —             | —                                      |
| `fuse_minimax_qk_norm`   | FP16/BF16                              | FP16/BF16                              | FP16/BF16                              | FP16/BF16     | —                                      |
| `fuse_attn_quant`        | FP8 static, NVFP4                      | FP8 static                             | FP8 static                             | —             | FP8 static                             |
| `fuse_attn_quant` (MLA)  | FP8 static/per-group, NVFP4            | FP8 static/per-group                   | FP8 static/per-group                   | —             | FP8 static (untested)                  |
| `fuse_rope_kvcache`      | —                                      | —                                      | —                                      | —             | FP16/BF16 (AITER)                      |
| `enable_qk_norm_rope`    | FP16/BF16                              | FP16/BF16                              | FP16/BF16                              | FP16/BF16     | —                                      |
| `enable_sp`              | FP16/BF16, FP8 static (needs config)   | FP16/BF16, FP8 static (primary target) | FP16/BF16 (needs config)               | FP16/BF16     | —                                      |
| `fuse_gemm_comms`        | FP16/BF16, FP8 static (needs config)   | FP16/BF16, FP8 static (primary target) | FP16/BF16 (needs config)               | FP16/BF16     | —                                      |
| `fuse_norm_quant`        | FP8 static/per-token/per-group         | FP8 static/per-token/per-group         | FP8 static/per-token/per-group         | —             | FP8 static/per-token/per-group         |
| `fuse_act_quant`         | FP8 static, NVFP4                      | FP8 static, FP8 per-group (128/64)     | FP8 static, FP8 per-group (128/64)     | —             | FP8 per-group (AITER)                  |
| `fuse_act_padding`       | —                                      | —                                      | —                                      | —             | FP16/BF16 (AITER)                      |
| `fuse_mla_dual_rms_norm` | —                                      | —                                      | —                                      | —             | BF16 (AITER)                           |

**Notes:**
- `fuse_attn_quant` support depends on attention backend (not all backends support fused quantization output).
- `enable_sp` and `fuse_gemm_comms` are only autoconfigured for SM90; other architectures require setting `PassConfig.sp_min_token_num` explicitly.
- SM100 support for AsyncTP also requires `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel`.

## Fusion Profitability

Fusions are **not universally beneficial**. vLLM applies fusions selectively based on:

### Token Count Sensitivity

- **Low-token fusions** (`num_tokens` small):
  - AllReduce + RMSNorm (≤ 64 MB tensor for TP=2)
  - RoPE + KV-Cache (≤ 256 tokens by default)
  - QK Norm + RoPE
  - MiniMax QK Norm
  - Reason: Kernel launch overhead dominates at small batch sizes

- **High-token fusions** (`num_tokens` large):
  - Sequence Parallelism + AsyncTP (≥ `sp_min_token_num`, typically 512-1024 tokens)
  - Reason: Communication-computation overlap only profitable when GEMM is large enough

- **Always-on fusions** (all `num_tokens`):
  - RMSNorm + Quant
  - SiLU+Mul + Quant
  - Attention + Quant
  - MLA Dual RMSNorm
  - Reason: Memory bandwidth savings outweigh costs across all batch sizes

### Graph Visibility

Some fusions require **full model graph** to be visible:
- `fuse_attn_quant` (requires Inductor partition or `splitting_ops=[]`)
- `enable_sp` (requires Inductor partition OR piecewise compilation with static divisible sizes)
- `fuse_gemm_comms` (requires SP, inherits SP's constraints)

Piecewise compilation mode splits the graph at `splitting_ops` boundaries, preventing cross-boundary fusions.

### Hardware-Specific Enablement

- **FlashInfer dependency:** AllReduce + RMSNorm (Hopper/Blackwell only)
- **AITER dependency:** RoPE + KV-Cache, RMSNorm + Padding, MLA Dual RMSNorm (ROCm only)
- **Symmetric memory:** AsyncTP (CUDA only, requires `torch.distributed._symmetric_memory`)

### Inductor Fusion Competition

On NVIDIA platforms, Inductor's native fusion (Triton codegen) can outperform custom CUDA kernels for some operations:

- **`fuse_norm_quant`:** Only enabled when either `rms_norm` or `quant_fp8` uses a custom kernel (otherwise Inductor's fusion is faster)
- **`fuse_act_quant`:** Same logic, except NVFP4 always uses custom op

This is controlled by [[CustomOp System]] dispatch: when custom ops are disabled (e.g., `--compilation_config.custom_ops '["none"]'`), Inductor generates its own fused kernels.

## Tuning Performance

> "Speedup depends heavily on the exact model, batch size, and hardware. If tuning performance by hand, always benchmark your exact use-case with and without the fusion to verify the impact."

**Benchmarking workflow:**

1. Establish baseline:
   ```bash
   vllm bench latency --model=meta-llama/Llama-3.1-70B-Instruct \
     -O2 -cc.pass_config.fuse_allreduce_rms=False
   ```

2. Enable fusion:
   ```bash
   vllm bench latency --model=meta-llama/Llama-3.1-70B-Instruct \
     -O2 -cc.pass_config.fuse_allreduce_rms=True
   ```

3. Compare TTFT, decode latency, throughput.

4. Use `TORCH_TRACE=~/trace_dir` + `tlparse` to verify fusion applied (see [[torch.compile Integration#Debugging]]).

**Key metrics to track:**
- Time-to-first-token (TTFT): sensitive to low-token fusions (AllReduce+RMS, RoPE+KV)
- Decode latency: sensitive to high-token fusions (AsyncTP), normalization+quant fusions
- Throughput: sensitive to all fusions (memory bandwidth reduction)
- Compilation time: auto-tuning adds 30s-5min (see [[Optimization Levels#Auto-tuning]])

## Implementation Details

### Inductor Pass Pipeline

Fusions are implemented as custom Inductor passes registered in vLLM's compilation pipeline:

1. **TorchDynamo** captures model forward pass into FX graph (dynamic on `num_tokens`)
2. **vLLM graph splitting** (optional, if piecewise mode): splits graph at `splitting_ops` boundaries
3. **Inductor passes** (vLLM custom):
   - Pattern-match fusion opportunities in FX graph
   - Replace matched subgraphs with fused op nodes
   - Example: `AllReduce` → `RMSNorm` → `QuantFP8` becomes single `AllReduceRMSQuantFused` node
4. **TorchInductor** lowers fused ops to Triton/CUDA kernels
5. **Compilation cache** saves compiled artifacts for reuse

See [[torch.compile Integration]] for full pipeline details.

### PassConfig Defaults by Optimization Level

| Fusion                   | -O0  | -O1       | -O2       | -O3       |
| ------------------------ | ---- | --------- | --------- | --------- |
| `fuse_allreduce_rms`     | Off  | Off       | On (TP>1) | On (TP>1) |
| `fuse_attn_quant`        | Off  | Off       | Off       | Off       |
| `fuse_rope_kvcache`      | Off  | On (AITER)| On (AITER)| On (AITER)|
| `enable_qk_norm_rope`    | Off  | Off       | Off       | Off       |
| `enable_sp`              | Off  | Off       | Off       | Off       |
| `fuse_gemm_comms`        | Off  | Off       | Off       | Off       |
| `fuse_norm_quant`        | Off  | Cond.     | Cond.     | Cond.     |
| `fuse_act_quant`         | Off  | Cond.     | Cond.     | Cond.     |
| `fuse_act_padding`       | Off  | On (AITER)| On (AITER)| On (AITER)|
| `fuse_mla_dual_rms_norm` | Off  | On (AITER)| On (AITER)| On (AITER)|
| `fuse_minimax_qk_norm`   | Off  | Off       | Off       | Off       |

**"Cond."** = Conditional on custom kernel usage (see Inductor Fusion Competition above).

See [[Optimization Levels]] for full flag matrix.

## Code Locations

- **Fusion passes:** [`vllm/compilation/passes/fusion/`](https://github.com/vllm-project/vllm/tree/main/vllm/compilation/passes/fusion)
- **PassConfig definition:** [`vllm/config/compilation.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py)
- **CUDA kernels:** [`csrc/layernorm_quant_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_quant_kernels.cu), [`csrc/quantization/fused_kernels/`](https://github.com/vllm-project/vllm/tree/main/csrc/quantization/fused_kernels), [`csrc/minimax_reduce_rms_kernel.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/minimax_reduce_rms_kernel.cu)
- **Benchmarks:** [`benchmarks/kernels/benchmark_fused_collective.py`](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_fused_collective.py)

## Related Concepts

- [[torch.compile Integration]] — vLLM's custom Inductor pass pipeline
- [[CUDA Graphs]] — kernel capture/replay after fusion
- [[Optimization Levels]] — high-level fusion presets
- [[Quantization]] — FP8/NVFP4 integration with fusions
- [[Tensor Parallelism]] — AllReduce, ReduceScatter, AllGather fusions
- [[CustomOp System]] — custom op vs Inductor fusion dispatch
- [[Attention Backends]] — attention backend compatibility with fusions
- AITER Backend — ROCm-specific fusion backend
- FlashInfer Backend — NVIDIA-specific fusion backend

## Further Reading

- Source: [[vllm-fusions-design]]
- Tracking issue: [vLLM #36066 (Fusion Support Matrix)](https://github.com/vllm-project/vllm/issues/36066)
- Blog: [Introduction to vLLM-torch.compile](https://blog.vllm.ai/2025/08/20/torch-compile.html)
