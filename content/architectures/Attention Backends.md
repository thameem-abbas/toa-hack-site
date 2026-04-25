---
title: Attention Backends
type: architecture
created: 2026-04-25
tags: [vllm, attention, backends, flashattention, flashinfer, triton]
---

# Attention Backends

vLLM's pluggable attention backend system enables automatic selection of the optimal attention implementation based on hardware capabilities, model configuration, and feature requirements.

## Overview

vLLM supports 13+ standard attention backends and 15+ MLA-specific backends, with automatic selection based on priority-ordered lists that vary by compute capability. Each backend is validated against configuration constraints (dtype, head size, block size, attention type) before use.

## Backend Selection

### Automatic Selection (Default)

When no backend is explicitly specified, vLLM:

1. Iterates through backends in **priority order** (hardware-dependent)
2. Validates each backend via `AttentionBackend.validate_configuration()`
3. Selects the **first compatible** backend
4. Raises error if no backend is compatible, listing all incompatibility reasons

### Manual Selection

Two methods to explicitly set backend:

**Command line:**
```bash
# Simple flag
vllm serve <model> --attention-backend FLASH_ATTN

# Structured config (dot notation or JSON)
vllm serve <model> --attention-config.backend FLASH_ATTN
vllm serve <model> -ac '{"backend": "FLASH_ATTN"}'
```

**Python API:**
```python
from vllm import LLM
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

# Method 1: AttentionConfig with enum
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    attention_config=AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN),
)

# Method 2: attention_backend parameter with string
llm = LLM(model="Qwen/Qwen3-0.6B", attention_backend="FLASH_ATTN")
```

Manual selection validates compatibility and raises `ValueError` with specific reason if unsupported:
```
ValueError: Selected backend FLASHMLA is not valid for this configuration.
Reason: ['compute capability not supported']
```

## Backend Priority (CUDA)

Priority order differs by compute capability and attention type.

### Standard Attention (MHA, MQA, GQA)

**Blackwell (SM 10.x):**
1. `FLASHINFER` (uses TRTLLM on Blackwell, supports attention sink)
2. `FLASH_ATTN` (FA4 on SM100+)
3. `TRITON_ATTN`
4. `FLEX_ATTENTION`
5. `TURBOQUANT`

**Ampere/Hopper (SM 8.x-9.x):**
1. `FLASH_ATTN` (FA3 on SM90, FA2 on SM80-89)
2. `FLASHINFER` (native implementation, no TRTLLM)
3. `TRITON_ATTN`
4. `FLEX_ATTENTION`
5. `TURBOQUANT`

### MLA Attention (DeepSeek-style)

MLA uses **separate backends** for prefill and decode phases.

**Blackwell (SM 10.x) decode backends:**
1. `FLASHINFER_MLA`
2. `CUTLASS_MLA`
3. `FLASH_ATTN_MLA`
4. `FLASHMLA`
5. `TRITON_MLA`
6. `FLASHINFER_MLA_SPARSE` (preferred for FP8 KV cache or low query-head counts ≤16)
7. `FLASHMLA_SPARSE`

**Ampere/Hopper (SM 8.x-9.x) decode backends:**
1. `FLASH_ATTN_MLA`
2. `FLASHMLA`
3. `FLASHINFER_MLA`
4. `TRITON_MLA`
5. `FLASHMLA_SPARSE`

**MLA prefill backends** (runtime selection):
- **TRT-LLM Ragged** (default on SM100, DeepSeek R1 dims only): TensorRT-LLM ragged attention
  - Disable: `-ac.use_trtllm_ragged_deepseek_prefill=0`
- **FlashInfer** (SM100, DeepSeek R1 dims only): FlashInfer CUTLASS backend
  - Enable: `-ac.disable_flashinfer_prefill=0`
  - Disable: `-ac.disable_flashinfer_prefill=1`
- **cuDNN** (SM100): cuDNN-based attention
  - Enable: `-ac.use_cudnn_prefill=1`
- **FlashAttention** (default fallback): FlashAttention varlen (FA3 on SM90, FA2 otherwise)

## Standard Attention Backends

### FlashAttention (`FLASH_ATTN`)

**Versions:**
- **FA2** (SM80+, Ampere/Ada): fp16/bf16, FP16 KV cache, block sizes %16, any head size
- **FA3** (SM90, Hopper): fp16/bf16, FP8 KV cache support (fp8_e4m3, fp8_e5m2), attention sink
- **FA4** (SM100+, Blackwell): fp16/bf16, FP16 KV cache, attention sink

**Configuration:**
```bash
# Specify version (default: FA4 on SM100+, FA3 on SM90, FA2 otherwise)
vllm serve <model> --attention-config.flash_attn_version=3
```

**Features:**
- All attention types: Decoder, Encoder, Encoder-Decoder
- Decode Context Parallelism (DCP) support
- Block sizes: multiples of 16
- Head sizes: any

**Limitations:**
- FA2: no FP8 KV cache, no attention sink
- FA3: Hopper-only (SM90)
- FA4: Blackwell-only (SM100+), no FP8 KV cache

### FlashInfer (`FLASHINFER`)

**Versions:**
- **Native** (SM70-90): fp16/bf16, FP8 KV cache support, block sizes 16/32/64, head sizes 64/128/256
- **TRTLLM** (SM100, Blackwell): TensorRT-LLM backend, attention sink support
  - Disable: `--attention-config.use_trtllm_attention=0`

**Features:**
- Decoder-only attention
- Decode Context Parallelism (DCP) support
- FP8 KV cache: fp8_e4m3, fp8_e5m2

**Limitations:**
- Fixed head sizes: 64, 128, 256
- No multimodal prefix support
- TRTLLM version: Blackwell-only, attention sink only on TRTLLM

### Triton Attention (`TRITON_ATTN`)

**Features:**
- All attention types: Decoder, Encoder, Encoder-Decoder
- All dtypes: fp16, bf16, fp32
- FP8 KV cache: fp8_e4m3, fp8_e5m2, int8_per_token_head, fp8_per_token_head
- Attention sink support
- Multimodal prefix full attention support
- Any head size, block sizes %16
- Any compute capability

**Limitations:**
- No Decode Context Parallelism (DCP) support

### Flex Attention (`FLEX_ATTENTION`)

**Features:**
- PyTorch native flex_attention
- Decoder and Encoder-Only attention
- Multimodal prefix full attention support
- Any block size, any head size
- All dtypes: fp16, bf16, fp32

**Limitations:**
- No Decode Context Parallelism (DCP)
- No attention sink

### ROCm Backends

**`ROCM_AITER_UNIFIED_ATTN`:**
- All attention types
- Attention sink support
- Multimodal prefix support
- Block sizes %16, any head size

**`ROCM_AITER_FA`:**
- Decoder-only
- FP8 KV cache support
- Block sizes 16/32, head sizes 64/128/256

**`ROCM_ATTN`:**
- Decoder, Encoder, Encoder-Only
- FP8 KV cache support
- Multimodal prefix support
- Block sizes %16, head sizes 32-256

### CPU Backend (`CPU_ATTN`)

**Features:**
- All attention types
- All dtypes: fp16, bf16, fp32
- Any block size
- Head sizes: 32, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512

**Limitations:**
- No attention sink
- No multimodal prefix
- No DCP

## MLA Backends

### Decode Backends

**`CUTLASS_MLA` (Blackwell SM100):**
- Block size: 128
- FP8 KV cache support (fp8_e4m3)
- DCP support

**`FLASHINFER_MLA` (Blackwell SM100):**
- Block sizes: 32, 64
- FP8 KV cache support (fp8_e4m3)
- No DCP

**`FLASHINFER_MLA_SPARSE` (Blackwell SM100):**
- Sparse MLA attention
- Head size: 576 (DeepSeek)
- FP8 KV cache support
- Preferred for: FP8 KV cache or low query-head counts (≤16)

**`FLASHMLA` (Hopper/Blackwell SM90-100):**
- Block size: 64
- FP8 KV cache support (fp8_e4m3)
- DCP support

**`FLASHMLA_SPARSE` (Hopper/Blackwell SM90-100):**
- Sparse MLA attention
- Head size: 576
- FP8 KV cache: bfloat16, fp8_ds_mla
- bf16 model dtype only

**`FLASH_ATTN_MLA` (Hopper SM90):**
- Block sizes %16
- FP16 KV cache only
- DCP support

**`TRITON_MLA` (any compute capability):**
- Block sizes %16
- FP8 KV cache support (fp8_e4m3)
- DCP support

### Prefill Backends

See "MLA Attention (DeepSeek-style)" section under "Backend Priority" above.

## Validation and Configuration

### AttentionBackend.validate_configuration()

Each backend implements validation to check:
- Model dtype compatibility (fp16, bf16, fp32)
- KV cache dtype support (auto, fp8, int8, etc.)
- Block size constraints
- Head size requirements
- Compute capability (CUDA SM version)
- Attention type (Decoder, Encoder, Encoder-Decoder)
- Feature support (attention sink, sparse attention, multimodal prefix, DCP)

Returns list of incompatibility reasons if validation fails.

### AttentionCGSupport Enum

Backends declare CUDA graph compatibility:
- `FULL`: fully compatible
- `PARTIAL`: partially compatible (some constraints)
- `NONE`: incompatible

Used by [[CUDA Graphs]] to determine capture eligibility.

## Feature Support Matrix

| Feature | Description | Backends with Support |
|---------|-------------|----------------------|
| **Attention Sink** | StreamingLLM attention sink | FA3, FA4, FLASHINFER (TRTLLM), TRITON_ATTN, ROCM_AITER_UNIFIED_ATTN |
| **Sparse Attention** | Sparse MLA (DeepSeek) | FLASHINFER_MLA_SPARSE, FLASHMLA_SPARSE, ROCM_AITER_MLA_SPARSE, XPU_MLA_SPARSE |
| **Multimodal Prefix** | Full attention for multimodal prefixes | FLEX_ATTENTION, TRITON_ATTN, ROCM_AITER_UNIFIED_ATTN, ROCM_ATTN |
| **DCP** | Decode Context Parallelism | FLASHINFER, FLASH_ATTN, FLASH_ATTN_DIFFKV, CUTLASS_MLA, FLASHMLA, FLASH_ATTN_MLA, TRITON_MLA |
| **FP8 KV Cache** | FP8 quantized KV cache | FA3, FLASHINFER, TRITON_ATTN, ROCM backends, MLA backends (most) |
| **Encoder-Decoder** | Sequence-to-sequence models | FLASH_ATTN, TRITON_ATTN, ROCM_AITER_UNIFIED_ATTN, ROCM_ATTN, CPU_ATTN |

## Cross-References

- [[vLLM Engine]] — backend selection and validation in engine initialization
- [[V1 Architecture]] — attention backend integration in V1 model runner
- [[CUDA Graphs]] — `AttentionCGSupport` enum for graph compatibility checking
- [[Hybrid KV Cache Manager]] — KV cache block size constraints from backend
- [[PagedAttention]] — block-based KV cache attention mechanism
- [[Quantized KV Cache]] — FP8/INT8 KV cache support in attention backends
- [[Plugin System]] — OOT hardware attention backends via platform plugins
- [[Model Runner V2]] — backend integration in MRV2 architecture
- [[Disaggregated Serving]] — backend selection in prefill vs decode instances

## See Also

- [[torch.compile Integration]] — interaction between Inductor and attention backends
- [[Kernel Fusions]] — attention+quant fusion optimization
- [[Speculative Decoding]] — attention backend requirements for draft model verification
- Multi-Modal Models — multimodal prefix full attention support
