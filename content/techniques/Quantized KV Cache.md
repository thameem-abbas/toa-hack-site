---
title: Quantized KV Cache
type: technique
created: 2026-04-25
---

# Quantized KV Cache

Quantized KV Cache reduces the memory footprint of the key-value cache in attention mechanisms by quantizing cached keys and values from FP16/BF16 to FP8, achieving ~50% KV cache memory reduction. This optimization enables longer context windows, higher batch sizes, and improved throughput, and can be combined with weight/activation quantization for compound memory savings.

## Overview

The KV cache stores the key and value vectors for all previous tokens in a sequence, enabling efficient autoregressive generation without recomputing attention for past tokens. For long contexts, the KV cache can consume more memory than the model weights.

### Memory Footprint

For a typical transformer model:
- **KV cache size**: `2 × batch_size × num_layers × num_kv_heads × seq_length × head_dim × dtype_bytes`
- **Example** (Llama 3 70B, batch_size=16, seq_length=4096, FP16):
  - `2 × 16 × 80 × 8 × 4096 × 128 × 2 bytes ≈ 10.7 GB`
- **With FP8 quantization**: `10.7 GB → 5.4 GB` (~50% reduction)

### Orthogonality

KV cache quantization is orthogonal to weight/activation quantization:
- **Weight quantization** ([[FP8 Quantization]], [[INT8 W8A8]], [[INT4 W4A16]]): Reduces model weight memory
- **KV cache quantization**: Reduces attention cache memory
- **Combined**: Maximize total memory savings

## FP8 KV Cache Formats

vLLM supports two FP8 formats for KV cache quantization:

### FP8 E4M3
- **Structure**: 1 sign bit, 4 exponent bits, 3 mantissa bits
- **Range**: ±448
- **Precision**: Higher precision, limited range
- **Support**: CUDA 11.8+, ROCm (AMD GPUs)
- **Use case**: Default for most models

### FP8 E5M2
- **Structure**: 1 sign bit, 5 exponent bits, 2 mantissa bits
- **Range**: ±57344
- **Precision**: Lower precision, wider range
- **Support**: CUDA 11.8+
- **Use case**: Models with high dynamic range in attention

**vLLM default**: FP8 E4M3 (`kv_cache_dtype="fp8"`)

## Quantization Strategies

### Per-Tensor Quantization
- **Scale granularity**: Single scale per Q, K, V tensor
- **Scales**: `q_scale = [1]`, `k_scale = [1]`, `v_scale = [1]`
- **Accuracy**: Good for most models
- **Compatibility**: All attention backends (FlashAttention, FlashInfer, xFormers, etc.)

### Per-Attention-Head Quantization
- **Scale granularity**: One scale per attention head
- **Scales**: `q_scale = [num_heads]`, `k_scale = [num_kv_heads]`, `v_scale = [num_kv_heads]`
- **Accuracy**: Higher than per-tensor (captures per-head variations)
- **Compatibility**: **Flash Attention backend only**
- **Requirement**: Calibration via [[llm-compressor]]

**Note**: For models with grouped-query attention (GQA), `num_kv_heads < num_heads` (e.g., Llama 3: 32 heads, 8 KV heads).

## Calibration Modes

### 1. No Calibration (Default Scales)

All quantization scales set to 1.0.

```python
from vllm import LLM

llm = LLM(
    "meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
    calculate_kv_scales=False
)
```

**Pros**:
- Zero calibration overhead
- Immediate deployment

**Cons**:
- Suboptimal accuracy (scales may not match actual distribution)
- Risk of clipping for high-magnitude values

**Use case**: Quick prototyping, models known to work well with unit scales

### 2. Random Token Calibration (On-the-Fly)

Scales estimated from a single batch of random tokens during warmup.

```python
from vllm import LLM

llm = LLM(
    "meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True
)
```

**Behavior**:
- vLLM generates random tokens during initialization
- Computes min/max for Q, K, V tensors
- Derives per-tensor scales
- Fixes scales for all subsequent inference

**Pros**:
- No external calibration data required
- Fast warmup

**Cons**:
- Random tokens may not represent deployment distribution
- Per-tensor only (no per-head support)

**Use case**: Development, testing, models without available calibration data

### 3. Dataset Calibration with llm-compressor (Recommended)

Scales estimated using representative calibration dataset.

```python
# See full workflow below
```

**Pros**:
- Highest accuracy (representative data)
- Supports per-attention-head quantization
- Scales stored in model config

**Cons**:
- Requires offline calibration step
- Needs representative dataset

**Use case**: Production deployments, maximum accuracy

## Dataset Calibration Workflow

### Prerequisites
```bash
pip install llmcompressor
```

### Step 1: Load Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### Step 2: Prepare Calibration Data
```python
from datasets import load_dataset

NUM_CALIB_SAMPLES = 512
MAX_SEQ_LEN = 2048
DATASET_ID = "HuggingFaceH4/ultrachat_200k"

ds = load_dataset(DATASET_ID, split=f"train_sft[:{NUM_CALIB_SAMPLES}]")
ds = ds.shuffle(seed=42)

def process_and_tokenize(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)
```

### Step 3: Configure Quantization Recipe

#### Option A: Per-Tensor Quantization
```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

fp8_args = QuantizationArgs(num_bits=8, type="float", strategy="tensor")

recipe = QuantizationModifier(
    config_groups={
        "attention": QuantizationScheme(
            targets=["LlamaAttention"],  # Model-specific layer name
            input_activations=fp8_args,   # Quantizes queries (Q)
        )
    },
    kv_cache_scheme=fp8_args,  # Quantizes keys (K) and values (V)
)
```

#### Option B: Per-Attention-Head Quantization (Flash Attention only)
```python
fp8_args = QuantizationArgs(num_bits=8, type="float", strategy="attn_head")

recipe = QuantizationModifier(
    config_groups={
        "attention": QuantizationScheme(
            targets=["LlamaAttention"],
            input_activations=fp8_args,
        )
    },
    kv_cache_scheme=fp8_args,
)
```

### Step 4: Apply Calibration
```python
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQ_LEN,
    num_calibration_samples=NUM_CALIB_SAMPLES,
)
```

### Step 5: Save Calibrated Model
```python
STRATEGY = "tensor"  # or "attn_head"
save_dir = f"Llama-3.1-8B-Instruct-kvattn-fp8-{STRATEGY}"
model.save_pretrained(save_dir, save_compressed=True)
tokenizer.save_pretrained(save_dir)
```

### Step 6: Deploy in vLLM
```python
from vllm import LLM

llm = LLM(f"./Llama-3.1-8B-Instruct-kvattn-fp8-{STRATEGY}")
# Scales loaded from model config automatically
```

## Flash Attention 3 Integration

When using Flash Attention 3 backend with FP8 KV cache:
- **Queries (Q) also quantized to FP8**: Full FP8 attention computation
- **Attention performed in FP8 domain**: Maximum efficiency
- **Hardware acceleration**: Optimized FP8 attention kernels on Hopper/Ada

**Enable Flash Attention 3**:
```python
llm = LLM(
    model_id,
    kv_cache_dtype="fp8",
    enforce_eager=False  # Allows Flash Attention selection
)
```

**Benefit**: Higher throughput for attention-bound workloads

## Combining with Weight Quantization

### FP8 Weights + FP8 KV Cache
```python
llm = LLM(
    "neuralmagic/Meta-Llama-3-8B-Instruct-FP8",  # Pre-quantized weights
    kv_cache_dtype="fp8",
    calculate_kv_scales=True
)
```

**Memory savings**: ~2× from weights + ~50% from KV cache = ~2.5× total reduction

### INT8 Weights + FP8 KV Cache
```python
llm = LLM(
    "./Meta-Llama-3-8B-Instruct-W8A8",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True
)
```

**Memory savings**: ~2× from weights + ~50% from KV cache = ~2.5× total reduction

### INT4 Weights + FP8 KV Cache
```python
llm = LLM(
    "./Meta-Llama-3-8B-Instruct-W4A16-G128",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True
)
```

**Memory savings**: ~4× from weights + ~50% from KV cache = ~4.5× total reduction

**Use case**: Maximum memory compression for fitting large models on smaller GPUs

## Configuration Options

### kv_cache_dtype
- `"auto"`: Use model's default dtype (FP16/BF16)
- `"fp8"`: Use FP8 E4M3 (default)
- `"fp8_e4m3"`: Explicitly use E4M3 format
- `"fp8_e5m2"`: Explicitly use E5M2 format

### calculate_kv_scales
- `False`: Use default scales (1.0)
- `True`: Estimate scales from random tokens during warmup

**Example**:
```python
llm = LLM(
    model_id,
    kv_cache_dtype="fp8_e5m2",
    calculate_kv_scales=True
)
```

## Performance Metrics

### Memory Reduction
- **KV cache**: ~50% reduction (FP16/BF16 → FP8)
- **Compound with weight quantization**: 2.5-4.5× total memory reduction
- **Longer contexts**: Enables 2× longer sequences within same memory budget
- **Higher batch sizes**: Enables 2× batch size within same memory budget

### Throughput Improvements
- **Batch size increase**: Higher throughput from larger batches
- **Context length increase**: Better GPU utilization on long-context tasks
- **Flash Attention 3**: Additional speedup from FP8 attention computation

### Accuracy Impact
- **Per-tensor with calibration**: Minimal accuracy loss (<0.5% on most benchmarks)
- **Per-attention-head**: Negligible accuracy loss (<0.1%)
- **No calibration**: 1-2% accuracy loss (model-dependent)

## Use Cases

### Ideal for Quantized KV Cache
- **Long-context generation**: Enabling 8K, 16K, 32K token contexts
- **High-throughput serving**: Maximizing batch size for better GPU utilization
- **Multi-turn conversations**: Reducing memory for long chat histories
- **Memory-constrained deployments**: Fitting larger models on smaller GPUs

### Not ideal for Quantized KV Cache
- **Short sequences (<512 tokens)**: KV cache is small, quantization overhead not worth it
- **Ultra-low latency**: FP8 conversion adds minimal overhead but may be unacceptable for <1ms requirements

## Best Practices

1. **Use dataset calibration** for production deployments (highest accuracy)
2. **Match calibration data to deployment**: Use representative sequences and templates
3. **Start with per-tensor**: Simpler, broader compatibility
4. **Use per-attention-head** if using Flash Attention and need maximum accuracy
5. **Combine with weight quantization**: Maximize total memory savings
6. **Monitor accuracy on long contexts**: FP8 errors can accumulate over very long sequences
7. **Use FP8 E4M3**: Default format works well for most models

## Troubleshooting

### Issue: Accuracy degraded on long contexts
**Cause**: Quantization errors accumulate over long sequences

**Solutions**:
1. Use dataset calibration instead of random token calibration
2. Use per-attention-head quantization (Flash Attention)
3. Use FP8 E5M2 (wider range) instead of E4M3
4. Monitor and validate on representative long-context tasks

### Issue: Out of memory despite KV cache quantization
**Cause**: Model weights or activations still dominate memory

**Solutions**:
1. Combine with weight quantization ([[FP8 Quantization]], [[INT8 W8A8]], [[INT4 W4A16]])
2. Reduce batch size
3. Reduce max sequence length
4. Use [[Prefix Caching]] to reduce redundant KV cache

### Issue: Slower inference after enabling KV cache quantization
**Cause**: Quantization/dequantization overhead exceeds memory bandwidth savings

**Solutions**:
1. Only enable for long sequences (>1024 tokens)
2. Use Flash Attention 3 for FP8-native attention
3. Increase batch size to amortize overhead

### Issue: Incompatibility with attention backend
**Error**: Per-attention-head quantization not supported

**Solution**: Use per-tensor quantization (broader compatibility) or switch to Flash Attention backend

## Limitations

- **Accuracy risk**: Small accuracy loss possible (typically <1%)
- **Long context accumulation**: Errors can accumulate over very long sequences (>16K tokens)
- **Calibration overhead**: Dataset calibration requires offline step
- **Backend dependency**: Per-attention-head requires Flash Attention
- **CUDA version**: Requires CUDA 11.8+ for FP8 support

## Cross-References

### Related Techniques
- [[Quantization]] — Overview of quantization methods
- [[FP8 Quantization]] — FP8 weight/activation quantization
- [[INT8 W8A8]] — INT8 weight/activation quantization
- [[INT4 W4A16]] — INT4 weight-only quantization

### Related Concepts
- [[KV Cache]] — Key-value cache in attention mechanisms
- [[Prefix Caching]] — Reusing KV cache for shared prefixes
- [[PagedAttention]] — KV cache memory management in vLLM

### Related Tools
- [[llm-compressor]] — Calibration toolkit
- [[vLLM]] — Inference engine

### Related Architectures
- [[Hybrid KV Cache Manager]] — KV cache management for mixed attention types

## References

- Source: [[vllm-quantization]]
- llm-compressor examples: https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_kv_cache
- FP8 specification: https://arxiv.org/abs/2209.05433
