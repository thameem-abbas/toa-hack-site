---
title: INT4 W4A16
type: technique
created: 2026-04-25
---

# INT4 W4A16

INT4 W4A16 quantization reduces model weights from FP16/BF16 to 4-bit integers while keeping activations in FP16, achieving 4× memory reduction. This method is particularly useful for memory-constrained deployments and low queries-per-second (QPS) workloads where latency is acceptable.

## Overview

INT4 W4A16 is a weight-only quantization method:
- **Weights**: Quantized to 4-bit signed integers (INT4)
- **Activations**: Remain in FP16/BF16 precision
- **Quantization**: Group-wise with per-group scales (default group_size=128)
- **Algorithm**: GPTQ (Generalized Post-Training Quantization) with calibration data

The technique trades compute overhead (dequantizing weights during matmuls) for significant memory savings, making it ideal for batch size 1 or low QPS scenarios.

## Hardware Support

| Platform | Support | Compute Capability | Kernel | Notes |
|----------|---------|-------------------|--------|-------|
| NVIDIA Ampere | ✅ | SM 8.0/8.6 | Marlin | Native support |
| NVIDIA Ada | ✅ | SM 8.9 | Marlin | FP8 preferred for performance |
| NVIDIA Hopper | ✅ | SM 9.0 | Marlin | FP8 preferred for performance |
| NVIDIA Blackwell | ✅ | SM 10.0 | Marlin | Supported (unlike INT8) |
| NVIDIA Turing | ✅* | SM 7.5 | Marlin | Limited support |
| AMD MI300+ | ❌ | — | — | Not supported |
| Intel GPU | ✅ | — | CPU fallback | Performance varies |
| x86 CPU | ✅ | — | CPU kernels | Slow |

*Turing does not support Marlin MXFP4 variant

**Recommended platforms**: Ampere, Ada, Hopper, Blackwell

## Quantization Scheme

### Weight Quantization (Group-wise)
- **Precision**: INT4 (-8 to 7, or 0 to 15 unsigned)
- **Granularity**: Group-wise (default group_size=128 elements per group)
- **Method**: GPTQ with calibration data
- **Scale storage**: FP16 scales per group

### Group-wise Quantization

Instead of a single scale per tensor or channel, group-wise quantization uses one scale per group of `group_size` consecutive weights.

**Example** (group_size=128):
- Weight tensor of shape [4096, 11008]
- Each row divided into 11008 / 128 = 86 groups
- 4096 × 86 = 352,256 scales (FP16)

**Benefits**:
- Higher accuracy than per-tensor or per-channel
- Captures local weight variations
- Modest scale storage overhead

**Trade-off**:
- Smaller group_size = higher accuracy, more scale overhead
- Larger group_size = lower accuracy, less scale overhead
- Default 128 is empirically optimal for most models

### Activation Quantization (None)
- **Precision**: FP16 or BF16 (no quantization)
- **Reason**: Activation quantization to INT4 causes severe accuracy loss

## GPTQ Algorithm

GPTQ (Generalized Post-Training Quantization) is a second-order optimization method for weight quantization.

### Algorithm Overview
1. For each layer, process weights column-by-column
2. For each column:
   - Quantize the weight
   - Compute quantization error
   - Distribute error to remaining unquantized weights using Hessian inverse
3. Use `actorder` to determine column processing order

### Key Hyperparameters

#### dampening_frac
Controls the influence of the GPTQ algorithm:
```python
GPTQModifier(dampening_frac=0.01)
```

- **Lower values (0.001-0.01)**: Higher GPTQ influence, better accuracy, risk of numerical instability
- **Higher values (0.1-1.0)**: Lower GPTQ influence, more stable, lower accuracy
- **Default**: 0.01

**Best practice**: Start at 0.01; if quantization fails with numerical errors, increase to 0.05 or 0.1.

#### actorder
Determines the order in which weights are quantized:
```python
GPTQModifier(actorder="weight")
```

- **actorder="weight"**: Order by weight magnitude (recommended)
- **actorder=None**: Sequential order
- **Default**: None

**Benefit**: `actorder="weight"` improves accuracy by quantizing less important weights first, without added inference latency.

**Best practice**: Always set `actorder="weight"` for production models.

## Quantization Workflow

### Prerequisites
```bash
pip install llmcompressor
pip install vllm "lm-eval[api]>=0.4.11"
```

### Step 1: Load Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### Step 2: Prepare Calibration Data

INT4 W4A16 **requires calibration data** for GPTQ optimization.

```python
from datasets import load_dataset

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

# Preprocess: apply chat template
def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

# Tokenize
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False
    )
ds = ds.map(tokenize, remove_columns=ds.column_names)
```

**Calibration data best practices**:
- **High variety**: Ensure diverse samples to prevent overfitting to specific use cases
- **Representative**: Match deployment data distribution
- **512 samples**: Good starting point (increase to 1024 if accuracy drops)

### Step 3: Apply Quantization (Simple)

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# Simple W4A16 recipe
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
```

### Step 4: Apply Quantization (Advanced)

For production deployments, use expanded recipe with tuned hyperparameters:

```python
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)

recipe = GPTQModifier(
    targets="Linear",
    config_groups={
        "config_group": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                group_size=128,
                symmetric=True,
                dynamic=False,
                actorder="weight",  # Important: improves accuracy
            ),
        ),
    },
    ignore=["lm_head"],
    update_size=NUM_CALIBRATION_SAMPLES,
    dampening_frac=0.01,  # Tune if needed
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
```

### Step 5: Save Quantized Model

```python
SAVE_DIR = "Meta-Llama-3-8B-Instruct-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

### Step 6: Deploy in vLLM

```python
from vllm import LLM

llm = LLM("./Meta-Llama-3-8B-Instruct-W4A16-G128")
result = llm.generate("Hello, my name is")
print(result[0].outputs[0].text)
```

## Accuracy Evaluation

```bash
lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-W4A16-G128",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

**Critical**: Include `add_bos_token=true` for accurate evaluation.

## Performance Metrics

### Memory Savings
- **Model weights**: 4× reduction (FP16 → INT4)
- **Activations**: No reduction (remain in FP16)
- **Total**: ~4× reduction for weight-dominated models
- **Example**: Llama 3 8B: ~14 GB → ~3.5 GB

### Throughput
- **Low QPS**: Excellent (minimal dequantization overhead amortization needed)
- **High QPS**: Moderate to poor (dequantization overhead becomes bottleneck)
- **Batch size 1**: Comparable to FP16 latency
- **Large batches**: Slower than [[FP8 Quantization]] or [[INT8 W8A8]]

### Accuracy
- **With tuned hyperparameters**: Typically 2-5% accuracy drop vs FP16
- **Without tuning**: 5-10% accuracy drop
- **Model-dependent**: Some models quantize better than others (e.g., Llama better than GPT-NeoX)

### Latency Characteristics
- **TTFT (Time to First Token)**: Minimal impact (prefill phase compute-bound)
- **ITL (Inter-Token Latency)**: 10-30% increase due to dequantization overhead
- **Marlin kernel**: Optimized for INT4, reduces overhead on Ampere+

## Marlin Kernel

The Marlin kernel is a high-performance kernel for INT4 weight-only quantization.

### Features
- **Sparse matrix multiplication**: Optimized for 4-bit weights
- **Group-wise dequantization**: Fused into matmul kernel
- **Hardware acceleration**: Utilizes Ampere/Ada/Hopper Tensor Cores
- **Compatibility**: GPTQ, AWQ, FP8 Marlin (W8A16), MXFP4 (not on Turing)

### Performance
- **Speedup**: Up to 2× faster than naive INT4 matmul
- **Memory bandwidth**: Reduces DRAM reads by 4× (INT4 vs FP16)
- **Use case**: Enables large models on memory-constrained GPUs

## Comparison with AWQ and GPTQ

All three methods produce INT4 weight-only models but differ in quantization algorithm:

| Method | Algorithm | Calibration | Accuracy | Speed |
|--------|-----------|-------------|----------|-------|
| [[INT4 W4A16]] (GPTQ) | Second-order optimization | Required | High | Moderate |
| [[AWQ]] | Activation-aware scaling | Required | High | Moderate |
| [[GPTQ]] (standalone) | Same as INT4 W4A16 | Required | High | Moderate |

**In vLLM**:
- All three use Marlin kernel for inference
- Performance differences are minimal (same kernel, same format)
- Choice depends on offline quantization accuracy and tooling preference

## Best Practices

### 1. Calibration Data
- **Start with 512 samples**: Increase to 1024 if accuracy drops
- **High variety**: Prevent overfitting to specific patterns
- **Representative**: Match deployment data distribution
- **Chat template**: Use model's training chat/instruction template

### 2. Hyperparameter Tuning

#### dampening_frac
- **Start at 0.01**: Good default
- **If quantization fails**: Increase to 0.05 or 0.1 (more stable)
- **If accuracy is poor**: Decrease to 0.005 (riskier, higher accuracy potential)

#### actorder
- **Always use actorder="weight"**: Improves accuracy without inference overhead
- **Exception**: Skip if quantization fails (rare)

#### group_size
- **Default 128**: Empirically optimal for most models
- **Smaller (64)**: Higher accuracy, more scale overhead, slower inference
- **Larger (256)**: Lower accuracy, less scale overhead, faster inference

### 3. Deployment Scenario
- **Low QPS (batch size 1-4)**: INT4 W4A16 is ideal
- **High QPS (batch size 8+)**: Consider [[FP8 Quantization]] or [[INT8 W8A8]]
- **Memory-constrained**: INT4 provides maximum weight compression

### 4. Model Selection
- **Test before deploying**: Some models quantize poorly to INT4
- **Use pre-quantized models**: NeuralMagic provides tested INT4 models
- **Monitor accuracy**: Always evaluate on representative tasks

## Combining INT4 with Other Optimizations

### INT4 + Quantized KV Cache
```python
llm = LLM(
    "./Meta-Llama-3-8B-Instruct-W4A16-G128",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True
)
```

**Benefit**: Additional ~50% KV cache memory reduction

### INT4 + Prefix Caching
```python
llm = LLM(
    "./Meta-Llama-3-8B-Instruct-W4A16-G128",
    enable_prefix_caching=True
)
```

**Benefit**: Reduced TTFT for shared-prefix requests

### INT4 + Tensor Parallelism
```python
llm = LLM(
    "./Meta-Llama-3-8B-Instruct-W4A16-G128",
    tensor_parallel_size=2
)
```

**Benefit**: Enable larger models on multi-GPU systems

## Troubleshooting

### Issue: Accuracy significantly degraded
**Solutions**:
1. Increase calibration samples (512 → 1024)
2. Set `actorder="weight"`
3. Decrease `dampening_frac` (0.01 → 0.005)
4. Decrease `group_size` (128 → 64)
5. Use more representative calibration data

### Issue: Quantization fails with numerical errors
**Error**: Hessian inversion fails, NaN values

**Solutions**:
1. Increase `dampening_frac` (0.01 → 0.1)
2. Remove `actorder` (set to None)
3. Check calibration data for extreme values
4. Try different random seed for dataset shuffle

### Issue: Slow inference at high batch sizes
**Cause**: Dequantization overhead dominates at high throughput

**Solutions**:
1. Use [[FP8 Quantization]] instead (Ada/Hopper/MI300+)
2. Use [[INT8 W8A8]] instead (Turing/Ampere)
3. Reduce batch size (INT4 optimized for low QPS)

### Issue: Out of memory during quantization
**Solutions**:
1. Reduce `MAX_SEQUENCE_LENGTH` (2048 → 1024)
2. Reduce `NUM_CALIBRATION_SAMPLES` (512 → 256)
3. Use `device_map="auto"` for model loading
4. Quantize on a larger GPU

## Pre-Quantized Models

NeuralMagic provides pre-quantized INT4 models:

```python
from vllm import LLM
llm = LLM("neuralmagic/Meta-Llama-3-8B-Instruct-INT4")
```

**Collection**: https://huggingface.co/collections/neuralmagic/int4-llms-for-vllm-668ec34bf3c9fa45f857df2c

## Use Cases

### Ideal for INT4 W4A16
- **Memory-constrained deployments**: Fitting large models on smaller GPUs
- **Low QPS workloads**: Batch size 1-4 (chatbots, assistants)
- **Edge devices**: Limited VRAM (e.g., RTX 3090, RTX 4090)
- **Cost optimization**: Using cheaper GPUs (e.g., L4 instead of A100)

### Not ideal for INT4 W4A16
- **High throughput**: Large batch sizes (use FP8 or INT8)
- **Strict accuracy requirements**: <1% accuracy loss (use FP8)
- **Modern GPUs with FP8**: Ada/Hopper/MI300+ (use FP8 instead)

## Limitations

- **Accuracy loss**: 2-10% depending on model and tuning
- **Dequantization overhead**: Slower at high batch sizes
- **Calibration required**: Offline quantization step with representative data
- **Model-dependent**: Some models quantize poorly to INT4
- **Higher latency**: Compared to FP8/INT8 on modern GPUs

## Cross-References

### Related Techniques
- [[Quantization]] — Overview of quantization methods
- [[FP8 Quantization]] — Preferred method for modern GPUs
- [[INT8 W8A8]] — Alternative 8-bit quantization
- [[Quantized KV Cache]] — Orthogonal KV cache memory optimization
- [[AWQ]] — Alternative INT4 quantization method
- [[GPTQ]] — Standalone GPTQ implementation

### Related Concepts
- [[KV Cache]] — Key-value cache in attention
- [[Tensor Parallelism]] — Distributing model across GPUs

### Related Tools
- [[llm-compressor]] — Quantization toolkit
- [[vLLM]] — Inference engine

### Related Architectures
- [[CustomOp System]] — Platform-specific INT4 kernel dispatch

## References

- Source: [[vllm-quantization]]
- GPTQ paper: https://arxiv.org/abs/2210.17323
- Marlin kernel: https://github.com/IST-DASLab/marlin
- NeuralMagic INT4 collection: https://huggingface.co/collections/neuralmagic/int4-llms-for-vllm-668ec34bf3c9fa45f857df2c
- llm-compressor examples: https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a16/llama3_example.py
