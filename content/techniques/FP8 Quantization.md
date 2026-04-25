---
title: FP8 Quantization
type: technique
created: 2026-04-25
---

# FP8 Quantization

FP8 (8-bit floating point) quantization reduces model weights and activations from FP16/BF16 precision to 8-bit floating point formats, achieving 2× memory reduction and up to 1.6× throughput improvement with minimal accuracy impact. vLLM supports FP8 quantization on NVIDIA Ada Lovelace/Hopper GPUs and AMD MI300+ GPUs.

## Overview

FP8 quantization in vLLM comes in two modes:
- **W8A8**: Both weights and activations quantized to FP8 (requires Ada/Hopper/MI300+)
- **W8A16**: Weight-only FP8 quantization with FP16 activations (Turing/Ampere via Marlin kernels)

The technique provides the best accuracy-performance tradeoff among quantization methods, making it the recommended approach for modern GPU deployments.

## FP8 Number Formats

FP8 has two standard representations defined in the FP8 specification:

### FP8 E4M3
- **Structure**: 1 sign bit, 4 exponent bits, 3 mantissa bits
- **Range**: ±448
- **Special values**: NaN (no infinities)
- **Use case**: Model weights (higher precision, limited range)

### FP8 E5M2
- **Structure**: 1 sign bit, 5 exponent bits, 2 mantissa bits
- **Range**: ±57344
- **Special values**: ±Inf, NaN
- **Use case**: Gradients during training (wider dynamic range, lower precision)

**vLLM default**: FP8 E4M3 for both weights and activations in inference.

## Quantization Schemes

### FP8_DYNAMIC (Recommended)
- **Weight quantization**: Static, per-channel scales computed once during quantization
- **Activation quantization**: Dynamic, per-token scales computed during each forward pass
- **Calibration**: No calibration data required (weights use round-to-nearest, activations are dynamic)
- **Accuracy**: Minimal loss compared to FP16 baseline
- **Throughput**: Up to 1.6× improvement on Hopper/Ada GPUs

### Static per-tensor
- **Weight quantization**: Single scale for entire weight tensor
- **Activation quantization**: Single scale for entire activation tensor
- **Use case**: Simplest scheme, lower accuracy than per-channel

### Static per-channel
- **Weight quantization**: One scale per output channel
- **Activation quantization**: Typically combined with dynamic per-token activations
- **Use case**: Default for weight quantization in FP8_DYNAMIC scheme

## Hardware Support

### NVIDIA GPUs
| GPU Generation | Compute Capability | W8A8 Support | W8A16 Support | Kernel |
|----------------|-------------------|--------------|---------------|--------|
| Volta | SM 7.0 | ❌ | ❌ | — |
| Turing | SM 7.5 | ❌ | ✅ | Marlin |
| Ampere | SM 8.0/8.6 | ❌ | ✅ | Marlin |
| Ada Lovelace | SM 8.9 | ✅ | ✅ | Native FP8, Marlin |
| Hopper | SM 9.0 | ✅ | ✅ | Native FP8, Marlin |
| Blackwell | SM 10.0 | ✅ | ✅ | Native FP8, Marlin |

### AMD GPUs
- **MI300+**: Full W8A8 support with native FP8 instructions
- **Older generations**: Not supported

### Other Platforms
- **Intel GPU**: ❌
- **x86 CPU**: ❌
- **TPU**: See TPU-specific documentation

## Quantization Workflow

### Option 1: Online Dynamic Quantization (Simplest)

No calibration or offline quantization required. vLLM quantizes on-the-fly during model loading.

```python
from vllm import LLM

# Automatic FP8 quantization at runtime
llm = LLM("meta-llama/Llama-3-8B-Instruct", quantization="fp8")
result = llm.generate("Hello, my name is")
```

**Behavior**:
- All `Linear` layers (except `lm_head`) quantized to FP8 E4M3
- Weights: Per-tensor static scales
- Activations: Per-tensor dynamic scales computed during forward pass
- Memory: 2× reduction
- Throughput: Limited improvement due to dynamic activation quantization overhead

**Use case**: Quick deployment without offline quantization step; acceptable for prototyping.

### Option 2: Offline Quantization with llm-compressor (Recommended)

Pre-quantize the model using [[llm-compressor]] for optimal performance.

#### Step 1: Install llm-compressor
```bash
pip install llmcompressor
```

#### Step 2: Load model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

#### Step 3: Apply FP8_DYNAMIC quantization
```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],  # Skip final projection layer
)

oneshot(model=model, recipe=recipe)
```

**Why no calibration data?**
- Weights quantized via round-to-nearest (RTN) — deterministic, no data needed
- Activations use dynamic per-token quantization — scales computed at runtime

#### Step 4: Save quantized model
```python
SAVE_DIR = "Meta-Llama-3-8B-Instruct-FP8-Dynamic"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

#### Step 5: Deploy in vLLM
```python
from vllm import LLM

llm = LLM("./Meta-Llama-3-8B-Instruct-FP8-Dynamic")
result = llm.generate("Hello my name is")
```

### Option 3: Use Pre-Quantized Models

HuggingFace hosts a collection of pre-quantized FP8 models optimized for vLLM:

```python
from vllm import LLM

# Directly use NeuralMagic's pre-quantized models
llm = LLM("neuralmagic/Meta-Llama-3-8B-Instruct-FP8")
```

**Collection**: https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127

## Kernel Backends

### Native FP8 Tensor Cores (Ada/Hopper/MI300)
- **Hardware acceleration**: 4th-gen Tensor Cores (NVIDIA), Matrix Cores (AMD)
- **Performance**: Up to 1.6× throughput improvement
- **Precision**: Full FP8 matmul operations

### Marlin Kernel (Turing/Ampere weight-only)
- **Mode**: W8A16 (weights in FP8, activations in FP16)
- **Performance**: Moderate speedup over FP16
- **Compatibility**: Fallback for GPUs without native FP8 support

### CUTLASS Backend
- **Use case**: Alternative high-performance kernel implementation
- **Availability**: Configurable via vLLM compilation flags

## Combining FP8 with Other Optimizations

### FP8 + Quantized KV Cache
```python
llm = LLM(
    "neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True
)
```

**Benefit**: Compound memory savings — 2× from weights, ~50% from KV cache

### FP8 + Kernel Fusions
FP8 quantization integrates with [[Kernel Fusions]] in vLLM:
- **Attention+Quant fusion**: 3-7% speedup (requires full graph mode)
- **RMSNorm+Quant fusion**: 1-4% speedup
- **SiLU+Mul+Quant fusion**: 1-4% speedup

Enable via `torch.compile` integration:
```python
llm = LLM(
    "neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    enforce_eager=False  # Enable CUDA graphs and fusions
)
```

### FP8 + Prefix Caching
```python
llm = LLM(
    "neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    enable_prefix_caching=True
)
```

**Benefit**: Reduced TTFT for requests with shared prefixes

### FP8 MoE Models
FP8 quantization is supported for [[Mixture of Experts]] models via [[FusedMoE Modular Kernel]]:

```python
# FP8 quantized MoE (e.g., Mixtral, DeepSeek-V3)
llm = LLM("neuralmagic/Mixtral-8x7B-Instruct-FP8")
```

**FP8 MoE benefits**:
- 2× reduction in expert memory footprint
- 1.2-1.5× speedup for MoE layers
- All experts GPU-resident (no offloading)

## Accuracy Evaluation

FP8 quantization typically has minimal accuracy impact, but always evaluate on downstream tasks:

### Install lm-eval
```bash
pip install vllm "lm-eval[api]>=0.4.11"
```

### Run evaluation
```bash
MODEL=./Meta-Llama-3-8B-Instruct-FP8-Dynamic
lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size auto \
  --limit 250
```

**Important**: Include `add_bos_token=True` — quantized models can be sensitive to BOS token presence.

### Example Results (Llama 3 8B on GSM8K)
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
| --- |------:| -------------- |-----:| --------- | - |----:| - |-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.768|±  |0.0268|
```

**Observation**: FP8 typically within 1% of FP16 baseline on most benchmarks.

## Performance Metrics

### Memory Savings
- **Model weights**: 2× reduction (FP16 → FP8)
- **Activations**: No reduction (dynamic quantization)
- **Total**: ~2× end-to-end memory reduction for weight-dominated models

### Throughput Improvements
- **Hopper H100**: Up to 1.6× throughput improvement
- **Ada L40S/L4**: Up to 1.5× throughput improvement
- **AMD MI300X**: Up to 1.4× throughput improvement
- **Turing/Ampere (W8A16)**: 1.1-1.2× throughput improvement

**Factors affecting speedup**:
- Batch size (larger = better amortization of quantization overhead)
- Sequence length (longer = more matmul-bound = higher speedup)
- Model size (larger = more time in matmuls = higher speedup)

### Latency Characteristics
- **TTFT (Time to First Token)**: Minimal impact (prefill phase)
- **ITL (Inter-Token Latency)**: 10-30% reduction on Ada/Hopper
- **Online dynamic quantization**: Limited latency improvement due to dynamic scales

## Troubleshooting

### Model fails to load with FP8 quantization
**Issue**: GPU does not support native FP8
**Solution**: Check compute capability; use W8A16 Marlin on Turing/Ampere or upgrade to Ada/Hopper

### Accuracy significantly degraded
**Issue**: FP8 may not preserve accuracy for all models
**Solution**: 
- Try [[INT8 W8A8]] with SmoothQuant calibration
- Use per-channel weight quantization (default in FP8_DYNAMIC)
- Check if model was pre-trained with quantization-aware training

### Lower throughput than expected
**Issue**: Dynamic activation quantization overhead
**Solution**: Use offline-quantized model with static scales where possible

### Out of memory errors
**Issue**: FP8 reduces memory but may still exceed GPU capacity
**Solution**: Combine with [[Quantized KV Cache]] or use [[INT4 W4A16]]

## Best Practices

1. **Use offline quantization** with llm-compressor for production deployments
2. **Evaluate on representative tasks** before deploying (add_bos_token=True)
3. **Combine with KV cache quantization** for maximum memory savings
4. **Enable CUDA graphs** (default in vLLM V1) for kernel fusion benefits
5. **Use pre-quantized models** from NeuralMagic collection when available
6. **Monitor accuracy on long contexts** — FP8 errors can accumulate

## Limitations

- **Hardware requirement**: Requires Ada Lovelace (SM89+), Hopper (SM90+), or AMD MI300+ for W8A8
- **Turing/Ampere limitation**: W8A16 only (no activation quantization)
- **Model compatibility**: Some custom architectures may not support FP8 quantization
- **Accuracy risk**: Small accuracy loss possible (typically <1% but model-dependent)

## Cross-References

### Related Techniques
- [[Quantization]] — Overview of quantization methods
- [[INT8 W8A8]] — Alternative 8-bit quantization for older GPUs
- [[INT4 W4A16]] — 4-bit quantization for higher memory savings
- [[Quantized KV Cache]] — FP8 KV cache quantization (orthogonal optimization)
- [[Kernel Fusions]] — Fusion optimizations for quantized kernels

### Related Concepts
- [[KV Cache]] — Key-value cache in attention
- [[Mixture of Experts]] — MoE models with FP8 experts

### Related Tools
- [[llm-compressor]] — Quantization toolkit
- [[vLLM]] — Inference engine

### Related Architectures
- [[FusedMoE Modular Kernel]] — MoE kernel with FP8 support
- [[CustomOp System]] — Platform-specific FP8 kernel dispatch

## References

- Source: [[vllm-quantization]]
- FP8 specification: https://arxiv.org/abs/2209.05433
- NeuralMagic FP8 collection: https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127
- llm-compressor GitHub: https://github.com/vllm-project/llm-compressor
