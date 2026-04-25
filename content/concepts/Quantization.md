---
title: Quantization
type: concept
created: 2026-04-25
---

# Quantization

Quantization is the process of reducing the numerical precision of model weights and/or activations from higher-precision formats (FP32, FP16, BF16) to lower-precision formats (FP8, INT8, INT4) to reduce memory footprint and accelerate inference. The core tradeoff is between model size/speed and accuracy.

## Overview

Quantization enables running large language models on hardware with limited memory by compressing the model's parameters and intermediate activations. A typical FP16 model with 7B parameters requires ~14 GB of memory; quantizing to FP8 reduces this to ~7 GB, and INT4 to ~3.5 GB.

## Quantization Dimensions

### Weight vs Activation Quantization

- **Weight-only quantization** (W4A16, W8A16): Only model weights are quantized; activations remain in higher precision (FP16/BF16). Reduces memory footprint but provides limited compute acceleration.
- **Weight-activation quantization** (W8A8, W4A8): Both weights and activations are quantized. Provides both memory savings and compute acceleration via specialized kernels (INT8 tensor cores, FP8 tensor cores).

### Static vs Dynamic Quantization

- **Static quantization**: Quantization scales are computed once during a calibration phase and fixed during inference. Used for weights and optionally for activations.
- **Dynamic quantization**: Quantization scales are computed on-the-fly during each forward pass. Common for activations (per-token dynamic quantization).

### Quantization Granularity

- **Per-tensor**: Single scale for the entire tensor
- **Per-channel**: One scale per output channel (for weights)
- **Per-token**: One scale per token (for activations)
- **Per-attention-head**: One scale per attention head (for KV cache)
- **Group-wise**: One scale per group of weights (e.g., group_size=128 for INT4)

## Quantization Methods in vLLM

vLLM supports 13+ quantization formats across different hardware platforms:

### FP8 Quantization
- **Formats**: FP8 E4M3 (±448 range, 3-bit mantissa) and FP8 E5M2 (±57344 range, 2-bit mantissa)
- **Methods**: [[FP8 Quantization]] (W8A8), weight-only FP8 (W8A16 via Marlin)
- **Hardware**: NVIDIA Ada Lovelace (SM89+), Hopper (SM90+), AMD MI300+
- **Benefits**: 2× memory reduction, up to 1.6× throughput improvement
- **Use case**: Modern GPU deployments requiring high throughput with minimal accuracy loss

### INT8 Quantization
- **Method**: [[INT8 W8A8]] with SmoothQuant activation smoothing
- **Hardware**: NVIDIA Turing (SM75+), Ampere, Ada, Hopper; x86 CPU
- **Benefits**: 2× memory reduction vs FP16, good accuracy with calibration
- **Limitation**: Not supported on Blackwell (SM 10.0+)
- **Use case**: Deployments on Turing/Ampere GPUs where FP8 hardware is unavailable

### INT4 Quantization
- **Method**: [[INT4 W4A16]] with group-wise quantization (GPTQ algorithm)
- **Hardware**: NVIDIA Ampere (SM80+), Ada, Hopper, Blackwell
- **Benefits**: 4× memory reduction vs FP16, Marlin kernel acceleration
- **Limitation**: Requires calibration data; higher accuracy loss than FP8/INT8
- **Use case**: Memory-constrained deployments with low QPS (queries per second)

### Other Methods
- **[[AWQ]]**: Activation-aware weight quantization (INT4)
- **[[GPTQ]]**: Post-training quantization via approximate second-order optimization (INT4)
- **[[GGUF]]**: GGML Universal File format (mixed precision)
- **BitsAndBytes**: NF4/FP4 quantization for training and inference
- **ModelOpt**: NVIDIA's optimization toolkit
- **TorchAO**: PyTorch's built-in quantization
- **Quark**: AMD's quantization toolkit

### KV Cache Quantization
- **Method**: [[Quantized KV Cache]] — quantizing key-value cache to FP8
- **Benefits**: ~50% reduction in KV cache memory footprint
- **Orthogonality**: Can be combined with weight/activation quantization
- **Strategies**: Per-tensor or per-attention-head quantization
- **Use case**: Enabling longer context windows and higher batch sizes

## Hardware Compatibility

| Method | Volta (7.0) | Turing (7.5) | Ampere (8.0) | Ada (8.9) | Hopper (9.0) | AMD GPU | Intel GPU | x86 CPU |
|--------|-------------|--------------|--------------|-----------|--------------|---------|-----------|---------|
| FP8 W8A8 | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ (MI300+) | ❌ | ❌ |
| FP8 W8A16 (Marlin) | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| INT8 W8A8 | ❌ | ✅ | ✅ | ✅ | ✅* | ❌ | ❌ | ✅ |
| INT4 W4A16 (Marlin) | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| GPTQ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| AWQ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| GGUF | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |

*INT8 not supported on Blackwell (SM 10.0+)

## Calibration Requirements

### No Calibration Required
- **FP8 online quantization**: `quantization="fp8"` flag enables dynamic FP8 without calibration
- **Weight-only methods**: Weights can be quantized via round-to-nearest (RTN) without data

### Calibration Recommended
- **INT8 W8A8**: Requires 512+ samples for activation scale estimation
- **INT4 W4A16**: Requires 512+ samples for GPTQ weight updates
- **FP8 KV cache**: Dataset calibration via [[llm-compressor]] for per-attention-head quantization

### Calibration Data Selection
- Use data representative of deployment workload
- For instruction-tuned models: `ultrachat_200k` or similar chat datasets
- For fine-tuned models: Sample from training data
- Typical configuration: 512 samples, 2048 tokens max sequence length

## Configuration in vLLM

### Command-line flag
```bash
vllm serve meta-llama/Llama-3-8B-Instruct --quantization fp8
```

### Python API
```python
from vllm import LLM
llm = LLM("meta-llama/Llama-3-8B-Instruct", quantization="fp8")
```

### Pre-quantized models
```python
# Model already quantized offline via llm-compressor
llm = LLM("neuralmagic/Meta-Llama-3-8B-Instruct-FP8")
```

### KV cache quantization
```python
llm = LLM(
    "meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True  # or False for no calibration
)
```

## Quantization Workflow with llm-compressor

[[llm-compressor]] is the recommended toolkit for offline quantization in vLLM.

### 1. Load model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
```

### 2. Prepare calibration data (if needed)
```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(512))
ds = ds.map(lambda ex: tokenizer(tokenizer.apply_chat_template(ex["messages"], tokenize=False)))
```

### 3. Apply quantization
```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# FP8: no calibration needed
recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
oneshot(model=model, recipe=recipe)

# INT8: requires calibration data
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])
]
oneshot(model=model, dataset=ds, recipe=recipe)
```

### 4. Save and deploy
```python
model.save_pretrained("./Llama-3-8B-FP8", save_compressed=True)
tokenizer.save_pretrained("./Llama-3-8B-FP8")

# Use in vLLM
llm = LLM("./Llama-3-8B-FP8")
```

## Accuracy Evaluation

Quantized models are sensitive to evaluation configuration:

```bash
lm_eval --model vllm \
  --model_args pretrained=./Llama-3-8B-FP8,add_bos_token=True \
  --tasks gsm8k --num_fewshot 5 --batch_size auto --limit 250
```

**Key consideration**: Include `add_bos_token=True` for quantized models, as they can be sensitive to the presence of the beginning-of-sequence token.

## Performance Metrics

### Memory Savings
- **FP8 W8A8**: 2× reduction vs FP16
- **INT8 W8A8**: 2× reduction vs FP16
- **INT4 W4A16**: 4× reduction vs FP16
- **FP8 KV cache**: ~50% KV cache memory reduction (orthogonal to weight quantization)

### Throughput Improvements
- **FP8 W8A8**: Up to 1.6× on Ada/Hopper GPUs
- **INT8 W8A8**: Variable (depends on model size, batch size, sequence length)
- **INT4 W4A16**: Best for low QPS; higher QPS may see slowdowns due to dequantization overhead

### Accuracy Impact
- **FP8**: Minimal accuracy loss (typically <1% on benchmarks)
- **INT8**: Requires calibration; accuracy close to FP16 with SmoothQuant
- **INT4**: Higher accuracy loss; tune `dampening_frac` and `actorder` to mitigate

## Plugin Architecture

vLLM supports out-of-tree (OOT) quantization plugins via the `@register_quantization_config` decorator:

```python
from vllm.model_executor.layers.quantization import register_quantization_config

@register_quantization_config("my_quant")
class MyQuantConfig(QuantizationConfig):
    def get_name(self) -> str: return "my_quant"
    def get_supported_act_dtypes(self) -> list: return [torch.float16]
    def get_min_capability(cls) -> int: return 80  # Ampere+
    def get_quant_method(self, layer, prefix): ...
```

This enables hardware vendors and researchers to implement custom quantization schemes without modifying vLLM core.

## Best Practices

1. **Start with FP8** if hardware supports it (Ada/Hopper/MI300) — best accuracy/performance tradeoff
2. **Use calibration data** that matches deployment workload (chat template, sequence length distribution)
3. **Start with 512 samples** for calibration; increase if accuracy drops
4. **Combine with KV cache quantization** for maximum memory savings
5. **Tune hyperparameters** for INT4 (dampening_frac, actorder) if accuracy is unsatisfactory
6. **Evaluate with proper settings** (add_bos_token=True for quantized models)
7. **Consider weight-only** (W8A16, W4A16) for deployments where compute is not the bottleneck

## Limitations and Considerations

- **Hardware dependency**: FP8 requires modern GPUs; INT8 not supported on Blackwell
- **Calibration cost**: INT8/INT4 require offline calibration step with representative data
- **Accuracy-memory tradeoff**: Lower precision = higher memory savings but potential accuracy loss
- **Kernel availability**: Some quantization methods require specific kernels (Marlin for INT4/FP8, CUTLASS for INT8)
- **Model compatibility**: Some models may not support all quantization formats (check model card)

## Cross-References

### Related Techniques
- [[FP8 Quantization]] — FP8 E4M3/E5M2 weight-activation quantization
- [[INT8 W8A8]] — INT8 weight-activation quantization with SmoothQuant
- [[INT4 W4A16]] — INT4 weight-only quantization with GPTQ
- [[Quantized KV Cache]] — FP8 KV cache quantization
- [[AWQ]] — Activation-aware weight quantization
- [[GPTQ]] — Post-training quantization
- [[GGUF]] — GGML Universal File format

### Related Concepts
- [[KV Cache]] — Key-value cache in attention mechanisms
- [[Kernel Fusions]] — Kernel optimizations often combined with quantization
- [[Mixture of Experts]] — MoE models with quantized experts

### Related Tools
- [[llm-compressor]] — Primary quantization toolkit for vLLM
- [[vLLM]] — Inference engine with quantization support

### Related Architectures
- [[CustomOp System]] — Platform-specific operation dispatch for quantized kernels

## References

- Source: [[vllm-quantization]]
- HuggingFace FP8 collection: https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127
- HuggingFace INT8 collection: https://huggingface.co/collections/neuralmagic/int8-llms-for-vllm-668ec32c049dca0369816415
- HuggingFace INT4 collection: https://huggingface.co/collections/neuralmagic/int4-llms-for-vllm-668ec34bf3c9fa45f857df2c
