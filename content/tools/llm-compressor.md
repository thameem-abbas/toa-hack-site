---
title: llm-compressor
type: tool
category: quantization
created: 2026-04-25
---

# llm-compressor

**llm-compressor** is a unified quantization toolkit developed by the vLLM team (formerly Neural Magic, now part of Red Hat) for creating compressed models optimized for vLLM inference. It consolidates multiple quantization methods (GPTQ, AWQ, FP8, INT8, SmoothQuant) into a single, actively maintained library.

## Overview

llm-compressor is the successor to AutoGPTQ and AutoAWQ, providing a modern, unified API for post-training quantization of large language models. It is the **recommended** tool for quantizing models for vLLM deployment.

**Key features:**
- **Unified API:** Single interface for GPTQ, AWQ, FP8, INT8, SmoothQuant
- **One-shot quantization:** Fast quantization without calibration (for some methods)
- **Calibration-based flows:** Traditional PTQ with calibration datasets
- **Direct vLLM compatibility:** Models ready for vLLM without conversion
- **Active maintenance:** Developed and maintained by vLLM core team
- **Production-ready:** Used in production by Red Hat and vLLM users

## Supported Quantization Methods

### Weight Quantization

**GPTQ:**
- INT4/INT8 weight quantization
- Hessian-based error compensation
- Group-wise scaling (configurable group size)
- Compatible with Marlin/Machete backends in vLLM

**AWQ:**
- INT4 weight quantization
- Activation-aware salient weight protection
- Group-wise scaling
- Compatible with Marlin backend in vLLM

### Weight + Activation Quantization

**FP8:**
- FP8 E4M3 for weights and activations
- Per-tensor or per-channel quantization
- Static or dynamic quantization
- Optimized for NVIDIA Hopper GPUs (H100+)

**INT8 (W8A8):**
- INT8 for both weights and activations
- Per-tensor or per-channel quantization
- SmoothQuant algorithm for activation quantization
- Broader hardware support (Ampere+)

### Calibration Techniques

**SmoothQuant:**
- Smooths activation outliers to enable INT8 quantization
- Per-channel scaling factors
- Migrates quantization difficulty from activations to weights
- Used as preprocessing for W8A8 quantization

## Installation

```bash
# Install from PyPI
pip install llm-compressor

# Or install from source
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
```

## Usage

### GPTQ Quantization

```python
from llm_compressor.transformers import oneshot
from transformers import AutoTokenizer

# Load model and tokenizer
model_name = "meta-llama/Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure GPTQ quantization
recipe = """
quant_stage:
  quant_modifiers:
    GPTQModifier:
      sequential_update: true
      ignore: ["lm_head"]
      config_groups:
        group_0:
          weights:
            num_bits: 4
            type: int
            symmetric: true
            group_size: 128
          targets: ["Linear"]
"""

# One-shot quantization
oneshot(
    model=model_name,
    dataset="open_platypus",
    recipe=recipe,
    output_dir="./llama3-8b-gptq",
    max_seq_length=2048,
    num_calibration_samples=512,
)

# Model ready for vLLM
# vllm serve ./llama3-8b-gptq
```

### AWQ Quantization

```python
from llm_compressor.transformers import oneshot

# Configure AWQ quantization
recipe = """
quant_stage:
  quant_modifiers:
    AWQModifier:
      sequential_update: true
      ignore: ["lm_head"]
      config_groups:
        group_0:
          weights:
            num_bits: 4
            type: int
            symmetric: false
            group_size: 128
          targets: ["Linear"]
"""

# One-shot quantization
oneshot(
    model="meta-llama/Llama-3-8B",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="./llama3-8b-awq",
    max_seq_length=2048,
    num_calibration_samples=512,
)

# Use with vLLM
# vllm serve ./llama3-8b-awq --quantization awq
```

### FP8 Quantization

```python
from llm_compressor.transformers import oneshot

# Configure FP8 quantization
recipe = """
quant_stage:
  quant_modifiers:
    FP8Modifier:
      ignore: ["lm_head"]
      config_groups:
        group_0:
          weights:
            num_bits: 8
            type: float
            strategy: tensor
          input_activations:
            num_bits: 8
            type: float
            strategy: tensor
          targets: ["Linear"]
"""

# One-shot quantization (FP8 can be one-shot or calibrated)
oneshot(
    model="meta-llama/Llama-3-8B",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="./llama3-8b-fp8",
    max_seq_length=2048,
    num_calibration_samples=512,
)

# Use with vLLM (auto-detected)
# vllm serve ./llama3-8b-fp8
```

### INT8 W8A8 with SmoothQuant

```python
from llm_compressor.transformers import oneshot

# Configure INT8 with SmoothQuant
recipe = """
quant_stage:
  quant_modifiers:
    SmoothQuantModifier:
      smoothing_strength: 0.5
      ignore: ["lm_head"]
    QuantizationModifier:
      ignore: ["lm_head"]
      config_groups:
        group_0:
          weights:
            num_bits: 8
            type: int
            symmetric: true
          input_activations:
            num_bits: 8
            type: int
            symmetric: false
          targets: ["Linear"]
"""

# Calibration-based quantization
oneshot(
    model="meta-llama/Llama-3-8B",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="./llama3-8b-int8",
    max_seq_length=2048,
    num_calibration_samples=512,
)

# Use with vLLM
# vllm serve ./llama3-8b-int8 --quantization int8
```

## Quantization Workflows

### One-Shot Quantization

**Fast quantization without calibration data:**
- Supported for: AWQ, some FP8 configs
- Uses synthetic data or model's own weights
- 5-10× faster than calibration-based
- Slightly lower accuracy than calibrated

```python
oneshot(
    model="meta-llama/Llama-3-8B",
    dataset=None,  # No calibration data
    recipe=awq_recipe,
    output_dir="./llama3-8b-awq-oneshot",
)
```

### Calibration-Based Quantization

**Traditional PTQ with calibration dataset:**
- Supported for: GPTQ, AWQ, FP8, INT8
- Requires representative calibration data (128-1024 samples)
- Better accuracy than one-shot
- Slower quantization

```python
oneshot(
    model="meta-llama/Llama-3-8B",
    dataset="open_platypus",  # Calibration dataset
    recipe=gptq_recipe,
    output_dir="./llama3-8b-gptq",
    num_calibration_samples=512,
)
```

### Custom Calibration Data

```python
from datasets import load_dataset

# Load custom dataset
dataset = load_dataset("allenai/c4", split="train", streaming=True)
dataset = dataset.shuffle(seed=42).take(1024)

oneshot(
    model="meta-llama/Llama-3-8B",
    dataset=dataset,
    recipe=recipe,
    output_dir="./llama3-8b-quantized",
)
```

## Advanced Features

### Per-Module Quantization

Quantize different layers with different settings:

```python
recipe = """
quant_stage:
  quant_modifiers:
    GPTQModifier:
      ignore: ["lm_head"]
      config_groups:
        group_attention:
          weights:
            num_bits: 4
            group_size: 128
          targets: ["self_attn.*"]
        group_mlp:
          weights:
            num_bits: 8
            group_size: 64
          targets: ["mlp.*"]
"""
```

### Mixed Precision

Combine quantization methods:

```python
# FP8 for most layers, FP16 for sensitive layers
recipe = """
quant_stage:
  quant_modifiers:
    FP8Modifier:
      ignore: ["lm_head", "model.embed_tokens"]
      config_groups:
        group_0:
          weights:
            num_bits: 8
            type: float
          targets: ["Linear"]
"""
```

### KV Cache Quantization

Quantize KV cache in addition to weights:

```python
# Combine weight quantization with KV cache quantization
recipe = """
quant_stage:
  quant_modifiers:
    AWQModifier:
      # ... AWQ config for weights
    KVCacheQuantModifier:
      num_bits: 8
      type: float  # or int
"""
```

## Integration with vLLM

llm-compressor produces models directly compatible with vLLM:

**GPTQ models:**
```bash
vllm serve ./llama3-8b-gptq  # Auto-detects GPTQ
# or explicitly:
vllm serve ./llama3-8b-gptq --quantization gptq_marlin
```

**AWQ models:**
```bash
vllm serve ./llama3-8b-awq --quantization awq
# or with Marlin backend:
vllm serve ./llama3-8b-awq --quantization awq_marlin
```

**FP8 models:**
```bash
vllm serve ./llama3-8b-fp8  # Auto-detects FP8
```

**INT8 models:**
```bash
vllm serve ./llama3-8b-int8 --quantization int8
```

## Performance Characteristics

**Quantization speed:**
- One-shot AWQ: 5-15 minutes (Llama-3-8B)
- Calibrated GPTQ: 30-60 minutes (Llama-3-8B)
- FP8: 10-30 minutes (Llama-3-8B)
- Scales with model size and calibration dataset

**Model quality:**
- GPTQ: Best INT4 accuracy (Hessian-based)
- AWQ: Fast with good INT4 accuracy
- FP8: Near-lossless for most models
- INT8: Excellent accuracy with SmoothQuant

## Comparison with Other Tools

**llm-compressor vs AutoGPTQ:**
- llm-compressor: Modern, unified, actively maintained by vLLM team
- AutoGPTQ: Legacy, GPTQ-only, less maintained
- Recommendation: Use llm-compressor

**llm-compressor vs AutoAWQ:**
- llm-compressor: Includes AWQ + other methods, vLLM-native
- AutoAWQ: Deprecated, AWQ-only
- Recommendation: Use llm-compressor

**llm-compressor vs [[AMD Quark]]:**
- llm-compressor: NVIDIA-focused, vLLM-native
- Quark: AMD-focused (MI-series GPUs), supports MXFP4/MXFP6
- Recommendation: llm-compressor for NVIDIA, Quark for AMD

**llm-compressor vs [[NVIDIA Model Optimizer]]:**
- llm-compressor: Open-source, vLLM-native, Red Hat supported
- ModelOpt: NVIDIA official, supports VLMs and diffusion models
- Recommendation: llm-compressor for LLMs, ModelOpt for VLMs/diffusion

## Limitations

1. **NVIDIA-centric:** Optimized for NVIDIA GPUs, limited AMD support
2. **Calibration data required:** Most methods need representative dataset
3. **Quantization time:** Can be slow for large models (70B+)
4. **Memory overhead:** Quantization process requires full model in memory
5. **Recipe syntax:** YAML recipe syntax has learning curve

## Use Cases

**When to use llm-compressor:**
1. **vLLM deployment:** Primary tool for quantizing models for vLLM
2. **Production serving:** Red Hat-supported, production-ready
3. **Unified workflow:** Need multiple quantization methods in one tool
4. **NVIDIA GPUs:** Optimized for A100/H100 serving
5. **Modern API:** Prefer maintained, actively developed tools

## Future Directions

- **GGUF export:** Support exporting to GGUF format
- **QAT support:** Quantization-aware training workflows
- **AutoQuant:** Automatic quantization method selection
- **ROCm optimization:** Better AMD GPU support
- **VLM support:** Vision-language model quantization
- **Faster calibration:** Smarter dataset selection and size

## References

- **GitHub:** https://github.com/vllm-project/llm-compressor
- **Documentation:** https://github.com/vllm-project/llm-compressor/tree/main/examples
- **AWQ examples:** https://github.com/vllm-project/llm-compressor/tree/main/examples/awq
- **GPTQ examples:** https://github.com/vllm-project/llm-compressor/tree/main/examples/gptq
- **FP8 examples:** https://github.com/vllm-project/llm-compressor/tree/main/examples/fp8

## Cross-References

- [[Quantization]] — Parent concept
- [[AWQ]] — Supported quantization method
- [[GPTQ]] — Supported quantization method
- [[FP8 Quantization]] — Supported quantization method
- [[INT8 W8A8]] — Supported quantization method (if exists)
- [[SmoothQuant]] — Supported calibration technique (if exists)
- [[vLLM]] — Target inference engine
- [[AutoGPTQ]] — Deprecated predecessor
- [[AutoAWQ]] — Deprecated predecessor
- [[AMD Quark]] — AMD alternative
- [[NVIDIA Model Optimizer]] — NVIDIA alternative
- [[lm-eval Harness]] — Evaluation tool
