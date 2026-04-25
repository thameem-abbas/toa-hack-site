---
title: GPTQ (Post-Training Quantization)
type: technique
category: quantization
created: 2026-04-25
---

# GPTQ (Post-Training Quantization)

**GPTQ** is a post-training quantization method that uses approximate second-order information (Hessian) to minimize quantization error, achieving high-quality INT4/INT8 weight quantization with group-wise scaling.

## Overview

GPTQ quantizes model weights from FP16/BF16 to INT4 (4-bit) or INT8 (8-bit) integers using layer-wise quantization with Hessian-based error compensation. Unlike simple rounding or activation-aware methods, GPTQ leverages second-order derivatives to minimize the impact of quantization on model outputs.

**Key characteristics:**
- **Hessian-based:** Uses approximate second-order information for optimal quantization
- **Layer-wise quantization:** Processes one layer at a time with error correction
- **Group-wise scaling:** Per-group scale factors (typical group size: 128)
- **Weight-only:** Weights quantized to INT4/INT8, activations remain FP16
- **No retraining:** Pure post-training quantization (PTQ)

## Architecture

### Quantization Process

GPTQ uses a block-wise quantization algorithm:

1. **Hessian estimation:** Compute approximate Hessian H = 2X^TX for each layer
2. **Cholesky decomposition:** Factor H = LL^T for stable inversion
3. **Block-wise quantization:** For each block of weights:
   - Quantize weights in block
   - Compute quantization error
   - Compensate error in remaining weights using Hessian inverse
4. **Group-wise scaling:** Apply per-group scale factors

### Configuration

Standard GPTQ configuration:
```python
from gptqmodel import QuantizeConfig

quant_config = QuantizeConfig(
    bits=4,           # 4-bit or 8-bit quantization
    group_size=128,   # Group size for scaling
)
```

**Dynamic per-module quantization:** GPTQModel supports different quantization parameters for different layers/modules, allowing fine-grained control over the quantization-accuracy tradeoff.

## Usage in vLLM

### Loading Pre-Quantized Models

vLLM auto-detects GPTQ models from config and supports two execution modes:

**1. Standard GPTQ mode:**
```bash
vllm serve ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2
```

**2. GPTQ with Marlin backend (faster):**
```bash
vllm serve <model_id> --quantization gptq_marlin
```

The Marlin backend provides optimized INT4 kernels for NVIDIA Ampere/Hopper GPUs.

**3. GPTQ with Machete backend:**
Machete is a vLLM-optimized backend for GPTQ models, providing world-class throughput.

### Python API

```python
from vllm import LLM, SamplingParams

# vLLM auto-detects GPTQ quantization from model config
llm = LLM(model="ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2")

sampling_params = SamplingParams(temperature=0.6, top_p=0.9)
outputs = llm.generate(prompts, sampling_params)
```

## Quantizing Models

### Using GPTQModel (Recommended)

GPTQModel is the successor to AutoGPTQ, developed by ModelCloud.AI:

```python
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

# Load calibration dataset
calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
).select(range(1024))["text"]

# Configure quantization
quant_config = QuantizeConfig(bits=4, group_size=128)

# Load model
model = GPTQModel.load(model_id, quant_config)

# Quantize (increase batch_size to speed up)
model.quantize(calibration_dataset, batch_size=2)

# Save quantized model
model.save(quant_path)
```

**Key advantages of GPTQModel:**
- Dynamic per-module quantization (different configs per layer)
- INT4 and INT8 support
- Marlin and Machete backend compatibility
- Active maintenance by ModelCloud.AI team
- Integrated vLLM support

### Using llm-compressor

The vLLM team's [[llm-compressor]] also supports GPTQ quantization:

```bash
# See llm-compressor documentation for GPTQ workflow
# https://github.com/vllm-project/llm-compressor
```

llm-compressor provides unified API across quantization methods (GPTQ, AWQ, FP8).

## Performance Characteristics

**Memory savings:**
- INT4: ~4× weight memory reduction
- INT8: ~2× weight memory reduction
- Total model: 3-3.5× (INT4), 1.8-2× (INT8)

**Throughput:**
- Standard GPTQ: 1.5-2× vs FP16 (INT4)
- GPTQ + Marlin: 2-2.5× vs FP16 (optimized INT4 kernels)
- GPTQ + Machete: 2.5-3× vs FP16 (vLLM-optimized backend)

**Accuracy:**
- INT4: Minimal degradation (<1-2% on most benchmarks)
- INT8: Near-lossless (<0.5% degradation)
- Hessian-based compensation provides better accuracy than naive quantization
- Model-dependent; larger models typically more robust

**Hardware support:**
- CUDA: Ampere (A100+), Hopper (H100+)
- ROCm: AMD MI-series GPUs
- Marlin/Machete: NVIDIA Ampere+ only

## Optimized Execution Backends

### Marlin

Marlin is a high-performance INT4 GEMM kernel developed by IST Austria and integrated into vLLM:

- Fused dequantization + GEMM
- Optimized for Tensor Cores (Ampere/Hopper)
- Support for both AWQ and GPTQ formats
- 1.5-2× speedup over standard GPTQ kernels

### Machete

Machete is a vLLM-optimized backend specifically for GPTQ:

- Developed by vLLM/NeuralMagic (now Red Hat)
- Highly optimized for batched inference
- World-class throughput for GPTQ models
- Support for dynamic per-module quantization

## Comparison with Other Methods

**GPTQ vs [[AWQ]]:**
- GPTQ: Hessian-based, slower quantization, sometimes better accuracy
- AWQ: Activation-aware, simpler calibration, faster quantization
- Both: INT4 weight-only, group-wise quantization, Marlin backend support
- Choice: AWQ for speed, GPTQ for best accuracy

**GPTQ vs [[FP8 Quantization]]:**
- GPTQ: INT4 weights only, ~4× memory reduction
- FP8: FP8 weights + activations, ~2× memory reduction, better accuracy
- FP8 requires Hopper+ GPUs, GPTQ works on Ampere+

**GPTQ vs [[GGUF]]:**
- GPTQ: Optimized for vLLM/GPU serving, production-ready
- GGUF: File format from llama.cpp, experimental in vLLM
- GPTQ has better vLLM integration and performance

## Model Availability

**Hugging Face Hub:**
- 5000+ GPTQ models available (as of 2026-04)
- Search: `https://huggingface.co/models?search=gptq`
- Popular series: TheBloke collections, ModelCloud quantized models

**Naming convention:**
- Typically includes "GPTQ" or "gptq" in model name
- Examples: `TheBloke/Llama-2-7B-Chat-GPTQ`, `ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2`

## Technical Details

### GPTQ Algorithm

For layer weights W ∈ R^(d_out × d_in):

1. **Hessian estimation:**
   - H = 2X^TX / n, where X is calibration activations
   - Add damping: H ← H + λI for numerical stability

2. **Cholesky decomposition:**
   - H = LL^T
   - Compute H^(-1) efficiently via forward/backward substitution

3. **Block-wise quantization:**
   ```
   for block b in range(0, d_in, block_size):
       for column i in block:
           w_q[i] = quantize(w[i])           # Quantize column i
           error = w[i] - dequantize(w_q[i])  # Compute error
           w[i+1:] -= error * H^(-1)[i,i+1:] / H^(-1)[i,i]  # Compensate
   ```

4. **Group-wise scaling:**
   - Divide weights into groups of size g (typically 128)
   - Compute per-group scale: `s_g = max(|W_g|) / (2^(bits-1) - 1)`
   - Store: `{W_q, s_g}` for each group

### Dynamic Per-Module Quantization

GPTQModel supports different quantization configs per module:

```python
# Example: Different bit-widths for attention vs FFN
quant_config = {
    "attention_layers": QuantizeConfig(bits=4, group_size=128),
    "ffn_layers": QuantizeConfig(bits=8, group_size=64),
}
```

This allows fine-grained control over the accuracy-memory tradeoff.

## Integration Points

**Related vLLM components:**
- [[Quantization]] — Broader quantization framework
- [[Kernel Fusions]] — Fusion opportunities with GPTQ dequantization
- [[torch.compile Integration]] — Potential for Inductor-based GPTQ kernels
- [[vLLM Engine]] — Integration with model loading and execution

**Toolchain:**
- [[llm-compressor]] — Unified quantization toolkit (vLLM team)
- AutoGPTQ — Legacy GPTQ library (deprecated)
- lm-eval Harness — Accuracy evaluation

## Limitations

1. **Slow quantization:** Hessian computation and inversion are expensive
2. **Calibration data:** Requires representative dataset (typically 128-1024 samples)
3. **Memory overhead:** Quantization process requires loading full model + Hessian
4. **Group size tradeoff:** Smaller groups = better accuracy but higher memory overhead
5. **INT4 precision:** Not suitable for all models/tasks

## Advanced Features

### Quantization-Aware Training (QAT)

While GPTQ is a PTQ method, GPTQModel supports QAT-style fine-tuning:

```python
# Fine-tune quantized model on task-specific data
model.train()  # Enable training mode
# ... standard PyTorch training loop
```

### Mixed Precision

Combine GPTQ with other techniques:

- **GPTQ + FP8 activations:** Hybrid quantization for better throughput
- **GPTQ + KV cache quantization:** Reduce memory for long contexts
- **Per-layer mixed precision:** Critical layers in higher precision

## Future Directions

- **W4A8 quantization:** Combine INT4 weights with INT8 activations
- **Improved backends:** Better Marlin/Machete integration, ROCm optimization
- **Automated calibration:** Smarter dataset selection and size
- **GGUF interoperability:** Convert between GPTQ and GGUF formats
- **Dynamic quantization:** Runtime adaptation based on input distribution

## References

- **GPTQModel:** https://github.com/ModelCloud/GPTQModel
- **AutoGPTQ (legacy):** https://github.com/PanQiWei/AutoGPTQ
- **Marlin kernels:** https://github.com/IST-DASLab/marlin
- **vLLM GPTQ docs:** `docs/features/quantization/gptqmodel.md`
- **Original GPTQ paper:** Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"

## Cross-References

- [[Quantization]] — Parent concept
- [[AWQ]] — Alternative INT4 method
- [[FP8 Quantization]] — Higher-precision alternative
- [[GGUF]] — Cross-platform quantization format
- [[llm-compressor]] — Unified quantization tool
- [[INT4 W4A16]] — Technical specification (if exists)
- Marlin — Optimized execution backend (if exists)
- [[vLLM]] — Inference engine
