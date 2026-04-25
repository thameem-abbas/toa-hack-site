---
title: AWQ (Activation-Aware Weight Quantization)
type: technique
category: quantization
created: 2026-04-25
---

# AWQ (Activation-Aware Weight Quantization)

**Activation-Aware Weight Quantization (AWQ)** is a weight-only INT4 quantization technique that protects salient weights based on activation magnitudes, achieving better accuracy than naive quantization at 4-bit precision.

## Overview

AWQ quantizes model weights from FP16/BF16 to INT4 (4-bit integers), effectively reducing model memory footprint by ~4× while maintaining model quality. Unlike methods that treat all weights equally, AWQ identifies and protects "salient" weights—those with high activation magnitudes—from aggressive quantization.

**Key characteristics:**
- **Weight-only quantization:** Weights stored as INT4, activations remain FP16
- **Group-wise scaling:** Typical group size 128 (configurable)
- **Activation-aware:** Protects salient weights based on per-channel activation statistics
- **Zero-point support:** Optional zero-point for asymmetric quantization
- **GEMM version:** Uses INT4 GEMM kernels for efficient execution

## Architecture

### Quantization Config

Standard AWQ configuration:
```python
quant_config = {
    "zero_point": True,      # Enable asymmetric quantization
    "q_group_size": 128,     # Group size for quantization
    "w_bit": 4,              # Weight bit-width
    "version": "GEMM"        # Use GEMM-based kernels
}
```

### Salient Weight Protection

AWQ computes per-channel activation magnitudes during calibration and uses these to determine which weights to protect:

1. **Calibration:** Run model on calibration dataset to collect activation statistics
2. **Saliency calculation:** Compute per-channel activation magnitudes
3. **Scaling:** Apply per-channel scaling to protect salient weights before quantization
4. **Quantization:** Quantize scaled weights to INT4 with group-wise quantization

## Usage in vLLM

### Loading Pre-Quantized Models

vLLM supports two AWQ execution modes:

**1. Standard AWQ mode:**
```bash
vllm serve TheBloke/Llama-2-7b-Chat-AWQ --quantization awq
```

**2. AWQ with Marlin backend (faster):**
```bash
vllm serve TheBloke/Llama-2-7b-Chat-AWQ --quantization awq_marlin
```

The Marlin backend provides optimized INT4 kernels for Ampere/Hopper GPUs, delivering higher throughput than standard AWQ kernels.

### Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="TheBloke/Llama-2-7b-Chat-AWQ",
    quantization="awq"  # or "awq_marlin" for Marlin backend
)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)
```

## Quantizing Models

### Using AutoAWQ (Deprecated)

> **Warning:** AutoAWQ library is deprecated. Functionality has been migrated to [[llm-compressor]].

Historical AutoAWQ workflow:
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    low_cpu_mem_usage=True,
    use_cache=False
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Quantize with calibration data
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized("mistral-instruct-v0.2-awq")
tokenizer.save_pretrained("mistral-instruct-v0.2-awq")
```

### Using llm-compressor (Recommended)

The vLLM team now maintains AWQ quantization in [[llm-compressor]]:

```bash
# See llm-compressor documentation for current workflow
# https://github.com/vllm-project/llm-compressor/tree/main/examples/awq
```

llm-compressor provides:
- Unified API across quantization methods (AWQ, GPTQ, FP8)
- Direct vLLM compatibility
- Active maintenance by vLLM team
- One-shot and calibration-based flows

## Performance Characteristics

**Memory savings:**
- Weights: 4× reduction (FP16 → INT4)
- Total model: ~3-3.5× (weights dominate large models)

**Throughput:**
- Standard AWQ: 1.5-2× vs FP16 (memory bandwidth bound)
- AWQ + Marlin: 2-2.5× vs FP16 (optimized INT4 kernels)

**Accuracy:**
- Minimal degradation on most benchmarks (<1-2% vs FP16)
- Better than naive INT4 quantization due to salient weight protection
- Model-dependent; some models more robust than others

**Hardware support:**
- CUDA: Ampere (A100+), Hopper (H100+)
- ROCm: AMD MI-series GPUs
- Marlin backend: NVIDIA Ampere+ only

## Comparison with Other Methods

**AWQ vs [[GPTQ]]:**
- AWQ: Activation-aware, simpler calibration, faster quantization
- GPTQ: Hessian-based (second-order), slower quantization, sometimes better accuracy
- Both: INT4 weight-only, group-wise quantization, Marlin backend support

**AWQ vs [[FP8 Quantization]]:**
- AWQ: INT4 weights only, FP16 activations, ~4× memory reduction
- FP8: FP8 weights + activations, ~2× memory reduction, better accuracy
- FP8 requires hardware support (Hopper+), AWQ works on older GPUs

**AWQ vs [[GGUF]]:**
- AWQ: Optimized for vLLM/GPU serving, single quantization type (INT4)
- GGUF: File format from llama.cpp, multiple quantization types, cross-platform
- GGUF support in vLLM is experimental; AWQ is production-ready

## Model Availability

**Hugging Face Hub:**
- 6500+ AWQ models available (as of 2026-04)
- Search: `https://huggingface.co/models?search=awq`
- Popular series: TheBloke collections, Meta Llama AWQ, Mistral AWQ

**Naming convention:**
- Typically includes "AWQ" in model name
- Examples: `TheBloke/Llama-2-7b-Chat-AWQ`, `casperhansen/mistral-7b-instruct-v0.1-awq`

## Technical Details

### Quantization Formula

For weight matrix W, activation magnitude S, and group size g:

1. Compute per-channel activation scale: `s_c = max(|A_c|)` where A is activation
2. Apply scaling: `W'_c = W_c * s_c^α` (α typically 0.5-1.0)
3. Quantize to INT4: `W_q = round(W' / scale) + zero_point`
4. Store: `{W_q, scale, zero_point, s_c}` per group

### Kernel Execution

Standard AWQ kernel flow:
1. Load INT4 weights from memory
2. Dequantize to FP16: `W_fp16 = (W_q - zero_point) * scale / s_c^α`
3. Compute FP16 GEMM: `Y = W_fp16 @ X`

Marlin kernel flow (optimized):
1. Fused dequantization + GEMM
2. Tiled execution for better memory locality
3. Optimized for Tensor Cores (Ampere/Hopper)

## Integration Points

**Related vLLM components:**
- [[Quantization]] — Broader quantization framework
- [[Kernel Fusions]] — Fusion opportunities with AWQ dequantization
- [[torch.compile Integration]] — Potential for Inductor-based AWQ kernels
- [[vLLM Engine]] — Integration with model loading and execution

**Toolchain:**
- [[llm-compressor]] — Recommended quantization toolkit
- AutoGPTQ — Alternative INT4 method (GPTQ)
- lm-eval Harness — Accuracy evaluation

## Limitations

1. **INT4 precision:** Not suitable for all models (some accuracy-sensitive tasks)
2. **Weight-only:** Activations remain FP16 (still memory bandwidth bound for decode)
3. **Calibration required:** Needs representative dataset for activation statistics
4. **Group size tradeoff:** Smaller groups = better accuracy but higher memory overhead
5. **Model coverage:** Some architectures not supported (e.g., encoder-only models)

## Future Directions

- **W4A8:** Combine AWQ weights with INT8 activations
- **Mixed precision:** Per-layer bit-width selection
- **Improved kernels:** Better Marlin integration, ROCm optimization
- **Automated calibration:** Smarter calibration dataset selection
- **GGUF bridge:** Better interoperability with llama.cpp ecosystem

## References

- **AutoAWQ (deprecated):** https://github.com/casper-hansen/AutoAWQ
- **llm-compressor AWQ examples:** https://github.com/vllm-project/llm-compressor/tree/main/examples/awq
- **Marlin kernels:** vLLM's optimized INT4 GEMM backend
- **vLLM AWQ docs:** `docs/features/quantization/auto_awq.md`

## Cross-References

- [[Quantization]] — Parent concept
- [[GPTQ]] — Alternative INT4 method
- [[FP8 Quantization]] — Higher-precision alternative
- [[GGUF]] — Cross-platform quantization format
- [[llm-compressor]] — Recommended quantization tool
- [[INT4 W4A16]] — Technical specification (if exists in vault)
- Marlin — Optimized execution backend (if exists in vault)
- [[vLLM]] — Inference engine
