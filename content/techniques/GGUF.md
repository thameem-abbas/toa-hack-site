---
title: GGUF (GPT-Generated Unified Format)
type: technique
category: quantization
created: 2026-04-25
---

# GGUF (GPT-Generated Unified Format)

**GGUF** is a file format from the llama.cpp ecosystem that supports multiple quantization types (Q4_0, Q4_K_M, Q8_0, etc.) for cross-platform LLM deployment. vLLM provides experimental support for loading and serving GGUF models.

## Overview

GGUF is the successor to GGML, designed as a unified format for storing quantized language models. Originally developed for llama.cpp (CPU inference), GGUF models can now be accelerated on GPUs via vLLM, though support is currently experimental and under-optimized.

**Key characteristics:**
- **File format:** Single-file or multi-file format with embedded metadata
- **Multiple quantization types:** Q4_0, Q4_1, Q4_K_S, Q4_K_M, Q8_0, etc.
- **Cross-platform:** Used by llama.cpp, Ollama, LM Studio, and now vLLM
- **Experimental in vLLM:** Under-optimized, may be incompatible with some vLLM features
- **Memory-focused:** Primary benefit is reduced memory footprint

## Supported Quantization Types

GGUF supports many quantization formats. vLLM supports a subset:

**Common types:**
- **Q4_0:** Simple 4-bit quantization, no zero-point
- **Q4_1:** 4-bit with zero-point
- **Q4_K_S:** 4-bit with k-quants (small, optimized for size)
- **Q4_K_M:** 4-bit with k-quants (medium, balanced)
- **Q8_0:** 8-bit quantization

**K-quants explanation:**
K-quants use mixed precision within blocks for better quality-size tradeoff:
- **Q4_K_S:** Smallest, most aggressive quantization
- **Q4_K_M:** Medium quality, commonly used
- **Q4_K_L:** Larger, better quality (if supported)

vLLM's GGUF support is evolving; check documentation for current type coverage.

## Usage in vLLM

> **Warning:** GGUF support in vLLM is highly experimental and under-optimized. Use for memory reduction, not peak performance.

### Loading from Hugging Face

Use `repo_id:quant_type` format:

```bash
# Load Q4_K_M quantized model from Hugging Face
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M --tokenizer Qwen/Qwen3-0.6B
```

**Tokenizer recommendation:** Use the base model's tokenizer instead of GGUF's embedded tokenizer:
- GGUF tokenizer conversion is slow and unstable
- Especially problematic for models with large vocab size
- Always specify `--tokenizer <base_model_id>`

### Tensor Parallelism

GGUF models support tensor parallelism in vLLM:

```bash
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
   --tokenizer Qwen/Qwen3-0.6B \
   --tensor-parallel-size 2
```

This allows distributing GGUF models across multiple GPUs.

### Local Files

Download and serve local GGUF files:

```bash
wget https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf
vllm serve ./Qwen3-0.6B-Q4_K_M.gguf --tokenizer Qwen/Qwen3-0.6B
```

### Manual Config Path

If Hugging Face doesn't support your model's config:

```bash
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
   --tokenizer Qwen/Qwen3-0.6B \
   --hf-config-path Qwen/Qwen3-0.6B
```

This provides a HuggingFace-compatible config for metadata conversion.

### Python API

```python
from vllm import LLM, SamplingParams

# Create LLM with GGUF model
llm = LLM(
    model="unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
    tokenizer="Qwen/Qwen3-0.6B",
)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Use chat() or generate()
outputs = llm.chat(conversation, sampling_params)
```

## File Format Constraints

### Single-File Requirement

> **Warning:** vLLM only supports single-file GGUF models.

If you have a multi-file GGUF model:

```bash
# Use gguf-split tool to merge files
# https://github.com/ggerganov/llama.cpp/pull/6135
gguf-split --merge input_part1.gguf input_part2.gguf --output merged.gguf
```

### Metadata Handling

GGUF embeds model metadata (architecture, vocab, hyperparameters) in the file header. vLLM relies on HuggingFace to convert this metadata to a model config.

**If conversion fails:**
- Manually create a HuggingFace config.json
- Use `--hf-config-path` to specify it
- Or use a similar model's config as reference

## Performance Characteristics

**Memory savings:**
- Q4_0/Q4_1: ~4× weight memory reduction
- Q4_K_M: ~4× with better quality than Q4_0
- Q8_0: ~2× weight memory reduction

**Throughput (experimental):**
- Significantly slower than native vLLM quantization ([[AWQ]], [[GPTQ]], [[FP8 Quantization]])
- Under-optimized kernels; GGUF was designed for llama.cpp
- Use for memory reduction when throughput is not critical

**Accuracy:**
- Q4_K_M: Good balance of size and quality
- Q8_0: Near-lossless
- Q4_0: More aggressive, higher degradation

**Hardware support:**
- CUDA: NVIDIA GPUs (experimental)
- ROCm: AMD GPUs (if supported)
- Tensor parallel: Multi-GPU support

## Comparison with Other Methods

**GGUF vs [[AWQ]]:**
- GGUF: File format, multiple quant types, cross-platform, experimental in vLLM
- AWQ: vLLM-native, optimized INT4, production-ready, Marlin backend
- Choice: AWQ for vLLM production, GGUF for cross-platform compatibility

**GGUF vs [[GPTQ]]:**
- GGUF: File format, llama.cpp ecosystem, experimental in vLLM
- GPTQ: vLLM-native, Hessian-based INT4, Marlin/Machete backends
- Choice: GPTQ for vLLM production, GGUF for portability

**GGUF vs [[FP8 Quantization]]:**
- GGUF: INT4/INT8, broader hardware support, lower memory
- FP8: Higher precision, better accuracy, Hopper+ only
- FP8 is production-ready in vLLM; GGUF is experimental

## Use Cases

**When to use GGUF in vLLM:**
1. **Cross-platform workflows:** Models used in both llama.cpp and vLLM
2. **Memory constraints:** Need to reduce footprint, throughput not critical
3. **Model availability:** Model only available in GGUF format
4. **Experimentation:** Testing GGUF model quality before re-quantizing with AWQ/GPTQ

**When NOT to use GGUF:**
1. **Production serving:** Use [[AWQ]] or [[GPTQ]] with Marlin backend
2. **High throughput:** GGUF is under-optimized in vLLM
3. **Feature compatibility:** GGUF may conflict with vLLM features (prefix caching, etc.)

## Limitations

1. **Experimental status:** Under-optimized, may have bugs or incompatibilities
2. **Single-file only:** Multi-file GGUF models require merging
3. **Tokenizer issues:** Embedded tokenizer conversion is slow/unstable
4. **Performance:** Slower than native vLLM quantization methods
5. **Feature compatibility:** May not work with all vLLM features (e.g., [[Prefix Caching]], [[Kernel Fusions]])
6. **Model coverage:** Not all GGUF quantization types may be supported

## Model Availability

**Hugging Face Hub:**
- Many GGUF models from llama.cpp community
- Common repos: unsloth, TheBloke, bartowski, etc.
- Search: `https://huggingface.co/models?search=gguf`

**Naming convention:**
- Repo name includes "GGUF" (e.g., `unsloth/Qwen3-0.6B-GGUF`)
- File name includes quant type (e.g., `Qwen3-0.6B-Q4_K_M.gguf`)

**Quantization type selection:**
- Q4_K_M: Recommended for balance of size and quality
- Q8_0: Near-lossless, 2× compression
- Q4_0: Smallest, lower quality

## Integration Points

**Related vLLM components:**
- [[Quantization]] — Broader quantization framework
- [[Tensor Parallelism]] — Multi-GPU support for GGUF
- [[vLLM Engine]] — Model loading and serving

**Alternative quantization methods:**
- [[AWQ]] — vLLM-native INT4 (recommended)
- [[GPTQ]] — vLLM-native INT4 with Hessian
- [[FP8 Quantization]] — Higher precision, better accuracy

**Toolchain:**
- **llama.cpp:** Original GGUF ecosystem
- **gguf-split:** Tool for merging multi-file models
- **Ollama, LM Studio:** Other GGUF runtimes
- [[lm-eval Harness]] — Accuracy evaluation

## Future Directions

- **Performance optimization:** Better kernels for GGUF in vLLM
- **Quantization type coverage:** Support more GGUF quant types
- **Multi-file support:** Native multi-file GGUF loading
- **Conversion tools:** Easy GGUF ↔ AWQ/GPTQ conversion
- **Feature parity:** Better integration with vLLM features (prefix caching, fusions, etc.)
- **Production readiness:** Move from experimental to stable

## References

- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **GGUF specification:** https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **gguf-split tool:** https://github.com/ggerganov/llama.cpp/pull/6135
- **vLLM GGUF docs:** `docs/features/quantization/gguf.md`

## Cross-References

- [[Quantization]] — Parent concept
- [[AWQ]] — vLLM-native INT4 alternative
- [[GPTQ]] — vLLM-native INT4 alternative
- [[FP8 Quantization]] — Higher-precision alternative
- [[Tensor Parallelism]] — Multi-GPU support
- [[vLLM]] — Inference engine
- [[llm-compressor]] — vLLM quantization toolkit (does not produce GGUF)
