---
title: vLLM Quantization Formats (AWQ, GPTQ, GGUF, Quark, ModelOpt)
type: source
category: quantization
created: 2026-04-25
sources:
  - file:///tmp/vllm/docs/features/quantization/auto_awq.md
  - file:///tmp/vllm/docs/features/quantization/gptqmodel.md
  - file:///tmp/vllm/docs/features/quantization/gguf.md
  - file:///tmp/vllm/docs/features/quantization/quark.md
  - file:///tmp/vllm/docs/features/quantization/modelopt.md
---

# vLLM Quantization Formats

Source summary for vLLM's support of multiple quantization formats: AWQ, GPTQ, GGUF, AMD Quark, and NVIDIA ModelOpt.

## Sources

1. `docs/features/quantization/auto_awq.md` — AWQ (Activation-Aware Weight Quantization)
2. `docs/features/quantization/gptqmodel.md` — GPTQModel (successor to AutoGPTQ)
3. `docs/features/quantization/gguf.md` — GGUF (llama.cpp format)
4. `docs/features/quantization/quark.md` — AMD Quark (MXFP4/MXFP6/FP8)
5. `docs/features/quantization/modelopt.md` — NVIDIA Model Optimizer

## Key Claims

### AWQ (Activation-Aware Weight Quantization)

**Deprecation notice:**
- AutoAWQ library is deprecated
- Functionality migrated to [[llm-compressor]] (vLLM project)
- Recommended workflow: llm-compressor AWQ examples

**Core technique:**
- INT4 weight quantization (BF16/FP16 → INT4)
- Activation-aware salient weight protection
- Group-wise quantization (typical group size: 128)
- Zero-point support for asymmetric quantization

**vLLM usage:**
- `--quantization awq` — Standard AWQ mode
- `--quantization awq_marlin` — AWQ with Marlin backend (faster)
- Auto-detection from model config

**Model availability:**
- 6500+ AWQ models on Hugging Face
- Examples: `TheBloke/Llama-2-7b-Chat-AWQ`

**Quantization config:**
```python
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}
```

### GPTQ (Post-Training Quantization)

**Library:**
- GPTQModel by ModelCloud.AI (successor to AutoGPTQ)
- INT4/INT8 weight quantization
- Hessian-based error compensation

**Advanced features:**
- Dynamic per-module quantization (different configs per layer)
- Marlin and Machete backend support
- Optimized for Ampere (A100+) and Hopper (H100+) GPUs

**vLLM usage:**
- Auto-detection from model config
- `--quantization gptq_marlin` — GPTQ with Marlin backend
- Machete backend for world-class throughput

**Model availability:**
- 5000+ GPTQ models on Hugging Face
- Example: `ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2`

**Quantization process:**
```python
from gptqmodel import GPTQModel, QuantizeConfig

quant_config = QuantizeConfig(bits=4, group_size=128)
model = GPTQModel.load(model_id, quant_config)
model.quantize(calibration_dataset, batch_size=2)
model.save(quant_path)
```

**Performance:**
- Marlin/Machete kernels: world-class inference performance
- Dynamic quantization: fine-grained accuracy-memory tradeoff

### GGUF (llama.cpp Format)

**Warning:**
- Experimental and under-optimized in vLLM
- May be incompatible with other vLLM features
- Use for memory reduction, not peak performance

**Constraints:**
- Single-file GGUF models only
- Multi-file models require merging with `gguf-split` tool
- Recommend using base model tokenizer (GGUF tokenizer conversion is slow/buggy)

**Supported quantization types:**
- Q4_0, Q4_1, Q4_K_S, Q4_K_M, Q8_0, etc.
- K-quants: mixed precision within blocks

**vLLM usage:**
```bash
# HuggingFace with quant type
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M --tokenizer Qwen/Qwen3-0.6B

# Tensor parallelism supported
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
   --tokenizer Qwen/Qwen3-0.6B \
   --tensor-parallel-size 2

# Local file
vllm serve ./Qwen3-0.6B-Q4_K_M.gguf --tokenizer Qwen/Qwen3-0.6B
```

**Manual config path:**
```bash
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
   --tokenizer Qwen/Qwen3-0.6B \
   --hf-config-path Qwen/Qwen3-0.6B
```

### AMD Quark

**Purpose:**
- Quantization toolkit for AMD GPUs (MI-series)
- Supports AWQ, GPTQ, Rotation, SmoothQuant algorithms
- Weight, activation, and KV-cache quantization

**Installation:**
```bash
pip install amd-quark
```

**Quantization process (5 steps):**
1. Load model (Transformers)
2. Prepare calibration dataloader (PyTorch DataLoader)
3. Set quantization configuration
4. Quantize and export (HuggingFace safetensors)
5. Evaluate in vLLM

**FP8 example:**
```python
# FP8 per-tensor quantization on weight, activation, kv-cache
# Algorithm: AutoSmoothQuant

from quark.torch.quantization import (
    Config, QuantizationConfig,
    FP8E4M3PerTensorSpec,
    load_quant_algo_config_from_file
)

FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(
    observer_method="min_max",
    is_dynamic=False,
).to_quantization_spec()

global_quant_config = QuantizationConfig(
    input_tensors=FP8_PER_TENSOR_SPEC,
    weight=FP8_PER_TENSOR_SPEC,
)

kv_cache_quant_config = {
    name: QuantizationConfig(
        input_tensors=global_quant_config.input_tensors,
        weight=global_quant_config.weight,
        output_tensors=FP8_PER_TENSOR_SPEC,
    )
    for name in kv_cache_layer_names
}

algo_config = load_quant_algo_config_from_file(
    "examples/torch/language_modeling/llm_ptq/models/llama/autosmoothquant_config.json"
)

quant_config = Config(
    global_quant_config=global_quant_config,
    layer_quant_config=layer_quant_config,
    kv_cache_quant_config=kv_cache_quant_config,
    exclude=["lm_head"],
    algo_config=algo_config,
)
```

**vLLM usage:**
```python
llm = LLM(
    model="Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant",
    kv_cache_dtype="fp8",
    quantization="quark",
)
```

**Quantization script:**
```bash
python3 quantize_quark.py --model_dir meta-llama/Llama-2-70b-chat-hf \
                          --output_dir /path/to/output \
                          --quant_scheme w_fp8_a_fp8 \
                          --kv_cache_dtype fp8 \
                          --quant_algo autosmoothquant \
                          --num_calib_data 512 \
                          --model_export hf_format \
                          --tasks gsm8k
```

**OCP MX (MXFP4, MXFP6) support:**
- Compliant with Open Compute Project specification
- Dynamic quantization for activations
- 2.5-4× memory savings vs FP16/BF16
- Simulated execution on devices without native OCP MX support (MI325, MI300, MI250)

**Quantization example:**
```bash
vllm serve fxmarty/qwen_1.5-moe-a2.7b-mxfp4 --tensor-parallel-size 1

# Or FP6 activations + FP4 weights:
vllm serve fxmarty/qwen1.5_moe_a2.7b_chat_w_fp4_a_fp6_e2m3 --tensor-parallel-size 1
```

**Mixed precision (layerwise AMP):**
- Mixed scheme of {MXFP4, FP8} currently supported
- Future: {MXFP4, MXFP6, FP8, BF16/FP16} combinations
- Balance between maximizing accuracy and throughput

**Example models:**
- `amd/Llama-2-70b-chat-hf-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8`
- `amd/Mixtral-8x7B-Instruct-v0.1-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8`
- `amd/Qwen3-8B-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8`

### NVIDIA Model Optimizer

**Purpose:**
- Post-Training Quantization (PTQ) and Quantization Aware Training (QAT)
- LLMs, Vision Language Models (VLMs), diffusion models
- NVIDIA GPU optimization

**Installation:**
```bash
pip install nvidia-modelopt
```

**Supported formats:**
- `FP8`: per-tensor weight scale (+ optional static activation scale)
- `FP8_PER_CHANNEL_PER_TOKEN`: per-channel weight, dynamic per-token activation
- `FP8_PB_WO` (or `fp8_pb_wo`): block-scaled FP8 weight-only (128×128 blocks)
- `NVFP4`: ModelOpt NVFP4 (use `quantization="modelopt_fp4"`)
- `MXFP8`: ModelOpt MXFP8 (use `quantization="modelopt_mxfp8"`)

**Detection:**
- vLLM detects ModelOpt checkpoints via `hf_quant_config.json`
- `quantization.quant_algo` field specifies format

**Quantization workflow:**
```python
import modelopt.torch.quantization as mtq
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("<path_or_model_id>")

# Select quantization config
config = mtq.FP8_DEFAULT_CFG

# Define calibration forward loop
def forward_loop(model):
    for data in calib_set:
        model(data)

# PTQ with in-place replacement
model = mtq.quantize(model, config, forward_loop)

# Export to HuggingFace checkpoint
import torch
from modelopt.torch.export import export_hf_checkpoint

with torch.inference_mode():
    export_hf_checkpoint(
        model,
        export_dir,
    )
```

**vLLM usage:**
```python
llm = LLM(
    model="nvidia/Llama-3.1-8B-Instruct-FP8",
    quantization="modelopt",
    trust_remote_code=True
)
```

**OpenAI-compatible server:**
```bash
vllm serve <path_to_exported_checkpoint> \
  --quantization modelopt \
  --host 0.0.0.0 --port 8000
```

**Testing:**
```bash
export VLLM_TEST_MODELOPT_FP8_PC_PT_MODEL_PATH=<path_to_fp8_pc_pt_checkpoint>
export VLLM_TEST_MODELOPT_FP8_PB_WO_MODEL_PATH=<path_to_fp8_pb_wo_checkpoint>
pytest -q tests/quantization/test_modelopt.py
```

## Technical Insights

### Quantization Method Comparison

**AWQ vs GPTQ:**
- AWQ: Activation-aware, simpler calibration, faster quantization
- GPTQ: Hessian-based, slower quantization, sometimes better accuracy
- Both: INT4 weight-only, group-wise quantization, Marlin backend support

**GGUF vs AWQ/GPTQ:**
- GGUF: File format, cross-platform, experimental in vLLM
- AWQ/GPTQ: vLLM-native, optimized kernels, production-ready
- GGUF use case: cross-platform workflows (llama.cpp ↔ vLLM)

**AMD Quark vs NVIDIA ModelOpt:**
- Quark: AMD GPUs (MI-series), MXFP4/MXFP6/FP8
- ModelOpt: NVIDIA GPUs, FP8/NVFP4, VLM support
- Platform-specific optimization

### Execution Backends

**Marlin:**
- Optimized INT4 GEMM kernels
- Supports both AWQ and GPTQ
- NVIDIA Ampere/Hopper GPUs
- 1.5-2× speedup over standard kernels

**Machete:**
- vLLM-optimized GPTQ backend
- World-class batched inference throughput
- Developed by vLLM/NeuralMagic (Red Hat)

**Quark fused kernels:**
- Dequantization from FP4/FP6 to half precision on-the-fly
- Useful for evaluating or serving MXFP4/MXFP6 models on devices without native support

### Calibration Approaches

**GPTQ:**
- Hessian estimation: H = 2X^TX
- Block-wise quantization with error compensation
- Requires 128-1024 calibration samples

**AWQ:**
- Per-channel activation magnitude statistics
- Salient weight protection via scaling
- Faster calibration than GPTQ

**Quark:**
- Algorithm-specific configs (AutoSmoothQuant, etc.)
- JSON config files for quantization algorithms
- Supports multiple quantization schemes (w_fp8_a_fp8, w_mxfp4_a_mxfp4, etc.)

**ModelOpt:**
- Calibration via forward loop
- PTQ and QAT support
- VLM-specific calibration

## Integration Points

**vLLM quantization flags:**
- `--quantization awq` — AWQ mode
- `--quantization awq_marlin` — AWQ with Marlin
- `--quantization gptq_marlin` — GPTQ with Marlin
- `--quantization quark` — AMD Quark models
- `--quantization modelopt` — NVIDIA ModelOpt models
- `--quantization modelopt_fp4` — NVIDIA NVFP4
- `--quantization modelopt_mxfp8` — NVIDIA MXFP8
- Auto-detection from `hf_quant_config.json` or model config

**KV cache quantization:**
- Quark: `kv_cache_dtype="fp8"`
- vLLM: `--kv-cache-dtype fp8` or `--kv-cache-dtype int8`

**Tensor parallelism:**
- GGUF: Supported (`--tensor-parallel-size N`)
- AWQ/GPTQ: Supported
- Quark: Supported

## Recommendations

**For vLLM production:**
1. **INT4:** Use [[llm-compressor]] (AWQ or GPTQ) → Marlin backend
2. **FP8:** Use [[llm-compressor]] (FP8) for Hopper GPUs
3. **AMD GPUs:** Use AMD Quark (MXFP4/FP8)
4. **Cross-platform:** Use GGUF (experimental)
5. **VLMs:** Use NVIDIA ModelOpt

**Quantization toolkit selection:**
- vLLM-native: [[llm-compressor]] (recommended)
- AMD GPUs: AMD Quark
- NVIDIA official: NVIDIA Model Optimizer
- Deprecated: AutoAWQ, AutoGPTQ

## Cross-References

**Techniques:**
- [[AWQ]] — Activation-aware weight quantization
- [[GPTQ]] — Hessian-based post-training quantization
- [[GGUF]] — llama.cpp file format
- [[FP8 Quantization]] — 8-bit floating-point quantization
- [[Quantization]] — Parent concept

**Tools:**
- [[llm-compressor]] — vLLM unified quantization toolkit
- AutoGPTQ — Deprecated GPTQ library
- AutoAWQ — Deprecated AWQ library
- [[vLLM]] — Inference engine

**Concepts:**
- [[Kernel Fusions]] — Quantization + kernel fusion opportunities
- [[Tensor Parallelism]] — Multi-GPU support for quantized models
- [[KV Cache]] — KV cache quantization for long contexts
