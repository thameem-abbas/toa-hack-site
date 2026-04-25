---
title: vLLM Quantization Documentation
type: source
created: 2026-04-25
source_urls:
  - file:///tmp/vllm/docs/features/quantization/README.md
  - file:///tmp/vllm/docs/features/quantization/fp8.md
  - file:///tmp/vllm/docs/features/quantization/int8.md
  - file:///tmp/vllm/docs/features/quantization/int4.md
  - file:///tmp/vllm/docs/features/quantization/quantized_kvcache.md
---

# vLLM Quantization Documentation

Comprehensive documentation covering vLLM's quantization support: FP8, INT8, INT4 weight/activation quantization and FP8 KV cache quantization.

## Core Claims

1. **Memory-accuracy tradeoff**: Quantization trades model precision for reduced memory footprint, allowing large models to run on more devices
2. **FP8 efficiency**: FP8 W8A8 achieves 2× memory reduction and up to 1.6× throughput improvement with minimal accuracy impact
3. **Hardware requirements**: FP8 W8A8 requires Ada Lovelace (SM89+) or Hopper (SM90+) for NVIDIA; Turing/Ampere supports weight-only FP8 via Marlin kernels
4. **INT8 calibration**: INT8 W8A8 requires calibration data for activation quantization, typically 512 samples from a representative dataset
5. **INT4 use case**: INT4 W4A16 optimized for low QPS workloads, reduces model size with group-wise quantization (group_size=128)
6. **KV cache savings**: FP8 KV cache quantization reduces memory by ~50%, enabling longer context windows and higher throughput
7. **Quantization granularity**: Supports per-tensor, per-channel, per-token (dynamic), and per-attention-head quantization strategies
8. **Plugin architecture**: OOT quantization plugins can be registered via `@register_quantization_config` decorator without modifying vLLM

## Hardware Compatibility Matrix

| Method | Volta (7.0) | Turing (7.5) | Ampere (8.0/8.6) | Ada (8.9) | Hopper (9.0) | AMD GPU | Intel GPU | x86 CPU |
|--------|-------------|--------------|------------------|-----------|--------------|---------|-----------|---------|
| AWQ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| GPTQ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Marlin (GPTQ/AWQ/FP8) | ❌ | ✅* | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| INT8 W8A8 | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| FP8 W8A8 | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ (MI300+) | ❌ | ❌ |
| BitsAndBytes | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| GGUF | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |

*Turing does not support Marlin MXFP4

**Blackwell limitation**: INT8 not supported on SM 10.0+, use FP8 instead

## Quantization Methods

### FP8 W8A8
- **FP8 E4M3**: 1 sign bit, 4 exponent bits, 3 mantissa bits; range: ±448
- **FP8 E5M2**: 1 sign bit, 5 exponent bits, 2 mantissa bits; range: ±57344, supports ±inf
- **Static weight quantization**: Per-channel scales for weights
- **Dynamic activation quantization**: Per-token scales computed during forward pass
- **Online quantization**: `quantization="fp8"` flag enables on-the-fly quantization without calibration
- **Marlin fallback**: Turing/Ampere GPUs use weight-only FP8 (W8A16) via Marlin kernels
- **Tool**: [[llm-compressor]] library for offline quantization with `FP8_DYNAMIC` scheme

### INT8 W8A8
- **Calibration required**: 512+ samples from representative dataset (e.g., `ultrachat_200k`)
- **SmoothQuant**: `smoothing_strength=0.8` parameter for activation smoothing
- **Dynamic per-token**: Activation scales computed per token during inference
- **Blackwell exclusion**: Not supported on RTX 6000 Blackwell (SM 10.0+)
- **Best practices**: Match calibration data to deployment data; use chat/instruction template

### INT4 W4A16
- **Group-wise quantization**: Default `group_size=128` for per-group weight scales
- **GPTQ algorithm**: Approximate second-order optimization for weight quantization
- **Hyperparameters**:
  - `dampening_frac=0.01`: Controls GPTQ influence (lower = higher accuracy, higher instability risk)
  - `actorder="weight"`: Activation ordering for improved accuracy
- **Marlin kernel**: Fast INT4 inference on Ampere/Ada/Hopper/Blackwell
- **Comparison**: Similar to AWQ/GPTQ methods (all use INT4 weights with FP16 activations)

### Quantized KV Cache
- **Memory reduction**: ~50% reduction in KV cache memory footprint
- **FP8 formats**: `fp8_e4m3` (CUDA 11.8+, ROCm), `fp8_e5m2` (CUDA 11.8+)
- **Quantization strategies**:
  - Per-tensor: Single scale per Q/K/V tensor (`q/k/v_scale = [1]`)
  - Per-attention-head: Scale per head (`q_scale = [num_heads]`, `k/v_scale = [num_kv_heads]`); Flash Attention only
- **Calibration modes**:
  - No calibration: `calculate_kv_scales=False` (scales = 1.0)
  - Random token: `calculate_kv_scales=True` (warmup estimation)
  - Dataset calibration: `llm-compressor` with `kv_cache_scheme` (recommended)
- **Flash Attention 3 integration**: Queries also quantized to FP8 for full FP8 attention

## Configuration Examples

### FP8 online quantization
```python
llm = LLM("facebook/opt-125m", quantization="fp8")
```

### FP8 offline with llm-compressor
```python
from llmcompressor.modifiers.quantization import QuantizationModifier
recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
oneshot(model=model, recipe=recipe)
```

### INT8 with SmoothQuant
```python
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])
]
```

### INT4 with custom GPTQ parameters
```python
recipe = GPTQModifier(
    targets="Linear",
    config_groups={
        "config_group": QuantizationScheme(
            weights=QuantizationArgs(
                num_bits=4, type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP, group_size=128,
                symmetric=True, actorder="weight"
            )
        )
    },
    dampening_frac=0.01
)
```

### FP8 KV cache with dataset calibration
```python
from llmcompressor.modifiers.quantization import QuantizationModifier
fp8_args = QuantizationArgs(num_bits=8, type="float", strategy="tensor")
recipe = QuantizationModifier(
    config_groups={"attention": QuantizationScheme(targets=["LlamaAttention"], input_activations=fp8_args)},
    kv_cache_scheme=fp8_args
)
```

## Plugin Architecture

### Custom quantization config registration
```python
from vllm.model_executor.layers.quantization import register_quantization_config

@register_quantization_config("my_quant")
class MyQuantConfig(QuantizationConfig):
    def get_name(self) -> str: ...
    def get_supported_act_dtypes(self) -> list: ...
    def get_min_capability(cls) -> int: ...
    def get_config_filenames() -> list[str]: ...
    def from_config(cls, config: dict): ...
    def get_quant_method(self, layer, prefix): ...
```

### Quantized linear method
```python
class MyQuantLinearMethod(UnquantizedLinearMethod):
    def create_weights(self, layer, *weight_args, **extra_weight_attrs): ...
    def apply(self, layer, x, bias=None): ...
```

### Quantized MoE method
```python
class MyQuantMoEMethod(FusedMoEMethodBase):
    def create_weights(self, layer, num_experts, hidden_size, ...): ...
    def apply(self, layer, router, x, router_logits): ...
    def get_fused_moe_quant_config(self, layer): ...
```

## Metrics

- **FP8 memory**: 2× reduction vs FP16/BF16
- **FP8 throughput**: Up to 1.6× improvement
- **KV cache memory**: ~50% reduction with FP8 quantization
- **Calibration samples**: 512 recommended starting point for INT8/INT4
- **Sequence length**: 2048 tokens recommended for calibration
- **Accuracy sensitivity**: Models sensitive to `bos` token presence in evaluation

## Cross-References

- [[Quantization]] — Main concept page
- [[FP8 Quantization]] — FP8 technique details
- [[INT8 W8A8]] — INT8 technique details
- [[INT4 W4A16]] — INT4 technique details
- [[Quantized KV Cache]] — KV cache quantization
- [[llm-compressor]] — Quantization toolkit
- [[AWQ]] — Activation-aware weight quantization
- [[GPTQ]] — Post-training quantization method
- [[GGUF]] — GGML Universal File format
- [[Kernel Fusions]] — Kernel optimization techniques
- [[KV Cache]] — Key-value cache concept
- [[Mixture of Experts]] — MoE quantization support

## Key Insights

1. vLLM supports 13 quantization methods (AWQ, BitsAndBytes, GGUF, GPTQ, INC, INT4, INT8, FP8, ModelOpt, Online, Quark, KV Cache, TorchAO) with platform-specific hardware requirements
2. FP8 offers best efficiency on modern GPUs (Ada/Hopper/MI300) with 2× memory reduction and minimal accuracy loss, while older GPUs fall back to weight-only via Marlin
3. INT8/INT4 require calibration data (512+ samples) for activation/weight quantization, with tunable hyperparameters (smoothing_strength, dampening_frac, actorder)
4. KV cache quantization to FP8 provides orthogonal memory savings (~50%) and can be combined with weight/activation quantization
5. Plugin architecture enables OOT quantization methods via decorator pattern, supporting custom linear and MoE quantization logic
