---
title: CustomOp System
type: architecture
created: 2026-04-25
tags: [plugin, dispatch, backend, extensibility]
---

# CustomOp System

Abstract class for dispatching vLLM operations to platform-specific backends and enabling Out-Of-Tree (OOT) hardware plugins to register optimized kernel implementations. The CustomOp system is vLLM's primary extensibility mechanism for hardware vendors.

## Overview

`CustomOp` manages two registries:
1. **`op_registry`** — vLLM's built-in operations
2. **`op_registry_oot`** — OOT plugin operations (hardware vendor extensions)

When an operation is invoked, CustomOp:
- Checks if the operation is **enabled** via `--compilation_config.custom_ops`
- If enabled → dispatches to platform-specific backend (`forward_cuda()`, `forward_hip()`, `forward_xpu()`, etc.)
- If disabled → falls back to PyTorch-native implementation (`forward_native()`)

This dual-path design allows:
- **Hardware vendors** to provide optimized kernels without modifying vLLM core
- **Inductor fusion** to take over when custom ops are disabled (often faster on NVIDIA)
- **Multi-platform support** with a single codebase

## Registration and Dispatch

### Registering a CustomOp

Built-in vLLM operations:

```python
from vllm.model_executor.custom_op import CustomOp

@CustomOp.register("mm_encoder_attn")
class MMEncoderAttention(CustomOp):
    def __init__(self, num_heads: int, head_size: int, ...):
        super().__init__()
        # Initialize operation

    def forward_native(self, query, key, value, ...):
        # PyTorch-native implementation (fallback)
        return torch.nn.functional.scaled_dot_product_attention(...)

    def forward_cuda(self, query, key, value, ...):
        # NVIDIA CUDA implementation (e.g., FlashAttention)
        return flash_attn_func(...)

    def forward_hip(self, query, key, value, ...):
        # AMD ROCm implementation
        return rocm_flash_attn(...)

    def forward_xpu(self, query, key, value, ...):
        # Intel XPU implementation
        return xpu_flash_attn(...)
```

OOT plugin operations:

```python
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.custom_op import CustomOp

@CustomOp.register_oot("MMEncoderAttention")
class CustomMMEncoderAttention(MMEncoderAttention):
    def __init__(self, ...):
        super().__init__(...)

    def forward_oot(self, query, key, value, ...):
        # Vendor-specific optimized kernel (e.g., Huawei Ascend NPU)
        return ascend_flash_attn(...)
```

**OOT registration flow:**
1. Plugin registers `CustomMMEncoderAttention` with key `"MMEncoderAttention"`
2. Entry added to `op_registry_oot`: `{"MMEncoderAttention": CustomMMEncoderAttention}`
3. When vLLM instantiates `MMEncoderAttention`, it checks `op_registry_oot` keys
4. If `"MMEncoderAttention"` found → instantiate `CustomMMEncoderAttention` instead
5. When called → dispatch to `forward_oot()` if OOT platform detected

### Dispatch Logic

When a CustomOp's `forward()` method is called:

```python
def forward(self, *args, **kwargs):
    if self.is_enabled():
        # Dispatch to platform-specific implementation
        if current_platform.is_cpu():
            return self.forward_cpu(*args, **kwargs)
        elif current_platform.is_cuda():
            return self.forward_cuda(*args, **kwargs)
        elif current_platform.is_rocm():
            return self.forward_hip(*args, **kwargs)  # fallback to forward_cuda() if not implemented
        elif current_platform.is_xpu():
            return self.forward_xpu(*args, **kwargs)
        elif current_platform.is_tpu():
            return self.forward_tpu(*args, **kwargs)
        elif current_platform.is_oot():
            return self.forward_oot(*args, **kwargs)
        else:
            return self.forward_native(*args, **kwargs)
    else:
        # Custom op disabled → use PyTorch-native implementation
        return self.forward_native(*args, **kwargs)
```

**Fallback hierarchy:**
1. Platform-specific method (e.g., `forward_hip()`)
2. CUDA fallback (ROCm uses `forward_cuda()` if `forward_hip()` not implemented)
3. Native fallback (`forward_native()` for all platforms)

**Note:** Derived classes can override this behavior via class inheritance.

## Enable/Disable Configuration

### Default Behavior

CustomOps are **enabled or disabled** based on `compilation_config.custom_ops`:

- If `compilation_config.backend == "inductor"` and `compilation_config.mode != CompilationMode.NONE`:
  - Append `"none"` to `custom_ops` → **disable all custom ops** (let Inductor generate fused Triton kernels)
- Otherwise:
  - Append `"all"` to `custom_ops` → **enable all custom ops**

**Rationale:** On NVIDIA platforms, Inductor's native fusion often outperforms custom CUDA kernels (see [[Kernel Fusions#Inductor Fusion Competition]]).

### Fine-Grained Control

Users can override defaults via CLI flags:

```bash
# Enable all custom ops
vllm serve --compilation_config.custom_ops '["all"]'

# Disable all custom ops
vllm serve --compilation_config.custom_ops '["none"]'

# Enable all except op1 (prefixed with -)
vllm serve --compilation_config.custom_ops '["all,-op1"]'

# Only enable op1 and op2 (prefixed with +)
vllm serve --compilation_config.custom_ops '["none,+op1,+op2"]'
```

**Constraints:**
- `"all"` and `"none"` cannot coexist
- User-set flags **override** optimization-level defaults

### Enforce-Enable Mechanism

Some operations are **force-enabled** regardless of configuration:

```python
class MMEncoderAttention(CustomOp):
    def __init__(self, ..., enforce_enable=True):
        super().__init__(enforce_enable=enforce_enable)
        # ...
```

**Use case:** Multi-modal models (ViT encoders) require device-specific optimized kernels for acceptable performance. Operations like `MMEncoderAttention` and `ApplyRotaryEmb` are enforce-enabled.

**Future plan:** This mechanism will be removed after vLLM adds a separate `compilation_config` for the multi-modal part.

## CustomOp Categories

vLLM has 12 categories of CustomOps (60+ total operations):

### 1. Attention

- `multi_head_latent_attention` (MLA for DeepSeek models)
- `mm_encoder_attn` (multi-modal encoder attention)
- `rel_pos_attention` (relative position attention)

**Related:** MLA Attention, Multi-Modal Models, [[Attention Backends]]

### 2. Activation

- `silu_and_mul`, `mul_and_silu` (SiLU + multiply, used in LLaMA FFN)
- `gelu_new`, `gelu_fast`, `quick_gelu` (GELU variants)
- `gelu_and_mul`, `gelu_and_mul_sparse` (GELU + multiply)
- `relu2`, `xielu` (ReLU variants)
- `swigluoai_and_mul`, `fatrelu_and_mul` (specialized activations)

**Related:** SiLU Activation, GELU, FFN Architecture

### 3. MM-Conv

- `conv2d`, `conv3d` (2D/3D convolution for vision models)

**Related:** Multi-Modal Models, Vision Encoders

### 4. Embedding

- `vocab_parallel_embedding` (vocabulary embedding with tensor parallelism)
- `parallel_lm_head` (language model head with tensor parallelism)

**Related:** [[Tensor Parallelism]], Vocabulary Projection

### 5. Linear

- `row_parallel_linear` (row-wise tensor-parallel linear layer)
- `column_parallel_linear` (column-wise tensor-parallel linear layer)
- `replicated_linear` (non-parallelized linear layer)

**Related:** [[Tensor Parallelism]], GEMM Optimization

### 6. Logits Processor

- `logits_processor` (logits manipulation for sampling)

**Related:** Sampling Strategies, Logit Bias

### 7. Mamba

- `mamba_mixer` (Mamba state-space model mixer)
- `mamba_mixer2` (Mamba2 variant)
- `mixer2_gated_rms_norm` (gated RMSNorm for Mamba2)
- `plamo2_mamba_mixer` (Plamo2-specific Mamba mixer)
- `short_conv` (short convolution for Mamba)

**Related:** Mamba Models, State-Space Models, Hybrid Attention

### 8. MoE

- `fused_moe` (fused mixture-of-experts)
- `modular_fused_moe` (modular MoE method)
- `unquantized_fused_moe` (unquantized MoE method)
- `transformers_fused_moe` (Transformers-compatible MoE)
- `grouped_topk` (grouped top-k routing)

**Related:** MoE Architecture, Expert Routing, Sparse Experts

### 9. Norm

- `rms_norm` (RMSNorm layer)
- `rms_norm_gated` (gated RMSNorm)
- `gemma_rms_norm` (Gemma-specific RMSNorm variant)

**Related:** RMSNorm, Layer Normalization

### 10. Quantization

- `quant_fp8` (FP8 quantization)

**Related:** [[Quantization]], [[FP8 Quantization]], [[Kernel Fusions]]

### 11. RoPE

- `rotary_embedding` (standard rotary positional embedding)
- `dual_chunk_rotary_embedding` (dual-chunk RoPE for long context)
- `apply_rotary_emb` (apply RoPE to tensors)

**Related:** RoPE, Positional Encodings, Long Context

### 12. Encoder

- `qwen2_decoder` (Qwen2 decoder layer)
- `mm_encoder_attn` (multi-modal encoder attention, duplicated from Attention category)
- `rel_pos_attention` (relative position attention, duplicated from Attention category)

**Related:** Qwen Models, Multi-Modal Models

## Integration with torch.compile

CustomOp interacts with [[torch.compile Integration]] in two modes:

### Mode 1: Custom Ops Enabled (Default for Eager/Non-Inductor)

```text
Model forward pass
  ↓
CustomOp.forward()
  ↓
Platform dispatch (forward_cuda(), forward_hip(), etc.)
  ↓
Vendor-optimized kernel (CUDA, HIP, XPU, TPU, OOT)
```

- Custom ops execute directly
- torch.compile sees custom ops as **opaque nodes** (does not trace into them)
- Fusion opportunities limited to boundaries around custom ops

### Mode 2: Custom Ops Disabled (Default for Inductor Mode)

```text
Model forward pass
  ↓
CustomOp.forward()
  ↓
forward_native() (PyTorch-native implementation)
  ↓
TorchDynamo traces through native operations
  ↓
Inductor generates fused Triton kernels
```

- Custom ops fall back to PyTorch-native implementation
- torch.compile traces **through** native operations (full graph visibility)
- Inductor applies its own fusions (often faster than custom CUDA kernels on NVIDIA)

**Example:** `rms_norm` + `quant_fp8` fusion

- **Custom ops enabled:** Two separate kernel launches (custom CUDA kernels)
- **Custom ops disabled:** TorchDynamo traces `rms_norm` and `quant_fp8` native implementations → Inductor fuses into single Triton kernel

See [[Kernel Fusions#Inductor Fusion Competition]] for when Inductor fusion outperforms custom ops.

## OOT Hardware Plugins

CustomOp enables hardware vendors to extend vLLM without forking:

### Official Plugins

- **[vllm-ascend](https://github.com/vllm-project/vllm-ascend)** — Huawei Ascend NPU
- **[vllm-spyre](https://github.com/vllm-project/vllm-spyre)** — Spyre accelerator
- **[vllm-gaudi](https://github.com/vllm-project/vllm-gaudi)** — Intel Gaudi
- **[vllm-neuron](https://github.com/vllm-project/vllm-neuron)** — AWS Neuron (Trainium/Inferentia)
- **[vllm-metal](https://github.com/vllm-project/vllm-metal)** — Apple Silicon (Metal)

### Non-Official Plugins

- **[vllm-metax](https://github.com/MetaX-MACA/vLLM-metax)** — MetaX GPU
- **[vllm-kunlun](https://github.com/baidu/vLLM-Kunlun)** — Baidu Kunlun XPU
- **[vllm-musa](https://github.com/MooreThreads/vllm-musa)** — Moore Threads GPU

See [[Plugin System]] for architectural details.

### Plugin Implementation Pattern

**Step 1:** Extend vLLM CustomOp

```python
# In plugin package (e.g., vllm_ascend)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.custom_op import CustomOp

@CustomOp.register_oot("SiluAndMul")
class AscendSiluAndMul(SiluAndMul):
    def __init__(self, ...):
        super().__init__(...)

    def forward_oot(self, x):
        # Ascend NPU optimized kernel
        import ascend_kernels
        return ascend_kernels.silu_and_mul(x)
```

**Step 2:** Bulk registration (optional)

```python
# In plugin package
from vllm.model_executor.custom_op import CustomOp

REGISTERED_CUSTOM_OPS = {
    "SiluAndMul": AscendSiluAndMul,
    "RMSNorm": AscendRMSNorm,
    "RotaryEmbedding": AscendRotaryEmbedding,
}

for op_name, op_cls in REGISTERED_CUSTOM_OPS.items():
    CustomOp.register_oot(_decorated_op_cls=op_cls, name=op_name)
```

**Step 3:** User runs vLLM with plugin

```bash
# Plugin installed: pip install vllm-ascend
vllm serve meta-llama/Llama-3.1-8B-Instruct
# vLLM auto-detects OOT platform → dispatches to forward_oot()
```

**Benefits:**
- No vLLM core modifications
- Plugin versioned independently
- Hardware vendors maintain their own kernels
- Users get optimized performance via `pip install vllm-{vendor}`

## Implementation Details

### CustomOp Base Class

**Location:** [`vllm/model_executor/custom_op.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/custom_op.py)

**Key methods:**

```python
class CustomOp:
    op_registry: Dict[str, Type[CustomOp]] = {}
    op_registry_oot: Dict[str, Type[CustomOp]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register built-in op"""
        def decorator(op_cls):
            cls.op_registry[name] = op_cls
            return op_cls
        return decorator

    @classmethod
    def register_oot(cls, name: str, _decorated_op_cls=None):
        """Decorator to register OOT op"""
        def decorator(op_cls):
            cls.op_registry_oot[name] = op_cls
            return op_cls
        if _decorated_op_cls:
            return decorator(_decorated_op_cls)
        return decorator

    def is_enabled(self) -> bool:
        """Check if custom op enabled via compilation_config.custom_ops"""
        # ...

    def forward(self, *args, **kwargs):
        """Main dispatch entry point"""
        if self.is_enabled():
            return self._dispatch_platform(*args, **kwargs)
        else:
            return self.forward_native(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        """PyTorch-native fallback (must be implemented by subclass)"""
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        """NVIDIA CUDA implementation (optional)"""
        return self.forward_native(*args, **kwargs)

    def forward_hip(self, *args, **kwargs):
        """AMD ROCm implementation (optional, fallback to forward_cuda)"""
        return self.forward_cuda(*args, **kwargs)

    # ... forward_cpu(), forward_xpu(), forward_tpu(), forward_oot()
```

### Configuration Parsing

**Location:** [`vllm/config/compilation.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py)

**CompilationConfig:**

```python
@dataclass
class CompilationConfig:
    backend: str = "inductor"  # "inductor" | "eager" | ...
    mode: CompilationMode = CompilationMode.DEFAULT
    custom_ops: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Auto-append "none" or "all"
        if self.backend == "inductor" and self.mode != CompilationMode.NONE:
            if "all" not in self.custom_ops and "none" not in self.custom_ops:
                self.custom_ops.append("none")
        else:
            if "all" not in self.custom_ops and "none" not in self.custom_ops:
                self.custom_ops.append("all")
```

**CustomOp.is_enabled() logic:**

```python
def is_enabled(self) -> bool:
    if self.enforce_enable:
        return True

    custom_ops = get_compilation_config().custom_ops
    op_name = self.__class__.__name__

    # Check explicit enable/disable
    if f"+{op_name}" in custom_ops:
        return True
    if f"-{op_name}" in custom_ops:
        return False

    # Check default
    if "all" in custom_ops:
        return True
    if "none" in custom_ops:
        return False

    # Should not reach here (post_init ensures "all" or "none")
    return True
```

## Use Cases

### Use Case 1: Multi-Platform Deployment

**Scenario:** Serve same model on NVIDIA A100 (CUDA), AMD MI300X (ROCm), and Intel Gaudi.

**Solution:**
1. vLLM core implements `forward_native()` (PyTorch-native, works everywhere)
2. NVIDIA-specific ops implement `forward_cuda()` (FlashAttention, custom CUDA kernels)
3. AMD-specific ops implement `forward_hip()` (ROCm kernels, AITER fusions)
4. Intel-specific ops via `vllm-gaudi` plugin implement `forward_oot()`

**Result:** Single codebase, platform-optimized performance.

### Use Case 2: Inductor Fusion vs Custom Kernels

**Scenario:** On NVIDIA H100, Inductor's Triton fusion outperforms custom CUDA kernels for `rms_norm` + `quant_fp8`.

**Solution:**
1. Run with `-cc.mode=3` (Inductor mode) → `custom_ops` auto-set to `["none"]`
2. `RMSNorm` and `QuantFP8` fall back to `forward_native()`
3. TorchDynamo traces through native ops → Inductor fuses into single Triton kernel
4. Benchmark confirms 10-15% speedup vs custom CUDA kernels

**Related:** [[Kernel Fusions#RMSNorm + Quantization]]

### Use Case 3: OOT Hardware Vendor Extension

**Scenario:** Huawei Ascend NPU vendor wants to support vLLM.

**Solution:**
1. Huawei implements `vllm-ascend` plugin package
2. Registers `AscendSiluAndMul`, `AscendRMSNorm`, etc. via `@CustomOp.register_oot()`
3. Implements `forward_oot()` with Ascend-optimized kernels
4. Users install: `pip install vllm vllm-ascend`
5. vLLM detects OOT platform → dispatches to `forward_oot()`

**Result:** Huawei maintains plugin independently, no vLLM core changes.

**Blog:** [Introducing vLLM Hardware Plugin, Best Practice from Ascend NPU](https://blog.vllm.ai/2025/05/12/hardware-plugin.html)

## Debugging

### Check if CustomOp is Enabled

**Set logging level:**

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.1-8B-Instruct
```

**Look for logs:**

```text
[CustomOp] SiluAndMul enabled: True (dispatching to forward_cuda)
[CustomOp] RMSNorm enabled: False (falling back to forward_native)
```

### Verify OOT Registration

**Check op_registry_oot:**

```python
from vllm.model_executor.custom_op import CustomOp

print(CustomOp.op_registry_oot)
# Output: {'MMEncoderAttention': <class 'AscendMMEncoderAttention'>, ...}
```

### Force Enable/Disable

**Force enable all custom ops:**

```bash
vllm serve --compilation_config.custom_ops '["all"]'
```

**Force disable specific op:**

```bash
vllm serve --compilation_config.custom_ops '["all,-SiluAndMul"]'
```

**Override in code:**

```python
from vllm import LLM
from vllm.config import CompilationConfig

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    compilation_config=CompilationConfig(
        custom_ops=["none", "+RotaryEmbedding"]  # Only enable RoPE
    )
)
```

## Limitations and Future Work

### Current Limitations

1. **Enforce-enable mechanism:** Multi-modal models require enforce-enable for some ops (breaks Inductor fusion). Will be removed after separate multi-modal `compilation_config`.

2. **OOT dispatch overhead:** OOT platform detection adds small runtime overhead (platform check on every `forward()` call). Could be optimized with dispatch caching.

3. **No partial fusion:** Cannot fuse across custom op boundaries (custom ops are opaque to TorchDynamo). Future work: expose custom op internals for Inductor fusion.

4. **No auto-tuning:** Custom ops don't participate in Inductor auto-tuning. Future work: integrate custom op selection into auto-tuning loop.

### Future Enhancements

- **Dynamic dispatch:** JIT-select between custom op and Inductor fusion based on input shape
- **Hybrid mode:** Allow Inductor to fuse around custom ops (e.g., fuse RMSNorm+Quant, but use custom FlashAttention)
- **Auto-registration:** Auto-discover and register OOT plugins (no manual `CustomOp.register_oot()` calls)
- **Per-layer configuration:** Fine-grained custom op enable/disable per model layer

## Related Concepts

- [[Plugin System]] — OOT hardware plugin architecture
- [[torch.compile Integration]] — custom op interaction with Inductor
- [[Kernel Fusions]] — fusion competition between custom ops and Inductor
- [[Optimization Levels]] — custom op enable/disable defaults
- [[Attention Backends]] — attention backend selection (separate from CustomOp dispatch)
- [[Quantization]] — FP8 quantization custom ops
- [[Tensor Parallelism]] — parallel linear/embedding custom ops
- Multi-Modal Models — enforce-enable mechanism for ViT encoders

## Further Reading

- Source: [[vllm-fusions-design]]
- Code: [`vllm/model_executor/custom_op.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/custom_op.py)
- Blog: [Introducing vLLM Hardware Plugin, Best Practice from Ascend NPU](https://blog.vllm.ai/2025/05/12/hardware-plugin.html)
- Plugin examples: [vllm-ascend](https://github.com/vllm-project/vllm-ascend), [vllm-gaudi](https://github.com/vllm-project/vllm-gaudi)
