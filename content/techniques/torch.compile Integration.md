---
title: torch.compile Integration
type: technique
tags: [compilation, performance, kernel-generation, torch-compile, inductor, dynamo]
related:
  - "CUDA Graphs"
  - "Optimization Levels"
  - "vLLM Engine"
  - "V1 Architecture"
---

# torch.compile Integration

vLLM V1 uses PyTorch's `torch.compile` as a critical component enabled by default to generate optimized GPU kernels and capture computation graphs. This integration provides significant performance improvements while maintaining compatibility with dynamic batch sizes through careful management of symbolic shapes and graph splitting.

## Overview

`torch.compile` in vLLM serves three primary purposes:

1. **Kernel Generation** — Dynamo captures Python code as FX graphs, Inductor compiles them to Triton/CUDA kernels
2. **Graph-Level Optimization** — Operator fusion, memory layout optimization, kernel auto-tuning
3. **CUDA Graph Enablement** — Compiled graphs can be captured as CUDA graphs for minimal CPU overhead

Unlike typical `torch.compile` usage, vLLM guarantees **all compilation finishes before serving requests**, preventing latency spikes during inference.

## Compilation Pipeline

### 1. Graph Capture (Dynamo)

Dynamo traces the model's `forward()` function and inlines called functions:

```python
# Traced files for Llama model:
vllm/model_executor/models/llama.py          # Model forward
vllm/attention/layer.py                      # Attention layer
vllm/model_executor/layers/activation.py     # Activation functions
vllm/model_executor/layers/linear.py         # Linear layers
vllm/model_executor/layers/rotary_embedding.py  # RoPE
torch/nn/modules/module.py                   # PyTorch module infrastructure
```

The traced computation graph captures:
- **Inputs**: input_ids, position_ids, model weights/buffers
- **Outputs**: final hidden states (before lm_head and sampling)
- **Symbolic shapes**: Only batch size (num_tokens) is dynamic

**Key Design Choice**: Attention is wrapped as `torch.ops.vllm.unified_attention_with_output`, a PyTorch custom op that Dynamo treats as opaque. This allows full-graph capture despite attention's complexity (KV cache interactions, variable shapes).

### 2. Graph Splitting

The computation graph is split by `splitting_ops` (typically the attention operation):

```
Subgraph 0: Input → First layer pre-attention
Subgraph 1: Attention operation (layer 0)
Subgraph 2: Post-attention layer 0 → Pre-attention layer 1
Subgraph 3: Attention operation (layer 1)
...
Subgraph N-1: Final layer post-attention → Output
```

For a model with `L` layers:
- **3 unique subgraph patterns**: first (pre-attention), middle (attention-to-attention), final (post-attention)
- **Total subgraphs**: `2L + 1` (L attention ops + L+1 inter-attention segments)

This splitting enables:
- **Piecewise compilation**: Reuse middle layer compilation for all layers
- **Piecewise CUDA graphs**: Exclude attention from CUDA graphs while capturing surrounding compute
- **Incremental warmup**: Compile/capture subgraphs independently

### 3. Inductor Compilation

Each subgraph is compiled by PyTorch Inductor to optimized kernels:

```
Subgraph (FX graph) → Inductor → Triton/CUDA kernels → Cached .py files
```

Compilation produces:
- **Transformed code**: `transformed_code.py` — Wrapper function that unpacks tensors and calls computation graph
- **Computation graph**: `computation_graph.py` — FX graph with shape annotations and submodules
- **Inductor cache**: `inductor_cache/xx/yyyy.py` — Compiled Triton kernels

**Example**: For a Llama-3.2-1B model with 16 layers:
- Subgraph 0: First layer (unique kernel)
- Subgraphs 1-15: Middle layers (same kernel, reused 15 times)
- Subgraph 16: Final layer (unique kernel)

### 4. Auto-Tuning (Optional)

When compiling for specific batch sizes (`compile_sizes=[1, 2, 4, 8]`), Inductor auto-tunes kernel parameters:

```
AUTOTUNE mm(8x2048, 2048x3072)
  triton_mm_4 0.0130 ms 100.0%  BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, num_stages=5, num_warps=2
  triton_mm_8 0.0134 ms  97.4%  BLOCK_K=128, BLOCK_M=16, BLOCK_N=64, num_stages=5, num_warps=4
  mm          0.0160 ms  81.6%  (cuBLAS baseline)
```

Auto-tuning can provide 20-30% speedups by finding optimal:
- Block sizes (`BLOCK_K`, `BLOCK_M`, `BLOCK_N`)
- Thread counts (`num_warps`)
- Pipeline depth (`num_stages`)

**Tradeoff**: Auto-tuning adds seconds to minutes of compile time. Disabled by default but recommended for production deployments where the cache can be pre-warmed.

## Compilation Cache

### Cache Key Generation

vLLM computes a hash from:

1. **Configuration**: All fields in `CompilationConfig`, `ModelConfig`, `ParallelConfig`, etc. (via `compute_hash()` methods)
2. **PyTorch State**: Torch version, CUDA version, device properties
3. **Source Code**: Hashes of all traced Python files (model code, vLLM layers, PyTorch modules)

This produces a cache directory like:
```
~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/
├── computation_graph.py
├── transformed_code.py
└── inductor_cache/
    ├── iw/ciwzrk3...py  (subgraph 0 kernel)
    ├── ly/clyfzxl...py  (subgraph 1-15 kernel)
    └── tf/ctfftkg...py  (subgraph 16 kernel)
```

### Cache Safety

The hash includes source code, so any change to model code or vLLM layers triggers cache miss and recompilation. This prevents:
- Incorrect results from stale kernels after code changes
- Cross-version compatibility issues

**Cache Portability**: The entire `torch_compile_cache/` directory can be copied between deployments with identical hardware and software, drastically reducing cold-start time.

### Cache Formats

```python
# Binary (default, faster load)
compilation_config=CompilationConfig(compile_cache_save_format="packed")

# Unpacked (human-readable, for debugging)
compilation_config=CompilationConfig(compile_cache_save_format="unpacked")
# or
export VLLM_COMPILE_CACHE_SAVE_FORMAT=unpacked
```

Unpacked format saves generated Triton code as `.py` files, allowing inspection and debugging.

## Dynamic Shapes

### The Guard Problem

PyTorch's `torch.compile` wants to specialize code for specific input shapes. When it encounters dynamic dimensions, it adds "guards" — runtime checks that trigger recompilation if the check fails.

Example problematic guard:
```python
if batch_size == 32:
    # Compiled kernel optimized for batch_size=32
else:
    # Recompile with new batch_size
```

For LLM inference, batch size varies constantly. Guards would cause:
- Compilation during serving (latency spikes)
- Explosion of compiled kernels (memory)

vLLM's solution: **Drop guards** using symbolic shapes, accepting tradeoffs in each mode.

### Three Dynamic Shape Modes

#### BACKED (Default)

```python
compilation_config=CompilationConfig(
    dynamic_shapes_config=DynamicShapesConfig(
        type=DynamicShapesType.BACKED
    )
)
```

- PyTorch treats dynamic dimensions as "backed" symbols
- **Specializes 0/1** — Automatically creates separate paths for batch_size=0, batch_size=1, batch_size>=2
- **Allows guards** — User code, Dynamo, Inductor, Autograd can all add guards
- **vLLM drops guards** — Uses custom pass to remove material guards after compilation
- **Risk**: Guard dropping may be unsound (correctness issues)
- **Benefit**: Maximum performance from specialization

**Use when**: You need peak performance and are willing to accept guard-dropping risks.

#### UNBACKED

```python
compilation_config=CompilationConfig(
    dynamic_shapes_config=DynamicShapesConfig(
        type=DynamicShapesType.UNBACKED
    )
)
```

- PyTorch treats dynamic dimensions as "unbacked" symbols
- **No specialization** — No 0/1 special cases
- **Guaranteed no guards** — Framework will not add guards
- **Risk**: Data-dependent errors if code branches on unbacked values without explicit handling
- **Tradeoff**: Missed optimizations (e.g., assuming inputs non-contiguous)

**Use when**: You need the strongest correctness guarantee against guards.

#### BACKED_SIZE_OBLIVIOUS (Experimental)

```python
compilation_config=CompilationConfig(
    dynamic_shapes_config=DynamicShapesConfig(
        type=DynamicShapesType.BACKED_SIZE_OBLIVIOUS
    )
)
```

- Treats backed symbols as unbacked wherever explicit unbacked handling exists
- **Mostly avoids 0/1 specialization** in framework code
- **Safer than BACKED** but still no guard guarantee
- **PyTorch experimental** — May be deprecated

**Use when**: You want a balance between avoiding guards and performance.

### Recommendation

- Development: `BACKED_SIZE_OBLIVIOUS` or `UNBACKED`
- Production (max performance): `BACKED` with thorough testing
- Production (max safety): `UNBACKED`

## Integration with CUDA Graphs

Compiled graphs serve as the foundation for [[CUDA Graphs]] capture:

### Piecewise CUDA Graphs

```
┌─────────────────────────────────────────┐
│ Model Forward                           │
│ ┌─────────────────────────────────────┐ │
│ │ Compiled Subgraph 0 (CUDA graph)    │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │ Attention 0 (eager)                 │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │ Compiled Subgraph 1 (CUDA graph)    │ │
│ └─────────────────────────────────────┘ │
│ ...                                      │
└─────────────────────────────────────────┘
```

- Attention runs in eager mode (not in CUDA graph)
- Surrounding compute captured as CUDA graphs
- Most compatible: works with all attention backends

### Full CUDA Graphs

```
┌─────────────────────────────────────────┐
│ Entire Model Forward (CUDA graph)       │
│ ┌─────────────────────────────────────┐ │
│ │ Compiled Subgraph 0                 │ │
│ │ Attention 0                          │ │
│ │ Compiled Subgraph 1                 │ │
│ │ Attention 1                          │ │
│ │ ...                                  │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

- Entire model captured as single CUDA graph
- Requires attention backend with CUDA graph support (e.g., FlashAttention v3)
- Lowest latency but less flexible

See [[CUDA Graphs]] for mode selection and compatibility.

## Interaction with Optimization Levels

[[Optimization Levels]] configure compilation behavior:

| Level | Compilation Mode | Typical Compile Time |
|-------|------------------|----------------------|
| **-O0** | `NONE` | 0s (no compilation) |
| **-O1** | `VLLM_COMPILE` (symbolic shapes) | 10-30s |
| **-O2** | `VLLM_COMPILE` (symbolic shapes) | 10-30s |
| **-O3** | `VLLM_COMPILE` (symbolic shapes) | 10-30s |

Additional `-O2`/`-O3` time comes from:
- Full+piecewise CUDA graph capture (not compilation)
- Additional kernel fusions

If `compile_sizes` is set (for auto-tuning):
```bash
vllm serve model --compilation-config '{"compile_sizes": [1, 2, 4, 8]}'
```
Add 30s-5min depending on model size and number of sizes.

## Debugging and Troubleshooting

### Enable Debug Logging

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.2-1B
```

Shows:
- Cache directory path
- Traced files
- Graph splitting details
- Inductor compilation progress

### Disable Compilation (Debugging)

```bash
vllm serve model -O0  # Sets mode=NONE
```

### Disable Cache (Force Recompile)

```bash
export VLLM_DISABLE_COMPILE_CACHE=1
vllm serve model
```

### Inspect Generated Kernels

```python
compilation_config = CompilationConfig(
    compile_cache_save_format="unpacked",
    debug_dump_path="/tmp/vllm_debug"
)
```

Generated kernels saved as readable `.py` files in debug path.

### Common Issues

1. **Compilation during serving** — Should never happen. If it does, cache key may be incorrectly computed.
2. **Guard failures** — Use `UNBACKED` or `BACKED_SIZE_OBLIVIOUS` mode.
3. **Performance regression after code change** — Cache miss triggered recompilation. Normal behavior.
4. **Out of memory during compilation** — Reduce `compile_sizes` or disable auto-tuning.

## Piecewise vs Full Compilation

### Piecewise Compilation (Default)

```python
CompilationConfig(
    splitting_ops=["vllm.unified_attention_with_output"]
)
```

**Advantages**:
- Attention flexibility (works with all backends)
- Faster compilation (reuse middle layer kernels)
- Compatible with piecewise CUDA graphs

**Disadvantages**:
- Incompatible with some custom passes (attention fusion, sequence parallelism)

### Full Compilation

```python
CompilationConfig(
    splitting_ops=[]  # No splitting
)
```

**Advantages**:
- Custom passes can optimize entire graph
- Potentially better kernel fusion

**Disadvantages**:
- Longer compilation (cannot reuse across layers)
- Requires attention backend CUDA graph support for full CUDA graphs
- Less tested than piecewise

**Automatic Fallback**: When attention fusion is enabled, vLLM automatically disables piecewise compilation by setting `splitting_ops=[]`.

### Future: Inductor Graph Partitioning

Experimental in PyTorch 2.9+:

```python
CompilationConfig(
    use_inductor_graph_partition=True
)
```

Allows splitting the graph **after Dynamo but in Inductor**, enabling:
- Custom passes that see the whole graph
- Piecewise CUDA graph capture
- Best of both worlds (but longer compile time)

## Performance Characteristics

### Kernel Fusion Examples

Inductor automatically fuses:
- Elementwise operations (ReLU, GELU, LayerNorm)
- Matrix multiplications with bias
- Quantization operations

Example from logs:
```
triton_mm_4: Fused matmul + bias + activation
  0.0130 ms vs 0.0160 ms (cuBLAS + separate activation)
  → 18% speedup
```

### Memory Overhead

Compilation itself adds minimal memory overhead (compiled kernels are small). Memory impact comes from:
- CUDA graph capture (allocated buffers for captured graphs)
- Auto-tuning (temporary buffers during benchmarking)

See [[CUDA Graphs]] for memory analysis.

### Compile Time Breakdown

For Llama-3.2-1B on A100:
- Dynamo graph capture: 1-2s
- Inductor compilation (symbolic shapes): 8-20s
- Auto-tuning (if enabled, per size): 30-120s
- Total (symbolic only): ~10-25s
- Total (with 4 auto-tune sizes): ~2-8min

Subsequent runs with warm cache: <1s

## Configuration Reference

### CompilationConfig

```python
from vllm.config import CompilationConfig, DynamicShapesConfig, DynamicShapesType

compilation_config = CompilationConfig(
    # Compilation mode
    mode="VLLM_COMPILE",  # or "NONE"
    
    # Dynamic shapes
    dynamic_shapes_config=DynamicShapesConfig(
        type=DynamicShapesType.BACKED  # or UNBACKED, BACKED_SIZE_OBLIVIOUS
    ),
    
    # Auto-tuning
    compile_sizes=[1, 2, 4, 8],  # Specific sizes to compile+tune (default: [])
    
    # Graph splitting
    splitting_ops=["vllm.unified_attention_with_output"],  # Piecewise compilation
    # splitting_ops=[],  # Full compilation
    
    # Cache
    compile_cache_save_format="packed",  # or "unpacked"
    # VLLM_DISABLE_COMPILE_CACHE env var to disable
    
    # CUDA graphs
    cudagraph_mode="FULL_AND_PIECEWISE",  # See [[CUDA Graphs]]
    cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 32, 64, 128, 256],
    
    # Debugging
    debug_dump_path=None,  # Path to save debug artifacts
)
```

### CLI Usage

```bash
# Via optimization level
vllm serve model -O2  # Default, uses VLLM_COMPILE

# Direct compilation config
vllm serve model \
  --compilation-config '{"mode": "VLLM_COMPILE", "compile_sizes": [1, 2, 4, 8]}'

# Dot notation for nested config
vllm serve model -cc.dynamic_shapes_config.type=unbacked
```

## Related Concepts

- [[CUDA Graphs]] — Uses compiled graphs for kernel capture
- [[Optimization Levels]] — Presets that configure compilation behavior
- [[Kernel Fusions]] — Operator fusion patterns enabled by Inductor
- [[vLLM Engine]] — Orchestrates compilation during initialization
- [[V1 Architecture]] — Multi-process design that enables piecewise compilation
- [[Chunked Prefill]] — Benefits from compiled kernels with dynamic batch sizes

## References

- [vLLM torch.compile blog post](https://blog.vllm.ai/2025/08/20/torch-compile.html) (referenced but future-dated in docs)
- PyTorch Dynamo: [pytorch.org/docs/stable/dynamo](https://pytorch.org/docs/stable/dynamo/index.html)
- PyTorch Inductor: [pytorch.org/docs/stable/torch.compiler](https://pytorch.org/docs/stable/torch.compiler.html)
- Triton Language: [github.com/openai/triton](https://github.com/openai/triton)
