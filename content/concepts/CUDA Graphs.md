---
title: CUDA Graphs
type: concept
tags: [cuda, optimization, latency, gpu, kernel-launch]
related:
  - "torch.compile Integration"
  - "vLLM Engine"
  - "V1 Architecture"
  - "Optimization Levels"
---

# CUDA Graphs

CUDA Graphs are a GPU programming feature that captures sequences of CUDA kernels and replays them with minimal CPU overhead. For LLM inference, they provide significant latency reductions in the decode phase, where many small kernels are launched repeatedly.

## What Are CUDA Graphs?

### Traditional CUDA Execution

```
CPU                          GPU
─────                        ────
Launch kernel A      ──────> Execute kernel A
  ↓ (overhead)
Launch kernel B      ──────> Execute kernel B
  ↓ (overhead)
Launch kernel C      ──────> Execute kernel C
  ↓ (overhead)
...
```

Each kernel launch incurs CPU overhead:
- Kernel argument setup
- Driver call
- CPU-GPU synchronization

For large kernels (like prefill attention on long sequences), this overhead is negligible (~1% of total time).

For small kernels (like decode attention on batch_size=1), overhead can be 20-50% of total time.

### CUDA Graph Execution

```
CPU                          GPU
─────                        ────
Capture phase:
  Launch kernel A    ──────> [Record] kernel A
  Launch kernel B    ──────> [Record] kernel B  
  Launch kernel C    ──────> [Record] kernel C
  ...
  End capture        ──────> Store graph

Replay phase:
  Replay graph       ──────> Execute A, B, C, ... (no CPU overhead)
  Replay graph       ──────> Execute A, B, C, ... (no CPU overhead)
  Replay graph       ──────> Execute A, B, C, ... (no CPU overhead)
  ...
```

**Capture**: Record GPU operations once  
**Replay**: Execute entire sequence with single CPU call

**Speedup**: 2-5x for decode phase, especially at low batch sizes.

## When CUDA Graphs Help

### Decode Phase (High Benefit)

Characteristics:
- Small batch size (often 1-32 tokens)
- Many kernel launches (attention, MLP, LayerNorm per layer)
- Each kernel is small (microseconds to low milliseconds)
- CPU launch overhead is significant fraction of total time

Example for Llama-3.1-8B (32 layers) decode with batch_size=8:
- Without CUDA graphs: 50+ kernel launches → ~15ms decode latency
- With CUDA graphs: 1 replay call → ~8ms decode latency
- **1.9x speedup**

### Prefill Phase (Low Benefit)

Characteristics:
- Large batch size (hundreds to thousands of tokens)
- Fewer kernel launches (dominated by large attention matmuls)
- Each kernel is large (tens to hundreds of milliseconds)
- CPU launch overhead is negligible (<1% of total time)

Example for Llama-3.1-8B prefill with 512 tokens:
- Without CUDA graphs: ~100ms prefill latency
- With CUDA graphs: ~98ms prefill latency
- **~2% speedup** (not worth the memory cost)

## vLLM CUDA Graph Modes

vLLM V1 introduces five modes controlled by `CompilationConfig.cudagraph_mode`:

### NONE

```python
cudagraph_mode = "NONE"
```

- **Behavior**: Disable CUDA graphs entirely. All kernels launched eagerly.
- **Use case**: Debugging, initial development, incompatible attention backends.
- **Latency**: Baseline (highest).
- **Memory**: Baseline (lowest).
- **Startup time**: Fastest (no capture).

### PIECEWISE

```python
cudagraph_mode = "PIECEWISE"
```

- **Behavior**: Capture CUDA graphs for computation between attention operations. Attention runs eagerly.
- **Compatibility**: Works with all attention backends (attention doesn't need to be graph-compatible).
- **Captures**: Separate graphs for different batch sizes (e.g., 1, 2, 4, 8, 16, 32, 64, 128, 256 tokens).
- **Latency**: Medium-low (attention not captured).
- **Memory**: Medium (one graph per captured size).
- **Startup time**: Medium (capture multiple sizes).

**Why exclude attention?**
- Attention has complex KV cache interactions
- Many backends use dynamic dispatch (not graph-friendly)
- Excluding attention preserves compatibility while capturing the rest

### FULL

```python
cudagraph_mode = "FULL"
```

- **Behavior**: Capture entire model forward (including attention) as single CUDA graph. Uniform decode batches reuse the same graph as non-uniform batches.
- **Compatibility**: Requires attention backend with `AttentionCGSupport.ALWAYS` or `AttentionCGSupport.UNIFORM_BATCH` (e.g., FlashAttention v3, Triton Attention).
- **Captures**: Separate graphs for different batch sizes, but fewer graphs than piecewise.
- **Latency**: Lowest (entire model captured).
- **Memory**: Low (fewer graphs than piecewise).
- **Startup time**: Low (fewer captures).
- **Tradeoff**: Best for small models or workloads with small prompts where prefill graph reuse is beneficial.

### FULL_DECODE_ONLY

```python
cudagraph_mode = "FULL_DECODE_ONLY"
```

- **Behavior**: Full CUDA graphs only for uniform decode batches (pure decode or speculative decode). No graphs for prefill/mixed batches.
- **Compatibility**: Requires attention backend with at least `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`.
- **Use case**: Disaggregated P/D setups where decode instances don't handle prefill.
- **Latency**: Lowest for decode, baseline for prefill.
- **Memory**: Low (only decode graphs, no prefill graphs).
- **Startup time**: Low.

**Why decode-only?**
- Prefill graphs consume significant memory
- Prefill gets minimal benefit from CUDA graphs
- In P/D separation, decode instances never see prefill

### FULL_AND_PIECEWISE (Default)

```python
cudagraph_mode = "FULL_AND_PIECEWISE"
```

- **Behavior**: Full CUDA graphs for uniform decode, piecewise CUDA graphs for prefill/mixed batches. Dispatcher selects mode dynamically based on batch composition.
- **Compatibility**: Requires piecewise compilation and attention backend with CUDA graph support.
- **Latency**: Lowest (best mode for each batch type).
- **Memory**: Highest (full decode graphs + piecewise prefill graphs).
- **Startup time**: Highest (capture both modes).

**Default in vLLM V1** because:
- Decode is latency-critical → full graphs maximize decode throughput
- Prefill benefits from piecewise graphs → better than eager
- Memory cost acceptable for most deployments

## CUDA Graph Dispatching

### Batch Descriptor

vLLM uses a `BatchDescriptor` to uniquely identify batches for CUDA graph selection:

```python
class BatchDescriptor(NamedTuple):
    num_tokens: int      # Padded token count
    num_reqs: int        # Number of requests
    uniform: bool        # All requests same query length?
    has_lora: bool       # Batch uses LoRA adapters?
```

**Uniform batches**:
- Pure decode: `max_query_len == 1`
- Speculative decode: `max_query_len == 1 + num_spec_tokens`

**Non-uniform batches**:
- Prefill: Variable query lengths
- Mixed: Prefill + decode in same batch

### Dispatcher Logic

```python
class CudagraphDispatcher:
    def dispatch(self, batch_descriptor: BatchDescriptor) -> (RuntimeMode, BatchDescriptor):
        # Priority: FULL > PIECEWISE > NONE
        
        if batch_descriptor.uniform and batch_descriptor in self.full_keys:
            return ("FULL", batch_descriptor)
        
        if batch_descriptor in self.piecewise_keys:
            return ("PIECEWISE", batch_descriptor)
        
        # No matching graph, fall back to eager
        return ("NONE", batch_descriptor)
```

**Single source of truth**: `CudagraphDispatcher` maintains all valid graph keys. Wrappers trust the dispatcher's decisions.

### Dispatch Priority Example

For `cudagraph_mode="FULL_AND_PIECEWISE"` with `cudagraph_capture_sizes=[1, 4, 8, 16]`:

| Batch | num_tokens | uniform | Dispatch Result |
|-------|-----------|---------|-----------------|
| Pure decode (bs=8) | 8 | True | **FULL** (exact match) |
| Pure decode (bs=12) | 16 | True | **FULL** (padded to 16) |
| Prefill (bs=4, len=128) | 512 | False | **PIECEWISE** (if captured) |
| Mixed (2 prefill + 2 decode) | variable | False | **PIECEWISE** or **NONE** |
| Pure decode (bs=256) | 256 | True | **NONE** (size not captured) |

**Padding**: Batches are padded up to the nearest captured size for CUDA graph reuse.

## Architecture Components

### CUDAGraphWrapper

Wraps a callable (model forward or subgraph) with capture/replay logic:

```python
class CUDAGraphWrapper:
    def __init__(self, callable, runtime_mode: Literal["FULL", "PIECEWISE"]):
        self.callable = callable
        self.runtime_mode = runtime_mode
        self.graph_cache = {}  # BatchDescriptor -> CUDAGraph
    
    def __call__(self, *args, **kwargs):
        # Get mode and descriptor from ForwardContext
        ctx_mode = get_forward_context().cudagraph_runtime_mode
        batch_desc = get_forward_context().batch_descriptor
        
        # If mode doesn't match this wrapper, pass through
        if ctx_mode != self.runtime_mode:
            return self.callable(*args, **kwargs)
        
        # If graph not cached, capture it
        if batch_desc not in self.graph_cache:
            self.graph_cache[batch_desc] = self._capture(batch_desc, args, kwargs)
        
        # Replay cached graph
        return self._replay(self.graph_cache[batch_desc], args, kwargs)
```

**Key insight**: Wrappers don't make dispatch decisions. They trust `ForwardContext` set by the dispatcher.

### Nested Wrapper Design

```
┌─────────────────────────────────────────────────────────┐
│ CUDAGraphWrapper(mode=FULL)                             │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Model Forward                                       │ │
│ │ ┌─────────────────────────────────────────────────┐ │ │
│ │ │ Compiled Subgraph 0                             │ │ │
│ │ │ ┌─────────────────────────────────────────────┐ │ │ │
│ │ │ │ CUDAGraphWrapper(mode=PIECEWISE)            │ │ │ │
│ │ │ │   (inactive when FULL mode)                 │ │ │ │
│ │ │ └─────────────────────────────────────────────┘ │ │ │
│ │ └─────────────────────────────────────────────────┘ │ │
│ │ Attention (custom op, not compiled)                 │ │
│ │ ┌─────────────────────────────────────────────────┐ │ │
│ │ │ Compiled Subgraph 1                             │ │ │
│ │ │ ┌─────────────────────────────────────────────┐ │ │ │
│ │ │ │ CUDAGraphWrapper(mode=PIECEWISE)            │ │ │ │
│ │ │ └─────────────────────────────────────────────┘ │ │ │
│ │ └─────────────────────────────────────────────────┘ │ │
│ │ ...                                                  │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Behavior by runtime mode**:

| Runtime Mode | FULL Wrapper | PIECEWISE Wrappers |
|--------------|--------------|---------------------|
| `FULL` | **Captures/replays** entire model | Inactive (pass through) |
| `PIECEWISE` | Inactive (pass through) | **Capture/replay** subgraphs |
| `NONE` | Inactive (pass through) | Inactive (pass through) |

This design allows both modes to coexist without conflicts.

## Attention Backend Compatibility

vLLM tracks each backend's CUDA graph support via `AttentionCGSupport` enum:

```python
class AttentionCGSupport(enum.Enum):
    ALWAYS = 3                          # FlashAttention v3, Triton Attention
    UNIFORM_BATCH = 2                   # FlashAttention v2, FlashMLA
    UNIFORM_SINGLE_TOKEN_DECODE = 1     # FlashInfer, Mamba, AITER MLA
    NEVER = 0                           # Most others
```

### Compatibility Table

| Backend | CUDA Graph Support | Notes |
|---------|-------------------|-------|
| FlashAttention v3 | `ALWAYS` | Unified kernel for prefill+decode |
| Triton Attention | `ALWAYS` | Separate kernels but both graph-compatible |
| FlashAttention v2 | `UNIFORM_BATCH` | Works for uniform batches, fallback to `FULL_AND_PIECEWISE` |
| FlashMLA | `UNIFORM_BATCH` | Multi-head latent attention |
| FlashInferMLA | `UNIFORM_BATCH` | |
| FlashInferMLASparse | `UNIFORM_BATCH` | |
| AITER FlashAttention | `UNIFORM_BATCH` | AMD ROCm backend |
| FlashInfer | `UNIFORM_SINGLE_TOKEN_DECODE` | Becomes `UNIFORM_BATCH` with TRTLLM on Blackwell |
| Mamba attention | `UNIFORM_SINGLE_TOKEN_DECODE` | SSM-based, not traditional attention |
| AITER MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| CUTLASS MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| Cascade Attention | `NEVER` | Long-context attention with chunked KV cache |
| All others | `NEVER` | |

### Automatic Mode Downgrade

If the attention backend doesn't support the requested CUDA graph mode, vLLM automatically downgrades:

| Requested Mode | Backend Support | Result |
|----------------|----------------|---------|
| `FULL` | `UNIFORM_BATCH` | `FULL_AND_PIECEWISE` (if piecewise compilation enabled) or `FULL_DECODE_ONLY` |
| `FULL` | `UNIFORM_SINGLE_TOKEN_DECODE` | `FULL_DECODE_ONLY` |
| `FULL` | `NEVER` | `PIECEWISE` (if piecewise compilation) or `NONE` |
| `FULL_AND_PIECEWISE` | `UNIFORM_SINGLE_TOKEN_DECODE` | `FULL_DECODE_ONLY` (piecewise removed) |
| `FULL_AND_PIECEWISE` | `NEVER` | `PIECEWISE` |

**Cascade Attention**: Always dispatches to `PIECEWISE` mode (or `NONE` if piecewise unavailable), regardless of configured mode.

## Memory Implications

### Memory vs Latency Tradeoff

CUDA graphs capture GPU memory allocations for intermediate tensors. Each captured graph requires:

1. **Input buffers**: Model weights, KV cache (shared across graphs)
2. **Intermediate buffers**: Activations, attention outputs (per graph)
3. **Output buffers**: Final hidden states (per graph)

**Memory cost per graph** ≈ (model activation memory for one forward pass)

Example for Llama-3.1-8B:
- Activation memory per forward: ~500MB
- Piecewise graphs (10 sizes): ~5GB additional memory
- Full graphs (10 sizes): ~5GB additional memory
- Full + piecewise: ~10GB additional memory

### vLLM V1 vs V0 Memory Usage

**V0** (legacy architecture):
- CUDA graphs applied more selectively
- Lower memory overhead for graph capture

**V1** (current architecture):
- More aggressive CUDA graph usage
- Higher memory overhead but better latency

**Recommendation**: If memory-constrained, use `FULL_DECODE_ONLY` instead of `FULL_AND_PIECEWISE`.

### Memory Optimization Strategies

1. **Reduce capture sizes**:
   ```python
   cudagraph_capture_sizes=[1, 8, 32, 128]  # Fewer sizes → less memory
   ```

2. **Use decode-only mode** in P/D setups:
   ```python
   cudagraph_mode="FULL_DECODE_ONLY"  # No prefill graphs
   ```

3. **Disable CUDA graphs** for memory-critical deployments:
   ```python
   cudagraph_mode="NONE"  # or use -O0
   ```

4. **Tune max batch size**:
   - Lower `max_num_seqs` reduces memory needed for largest graphs

## Interaction with Multimodal Models

Multimodal models (vision-language) have additional CUDA graph considerations:

### Vision Encoder CUDA Graphs

- Vision encoders (e.g., ViT in Qwen2-VL) can be captured separately
- See `docs/design/cuda_graphs_multimodal.md` for details

### Text-Image Interleaving

- Mixed batches with images+text may not be uniform
- Falls back to piecewise or eager execution

## Configuration

### Via CompilationConfig

```python
from vllm.config import CompilationConfig

compilation_config = CompilationConfig(
    cudagraph_mode="FULL_AND_PIECEWISE",  # NONE, PIECEWISE, FULL, FULL_DECODE_ONLY
    cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 32, 64, 128, 256],
)

llm = LLM(model="meta-llama/Llama-3.1-8B", compilation_config=compilation_config)
```

### Via CLI

```bash
# Via optimization level
vllm serve model -O2  # Uses FULL_AND_PIECEWISE

# Direct configuration
vllm serve model --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'

# Dot notation
vllm serve model -cc.cudagraph_mode=PIECEWISE
```

### Via Environment Variable

```bash
# Disable CUDA graphs for debugging
export VLLM_CUDAGRAPH_MODE=NONE
vllm serve model
```

## Debugging CUDA Graphs

### Enable Debug Logging

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve model
```

Logs show:
- CUDA graph capture for each size
- Dispatch decisions (FULL vs PIECEWISE vs NONE)
- Graph replay counts

### Disable CUDA Graphs

```bash
vllm serve model -O0  # Sets cudagraph_mode=NONE
```

### Verify Dispatch Behavior

Add logging to see which mode is selected:

```python
# In vllm/v1/cudagraph_dispatcher.py
logger.info(f"Dispatched batch {batch_descriptor} to {runtime_mode}")
```

### Common Issues

1. **Graph capture fails**:
   - Attention backend not compatible → Use `PIECEWISE` mode
   - Dynamic control flow in model → Cannot be captured

2. **Memory OOM during capture**:
   - Reduce `cudagraph_capture_sizes`
   - Use `FULL_DECODE_ONLY` instead of `FULL_AND_PIECEWISE`

3. **No speedup observed**:
   - Batch size not in `cudagraph_capture_sizes` → Falling back to eager
   - Prefill-only workload → CUDA graphs don't help much

## Relationship to torch.compile

CUDA graphs and [[torch.compile Integration]] are orthogonal:

| Feature | Purpose | Independent? |
|---------|---------|-------------|
| `torch.compile` | Generate optimized kernels via Inductor | Yes (can compile without graphs) |
| CUDA graphs | Reduce CPU launch overhead | Yes (can graph without compiling) |

**In practice**: vLLM uses both together:

1. **Compile** → Generate fast Triton kernels
2. **Capture** → Record compiled kernel launches as CUDA graph
3. **Replay** → Execute with minimal overhead

**Piecewise compilation enables piecewise CUDA graphs** by splitting the graph at attention operations.

**Full compilation** can work with full CUDA graphs or no graphs.

## Performance Characteristics

### Latency Impact (Llama-3.1-8B on A100)

| Workload | Eager | PIECEWISE | FULL | FULL_AND_PIECEWISE |
|----------|-------|-----------|------|--------------------|
| Decode (bs=1) | 12ms | 8ms | 6ms | 6ms |
| Decode (bs=8) | 15ms | 10ms | 8ms | 8ms |
| Prefill (512 tok) | 100ms | 95ms | 98ms | 95ms |
| Mixed (prefill+decode) | 120ms | 110ms | 120ms | 110ms |

**Key observations**:
- Decode sees 1.5-2x speedup with graphs
- Prefill sees minimal benefit
- Mixed batches benefit from piecewise graphs

### Throughput Impact

For decode-heavy workloads (long conversations):
- **FULL_AND_PIECEWISE**: 20-30% higher throughput than eager
- **PIECEWISE**: 10-20% higher throughput than eager

For prefill-heavy workloads (short queries):
- **Minimal impact**: <5% throughput difference

### Startup Time Impact

CUDA graph capture adds to startup time:

| Mode | Capture Time (Llama-3.1-8B) |
|------|---------------------------|
| `NONE` | 0s |
| `PIECEWISE` (10 sizes) | 5-10s |
| `FULL` (10 sizes) | 5-10s |
| `FULL_AND_PIECEWISE` | 10-20s |

**Mitigation**: Cache is persistent across restarts (no re-capture needed).

## Related Concepts

- [[torch.compile Integration]] — Compiles kernels that CUDA graphs capture
- [[Optimization Levels]] — -O0 disables graphs, -O2 enables FULL_AND_PIECEWISE
- [[vLLM Engine]] — Orchestrates CUDA graph capture during initialization
- [[V1 Architecture]] — Multi-process design with unified scheduler supports dynamic dispatch
- [[PagedAttention]] — KV cache interactions affect CUDA graph compatibility
- [[Chunked Prefill]] — Prefill chunking can affect batch uniformity

## References

- [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [vLLM PR #20059](https://github.com/vllm-project/vllm/pull/20059) — Full CUDA graphs implementation
- [[vllm-compilation-design]] — Source summary for this information
