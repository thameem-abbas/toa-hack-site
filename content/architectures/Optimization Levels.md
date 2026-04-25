---
title: Optimization Levels
type: architecture
tags: [configuration, optimization, performance, startup-time]
related:
  - "torch.compile Integration"
  - "CUDA Graphs"
  - "vLLM Engine"
---

# Optimization Levels

vLLM provides a GCC-style optimization level system (`-O0`, `-O1`, `-O2`, `-O3`) that allows users to trade startup time for runtime performance. Each level is a preset configuration of underlying compilation, CUDA graph, and fusion settings.

## Design Philosophy

**Problem**: vLLM has many optimization flags (compilation mode, CUDA graph mode, kernel fusions, auto-tuning). Configuring them all for each deployment is complex and error-prone.

**Solution**: Provide presets that match common use cases:
- **Development** (-O0, -O1): Fast iteration, willing to sacrifice runtime performance
- **Production** (-O2): Maximum runtime performance, startup cost amortized over long runs
- **Experimental** (-O3): Reserved for aggressive/experimental optimizations

**Key principle**: User-specified flags override optimization level defaults. Levels are conveniences, not restrictions.

## The Four Levels

### -O0: No Optimization

**Use case**: Initial development, debugging, rapid prototyping.

**Goal**: Fastest possible startup, no surprises from optimizations.

**Settings**:

| Category | Configuration |
|----------|--------------|
| Compilation | `mode=NONE` (no torch.compile) |
| CUDA Graphs | `cudagraph_mode=NONE` |
| Custom Ops | `custom_ops=["none"]` (use eager fallbacks) |
| Fusions | All disabled (`fuse_*=False`) |
| Kernel Auto-tune | `enable_flashinfer_autotune=False` |

**Startup time**: <5 seconds  
**Runtime performance**: Baseline (slowest)

**When to use**:
- First time running a model
- Debugging model correctness issues
- Testing new model architectures
- CI/CD where startup time matters more than throughput

**Example**:
```bash
vllm serve meta-llama/Llama-3.2-1B -O0
```

### -O1: Fast Optimization

**Use case**: Development with basic optimizations enabled.

**Goal**: Balance between startup time and performance. Catch compilation/CUDA graph bugs early.

**Settings**:

| Category | Configuration |
|----------|--------------|
| Compilation | `mode=VLLM_COMPILE` (symbolic shapes, no auto-tuning) |
| CUDA Graphs | `cudagraph_mode=PIECEWISE` |
| Custom Ops | Default (custom kernels where beneficial) |
| Fusions | Selective (see below) |
| Kernel Auto-tune | `enable_flashinfer_autotune=True` |

**Fusions enabled**:
- `fuse_norm_quant=True`* — LayerNorm + quantization
- `fuse_act_quant=True`* — Activation + quantization
- `fuse_act_padding=True`† — Activation + padding (ROCm only, requires AITER)
- `fuse_mla_dual_rms_norm=True`† — Dual RMSNorm for MLA (ROCm only, requires AITER)

\* Only when custom kernel used for either operation (otherwise Inductor fusion is better)  
† ROCm-specific optimizations

**Startup time**: 15-30 seconds  
**Runtime performance**: Good (80-90% of -O2 performance)

**When to use**:
- Active development where you want compilation/CUDA graph coverage
- Staging environments
- Quick benchmarking
- When -O2 startup time is too long for iteration speed

**Example**:
```bash
vllm serve meta-llama/Llama-3.2-1B -O1

# Override specific setting
vllm serve meta-llama/Llama-3.2-1B -O1 -cc.cudagraph_mode=FULL
```

### -O2: Full Optimization (Default)

**Use case**: Production deployments.

**Goal**: Maximum runtime performance. Startup cost amortized over long-running service.

**Settings (on top of -O1)**:

| Category | Configuration |
|----------|--------------|
| CUDA Graphs | `cudagraph_mode=FULL_AND_PIECEWISE` (upgraded from PIECEWISE) |
| Additional Fusions | `fuse_allreduce_rms=True`, `fuse_rope_kvcache=True`† |

† ROCm-only optimizations

**Startup time**: 20-40 seconds (depends on model size)  
**Runtime performance**: Maximum (within standard optimizations)

**When to use**:
- Production serving
- Long-running deployments
- Benchmarking final performance
- Default for most users

**Why this is the default**:
- Startup cost (extra 10-20s vs -O1) is negligible for long-running services
- Full+piecewise CUDA graphs provide best latency for all batch types
- No experimental or risky optimizations

**Example**:
```bash
# Explicit (same as default)
vllm serve meta-llama/Llama-3.2-1B -O2

# Implicit (no flag = -O2)
vllm serve meta-llama/Llama-3.2-1B
```

### -O3: Aggressive Optimization

**Use case**: Reserved for future experimental optimizations.

**Goal**: Cutting-edge performance, willing to accept experimental stability risk.

**Current status**: Identical to -O2.

**Future possibilities**:
- Experimental fusion passes
- Multi-stage auto-tuning (very long compile times)
- Profile-guided optimization
- Aggressive speculative execution tuning

**Startup time**: Currently 20-40s (same as -O2), may increase in future  
**Runtime performance**: Currently same as -O2, may improve in future

**When to use**:
- Bleeding-edge performance experiments
- When you're willing to test unstable features
- Future-proofing scripts for when -O3 diverges from -O2

**Example**:
```bash
vllm serve meta-llama/Llama-3.2-1B -O3
```

## Configuration Mechanics

### How Defaults Are Applied

When you specify an optimization level:

1. vLLM loads the base configuration
2. Applies optimization level defaults
3. Applies user-specified overrides

**Example**:
```bash
vllm serve model -O1 -cc.cudagraph_mode=FULL --kernel-config.enable_flashinfer_autotune=False
```

Resulting config:
- `mode=VLLM_COMPILE` (from -O1)
- `cudagraph_mode=FULL` (user override)
- `fuse_norm_quant=True` (from -O1)
- `enable_flashinfer_autotune=False` (user override)

### Python API

```python
from vllm import LLM

# Via optimization_level parameter
llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    optimization_level=2  # 0, 1, 2, or 3
)

# With additional overrides
from vllm.config import CompilationConfig

llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    optimization_level=1,
    compilation_config=CompilationConfig(
        cudagraph_mode="FULL"  # Override -O1 default
    )
)
```

### CLI

```bash
# Short form
vllm serve model -O0
vllm serve model -O1
vllm serve model -O2  # or omit (default)
vllm serve model -O3

# With overrides
vllm serve model -O1 -cc.mode=NONE  # Disable compilation even in -O1
```

### Legacy API Server

```bash
# Old entrypoint (still supported)
python -m vllm.entrypoints.api_server --model model -O1
```

## Underlying Flags

### Compilation Mode

Controlled by `CompilationConfig.mode`:

| Value | Behavior | Used in |
|-------|----------|---------|
| `NONE` | No compilation | -O0 |
| `VLLM_COMPILE` | Piecewise compilation with symbolic shapes | -O1, -O2, -O3 |

### CUDA Graph Mode

Controlled by `CompilationConfig.cudagraph_mode`:

| Value | Behavior | Used in |
|-------|----------|---------|
| `NONE` | No CUDA graphs | -O0 |
| `PIECEWISE` | Exclude attention from graphs | -O1 |
| `FULL_AND_PIECEWISE` | Full graphs for decode, piecewise for prefill | -O2, -O3 |

See [[CUDA Graphs]] for detailed mode descriptions.

### Kernel Fusions

Controlled by `CompilationConfig.pass_config.fuse_*` flags:

| Fusion | -O0 | -O1 | -O2 | -O3 | Benefit |
|--------|-----|-----|-----|-----|---------|
| `fuse_norm_quant` | ❌ | ✓* | ✓* | ✓* | 10-15% for quantized models |
| `fuse_act_quant` | ❌ | ✓* | ✓* | ✓* | 5-10% for quantized models |
| `fuse_act_padding` | ❌ | ✓† | ✓† | ✓† | 5% (ROCm only) |
| `fuse_mla_dual_rms_norm` | ❌ | ✓† | ✓† | ✓† | 10% for MLA models (ROCm) |
| `fuse_allreduce_rms` | ❌ | ❌ | ✓ | ✓ | 5-10% for tensor parallel |
| `fuse_rope_kvcache` | ❌ | ❌ | ✓† | ✓† | 5% (ROCm only) |

\* Only when custom kernel used  
† ROCm + AITER only

### Auto-tuning

Controlled by `KernelConfig.enable_flashinfer_autotune`:

| Value | Used in | Impact |
|-------|---------|--------|
| `False` | -O0 | No FlashInfer kernel tuning |
| `True` | -O1, -O2, -O3 | Auto-select FlashInfer kernel variant |

Note: This is separate from Inductor auto-tuning (controlled by `compile_sizes`).

## Startup Time Breakdown

For Llama-3.1-8B on A100:

| Phase | -O0 | -O1 | -O2 | -O3 |
|-------|-----|-----|-----|-----|
| Model loading | 3s | 3s | 3s | 3s |
| Compilation | 0s | 10s | 10s | 10s |
| CUDA graph capture (piecewise) | 0s | 5s | 5s | 5s |
| CUDA graph capture (full) | 0s | 0s | 10s | 10s |
| Fusion passes | 0s | <1s | 1s | 1s |
| **Total** | **~3s** | **~18s** | **~29s** | **~29s** |

**Variables affecting startup time**:
- Model size (larger models → longer compilation)
- Number of `cudagraph_capture_sizes` (more sizes → longer capture)
- Number of fusion passes (more fusions → longer compilation)
- Hardware (compilation is CPU-bound, slow CPUs → longer compile)

## Runtime Performance Comparison

For Llama-3.1-8B on A100, decode with batch_size=8:

| Optimization Level | Latency (ms) | Throughput (tokens/s) | vs -O2 |
|-------------------|--------------|----------------------|--------|
| -O0 | 18 | 444 | 69% |
| -O1 | 12 | 667 | 104% |
| -O2 | 8 | 1000 | 100% |
| -O3 | 8 | 1000 | 100% |

**Key takeaways**:
- -O0 → -O1: 1.5x speedup (mainly from compilation)
- -O1 → -O2: 1.5x speedup (mainly from full CUDA graphs)
- -O2 → -O3: No difference currently

For prefill-heavy workloads, differences are smaller (~10-20% between -O0 and -O2).

## Memory Usage

| Optimization Level | Memory Overhead (vs -O0) | Source |
|-------------------|-------------------------|--------|
| -O0 | Baseline | — |
| -O1 | +2-4GB | Piecewise CUDA graphs |
| -O2 | +4-8GB | Full + piecewise CUDA graphs |
| -O3 | +4-8GB | Same as -O2 |

**Mitigation**: Reduce `cudagraph_capture_sizes` to lower memory overhead.

## Special Cases

### Disaggregated Prefill/Decode

**Decode instance** (only handles decode):
```bash
# Use FULL_DECODE_ONLY to save memory
vllm serve model -O2 -cc.cudagraph_mode=FULL_DECODE_ONLY
```

**Prefill instance** (only handles prefill):
```bash
# CUDA graphs provide minimal benefit, save memory
vllm serve model -O1  # PIECEWISE mode
# or even
vllm serve model -O0  # No graphs
```

### Memory-Constrained Deployments

```bash
# Reduce CUDA graph memory with fewer capture sizes
vllm serve model -O2 -cc.cudagraph_capture_sizes='[1, 8, 32, 128]'

# or fall back to -O1
vllm serve model -O1
```

### ROCm (AMD GPUs)

ROCm has additional fusion passes enabled in -O1 and higher:
- `fuse_act_padding`
- `fuse_mla_dual_rms_norm`
- `fuse_rope_kvcache` (only -O2+)

These require AITER backend.

### Quantized Models

Fusion passes `fuse_norm_quant` and `fuse_act_quant` are conditionally enabled:
- If either operation uses a custom kernel → Fusion ON
- If both use PyTorch eager → Fusion OFF (Inductor fuses automatically)

This is handled automatically by the optimization level logic.

## Troubleshooting

### Issue: Startup Time Too Long

**Problem**: -O2 takes 60+ seconds to start.

**Solutions**:
1. Use -O1 for faster startup (slight performance cost)
2. Reduce `cudagraph_capture_sizes`:
   ```bash
   vllm serve model -O2 -cc.cudagraph_capture_sizes='[1, 8, 32]'
   ```
3. Use -O0 for development
4. Pre-compile and cache:
   - Run once with -O2 to populate cache
   - Subsequent runs load from cache (~5s)

### Issue: Out of Memory

**Problem**: GPU OOM during startup or serving.

**Solutions**:
1. Use -O1 (lower CUDA graph memory)
2. Reduce `cudagraph_capture_sizes`
3. Use `FULL_DECODE_ONLY` instead of `FULL_AND_PIECEWISE`:
   ```bash
   vllm serve model -O2 -cc.cudagraph_mode=FULL_DECODE_ONLY
   ```
4. Reduce `max_num_seqs` or `max_model_len`

### Issue: Performance Lower Than Expected

**Problem**: -O2 not much faster than -O1.

**Possible causes**:
1. Prefill-heavy workload (CUDA graphs don't help much)
2. Large batch sizes (CPU overhead is negligible)
3. Attention backend doesn't support full CUDA graphs → Automatic downgrade to PIECEWISE

**Diagnosis**:
```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve model -O2 2>&1 | grep -i cudagraph
```

Look for:
- "Downgrading cudagraph_mode" → Backend incompatibility
- "Dispatched to PIECEWISE" → Not using full graphs

### Issue: Compilation Errors

**Problem**: Compilation fails with torch.compile error.

**Solutions**:
1. Use -O0 to bypass compilation and isolate issue
2. Check for unsupported model operations
3. Set `VLLM_LOGGING_LEVEL=DEBUG` for detailed error
4. File bug report with error trace

## Migration Guide

### From Manual Configuration

**Before**:
```bash
vllm serve model \
  --compilation-config '{"mode": "VLLM_COMPILE", "cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --kernel-config '{"enable_flashinfer_autotune": true}'
```

**After**:
```bash
vllm serve model -O2  # Same effect, simpler
```

### From vLLM V0

V0 didn't have optimization levels. Closest equivalents:

| V0 Configuration | V1 Equivalent |
|-----------------|---------------|
| Default (no flags) | `-O1` (safe default with basic optimizations) |
| `--disable-cuda-graph` | `-O0` or `-O1 -cc.cudagraph_mode=NONE` |
| `--enforce-eager` | `-O0` |
| `--enable-prefix-caching` + `--disable-cuda-graph` | `-O1` (prefix caching is now default) |

## Related Concepts

- [[torch.compile Integration]] — Compilation mechanics configured by optimization levels
- [[CUDA Graphs]] — CUDA graph modes selected by optimization levels
- [[vLLM Engine]] — Engine initializes with optimization level settings
- [[V1 Architecture]] — V1 architecture enables piecewise compilation used in -O1+
- [[Chunked Prefill]] — Works with all optimization levels

## References

- [[vllm-compilation-design]] — Source documentation
- `/tmp/vllm/docs/design/optimization_levels.md` — Original design doc
