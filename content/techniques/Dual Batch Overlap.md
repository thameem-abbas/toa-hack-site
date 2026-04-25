---
title: Dual Batch Overlap
type: technique
created: 2026-04-25
related:
  - "Expert Parallelism"
  - "Mixture of Experts"
  - "CUDA Graphs"
  - "FusedMoE Modular Kernel"
  - "vLLM Engine"
---

# Dual Batch Overlap

Dual Batch Overlap (DBO) is a vLLM optimization that overlaps all-to-all communication in [[Mixture of Experts]] layers with surrounding computation by splitting the batch into two microbatches executed on separate CPU threads.

## Motivation

### Problem: Communication Bottleneck in EP

With [[Expert Parallelism]], MoE layers require all-to-all communication:
- **Dispatch**: Route tokens to GPU holding selected expert
- **Combine**: Gather expert outputs back to origin GPU

For decode (small batches), all-to-all latency can dominate:
- Typical all-to-all: 100-500 μs
- Typical expert compute: 200-800 μs
- **Without overlap**: Total = 200 μs (dispatch) + 500 μs (compute) + 200 μs (combine) = 900 μs
- **With overlap**: Total ≈ max(400 μs all-to-all, 500 μs compute) = 500 μs

**Speedup potential**: 1.4-1.8× for decode-heavy workloads.

## Mechanism

### Batch Splitting

`GPUModelRunner` splits batch into two microbatches:
1. Coordinate across all DP ranks: All must agree to microbatch
2. Pad token count to max across ranks (avoid empty microbatch)
3. Slice `CommonAttentionMetadata` in half

**Result**: Two microbatches with ~half the tokens each.

### Thread Ping-Pong

Two CPU threads (UBatch threads) execute microbatches in interleaved fashion:

```
Thread 0                          Thread 1
--------                          --------
Dispatch microbatch 0 (send)      (waiting)
Dispatch microbatch 0 (recv)  →   Dispatch microbatch 1 (send)
(waiting)                         Dispatch microbatch 1 (recv)
Compute microbatch 1          ←   (waiting)
(waiting)                         Compute microbatch 0
Combine microbatch 1 (send)   →   (waiting)
Combine microbatch 1 (recv)       Combine microbatch 0 (send)
(waiting)                     ←   Combine microbatch 0 (recv)
```

### Overlap Schedule

Current implementation in vLLM:

```python
# Schedule notation:
#   S  = Shared expert
#   A0 = MLA qkv projection
#   A1 = Core attention + output projection + MoE gate
#   D  = All-to-all Dispatch
#   C  = All-to-all Combine

# Compute timeline: |-A0₀-A1₀-||-MLP₁-||-S₁-MLP₀-||-S₀-A0₁-A1₁-|
# Communication:    |----D₁---||--D₀--||----C₁---||-----C₀-----|

# Execution order:
# 1. D₁ send          (thread 1 starts dispatch)
# 2. A0₀, A1₀         (thread 0 does attention while thread 1 waits for dispatch recv)
# 3. D₁ recv          (thread 1 receives dispatched tokens)
# 4. D₀ send          (thread 0 starts dispatch)
# 5. MLP₁             (thread 1 computes experts while thread 0 waits)
# 6. D₀ recv          (thread 0 receives dispatched tokens)
# 7. C₁ send          (thread 1 starts combine)
# 8. S₁, MLP₀         (shared expert on thread 1, experts on thread 0)
# 9. C₁ recv          (thread 1 receives combined outputs)
# 10. C₀ send         (thread 0 starts combine)
# 11. S₀, A0₁, A1₁    (shared expert + attention on thread 0)
# 12. C₀ recv         (thread 0 receives combined outputs)
```

**Key insight**: While one microbatch waits on communication, the other computes.

## Implementation Components

### UBatchWrapper

Model wrapper managing threads, contexts, and CUDA graphs for DBO.

**Responsibilities**:
- Spawn two UBatch threads
- Manage CUDA graph capture/replay for DBO
- Decide whether to enable DBO based on batch size thresholds

**Initialization**:
```python
UBatchWrapper(model, vllm_config, cudagraph_mode, device)
```

**Forward**:
```python
def forward(self, **model_args):
    if 'ubatch_slices' in forward_context:
        # Run with DBO: Split batch, spawn threads
        return self._forward_with_dbo(model_args)
    else:
        # Run without DBO: Standard forward pass
        return self.model.forward(**model_args)
```

**CUDA Graphs**:
- DBO requires Full CUDA graphs (capture entire model execution)
- `UBatchWrapper` manages graph capture for both microbatches
- Once captured, replay has no multithreading overhead

**Transparency**: `GPUModelRunner` doesn't know if DBO is active (wrapper handles it).

### UBatchContext

`ForwardContext` wrapper for thread synchronization.

**Creation**:
```python
make_ubatch_contexts(stream_0, stream_1, forward_ctx_0, forward_ctx_1, barrier)
```

Creates two `UBatchContext` instances with cross-references for ping-pong.

**Key methods**:

- **`dbo_yield()`**: Pause current thread, wake other thread
  - Uses CUDA events for synchronization
  - Ensures previous CUDA ops complete before switching
  
- **`dbo_register_recv_hook(callback)`**: Register callback for other thread to invoke
  - Typically used for async all-to-all receive
  - Other thread calls `dbo_maybe_run_recv_hook()` to execute
  
- **`dbo_maybe_run_recv_hook()`**: Run callback registered by other thread (if any)
  - Used to wait on all-to-all completion from other microbatch

**Synchronization pattern**:
```python
# In FusedMoEModularKernel.forward():

# Thread 0, microbatch 0
recv_hook_0 = prepare_finalize.prepare_no_receive(...)  # Start dispatch, don't wait
ubatch_ctx_1.dbo_register_recv_hook(recv_hook_0)       # Register for thread 1 to call
ubatch_ctx.dbo_yield()                                  # Yield to thread 1

# Thread 1, microbatch 1
recv_hook_1 = prepare_finalize.prepare_no_receive(...)  # Start dispatch, don't wait
ubatch_ctx_0.dbo_register_recv_hook(recv_hook_1)       # Register for thread 0 to call
ubatch_ctx.dbo_yield()                                  # Yield to thread 0

# Thread 0, microbatch 0 (resumed)
ubatch_ctx.dbo_maybe_run_recv_hook()                   # Wait on thread 1's dispatch
# ... expert computation ...
ubatch_ctx.dbo_yield()                                  # Yield to thread 1

# And so on...
```

### GPUModelRunner Integration

Decides whether to enable DBO based on token counts:

**Configuration**:
- `--dbo-decode-token-threshold`: Min tokens for decode-only batch (default varies)
- `--dbo-prefill-token-threshold`: Min tokens for batch with prefill (default varies)

**Decision logic**:
1. Check if all DP ranks can microbatch (coordination via distributed)
2. Check if token count exceeds threshold
3. If yes: Slice `CommonAttentionMetadata`, set `ubatch_slices` in `ForwardContext`
4. If no: Standard single-batch execution

**Uniformity**: All DP ranks must agree (can't have some with DBO, some without).

## Configuration

### Enabling DBO

Required flags:
```bash
vllm serve model \
  --enable-dbo \                    # Enable DBO
  --data-parallel-size N \          # N > 1 required
  --enable-expert-parallel \        # EP required
  --all2all-backend deepep_low_latency  # or deepep_high_throughput
```

**Prerequisites**:
- Data parallel size > 1 (need DP + EP for all-to-all)
- Expert parallelism enabled
- DeepEP backend (only async backends support DBO)

### Thresholds

```bash
vllm serve model \
  --enable-dbo \
  --dbo-decode-token-threshold 128 \   # Min tokens for decode-only batch
  --dbo-prefill-token-threshold 512    # Min tokens for batch with prefill
```

**Rationale**:
- Small batches: Overhead of multithreading > benefit of overlap
- Large batches: Communication latency amortized, less benefit from overlap
- Sweet spot: Medium batches where communication is significant but not amortized

### Example: DeepSeek-V2-Lite

```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --trust-remote-code \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --enable-dbo \
  --all2all-backend deepep_low_latency \
  --dbo-decode-token-threshold 64 \
  --dbo-prefill-token-threshold 256
```

**Note**: Requires at least 2 GPUs visible in `CUDA_VISIBLE_DEVICES`.

## Compatibility

### Supported Backends

Only async `FusedMoEPrepareAndFinalizeModular` backends:
- `deepep_high_throughput` (prefill-optimized)
- `deepep_low_latency` (decode-optimized, recommended for DBO)

**Requirement**: Backend must implement `prepare_no_receive()` (async dispatch).

### CUDA Graph Mode

DBO requires **Full CUDA graphs** (entire model captured):
- `FULL` or `FULL_DECODE_ONLY` cudagraph modes
- Partial graphs (`PIECEWISE`) don't support DBO
- Reason: Thread synchronization must be captured in graph

**Benefit**: Once captured, replay is synchronous (no multithreading overhead).

### Not Compatible With

- `--all2all-backend flashinfer_nvlink_*` (no async support)
- Single DP rank (`--data-parallel-size 1`)
- EP without DP
- Models without MoE layers

## Performance

### Speedup

Workload-dependent, typically:
- **Decode-heavy**: 1.3-1.8× speedup
- **Prefill-heavy**: 1.1-1.3× speedup
- **Mixed**: 1.2-1.5× speedup

**Best case**: Decode workload where all-to-all latency ≈ expert compute time (maximal overlap).

### Overhead

- **CUDA graph capture time**: +10-30% vs non-DBO (two microbatches to capture)
- **Memory**: Minimal (two sets of attention metadata)
- **CPU threads**: 2 worker threads (usually negligible)

### Tuning

Adjust thresholds based on workload:
- **Decode-only**: Lower `--dbo-decode-token-threshold` (e.g., 32-64)
- **Mixed prefill/decode**: Higher `--dbo-prefill-token-threshold` (e.g., 512-1024)
- **Profiling**: Use `nsys` to check if communication is hiding behind compute

## Limitations

### DP + EP Only

DBO requires data parallelism + expert parallelism:
- Cannot use with EP-only or TP-only deployments
- Reason: All-to-all overhead only significant with multi-rank communication

### Fixed Microbatch Split

Current implementation: Always split 50/50
- Cannot dynamically adjust split ratio based on routing imbalance
- Future: Adaptive microbatch sizing

### Shared Expert Overlap

Schedule includes shared expert computation during combine phase:
- Works for models like DeepSeek (has shared experts)
- No benefit for models without shared experts (e.g., Mixtral)

## Debugging

### Verify DBO is Active

Check logs for:
```
DBO enabled for batch with N tokens
```

If not appearing, DBO is disabled (threshold not met or incompatible config).

### Profiling with NVTX

vLLM emits NVTX markers for DBO events:
```bash
nsys profile -t cuda,nvtx vllm serve ... --enable-dbo
```

Look for:
- `dbo_yield` markers showing thread switches
- Overlap between `all2all_dispatch` and `expert_compute`

### Common Issues

**DBO not activating**:
- Check `--data-parallel-size > 1`
- Check `--enable-expert-parallel`
- Check `--all2all-backend` is DeepEP variant
- Check token count exceeds thresholds

**Lower than expected speedup**:
- Profiling may show compute << communication (need larger batch or faster kernels)
- Or communication << compute (not communication-bound, DBO doesn't help)

**CUDA graph capture errors**:
- Ensure `--compilation-config '{"cudagraph_mode": "FULL"}'` or similar
- Check for non-CUDA-graph-compatible operations in model

## Future Directions

### Adaptive Microbatch Sizing

Current: Fixed 50/50 split
Future: Adjust split based on routing imbalance (e.g., 60/40 if experts unevenly loaded)

### Multi-Microbatch Overlap

Current: 2 microbatches
Future: N microbatches for more overlap granularity

### Cross-Layer Overlap

Current: Overlap within MoE layer only
Future: Overlap MoE layer N with attention layer N+1

## References

- Concept: [[Expert Parallelism]], [[Mixture of Experts]]
- Architecture: [[FusedMoE Modular Kernel]]
- Related: [[CUDA Graphs]], [[vLLM Engine]]
- Source: [[vllm-moe-design]]
