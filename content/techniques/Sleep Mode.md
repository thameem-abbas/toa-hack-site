---
title: Sleep Mode
type: technique
created: 2026-04-25
tags: [memory-management, rlhf, gpu-sharing, weight-updates]
---

# Sleep Mode

vLLM's Sleep Mode allows temporarily releasing GPU memory (model weights + KV cache) without stopping the server or unloading containers. This enables GPU sharing between inference and training workloads, particularly for RLHF scenarios where inference and training alternate.

## Overview

Sleep Mode provides two levels:

**Level 1** (shallow sleep):
- Offload model weights to CPU RAM
- Discard KV cache
- Fast wake-up: restore weights from CPU backup
- Use case: pause inference, run training, resume inference with same model

**Level 2** (deep sleep):
- Discard model weights and KV cache (retain only model buffers like RoPE tensors in CPU)
- Free ~90%+ of GPU memory
- Use case: update model weights (RLHF training), load different model

## Key Benefits

- **GPU memory release**: Up to 90%+ of GPU memory freed for other tasks
- **Fast resume**: No full model reload (Level 1) or targeted weight reload (Level 2)
- **API control**: HTTP endpoints (`/sleep`, `/wake_up`) or Python API (`llm.sleep()`, `llm.wake_up()`)
- **Distributed support**: Works with tensor parallelism, pipeline parallelism, etc.
- **Fine-grained control**: Wake only weights or KV cache separately (Level 2)

Platforms: CUDA, ROCm

## Usage

### Offline Inference

Enable sleep mode:
```python
from vllm import LLM

llm = LLM("Qwen/Qwen3-0.6B", enable_sleep_mode=True)
```

**Level 1 (shallow sleep):**
```python
# Pause inference, free GPU memory
llm.sleep(level=1)

# ... run training on GPUs

# Resume inference
llm.wake_up()
```

**Level 2 (deep sleep with weight update):**
```python
# Deep sleep
llm.sleep(level=2)

# Reallocate weights memory only
llm.wake_up(tags=["weights"])

# Load updated weights in-place
llm.collective_rpc("reload_weights")

# Reallocate KV cache
llm.wake_up(tags=["kv_cache"])
```

### Online Serving

Enable development mode endpoints:
```bash
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B \
  --enable-sleep-mode \
  --port 8000
```

**Level 1:**
```bash
curl -X POST 'http://localhost:8000/sleep?level=1'
curl -X POST 'http://localhost:8000/wake_up'
```

**Level 2 with weight update:**
```bash
curl -X POST 'http://localhost:8000/sleep?level=2'

# Wake weights only
curl -X POST 'http://localhost:8000/wake_up?tags=weights'

# Load new weights
curl -X POST 'http://localhost:8000/collective_rpc' \
  -H 'Content-Type: application/json' \
  -d '{"method":"reload_weights"}'

# Wake KV cache
curl -X POST 'http://localhost:8000/wake_up?tags=kv_cache'
```

## RLHF Weight Update Pattern

Sleep Mode enables safe in-place weight updates for RLHF training:

### Problem

During RLHF, trainer produces updated weights that must be loaded into inference engine. Naive approach:
```python
# BAD: Risk of OOM
llm.wake_up()  # Allocates weights + KV cache
llm.load_weights()  # Peak memory = old weights + new weights + KV cache
```

### Solution

Fine-grained wake-up avoids OOM:
```python
# GOOD: Minimize peak memory
llm.sleep(level=2)  # Discard old weights + KV cache

llm.wake_up(tags=["weights"])  # Allocate weights memory only

llm.collective_rpc("reload_weights")  # Load new weights in-place
# Peak memory = new weights (no KV cache yet)

llm.wake_up(tags=["kv_cache"])  # Allocate KV cache with new weights
# Now ready for inference
```

This pattern minimizes peak GPU memory usage during weight synchronization.

### Weight Transfer Integration

See [[RLHF with vLLM]] for full workflow:
1. Generate rollouts with current policy
2. Trainer computes updated weights
3. Sleep inference engine (level 2)
4. Transfer weights from trainer to inference (NCCL/IPC backends)
5. Wake weights, reload, wake KV cache
6. Resume generation with new policy

## API Reference

### Python API

**`LLM.sleep(level: int)`**
- `level=1`: Offload weights to CPU, discard KV cache
- `level=2`: Discard weights and KV cache

**`LLM.wake_up(tags: list[str] | None = None)`**
- `tags=None`: Wake all resources (default)
- `tags=["weights"]`: Wake weights only
- `tags=["kv_cache"]`: Wake KV cache only

Note: `is_sleeping` returns `True` until all components awake.

**`LLM.collective_rpc(method: str)`**
- `method="reload_weights"`: Load weights in-place (after waking weights with Level 2)

### HTTP Endpoints

Requires `VLLM_SERVER_DEV_MODE=1`:

- `POST /sleep?level={1|2}` — Put model to sleep
- `POST /wake_up` — Wake all resources
- `POST /wake_up?tags=weights` — Wake weights only
- `POST /wake_up?tags=kv_cache` — Wake KV cache only
- `POST /collective_rpc` — RPC call (body: `{"method":"reload_weights"}`)
- `GET /is_sleeping` — Check sleep status

Warning: These endpoints bypass authentication. Do not expose to untrusted users.

## Memory Savings

Typical memory breakdown (70B model, FP16):

| Component | Memory | Level 1 | Level 2 |
|-----------|--------|---------|---------|
| Model weights | ~140 GB | → CPU RAM | ✓ Freed |
| KV cache | ~20-40 GB | ✓ Freed | ✓ Freed |
| CUDA kernels | ~2-5 GB | Kept | Kept |
| Model buffers (RoPE, etc.) | ~100-500 MB | Kept | → CPU RAM |

**Level 1 savings**: 20-40 GB (KV cache only)
**Level 2 savings**: 160-180 GB (~90% of total)

CPU RAM requirement (Level 1): Must fit model weights (~140 GB for 70B model).

## Distributed Workloads

Sleep Mode works with multi-GPU parallelism:

**Tensor Parallelism (TP):**
- Each GPU sleeps/wakes independently
- Weights sharded across GPUs in CPU RAM (Level 1)

**Pipeline Parallelism (PP):**
- Each stage sleeps/wakes independently
- Coordinate across stages for full model sleep

**Data Parallelism (DP):**
- Each DP rank has independent sleep state
- Load balancer should drain requests before sleeping

## ROCm-Specific Considerations

ROCm uses chunked virtual memory allocation. Configure chunk size via:

```bash
export VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE=256  # MB (default: 256)
```

**Tuning:**
- Larger chunks → faster performance
- Too large → OOM during wake-up
- Recommended: power of 2 (128, 256, 512 MB)

If OOM occurs, reduce chunk size.

## Performance

**Sleep latency** (70B model, single GPU):
- Level 1: 5-10 seconds (CPU memory copy)
- Level 2: 1-2 seconds (discard only)

**Wake latency** (70B model):
- Level 1 full: 5-10 seconds (GPU ← CPU copy)
- Level 2 weights only: 20-30 seconds (reload from disk/network)
- Level 2 KV cache only: 1-2 seconds (allocate blocks)

Latencies scale with model size and GPU count.

## Limitations

1. **KV cache discarded**: Prefix cache benefits lost on wake-up
2. **In-flight requests**: Must be drained (or use [[Async RL]] pause/resume API)
3. **CPU RAM requirement** (Level 1): Must fit model weights
4. **Dev mode required**: HTTP endpoints need `VLLM_SERVER_DEV_MODE=1`
5. **ROCm chunking**: Requires tuning chunk size for large models

## Use Cases

### RLHF Training

**Sequential pipeline:**
```
[Inference (vLLM)] → [Training (PyTorch)] → [Inference (vLLM)] → ...
```

GPU timeline:
```
Inference active ████████░░░░░░░░████████
Training active  ░░░░░░░░████████░░░░░░░░
Sleep mode       ░░░░░░░░▓▓░░░░░░▓▓░░░░░░
```

Sleep Mode enables GPU sharing without model unload/reload overhead.

### Cost Optimization

Multi-tenant serving with bursty traffic:
- Sleep low-QPS models during idle periods
- Wake on demand (5-10s latency acceptable for cold start)
- Oversubscribe GPU capacity (sleep unused models)

### Model Switching

Single-GPU server:
```python
# Serve model A
llm_a = LLM("model-a", enable_sleep_mode=True)
# ... serve requests

# Switch to model B
llm_a.sleep(level=2)
llm_b = LLM("model-b")  # Now enough GPU memory
```

## Async RL Integration

See [[Async RL]] for advanced pause/resume workflow:

**Pause/Resume API** coordinates with Sleep Mode:
```python
# Pause generation (keep requests in queue)
await engine.pause_generation(mode="keep", clear_cache=True)

# Sleep engine
await engine.sleep(level=2)

# ... train model, get new weights

# Wake engine
await engine.wake_up(tags=["weights"])
await engine.collective_rpc("reload_weights")
await engine.wake_up(tags=["kv_cache"])

# Resume generation
await engine.resume_generation()
```

Difference:
- **Pause**: Freezes scheduler, keeps requests in queue
- **Sleep**: Frees GPU memory

Combined: Safe mid-flight weight updates.

## Best Practices

1. **Drain requests before sleep**: Avoid interrupting in-flight generation
2. **Monitor CPU RAM** (Level 1): Ensure enough space for weights
3. **Use tags for weight updates** (Level 2): Avoid OOM during reload
4. **Profile sleep/wake latency**: Measure end-to-end overhead
5. **Coordinate with load balancer**: Drain traffic before sleeping
6. **Test ROCm chunk size**: Start with 256 MB, tune if needed

## Cross-References

- [[RLHF with vLLM]] — Sleep Mode in RLHF training pipeline
- [[Async RL]] — Pause/resume API for mid-flight weight updates
- [[vLLM Engine]] — Engine sleep/wake implementation
- [[Tensor Parallelism]] — Distributed sleep/wake
- [[Data Parallelism]] — Per-rank sleep state management
- [[Prefix Caching]] — Prefix cache discarded on sleep

## Related

- **CUDA Unified Memory**: Alternative to explicit sleep (automatic paging)
- **Model Offloading**: CPU offload during inference (different from sleep)
- **vLLM Swap Space** (V0): Deprecated KV cache swapping (replaced by prefix caching)
