---
title: RLHF with vLLM
type: technique
created: 2026-04-25
tags: [rlhf, training, alignment, weight-transfer, async-rl]
---

# RLHF with vLLM

Reinforcement Learning from Human Feedback (RLHF) is a technique for fine-tuning language models using human preference data to align model outputs with desired behaviors. vLLM serves as a high-performance inference backend for RLHF training, enabling fast rollout generation while the trainer optimizes policy weights.

## Overview

**RLHF Training Loop:**

1. **Rollout generation**: Policy model generates completions for prompts (inference)
2. **Reward calculation**: Reward model scores completions based on preferences
3. **Policy update**: Trainer computes gradients and updates policy weights (training)
4. **Weight sync**: New weights transferred to inference engine
5. Repeat

**vLLM's role**: Fast rollout generation (step 1) using optimized inference engine.

## Supported RL Frameworks

vLLM integrates with 11+ open-source RL libraries:

| Framework | Description | vLLM Integration |
|-----------|-------------|------------------|
| [TRL](https://github.com/huggingface/trl) | HuggingFace RL library | GRPO, PPO, DPO |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Ray-based distributed RLHF | PPO, GRPO |
| [verl](https://github.com/volcengine/verl) | ByteDance RLHF framework | PPO, GRPO |
| [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) | NVIDIA NeMo RL toolkit | PPO |
| [Unsloth](https://github.com/unslothai/unsloth) | Fast fine-tuning library | GRPO |
| [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl) | Distributed RL training | PPO |
| [Cosmos-RL](https://github.com/nvidia-cosmos/cosmos-rl) | NVIDIA Cosmos RL | PPO |
| [Open Instruct](https://github.com/allenai/open-instruct) | AllenAI instruction tuning | RLHF |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) | Multi-node RL | PPO |
| [PipelineRL](https://github.com/ServiceNow/PipelineRL) | Pipelined RL training | PPO |
| [ms-swift](https://github.com/modelscope/ms-swift) | ModelScope SWIFT | RLHF |

Most frameworks use vLLM for rollout generation, PyTorch/DeepSpeed for policy updates.

## Architecture Patterns

### Sequential (Standard)

Generation and training happen sequentially:

```
[Generate rollouts (vLLM)] → [Train policy (PyTorch)] → [Generate rollouts (vLLM)] → ...
```

**Pros:**
- Simple coordination
- No GPU contention

**Cons:**
- GPUs idle during alternating phases
- Lower throughput

**GPU utilization:**
```
Inference GPUs ████████░░░░░░░░████████░░░░░░░░
Training GPUs  ░░░░░░░░████████░░░░░░░░████████
```

### Async Pipelined (Advanced)

Generation and training overlap via async coroutines:

```
[Generate batch 1] → [Generate batch 2] → [Generate batch 3] → ...
       ↓                    ↓                    ↓
       [Train batch 1] → [Train batch 2] → [Train batch 3] → ...
```

**Pros:**
- Higher GPU utilization
- Increased throughput

**Cons:**
- Complex weight synchronization (weights updated mid-flight)
- Requires pause/resume API

See Async RL for implementation details.

### GPU Sharing (RLHF-Specific)

Inference and training share same GPUs via [[Sleep Mode]]:

```
Inference (vLLM) active  ████████░░░░░░░░████████
Training (PyTorch) active ░░░░░░░░████████░░░░░░░░
vLLM sleep mode          ░░░░░░░░▓▓░░░░░░▓▓░░░░░░
```

**Workflow:**
1. vLLM generates rollouts
2. vLLM sleeps (level 2), frees 90% GPU memory
3. PyTorch trains on freed GPUs
4. PyTorch updates weights
5. vLLM wakes, reloads weights, resumes inference

**Pros:**
- No separate GPU pools needed
- Cost-effective for small-scale RLHF

**Cons:**
- Sleep/wake overhead (5-30s per cycle)
- Not suitable for high-frequency updates

## Weight Synchronization

### Challenge

Policy weights updated by trainer must be transferred to vLLM inference engine:

**Problem:**
- Trainer and inference run on different GPUs/processes
- Weights must be synced without stopping inference
- Multi-GPU models require consistent cross-GPU updates

### Solutions

vLLM supports two weight transfer backends (see [Weight Transfer docs](https://github.com/vllm-project/vllm/tree/main/docs/training/weight_transfer)):

#### 1. NCCL Backend (Multi-GPU)

Use NVIDIA NCCL for GPU-to-GPU weight transfer:

**Setup:**
```python
from vllm import LLM
from vllm.distributed import init_distributed_environment

# Trainer side
trainer_llm = LLM(model="meta-llama/Llama-3-8B", tensor_parallel_size=4)

# Inference side
inference_llm = LLM(model="meta-llama/Llama-3-8B", tensor_parallel_size=4)

# Transfer weights (NCCL all-gather)
inference_llm.load_weights_from_trainer(trainer_llm)
```

**When to use:**
- Multi-GPU models (TP > 1)
- Cross-node weight transfer (InfiniBand)
- High bandwidth requirements

#### 2. IPC Backend (Same-GPU)

Use shared memory (inter-process communication) for same-GPU weight sharing:

**Setup:**
```python
# Enable IPC backend
inference_llm = LLM(
    model="meta-llama/Llama-3-8B",
    weight_transfer_backend="ipc"
)

# Weights shared via /dev/shm
trainer_llm.sync_weights_to_inference(inference_llm)
```

**When to use:**
- Single-GPU models
- Trainer and inference on same node
- Low latency requirements

### Weight Update Pattern with Sleep Mode

Avoid OOM during weight reload (Level 2 sleep):

```python
# 1. Sleep inference engine
inference_llm.sleep(level=2)  # Discard old weights + KV cache

# 2. Transfer weights from trainer
# (weights stored in CPU RAM or network buffer)

# 3. Wake weights memory only
inference_llm.wake_up(tags=["weights"])

# 4. Load new weights in-place
inference_llm.collective_rpc("reload_weights")

# 5. Wake KV cache
inference_llm.wake_up(tags=["kv_cache"])

# 6. Resume inference with new policy
```

This pattern minimizes peak GPU memory (avoids storing old + new weights simultaneously).

## Async RL Workflow

See Async RL for full details. Summary:

**1. Pause generation (keep requests in queue):**
```python
await inference_engine.pause_generation(mode="keep", clear_cache=True)
```

**2. Sleep engine (free GPU memory):**
```python
await inference_engine.sleep(level=2)
```

**3. Train policy (weights updated):**
```python
trainer.compute_gradients(batch)
trainer.update_weights()
```

**4. Sync weights to inference:**
```python
inference_engine.wake_up(tags=["weights"])
inference_engine.collective_rpc("reload_weights")
inference_engine.wake_up(tags=["kv_cache"])
```

**5. Resume generation:**
```python
await inference_engine.resume_generation()
```

**Key insight**: Requests paused with `mode="keep"` continue generating with new weights after resume. `clear_cache=True` ensures KV cache doesn't contain stale prefixes.

## Example: GRPO with TRL

Group Relative Policy Optimization (GRPO) with HuggingFace TRL:

```python
from trl import GRPOTrainer
from vllm import LLM

# Initialize vLLM for rollout generation
rollout_llm = LLM(
    model="Qwen/Qwen3-4B",
    tensor_parallel_size=4,
    enable_sleep_mode=True,
)

# Initialize TRL trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    rollout_generator=rollout_llm,  # Use vLLM for generation
)

# Training loop (TRL handles weight sync)
trainer.train()
```

See notebooks:
- [TRL GRPO with vLLM](https://huggingface.co/learn/cookbook/grpo_vllm_online_training)
- [Unsloth Qwen3-4B GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)

## Performance Optimization

### Rollout Generation

**Use [[Speculative Decoding]] for latency:**
- EAGLE: 1.8-2.5× speedup, low memory overhead
- N-gram: 1.2-1.8× speedup, zero overhead (good for high QPS)

**Use [[Quantization]] for memory:**
- FP8: 2× memory reduction, minimal accuracy loss
- INT4: 4× memory reduction, small accuracy loss

**Use [[Prefix Caching]]:**
- Cache system prompts (e.g., "You are a helpful assistant")
- 2-10× TTFT reduction for shared prefixes

### Weight Transfer

**NCCL optimization:**
- Use GPUDirect RDMA for cross-node transfer (reduces CPU bottleneck)
- Pipeline weight transfer with training (overlap communication)

**IPC optimization:**
- Pre-allocate shared memory buffers
- Use zero-copy semantics (mmap)

### Training Throughput

**Gradient accumulation:**
- Smaller batches per rollout generation
- Accumulate gradients across multiple batches
- Reduces peak memory during training

**Mixed precision:**
- Train in BF16 (reduced memory, faster compute)
- Keep policy in FP16/BF16 for inference quality

## Multi-GPU Deployment

### Tensor Parallelism

Policy model sharded across GPUs:

```python
# Trainer (TP=4)
trainer_llm = LLM(model="meta-llama/Llama-3-70B", tensor_parallel_size=4)

# Inference (TP=4)
inference_llm = LLM(model="meta-llama/Llama-3-70B", tensor_parallel_size=4)

# NCCL weight transfer (sharded all-gather)
inference_llm.load_weights_from_trainer(trainer_llm)
```

Each GPU holds 1/4 of weights. NCCL ensures consistent cross-GPU updates.

### Data Parallelism

Multiple DP ranks for throughput scaling:

```python
# 4 DP ranks × 2 TP size = 8 GPUs
inference_llm = LLM(
    model="meta-llama/Llama-3-8B",
    tensor_parallel_size=2,
    data_parallel_size=4,
)
```

Weight sync must broadcast to all DP ranks. Use internal load balancer (`data_parallel_backend="ray"`) for automatic coordination.

### Hybrid: TP + DP

Large-scale RLHF (e.g., Llama-70B):

```
8 nodes × 8 GPUs/node = 64 GPUs
TP=8 (within-node), DP=8 (across-node)
```

Weight sync:
1. Trainer updates policy on dedicated GPUs
2. NCCL broadcast to DP rank 0
3. DP rank 0 all-gathers to other DP ranks

## Monitoring

### vLLM Metrics

Track RLHF-specific metrics via [[vLLM Metrics]]:

**Rollout latency:**
- `vllm:e2e_request_latency_seconds` - E2E generation time
- `vllm:time_to_first_token_seconds` - TTFT (critical for RL responsiveness)

**Throughput:**
- `vllm:generation_tokens_total` - Total tokens generated
- `rate(vllm:generation_tokens_total[1m])` - Tokens/sec

**Weight sync timing:**
- Custom metrics for sleep/wake latency
- Weight transfer duration

**Resource usage:**
- `vllm:kv_cache_usage_perc` - KV cache saturation
- GPU memory usage (NVIDIA-SMI)

## Limitations

1. **Weight sync overhead**: 5-30s per update (depends on model size, backend)
2. **Stale KV cache**: Prefix cache invalidated on weight update (loses benefits)
3. **Async complexity**: Pause/resume requires careful request queue management
4. **Multi-DP coordination**: External load balancer requires manual pause/resume per rank

## Best Practices

1. **Profile weight sync latency**: Measure end-to-end overhead (sleep → transfer → wake)
2. **Batch rollout generation**: Amortize inference startup cost
3. **Use NCCL for TP models**: IPC only suitable for single-GPU
4. **Monitor KV cache hit rate**: Prefix caching benefits lost on weight update
5. **Test async pipeline**: Validate request continuity across weight updates
6. **Choose appropriate sleep level**: Level 1 if same model, Level 2 if weights updated
7. **Coordinate with load balancer**: Drain traffic before weight sync

## Use Cases

**Strong fit:**
- Online RLHF (PPO, GRPO) - frequent weight updates
- Instruction fine-tuning - alignment with human preferences
- Constitutional AI - iterative refinement

**Weak fit:**
- Offline RL (no inference during training)
- Pre-training (no inference backend needed)

## Cross-References

- [[Sleep Mode]] — GPU memory release during training phase
- Async RL — Pause/resume API for mid-flight weight updates
- [[vLLM Engine]] — Inference backend for rollout generation
- [[Data Parallelism]] — Multi-rank weight synchronization
- [[Tensor Parallelism]] — Sharded weight transfer via NCCL
- [[Speculative Decoding]] — Latency optimization for rollouts
- [[Prefix Caching]] — System prompt caching (invalidated on weight update)
- [[vLLM Metrics]] — Monitoring rollout generation performance

## Related

- **DPO (Direct Preference Optimization)**: Offline alignment (no vLLM needed)
- **Reward Modeling**: Training reward model (separate from policy inference)
- **PPO (Proximal Policy Optimization)**: On-policy RL algorithm (uses vLLM for rollouts)
- **GRPO (Group Relative Policy Optimization)**: Group-based preference learning
