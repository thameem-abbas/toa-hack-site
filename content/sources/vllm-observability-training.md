---
title: vLLM Observability & Training Integration
type: source-summary
created: 2026-04-25
batch: 13
sources:
  - file:///tmp/vllm/docs/design/metrics.md
  - file:///tmp/vllm/docs/features/reasoning_outputs.md
  - file:///tmp/vllm/docs/features/sleep_mode.md
  - file:///tmp/vllm/docs/training/rlhf.md
  - file:///tmp/vllm/docs/training/async_rl.md
---

# vLLM Observability & Training Integration

Summary of vLLM's production observability features (Prometheus metrics system) and RLHF training integration (sleep mode, async RL, weight synchronization).

## Sources

1. **`docs/design/metrics.md`** — Comprehensive metrics design document
2. **`docs/features/reasoning_outputs.md`** — Reasoning models (DeepSeek-R1, Qwen3, etc.)
3. **`docs/features/sleep_mode.md`** — GPU memory release for training/inference interleaving
4. **`docs/training/rlhf.md`** — RLHF integration overview and ecosystem
5. **`docs/training/async_rl.md`** — Async RL pipelining with pause/resume API

## Key Concepts

### vLLM Metrics

**Categories:**
- Server-level: Gauges/Counters tracking engine state (running requests, KV cache usage)
- Request-level: Histograms for SLO monitoring (TTFT, ITL, E2E latency)

**Access:**
- `/metrics` Prometheus endpoint (vllm: prefix, model_name label)
- LoggingStatLogger (INFO logs every 5s: throughput, cache hit rate, request counts)

**Design principles:**
- Metrics collected in API server process (not engine core) to minimize overhead
- Monotonic timestamps for interval calculations (immune to NTP clock skew)
- Same-process timestamp comparison (monotonic clocks differ across processes)

**Event timeline:**
- `QUEUED` → `SCHEDULED` → `NEW_TOKENS` (prefill) → `NEW_TOKENS` (decode) → finished
- `PREEMPTED` event for KV cache eviction

**Calculated intervals:**
- Queue: QUEUED → SCHEDULED
- Prefill: SCHEDULED → first NEW_TOKENS
- Decode: first NEW_TOKENS → last NEW_TOKENS
- Inter-token: successive NEW_TOKENS
- TTFT: arrival_time (frontend) → first token
- E2E: arrival_time → final token

**Important metrics:**
- `vllm:num_requests_running/waiting` (Gauge)
- `vllm:kv_cache_usage_perc` (Gauge, 0-1)
- `vllm:prefix_cache_queries/hits` (Counter) → derive hit rate
- `vllm:time_to_first_token_seconds` (Histogram)
- `vllm:inter_token_latency_seconds` (Histogram)
- `vllm:e2e_request_latency_seconds` (Histogram)
- `vllm:request_prefill/decode_time_seconds` (Histogram)
- `vllm:generation_tokens_total` (Counter)
- `vllm:spec_decode_num_accepted/draft_tokens` (Counter)

**KV cache residency (sampled):**
- `vllm:kv_block_lifetime_seconds` — allocation → eviction
- `vllm:kv_block_idle_before_evict_seconds` — last access → eviction
- `vllm:kv_block_reuse_gap_seconds` — time between touches

Enable with `--kv-cache-metrics-sample`.

**Deprecated metrics:**
- `vllm:num_requests_swapped`, `vllm:cpu_cache_usage_perc` (V0 CPU swapping, removed in V1)
- `vllm:time_in_queue_requests` (duplicate of `request_queue_time_seconds`)

**Future work:**
- Parallel sampling metrics (`request_params_n`, `request_max_num_generation_tokens`)
- Speculative decoding acceptance rate as counters (not gauge)
- Autoscaling metrics (saturation detection: queue time > ITL)
- OpenTelemetry tracing (separate from metrics aggregation)

**Multi-process mode:**
- Metrics in API server process
- Multiprocess mode only when `--api-server-count > 1`
- Python/process metrics (GC, memory, FDs) unavailable in multiprocess

**Grafana dashboard:**
- Reference example with critical metrics (TTFT p95, ITL, cache usage, throughput)
- PR [#2316](https://github.com/vllm-project/vllm/pull/2316) for background

**Prometheus client:**
- Switched from aioprometheus to prometheus_client (PR [#2730](https://github.com/vllm-project/vllm/pull/2730))
- HTTP metrics via prometheus_fastapi_instrumentator (PR [#15657](https://github.com/vllm-project/vllm/pull/15657))

**Key PRs:**
- Issue [#3616](https://github.com/vllm-project/vllm/issues/3616): "Even Better Observability" roadmap
- Issue [#10582](https://github.com/vllm-project/vllm/issues/10582): V1 metrics implementation

### Reasoning Models

**Definition:**
Models that produce internal reasoning tokens (`<think>...</think>`) before final answer. Output includes `reasoning` field (thinking process) and `content` field (final answer).

**Supported models (11+):**
- DeepSeek R1 series (`deepseek_r1`)
- Qwen3 series (`qwen3`, thinking on by default)
- QwQ-32B (`deepseek_r1`)
- IBM Granite 3.2 (`granite`, thinking off by default)
- ERNIE-4.5, GLM-4.5, Holo2, Hunyuan A13B, MiniMax-M2

**Usage:**
```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --reasoning-parser deepseek_r1
```

**Thinking budget:**
- Limit reasoning tokens via `thinking_token_budget` parameter
- Token counting from `reasoning_start_str` to `reasoning_end_str`
- When budget exhausted, vLLM forces `reasoning_end_str` emission
- Configure boundary tokens via `--reasoning-config` JSON

Example:
```bash
vllm serve Qwen/Qwen3-0.6B \
  --reasoning-parser qwen3 \
  --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}'
```

Request with budget:
```json
{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [...],
  "thinking_token_budget": 10
}
```

**Server-level defaults:**
- `--default-chat-template-kwargs '{"enable_thinking": false}'` (disable for Qwen3)
- Request-level `extra_body` overrides server defaults

**Streaming:**
- Reasoning content in `delta.reasoning` field
- OpenAI client supports extra attributes (use `hasattr` to check)

**Tool calling:**
- Reasoning available when both reasoning parser and tool calling enabled
- Tools parsed from `content` only (not from `reasoning`)

**Structured outputs:**
- Reasoning models support constrained decoding (JSON, regex)
- Structured output engines skip reasoning content (Reasoner class detects `</think>`)

**Limitations:**
- Reasoning content only for `/v1/chat/completions` endpoint
- Some models don't support tool calling in thinking mode (e.g., DeepSeek-V3.1)

**Implementation:**
- `ReasoningParser`: extract reasoning from output (streaming and non-streaming methods)
- `Reasoner`: detect reasoning boundaries for structured output engines
- `ReasoningParserManager.register_lazy_module()` for new parsers

**Performance:**
- 2-10× TTFT increase (depends on reasoning token count)
- Thinking budget controls overhead
- Strong fit: math, code, logic; weak fit: simple queries

**Migration:**
- `reasoning_content` → `reasoning` (breaking change)

### Sleep Mode

**Purpose:**
Temporarily release GPU memory (model weights + KV cache) without stopping server. Enables GPU sharing between inference and training (RLHF use case).

**Sleep levels:**
- **Level 1** (shallow): Offload weights to CPU RAM, discard KV cache (fast wake, same model)
- **Level 2** (deep): Discard weights + KV cache (retain buffers like RoPE in CPU), ~90% GPU memory freed

**Benefits:**
- GPU memory release: 90%+ freed
- Fast resume: no full model reload (L1) or targeted weight reload (L2)
- Distributed support: works with TP/PP/DP
- Fine-grained control: wake weights or KV cache separately (L2)

**Platforms:**
CUDA, ROCm

**Usage (offline):**
```python
from vllm import LLM
llm = LLM("Qwen/Qwen3-0.6B", enable_sleep_mode=True)

# Level 1
llm.sleep(level=1)
llm.wake_up()

# Level 2 with weight update
llm.sleep(level=2)
llm.wake_up(tags=["weights"])
llm.collective_rpc("reload_weights")
llm.wake_up(tags=["kv_cache"])
```

**Usage (online):**
Requires `VLLM_SERVER_DEV_MODE=1`:
```bash
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B --enable-sleep-mode
curl -X POST 'http://localhost:8000/sleep?level=2'
curl -X POST 'http://localhost:8000/wake_up?tags=weights'
curl -X POST 'http://localhost:8000/collective_rpc' -d '{"method":"reload_weights"}'
curl -X POST 'http://localhost:8000/wake_up?tags=kv_cache'
```

**RLHF weight update pattern:**
Fine-grained wake avoids OOM:
```python
llm.sleep(level=2)
llm.wake_up(tags=["weights"])  # No KV cache yet
llm.collective_rpc("reload_weights")  # Load new weights in-place
llm.wake_up(tags=["kv_cache"])  # Now allocate KV cache
```

Minimizes peak memory: avoids (old weights + new weights + KV cache) spike.

**Memory savings (70B FP16):**
- Weights: ~140 GB → CPU (L1) or freed (L2)
- KV cache: ~20-40 GB → freed (both levels)
- Level 1: 20-40 GB saved, Level 2: 160-180 GB saved (~90%)

**Performance:**
- Sleep: 1-10s (L2 faster, L1 copies to CPU)
- Wake: 5-30s (L1: CPU→GPU copy, L2: reload from disk/network + allocate)

**ROCm chunking:**
Virtual memory via chunks, configure size:
```bash
export VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE=256  # MB (default)
```
Larger → faster, too large → OOM. Power of 2 recommended.

**Limitations:**
- KV cache (prefix cache) discarded
- In-flight requests must be drained
- CPU RAM requirement (L1): must fit weights
- Dev mode required for HTTP endpoints

**Use cases:**
- RLHF: GPU sharing between inference and training
- Cost optimization: sleep low-QPS models during idle
- Model switching: single-GPU server

**Integration:**
- See blog post: [vLLM Sleep Mode](https://blog.vllm.ai/2025/10/26/sleep-mode.html)

### RLHF with vLLM

**Overview:**
RLHF training loop: rollout generation (inference) → reward calculation → policy update (training) → weight sync → repeat. vLLM provides fast rollout generation.

**Supported frameworks (11+):**
TRL, OpenRLHF, verl, NeMo-RL, Unsloth, Prime-RL, Cosmos-RL, Open Instruct, SkyRL, PipelineRL, ms-swift

**Architecture patterns:**
1. **Sequential**: Generate → train → generate (GPUs idle during alternating phases)
2. **Async pipelined**: Overlap generation and training (higher GPU utilization, requires pause/resume)
3. **GPU sharing**: Inference and training share GPUs via sleep mode

**Weight synchronization:**
- **NCCL backend** (multi-GPU): GPU-to-GPU transfer via NCCL all-gather
- **IPC backend** (same-GPU): Shared memory via /dev/shm

See [Weight Transfer docs](https://github.com/vllm-project/vllm/tree/main/docs/training/weight_transfer) for details.

**Weight update with sleep mode:**
```python
inference_llm.sleep(level=2)
inference_llm.wake_up(tags=["weights"])
inference_llm.collective_rpc("reload_weights")
inference_llm.wake_up(tags=["kv_cache"])
```

**GRPO example (TRL):**
```python
from trl import GRPOTrainer
from vllm import LLM

rollout_llm = LLM("Qwen/Qwen3-4B", tensor_parallel_size=4, enable_sleep_mode=True)
trainer = GRPOTrainer(model=model, args=args, rollout_generator=rollout_llm)
trainer.train()
```

**Notebooks:**
- [TRL GRPO with vLLM](https://huggingface.co/learn/cookbook/grpo_vllm_online_training)
- [Unsloth Qwen3-4B GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)

**Performance optimization:**
- Speculative decoding for latency (EAGLE: 1.8-2.5×, n-gram: 1.2-1.8×)
- Quantization for memory (FP8: 2×, INT4: 4×)
- Prefix caching for system prompts (2-10× TTFT reduction)

**Multi-GPU:**
- TP: Sharded weight transfer via NCCL
- DP: Broadcast to all ranks (internal LB auto-coordinates)

**Monitoring:**
- `vllm:e2e_request_latency_seconds`, `time_to_first_token_seconds`
- `vllm:generation_tokens_total` (throughput)
- Custom metrics for sleep/wake latency

**Limitations:**
- Weight sync overhead: 5-30s per update
- Stale KV cache: prefix cache invalidated on weight update
- Async complexity: pause/resume requires careful queue management

### Async Reinforcement Learning

**Problem:**
Standard RL: generation and training sequential → GPUs idle during alternating phases.

**Solution:**
One-off pipelining: separate generation and training into parallel coroutines. Model generates new samples while training on previous data.

**Complication:**
Weights updated mid-flight while requests in progress.

**Pause/Resume API:**
```python
await engine.pause_generation(mode="keep", clear_cache=True)
# ... sync weights
await engine.resume_generation()
```

**Modes:**
- `"abort"`: Abort all in-flight requests, return partial results (default)
- `"wait"`: Wait for all requests to finish before pausing
- `"keep"`: Freeze requests in queue, resume when `resume_generation` called

**HTTP endpoints:**
- `POST /pause?mode=keep`
- `POST /resume`

**Data parallelism:**
- Internal LB (`data_parallel_backend="ray"`): single pause/resume call (auto-coordinates)
- External LB: must send pause/resume to every instance individually

**Typical async RL flow:**
1. Start generating rollouts
2. Trainer produces new weights → pause generation (`mode="keep"`)
3. Sync weights (NCCL/IPC)
4. Resume generation (in-flight requests continue with new weights)
5. Repeat

**Key insight:**
Requests paused with `mode="keep"` produce tokens from old weights before pause, new weights after resume. `clear_cache=True` discards KV cache (all tokens after resume use new weights).

**Example:**
[Async RLHF example](https://github.com/vllm-project/vllm/tree/main/examples/rl) with `vllm.AsyncLLMEngine`, NCCL weight transfer, pause/resume validation.

## Pages Created

- **[[vLLM Metrics]]** (architecture) — Prometheus metrics system, event timeline, KV cache residency, autoscaling use cases
- **[[Reasoning Models]]** (concept) — Chain-of-thought models, thinking budget control, 11+ supported models (DeepSeek-R1, Qwen3, etc.)
- **[[Sleep Mode]]** (technique) — GPU memory release, 2 sleep levels, RLHF weight update pattern, ROCm chunking
- **[[RLHF with vLLM]]** (technique) — Training integration, 11+ RL frameworks, weight synchronization (NCCL/IPC), async pipelining
- **[[vllm-observability-training]]** (source summary, this page)

## Cross-References

### vLLM Metrics
- [[vLLM Engine]] — Scheduler/worker that emits metrics
- [[V1 Architecture]] — AsyncLLM.output_handler_loop metrics collection
- [[Prefix Caching]] — Prefix cache hit/miss metrics
- [[Speculative Decoding]] — Draft acceptance metrics
- [[KV Cache]] — Cache usage and residency
- [[LoRA]] — Per-adapter request tracking
- [[Data Parallelism]] — Internal LB metric aggregation

### Reasoning Models
- [[Structured Outputs]] — Constrained decoding on final answer
- [[vLLM Engine]] — Reasoning parser integration
- [[Speculative Decoding]] — Incompatible (reasoning tokens unpredictable)
- [[Multi-Token Prediction]] — Some reasoning models support MTP (DeepSeek-V3)

### Sleep Mode
- [[RLHF with vLLM]] — Sleep mode in training pipeline
- Async RL — Pause/resume for mid-flight weight updates
- [[vLLM Engine]] — Engine sleep/wake implementation
- [[Tensor Parallelism]] — Distributed sleep/wake
- [[Prefix Caching]] — Prefix cache discarded on sleep

### RLHF with vLLM
- [[Sleep Mode]] — GPU memory release during training
- Async RL — Pause/resume API for weight updates
- [[Data Parallelism]] — Multi-rank weight sync
- [[Tensor Parallelism]] — Sharded weight transfer
- [[Speculative Decoding]] — Latency optimization for rollouts
- [[vLLM Metrics]] — Monitoring rollout performance

### Async RL
- [[RLHF with vLLM]] — RL training workflow
- [[Sleep Mode]] — Combined pause + sleep for weight updates
- [[vLLM Engine]] — Pause/resume implementation
- [[Data Parallelism]] — Internal vs external LB coordination

## Key Insights

1. **Metrics architecture**: V1 collects metrics in API server (not engine core) to minimize overhead; uses monotonic timestamps for interval calculations (immune to NTP skew); frontend calculates TTFT from arrival_time to account for input processing.

2. **Reasoning models**: 11+ model families supported via pluggable ReasoningParser system; thinking budget limits reasoning tokens to control latency/cost; structured outputs and tool calling compatible (reasoning skipped by constraint engines).

3. **Sleep Mode**: Level 1 (CPU offload) for same-model resume, Level 2 (discard) for weight updates; fine-grained wake (`tags=["weights"]` then `tags=["kv_cache"]`) avoids OOM during weight reload; ROCm chunking requires tuning for large models.

4. **RLHF integration**: 11+ RL frameworks use vLLM for rollouts; weight sync via NCCL (multi-GPU) or IPC (same-GPU); async pipelining overlaps generation and training but requires pause/resume coordination.

5. **Async RL**: `pause_generation(mode="keep")` freezes requests in queue; weight update during pause; `resume_generation()` continues requests with new weights; `clear_cache=True` discards stale KV cache.

## Related Concepts

- **OpenTelemetry**: Distributed tracing (separate from metrics)
- **Grafana**: Metrics visualization and alerting
- **Kubernetes HPA**: Autoscaling based on vLLM metrics
- **Chain-of-Thought**: User-provided reasoning prompts (all models)
- **DPO**: Offline alignment (no vLLM inference needed)
- **PPO/GRPO**: On-policy RL algorithms using vLLM for rollouts
