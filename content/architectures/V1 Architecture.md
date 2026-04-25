---
title: V1 Architecture
type: architecture
created: 2026-04-25
tags: [vllm, v1, scheduler, unified-architecture]
---

# V1 Architecture

Major re-architecture of vLLM's core systems (scheduler, KV cache manager, worker, sampler, API server) for simplicity, performance, and unified optimization support. V0 is fully deprecated as of 2026.

## Motivation

vLLM V0 successfully supported 200+ models and 5+ hardware platforms, but grew increasingly complex:
- Features developed independently led to fragmented codebase
- Prefill/decode separation made it hard to combine optimizations (chunked prefill + prefix caching + spec decode)
- Technical debt accumulated (complex scheduling logic, prefill/decode mode switching)
- CPU overhead limited throughput in high-concurrency scenarios

V1 retains V0's stable components (models, GPU kernels, utilities) but re-architects core systems.

## Design Goals

### 1. Simple, Modular, Easy-to-Hack Codebase

- Clear separation of concerns (API server, scheduler, execution)
- Minimal cross-component dependencies
- Easy to add new features (e.g., new scheduling policy = small change to scheduler)

### 2. High Performance with Near-Zero CPU Overhead

- Busy loop in engine core (no asyncio overhead in hot path)
- Multi-process architecture (API server, engine core, workers run in separate processes)
- ZMQ sockets for IPC (lower latency than gRPC/HTTP for local communication)
- CUDA graphs for fixed batch sizes (GPU kernel launch overhead eliminated)

### 3. Combine Key Optimizations

**V0 problem:** Features conflicted due to prefill/decode separation:
- Chunked prefill: splits prefill into chunks
- Prefix caching: reuses KV cache for shared prefixes
- Spec decode: runs multiple tokens per step

Combining these required complex mode switching and scheduler state machines.

**V1 solution:** Unified scheduler treats prompt and output tokens identically (see #Unified Scheduler).

### 4. Zero Configs (Enable by Default)

- Chunked prefill: enabled by default (V0: conditionally enabled)
- Prefix caching: enabled when beneficial (V0: manual flag)
- CUDA graphs: automatic capture for observed batch sizes (V0: manual config)

## Unified Scheduler

The V1 scheduler is the most significant architectural change.

### Token Budget Allocation

**V0 approach:** Separate prefill and decode queues. Scheduler fills prefill batch, then decode batch. Switching between modes adds complexity.

**V1 approach:** Single unified queue. Scheduler allocates token budget as dictionary:

```python
schedule = {
    "request_1": 128,  # 128 tokens (could be prefill, decode, or mix)
    "request_2": 50,   # 50 tokens
    "request_3": 1,    # 1 token
}
```

**Benefits:**
- Prefill and decode treated the same (no mode switching)
- Chunked prefill trivial to implement (allocate partial prefill tokens)
- Prefix caching works naturally (skip cached tokens, allocate only new tokens)
- Spec decode fits cleanly (allocate multiple output tokens)

### Scheduling Policies

V1 supports multiple policies via `--scheduling-policy`:

- **FCFS (First-Come, First-Served):** Default. Requests processed in arrival order.
- **Priority:** Requests have assigned priorities. Higher priority processed first. FCFS as tie-breaker.

**Future:** Preemption policies (e.g., preempt low-priority requests to serve high-priority).

### KV Cache Management

Scheduler maintains:
- **Block allocator:** Manages free/occupied blocks (see [[PagedAttention]])
- **Request state:** Which blocks each request owns, how many tokens cached
- **Prefix cache index:** Hash of prompt → cached blocks (see [[Prefix Caching]])

## Differences from V0

### Chunked Prefill

**V0:** Conditionally enabled based on model characteristics (e.g., disabled for small models).

**V1:** Enabled by default whenever possible.

**Impact:** More consistent TTFT (time to first token) across requests. Long prefills don't block short decodes.

### CUDA Graphs

**V0:** Conservative memory allocation for CUDA graphs.

**V1:** More aggressive CUDA graph capture.

**Impact:** Higher GPU memory usage, lower kernel launch overhead. Trade memory for latency.

**Mitigation:** Adjust `--max-num-seqs` or `--max-num-batched-tokens` if OOM occurs.

### Logprobs Semantics

**V0:** Logprobs computed after applying all logit processors (temperature, top-p, penalties).

**V1 (default):** Logprobs computed from raw model output (before logit processors).

**Modes (via `--logprobs-mode`):**
- `raw_logprobs` (default): Raw model logprobs (before processors)
- `processed_logprobs`: After all processors (V0 behavior)
- `raw_logits`: Raw logits (before softmax)
- `processed_logits`: Processed logits (before softmax)

**Why changed:** Performance. Applying processors just to compute logprobs adds overhead. Most users want raw logprobs.

**Migration:** Use `--logprobs-mode processed_logprobs` for V0 behavior.

### Prompt Logprobs with Prefix Caching

**V0:** Cached prompt logprobs alongside KV cache.

**V1:** Does not cache prompt logprobs. Requests requiring prompt logprobs bypass prefix cache and recompute full prefill.

**Why changed:** Simplicity. Caching logprobs added significant complexity to cache manager.

**Impact:** Requests with `logprobs=True` don't benefit from prefix caching. Use `echo=False` if you don't need prompt logprobs.

## Feature Support Matrix

> [!note] Support Levels
> - 🟢 Functional: Fully operational, performance comparable to or better than V0
> - 🟡 In Progress: Planned, open PRs/RFCs exist
> - 🔴 Removed: Dropped from V1, only re-introduce if strong demand

### Hardware

| Platform | Support |
|----------|---------|
| NVIDIA GPUs | 🟢 |
| AMD GPUs (ROCm) | 🟢 |
| Intel GPUs | 🟢 |
| TPU | 🟢 |
| CPU | 🟢 |

**Plugins:** Additional platforms via [vllm-ascend](https://github.com/vllm-project/vllm-ascend), [vllm-spyre](https://github.com/vllm-project/vllm-spyre), [vllm-gaudi](https://github.com/vllm-project/vllm-gaudi), [vllm-openvino](https://github.com/vllm-project/vllm-openvino)

### Models

| Type | Support | Notes |
|------|---------|-------|
| Decoder-only | 🟢 | Llama, GPT, Mistral, Qwen, etc. |
| Encoder-decoder | 🟢 (Whisper), 🔴 (others) | BART/Florence via [bart-plugin](https://github.com/vllm-project/bart-plugin) |
| Pooling | 🟢 | Embedding models. Prefix caching + chunked prefill for last-pooling. |
| Mamba | 🟢 | Mamba-1, Mamba-2, hybrid models. No prefix caching yet. |
| Multimodal | 🟢 | Vision-language, audio models |

### Features

| Feature | Support | Notes |
|---------|---------|-------|
| Prefix Caching | 🟢 | Automatic for shared prompts |
| Chunked Prefill | 🟢 | Default enabled |
| LoRA | 🟢 | Multi-adapter serving |
| FP8 KV Cache | 🟢 | Quantized KV cache |
| Spec Decode | 🟢 | EAGLE, Medusa, draft models |
| Structured Output | 🟢 | Outlines/guidance backends |
| Concurrent Partial Prefills | 🟡 | [RFC #14003](https://github.com/vllm-project/vllm/issues/14003) |

### Removed Features

#### best_of Sampling

**V0:** Generate `best_of` sequences, return best by log probability.

**V1:** Removed due to low usage. [RFC #13361](https://github.com/vllm-project/vllm/issues/13361)

**Alternative:** Run multiple requests with different `temperature`/`top_p`, rank externally.

#### Per-Request Logits Processors

**V0:** Custom per-request functions to adjust logits (e.g., ban certain tokens for specific request).

**V1:** Removed. Global logits processors only (set at engine startup). [RFC #17799](https://github.com/vllm-project/vllm/issues/17799)

**Alternative:** Use structured output constraints or adjust request at application layer.

#### GPU ↔ CPU KV Cache Swapping

**V0:** Swap KV cache to CPU when GPU memory full, swap back when needed.

**V1:** Removed. Unified scheduler + better memory management eliminates need.

**Why:** Swapping added significant complexity. V1's efficient scheduling makes preemption rare.

#### Request-level Structured Output Backend

**V0:** Each request could specify outlines vs. guidance backend.

**V1:** Global backend selection only. Fallback to alternate backend if primary fails.

## Process Architecture

See [[vLLM Engine#V1 Process Architecture]] for full details.

**Key points:**
- API server, engine core, GPU workers run in separate processes
- ZMQ sockets for IPC (API server ↔ engine core)
- Engine core runs busy loop (low latency)
- GPU workers load model shards, execute forward passes
- DP coordinator (optional) for data parallelism load balancing

**Example process counts:**
- 4 GPUs, TP=4: 6 processes (1 API + 1 Core + 4 Workers)
- 8 GPUs, TP=2, DP=4: 17 processes (4 API + 4 Cores + 8 Workers + 1 Coordinator)

## Performance Improvements

V1 shows significant gains over V0, especially for:
- **Long context:** Chunked prefill + unified scheduler reduce prefill latency variance
- **High concurrency:** Near-zero CPU overhead allows more concurrent requests
- **Prefix-heavy workloads:** Automatic prefix caching (no manual flag)

**Benchmarks:** See [vLLM V1 blog post](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) (published Jan 27, 2025).

## Migration from V0

> [!warning] V0 Fully Deprecated
> V0 is no longer supported. All users must migrate to V1.

### Breaking Changes

1. **Logprobs semantics:** Use `--logprobs-mode processed_logprobs` for V0 behavior
2. **Removed features:** `best_of`, per-request logits processors, KV swapping, request-level structured output backend
3. **CUDA graph memory:** May need to reduce `--max-num-seqs` if OOM
4. **Prompt logprobs with prefix caching:** No longer cached. Disable `logprobs` if not needed.

### Compatible Changes

- API surface unchanged (OpenAI-compatible API same endpoints)
- Model support unchanged (same HuggingFace models)
- Deployment unchanged (same `vllm serve` command)

## Cross-References

**Concepts:**
- [[Chunked Prefill]] — breaking long prefills into chunks
- [[Prefix Caching]] — KV cache reuse for shared prefixes
- [[Continuous Batching]] — dynamic request scheduling
- [[PagedAttention]] — block-based KV cache
- [[Speculative Decoding]] — multi-token per step
- [[CUDA Graphs]] — kernel launch optimization

**Architectures:**
- [[vLLM Engine]] — process architecture, LLMEngine, workers

**Tools:**
- [[vLLM]] — high-level tool page

## See Also

- [vLLM V1 Blog Post](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)
- [V1 User Guide](https://docs.vllm.ai/en/stable/usage/v1_guide.html)
- [V0 Deprecation RFC #18571](https://github.com/vllm-project/vllm/issues/18571)
