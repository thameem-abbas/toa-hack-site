---
title: vLLM Engine
type: architecture
created: 2026-04-25
tags: [vllm, inference-engine, distributed-serving]
---

# vLLM Engine

Core inference engine architecture for vLLM, managing request scheduling, KV cache, distributed execution, and tokenization/detokenization.

## Overview

The vLLM engine is a multi-component system that orchestrates LLM inference from request intake through output generation. The V1 architecture separates concerns across multiple processes: API serving, scheduling, and GPU execution run in distinct processes communicating via ZMQ sockets.

## V1 Process Architecture

vLLM V1 uses a multi-process design to maximize throughput and minimize CPU overhead. Each process has a specific role in the inference pipeline.

### API Server Process

**Purpose:** Handle HTTP requests, input processing, and output streaming.

**Count:** 1 by default; scales to `DP` (data parallel size) automatically. Manual override via `--api-server-count`.

**Responsibilities:**
- Accept HTTP requests (OpenAI-compatible API)
- Tokenization of input prompts
- Multi-modal data loading (images, audio, etc.)
- Stream results back to clients
- ZMQ communication with engine cores (many-to-many topology)

**Configuration:**
- `VLLM_MEDIA_LOADING_THREAD_COUNT` (default: 8) controls media loading threads per API server

**Code:** `vllm/entrypoints/openai/api_server.py`, `vllm/v1/utils.py`

### Engine Core Process

**Purpose:** Schedule requests, manage KV cache, coordinate GPU workers.

**Count:** 1 per data parallel rank (e.g., DP=4 → 4 engine cores)

**Responsibilities:**
- Run unified scheduler (see [[V1 Architecture#Unified Scheduler]])
- Allocate/deallocate KV cache blocks
- Dispatch work to GPU worker processes
- Maintain request state machine
- Run busy loop for low latency

**Code:** `vllm/v1/engine/core.py`, `vllm/v1/engine/utils.py`

### GPU Worker Processes

**Purpose:** Load model weights and execute forward passes.

**Count:** 1 per GPU. Total workers = `DP × PP × TP`

**Responsibilities:**
- Load model shard weights (sharding happens at initialization, see [[#Class Hierarchy]])
- Execute model forward passes
- Manage GPU memory (CUDA graphs, workspace buffers)
- Communicate with owning engine core

**Code:** `vllm/v1/executor/multiproc_executor.py`, `vllm/v1/worker/gpu_worker.py`

### DP Coordinator Process (Conditional)

**Purpose:** Load balancing across data parallel ranks.

**Count:** 1 if DP > 1, else 0

**Responsibilities:**
- Distribute requests across DP ranks
- Coordinate synchronized forward passes for MoE models
- Balance request queues

**Code:** `vllm/v1/engine/coordinator.py`

### Process Count Formula

For `N` GPUs, tensor parallel size `TP`, pipeline parallel size `PP`, data parallel size `DP`, API server count `A`:

```
Total = A + DP + N + (1 if DP > 1 else 0)

where N = DP × PP × TP
```

**Examples:**

| Config | GPUs | TP | PP | DP | Processes | Breakdown |
|--------|------|----|----|----|-----------| ----------|
| Single-node, 4 GPUs | 4 | 4 | 1 | 1 | 6 | 1 API + 1 Core + 4 Workers |
| Multi-DP, 8 GPUs | 8 | 2 | 1 | 4 | 17 | 4 API + 4 Cores + 8 Workers + 1 Coordinator |

See V1 architecture diagrams in `docs/assets/design/arch_overview/`.

## LLMEngine

`LLMEngine` is the synchronous core orchestrator. It coordinates four stages:

1. **Input Processing:** Tokenization via HuggingFace tokenizer
2. **Scheduling:** Select requests to process in each step (see [[Continuous Batching]])
3. **Model Execution:** Distributed forward pass across workers
4. **Output Processing:** Detokenization (token IDs → text)

**Modes:**
- **Offline:** `LLM` class (`vllm/entrypoints/llm.py`) wraps `LLMEngine` for batch inference
- **Online:** `AsyncLLMEngine` wraps `LLMEngine` with asyncio event loop for concurrent request handling

**Code:** `vllm/engine/llm_engine.py`

### AsyncLLMEngine

Asynchronous wrapper around `LLMEngine`. Runs background loop to process requests concurrently, stream outputs to clients.

**Used by:**
- OpenAI-compatible API server (`vllm serve`)
- Demo API server (`vllm/entrypoints/api_server.py`)

**Code:** `vllm/engine/async_llm_engine.py`

## Worker

A worker is a process controlling one accelerator (GPU/TPU/etc.). Workers are identified by:
- `rank`: global orchestration index
- `local_rank`: device index on current node

**Example:** TP=2, PP=2 → 4 workers total
- Worker 0: rank=0, local_rank=0
- Worker 1: rank=1, local_rank=1
- Worker 2: rank=2, local_rank=0 (different node)
- Worker 3: rank=3, local_rank=1 (different node)

## Model Runner

Each worker has one model runner object.

**Responsibilities:**
- Load model from HuggingFace checkpoint
- Prepare input tensors (reshape, pad, slice for current batch)
- Capture CUDA graphs for fixed batch sizes
- Execute model forward pass
- Return logits/hidden states

**Code:** `vllm/worker/model_runner.py` (V0), `vllm/v1/worker/gpu_model_runner.py` (V1)

## Model

The actual `torch.nn.Module` instance inside each model runner.

**Key points:**
- vLLM supports 200+ model architectures (decoder-only, encoder-decoder, vision-language, etc.)
- See [[V1 Architecture#Models]] for model type support matrix
- Model initialization signature is uniform across all models (see [[#Class Hierarchy]])

## Class Hierarchy

vLLM's class hierarchy enforces three design principles:

### 1. Extensibility via VllmConfig

All components accept a single `VllmConfig` object containing engine-level global state. Benefits:
- Deep class hierarchies don't need constructor signature changes when adding new features
- Components access only the config fields they need
- Example: adding a new feature touching only model runner requires changing only `VllmConfig` class

**Structure:**
```python
VllmConfig = {
    model_config,
    cache_config,
    parallel_config,
    scheduler_config,
    device_config,
    lora_config,
    quant_config,
    ...
}
```

### 2. Uniformity via Keyword-Only Constructor

All vLLM models share a single constructor signature:

```python
def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

Benefits:
- Model runner doesn't need model-specific initialization logic
- Enables vision-language model composition (vision + language models composed via uniform interface)
- Out-of-tree models can use shim pattern for backwards compatibility

### 3. Sharding and Quantization at Initialization

Model weight transformations (tensor parallelism sharding, quantization) happen **during** initialization, not after.

**Why:** Memory efficiency for large models.

**Example:** 405B model (810GB weights) on 16×H100 (80GB each)
- **Post-init sharding (bad):** Each GPU loads full 810GB → OOM
- **Init-time sharding (good):** Each GPU loads only its 50GB shard → fits in memory

**Mechanism:**
- `prefix` argument indicates model position in checkpoint state dict (e.g., `""` for top-level, `"vision"` for vision submodule)
- Enables non-uniform quantization (different parts quantized differently)

## Entrypoints

### LLM Class (Offline Inference)

Python API for batch inference without a server.

**Usage:**
```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(prompts, SamplingParams(temperature=0.8))
```

**Code:** `vllm/entrypoints/llm.py`

### OpenAI-Compatible API Server (Online Serving)

Start with `vllm serve <model>` or `python -m vllm.entrypoints.openai.api_server`.

**Code:** `vllm/entrypoints/openai/api_server.py`, `vllm/entrypoints/cli/main.py`

**Deprecation:** Direct use of `python -m vllm.entrypoints.openai.api_server` is deprecated; prefer `vllm serve`.

## Cross-References

**Concepts:**
- [[PagedAttention]] — KV cache memory management
- [[KV Cache]] — attention caching mechanics
- [[Tensor Parallelism]] — model layer sharding
- [[Pipeline Parallelism]] — model stage pipelining
- [[Data Parallelism]] — request-level parallelism
- [[Continuous Batching]] — dynamic request scheduling
- [[Chunked Prefill]] — breaking long prefills into chunks
- [[Prefix Caching]] — reusing computed KV cache for shared prefixes

**Architectures:**
- [[V1 Architecture]] — V1 design goals, unified scheduler, differences from V0

**Tools:**
- [[vLLM]] — high-level tool page

**Papers:**
- [[Efficient Memory Management for Large Language Model Serving with PagedAttention]] — foundational paper

## See Also

- [vLLM Architecture Docs](https://docs.vllm.ai/en/stable/design/arch_overview.html)
- [Class Hierarchy Diagram](https://github.com/vllm-project/vllm/blob/main/docs/assets/design/hierarchy.png)
