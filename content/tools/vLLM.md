---
title: vLLM
type: tool
created: 2026-04-25
tags: [llm-serving, inference-engine, openai-api]
url: https://github.com/vllm-project/vllm
---

# vLLM

High-throughput, memory-efficient LLM serving engine with OpenAI-compatible API. Origin: UC Berkeley Sky Computing Lab. 2000+ contributors as of 2026.

## Overview

vLLM is a production-grade serving engine optimized for LLM inference throughput. It achieves 2-4× higher throughput than comparable systems through:
- [[PagedAttention]]: virtual memory for KV cache (60-80% memory waste reduction)
- [[Continuous Batching]]: dynamic request scheduling
- [[Chunked Prefill]]: breaking long prompts into chunks to reduce latency variance
- [[Prefix Caching]]: automatic reuse of KV cache for shared prefixes
- [[CUDA Graphs]]: kernel launch overhead elimination

## Key Capabilities

### Memory Management

**PagedAttention** (see [[PagedAttention]]):
- Block-based KV cache allocation (like OS virtual memory)
- Near-zero memory fragmentation
- Enables high request concurrency (more requests fit in GPU memory)

**Prefix Caching** (see [[Prefix Caching]]):
- Automatic detection of shared prefixes (system prompts, few-shot examples)
- Copy-on-write semantics for cached blocks
- Reduces redundant computation for repeated prompts

### Batching & Scheduling

**Continuous Batching** (see [[Continuous Batching]]):
- Add/remove requests from batch mid-flight
- No wasted compute on padding (unlike static batching)
- Higher GPU utilization

**Chunked Prefill** (see [[Chunked Prefill]]):
- Long prefills broken into chunks (e.g., 512 tokens per chunk)
- Interleave prefill chunks with decode tokens
- Reduces time-to-first-token (TTFT) variance

**Unified Scheduler (V1)** (see [[V1 Architecture#Unified Scheduler]]):
- Treats prompt and output tokens identically
- Token budget allocation: `{request_id: num_tokens}`
- Enables clean integration of chunked prefill + prefix caching + spec decode

### Model Support

**200+ model architectures:**
- Decoder-only: Llama, Mistral, GPT, Qwen, Phi, DeepSeek
- Encoder-decoder: Whisper (native), BART/Florence (plugin)
- Vision-language: LLaVA, Qwen-VL, Phi-3-Vision, InternVL
- Mamba: Mamba-1, Mamba-2, hybrid models (Jamba, Zamba)
- Pooling: Embedding models (BERT, sentence-transformers)

See [[V1 Architecture#Models]] for full support matrix.

### Quantization

**Supported methods:**
- **FP8:** W8A8 (weights + activations), KV cache quantization
- **INT8:** W8A8 via SmoothQuant
- **INT4:** GPTQ, AWQ
- **MXFP4:** Microscaling FP4 (see [[llm-compressor]])

**Integration:** Loads quantized checkpoints from HuggingFace. No quantization at runtime (use [[llm-compressor]] for quantization).

### Distributed Serving

**Tensor Parallelism (TP):** Shard model layers across GPUs (see [[Tensor Parallelism]])
- Single-node: `--tensor-parallel-size 4` (for 4 GPUs)
- Multi-node: NCCL all-reduce across nodes

**Pipeline Parallelism (PP):** Shard model stages across GPUs (see [[Pipeline Parallelism]])
- Example: 8 layers per GPU for 4-GPU pipeline
- Microbatching for pipeline bubbles

**Data Parallelism (DP):** Replicate model, shard requests (see [[Data Parallelism]])
- V1 adds DP coordinator for load balancing
- Example: `--data-parallel-size 4` (4 replicas)

### Speculative Decoding

Draft-based acceleration (see [[Speculative Decoding]]):
- **Draft models:** Small model generates candidates, large model verifies
- **EAGLE:** Tree-based speculation with learned draft heads
- **Medusa:** Multiple decoding heads
- **N-gram:** Simple draft via n-gram lookup

Typically 1.5-2.5× speedup for greedy decoding.

### LoRA Serving

Multi-adapter serving from single base model (see LoRA Serving):
- Load multiple LoRA adapters (e.g., 100 adapters)
- Per-request adapter selection
- Batched inference across different adapters

### Structured Output

Constrained decoding via guided backends:
- **Outlines:** CFG-based constrained generation
- **Guidance:** Grammar-based generation
- Automatic fallback if primary backend fails

## Interfaces

### Offline Inference (LLM Class)

Batch inference without server.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-8B")
prompts = ["Once upon a time", "In a galaxy far away"]
outputs = llm.generate(prompts, SamplingParams(temperature=0.8, max_tokens=100))

for output in outputs:
    print(output.outputs[0].text)
```

**Use cases:** Batch evaluation, offline benchmarking, dataset processing.

**Code:** `vllm/entrypoints/llm.py`

### Online Serving (vllm serve)

OpenAI-compatible API server.

```bash
vllm serve meta-llama/Llama-3-8B --tensor-parallel-size 4
```

**Endpoints:**
- `/v1/completions` (text completion)
- `/v1/chat/completions` (chat completion)
- `/v1/embeddings` (pooling models)

**Use cases:** Production serving, multi-user inference, API-based integration.

**Code:** `vllm/entrypoints/openai/api_server.py`

## Hardware Support

| Platform | Support | Notes |
|----------|---------|-------|
| NVIDIA GPUs | 🟢 | H100, A100, L40S, RTX 40xx, etc. |
| AMD GPUs | 🟢 | MI300X, MI250 via ROCm |
| Intel GPUs | 🟢 | Data Center GPU Max via Intel Extension for PyTorch |
| TPU | 🟢 | Google TPU v5e, v6e |
| CPU | 🟢 | AVX-512 optimizations |

**Plugins:** [vllm-ascend](https://github.com/vllm-project/vllm-ascend), [vllm-spyre](https://github.com/vllm-project/vllm-spyre), [vllm-gaudi](https://github.com/vllm-project/vllm-gaudi), [vllm-openvino](https://github.com/vllm-project/vllm-openvino)

## Architecture

See [[vLLM Engine]] and [[V1 Architecture]] for full architecture details.

**High-level flow:**
1. Client sends HTTP request to API server
2. API server tokenizes input, sends to engine core via ZMQ
3. Engine core schedules request, allocates KV cache blocks
4. Engine core dispatches work to GPU workers
5. GPU workers execute model forward pass
6. Engine core decodes output tokens, streams to API server
7. API server detokenizes and streams to client

**V1 multi-process architecture (see [[vLLM Engine#V1 Process Architecture]]):**
- API Server Process: HTTP handling, tokenization
- Engine Core Process: scheduler, KV cache manager
- GPU Worker Processes: model execution (1 per GPU)
- DP Coordinator Process (optional): data parallelism load balancing

## Performance Characteristics

**Strengths:**
- Highest throughput for decoder-only models (2-4× vs. baseline)
- Efficient memory use (60-80% reduction in KV cache waste)
- Low latency for high request concurrency (near-zero CPU overhead in V1)

**Trade-offs:**
- TTFT can be high for very long prompts (mitigated by chunked prefill)
- CUDA graph memory overhead (configurable via `--max-num-seqs`)
- Batch size limited by KV cache memory (no CPU swapping in V1)

**Ideal workloads:**
- High request concurrency (100+ concurrent users)
- Shared prefixes (chat templates, few-shot examples)
- Long context (prefix caching reuses common tokens)

## Configuration

Key flags for `vllm serve`:

**Parallelism:**
- `--tensor-parallel-size N`: TP across N GPUs
- `--pipeline-parallel-size N`: PP across N GPUs
- `--data-parallel-size N`: DP with N replicas

**Memory:**
- `--gpu-memory-utilization 0.9`: Target GPU memory fraction (default 0.9)
- `--max-num-seqs 256`: Max concurrent sequences
- `--max-num-batched-tokens 8192`: Max tokens per batch

**Scheduling:**
- `--scheduling-policy {fcfs,priority}`: FCFS or priority-based
- `--enable-prefix-caching`: Enable prefix caching (V1 default: enabled)
- `--enable-chunked-prefill`: Enable chunked prefill (V1 default: enabled)

**Quantization:**
- `--quantization {awq,gptq,fp8}`: Load quantized model
- `--kv-cache-dtype {auto,fp8}`: KV cache quantization

**Speculative decoding:**
- `--speculative-model <model>`: Draft model for spec decode
- `--num-speculative-tokens N`: Tokens per draft step

See [vLLM docs](https://docs.vllm.ai/en/stable/) for all flags.

## Installation

```bash
# From PyPI (recommended)
pip install vllm

# From source (for development)
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

**Requirements:**
- Python 3.8+
- CUDA 11.8+ (for NVIDIA GPUs)
- PyTorch 2.0+

## Cross-References

**Concepts:**
- [[PagedAttention]] — KV cache memory management
- [[Continuous Batching]] — dynamic scheduling
- [[Chunked Prefill]] — latency variance reduction
- [[Prefix Caching]] — KV cache reuse
- [[Tensor Parallelism]] — model layer sharding
- [[Pipeline Parallelism]] — model stage pipelining
- [[Data Parallelism]] — request-level parallelism
- [[Speculative Decoding]] — draft-based acceleration

**Architectures:**
- [[vLLM Engine]] — process architecture, LLMEngine, workers
- [[V1 Architecture]] — V1 design, unified scheduler

**Papers:**
- [[Efficient Memory Management for Large Language Model Serving with PagedAttention]] — vLLM paper, SOSP 2023

**Tools:**
- [[llm-compressor]] — quantization toolkit
- GuideLLM — benchmarking tool

## See Also

- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM Docs](https://docs.vllm.ai/)
- [vLLM Blog](https://blog.vllm.ai/)
- [vLLM Slack](https://inviter.co/vllm-slack)
- [vLLM Paper](https://arxiv.org/abs/2309.06180)
