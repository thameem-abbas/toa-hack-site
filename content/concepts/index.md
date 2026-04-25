---
title: Concepts
type: index
---

# Concepts

Foundational ideas behind LLM inference optimization.

## Attention & Memory
- [[KV Cache]] — key-value cache mechanics, memory footprint, eviction strategies
- [[PagedAttention]] — vLLM's virtual memory approach to KV cache management
- [[Prefix Caching]] — reusing computed prefixes across requests

## Parallelism
- [[Tensor Parallelism]] — splitting individual layers (weight matrices) across GPUs; Megatron-LM algorithm; all-reduce per layer; TP=GPUs per node
- [[Pipeline Parallelism]] — splitting model layers sequentially across GPUs/nodes; micro-batching; bubble overhead; PP=number of nodes
- [[Data Parallelism]] — replicating model across GPU groups; independent batches; ZMQ load balancing; DP scales throughput
- [[Expert Parallelism]] — distributing MoE experts across GPUs, all-to-all routing
- [[Context Parallelism]] — sharding long sequences across GPUs; prefill CP (ring attention, under development), decode CP (KV cache sharding along sequence dimension)

## Model Architectures
- [[Mixture of Experts]] — sparse model architecture with expert routing, top-K selection, load balancing

## Scheduling & Batching
- [[Continuous Batching]] — dynamic request scheduling vs static batching
- [[Chunked Prefill]] — breaking long prefills into chunks to reduce TTFT interference

## Profiling & Observability
- [[NVTX Profiling]] — NVIDIA Tools Extension markers for GPU timeline profiling

## GPU Optimization
- [[CUDA Graphs]] — Capturing GPU kernel sequences for replay with minimal CPU overhead

## Quantization
- [[Quantization]] — reducing precision of model weights/activations for memory savings and speedup

## Serving Patterns
- [[Disaggregated Serving]] — separating prefill and decode phases across GPU pools
- Model Routing — directing requests to specialized model instances
- LoRA Serving — serving multiple LoRA adapters from a single base model

## Optimization Strategies
- [[Speculative Decoding]] — draft model proposes tokens, target verifies in parallel; converts sequential generation into parallel verification for 1.5-3× latency reduction

## Reasoning & Generation
- [[Reasoning Models]] — Models that produce internal reasoning tokens (<think>...</think>) before final answer; 11+ model families (DeepSeek-R1, Qwen3, QwQ, Granite 3.2, etc.); thinking budget control; structured outputs and tool calling compatible
