---
title: Architectures
type: index
---

# Architectures

System designs for LLM inference at scale.

## Core Systems
- [[vLLM Engine]] — scheduler, worker, model runner, block manager internals
- [[V1 Architecture]] — vLLM V1 unified scheduler, multi-process design, differences from V0
- [[OpenAI-Compatible Server]] — FastAPI HTTP server implementing OpenAI APIs (/v1/completions, /v1/chat/completions, /v1/embeddings, /v1/audio/transcriptions, /v1/realtime) + vLLM custom endpoints (/v1/score, /generative_scoring, /tokenize, /rerank); chat template system, extra parameters via extra_body, Ray Serve LLM integration
- [[Model Runner V2]] — V2 model runner redesign: persistent batch decoupling, async-first execution, StagedWriteTensor, GPU-native input prep, Triton sampler, explicit CUDA graph management
- [[Attention Backends]] — Pluggable attention backend system: 13+ standard backends (FlashAttention-2/3/4, FlashInfer, Triton, flex_attention, ROCm AITER), 15+ MLA backends, automatic selection via priority lists, AttentionBackend.validate_configuration() pattern
- [[Logits Processors]] — Batch-granularity logits transformation: stateful processors, BatchUpdate synchronization (remove/add/move), argmax-invariant optimization, LogitsProcessor base class
- [[Plugin System]] — Python entry_points extensibility: general (models), platform (OOT hardware), IO processor (multimodal), stat logger; OOT platform plugins (vllm-ascend, vllm-gaudi, vllm-neuron, vllm-kunlun)
- [[vLLM Metrics]] — Prometheus-compatible metrics system: server-level (Gauges/Counters for state), request-level (Histograms for SLOs), KV cache residency, prefix cache hit rate, speculative decoding acceptance, event timeline (QUEUED→SCHEDULED→NEW_TOKENS), /metrics endpoint
- [[Optimization Levels]] — Four optimization levels (-O0 to -O3) trading startup time for runtime performance
- [[FusedMoE Modular Kernel]] — Three-component MoE architecture (Prepare/Finalize, Experts, Weight/Reduce), pluggable backends and kernels
- [[CustomOp System]] — Platform-specific operation dispatch (CUDA/ROCm/XPU/TPU/OOT), OOT hardware plugin registration, custom op vs Inductor fusion trade-offs
- [[Hybrid KV Cache Manager]] — per-layer KV cache allocation for hybrid models (Gemma, Llama 4, Jamba), unified page size with mixed attention types
- [[KV Cache Transfer]] — connectors for transferring KV cache between prefill and decode instances (P2pNcclConnector, NixlConnector, LMCache, Mooncake, FlexKV)
- [[llm-d]] — disaggregated inference on Kubernetes, prefill/decode pool separation, KV cache routing
- Inference Gateway — multi-model routing, A/B canary, per-pool autoscaling

## Deployment Patterns
- Single-Node Serving — one GPU or multi-GPU on single machine
- Multi-Node Distributed — cross-node tensor/pipeline parallelism
- Kubernetes Inference — HPA, custom metrics, Prometheus integration
