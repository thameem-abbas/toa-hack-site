---
title: vLLM / LLM-D Research Wiki
type: index
created: 2026-04-25
---

# vLLM / LLM-D Research Wiki

Research wiki covering LLM inference optimization — vLLM, llm-d, quantization, speculative decoding, distributed serving, and evaluation.

## Core Areas

### [[concepts/index|Concepts]]
Foundational ideas: KV cache, PagedAttention, tensor parallelism, pipeline parallelism, continuous batching, prefix caching.

### [[architectures/index|Architectures]]
System designs: vLLM engine internals, llm-d disaggregated serving, inference gateway patterns, model routing.

### [[techniques/index|Techniques]]
Optimization methods: quantization (FP8, MXFP4, GPTQ, AWQ), speculative decoding (EAGLE, Medusa, N-gram, draft models), LoRA serving, chunked prefill.

### [[papers/index|Papers]]
Key papers with summaries, claims, and links to concept pages.

### [[benchmarks/index|Benchmarks]]
Performance data: throughput, latency, quality metrics, cost comparisons across configurations.

### [[tools/index|Tools]]
Software: vLLM, llm-d, GuideLLM, lm-eval, llm-compressor, TRL, NemoClaw.

### [[open-questions/index|Open Questions]]
Unresolved problems, active research threads, gaps in current understanding.

### [[sources/nvtx-pytorch-hooks|Sources]]
Ingested source summaries with extracted claims and cross-references.

## Recent Activity

See [[log|Activity Log]] for ingestion history.
