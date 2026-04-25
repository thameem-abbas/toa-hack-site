---
title: Benchmarks
type: index
---

# Benchmarks

Performance data, methodology, and comparisons.

## Categories
- Throughput Benchmarks — tokens/sec, requests/sec across configurations
- Latency Benchmarks — TTFT, TPOT, E2E latency distributions
- Quality Benchmarks — perplexity, accuracy, lm-eval scores post-quantization
- Cost Benchmarks — $/1M tokens across GPU tiers and optimization combos

## Tools
- [[vLLM Benchmarking]] — Built-in CLI tools (vllm bench serve, throughput, mm-processor): 15+ datasets (ShareGPT, Spec Bench, SPEED-Bench, VisionArena, InstructCoder, synthetic), load patterns (request-rate, burstiness, max-concurrency), metrics (TTFT, TPOT, ITL, E2E, throughput), visualization (timeline HTML, dataset stats PNG)
- GuideLLM — scenario-based benchmarking (recommended for production)
- lm-eval — language model evaluation harness
