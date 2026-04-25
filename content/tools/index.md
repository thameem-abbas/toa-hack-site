---
title: Tools
type: index
---

# Tools

Software for LLM inference, optimization, and evaluation.

## Serving
- [[vLLM]] — high-throughput LLM serving engine, OpenAI-compatible API
- [[llm-d]] — Kubernetes-native distributed inference serving: disaggregated prefill/decode pools, KV cache routing, autoscaling, KServe LLMInferenceService integration
- [[NemoClaw]] — NVIDIA agentic framework with tool-calling

## Quantization
- [[llm-compressor]] — unified quantization toolkit (GPTQ, AWQ, FP8, INT8), vLLM-native, Red Hat supported
- [[AutoGPTQ]] — GPTQ quantization library (deprecated, use llm-compressor)
- [[AutoAWQ]] — AWQ quantization library (deprecated, use llm-compressor)

## Evaluation & Benchmarking
- [[GuideLLM]] — scenario-based LLM benchmarking
- [[lm-eval Harness]] — language model evaluation
- [[RAGAs]] — retrieval-augmented generation evaluation

## Training
- [[TRL]] — transformer reinforcement learning (DPO, reward modeling)
- [[PEFT]] — parameter-efficient fine-tuning (LoRA adapters)

## Profiling
- [[NVTX Profiling]] — NVIDIA Tools Extension for GPU timeline annotation (Nsight Systems / Nsight Compute)

## Infrastructure
- [[Kubernetes for Inference]] — kind, Helm, HPA, Prometheus
- [[Grafana Dashboards]] — inference monitoring and alerting
