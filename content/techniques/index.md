---
title: Techniques
type: index
---

# Techniques

Optimization methods for LLM inference.

## Quantization
- [[FP8 Quantization]] — FP8 W8A8 weight-activation quantization on Ada/Hopper/MI300+; 2× memory reduction, up to 1.6× throughput; static per-channel weights, dynamic per-token activations; W8A16 weight-only via Marlin on Turing/Ampere
- [[INT8 W8A8]] — INT8 weight-activation quantization for Turing/Ampere/CPU; SmoothQuant activation smoothing, 512+ sample calibration required; not supported on Blackwell (use FP8)
- [[INT4 W4A16]] — INT4 weight-only quantization with FP16 activations; 4× memory reduction, GPTQ algorithm, group-wise scales (group_size=128), Marlin kernel on Ampere+; optimized for low QPS workloads
- [[Quantized KV Cache]] — FP8 KV cache quantization; ~50% KV cache memory reduction, per-tensor or per-attention-head scales, calibration via llm-compressor; orthogonal to weight quantization
- MXFP4 — microscaling FP4, block-wise quantization
- [[GPTQ]] — Hessian-based INT4/INT8 post-training quantization, group-wise scaling, Marlin/Machete backends
- [[AWQ]] — Activation-aware weight quantization, INT4 weight-only, salient weight protection, Marlin backend
- [[GGUF]] — llama.cpp file format, multiple quantization types (Q4_K_M, Q8_0, etc.), experimental in vLLM
- Compound Quantization — stacking FP8 + speculative decoding + prefix caching

## Speculative Decoding
- [[EAGLE]] — lightweight prediction heads trained on target model's hidden states; no separate draft model needed, high acceptance rate (70-90%), 1.8-2.5× speedup
- [[Multi-Token Prediction]] — models natively trained to predict multiple tokens per forward pass (DeepSeek-V3, Qwen3, MiMo); zero overhead, simplest setup, 2.0-2.8× speedup
- [[Draft Model Speculation]] — smaller model from same family generates candidates, target verifies; classic approach, 60-80% acceptance, 2.0-3.0× speedup at low QPS
- [[N-gram Speculation]] — heuristic pattern matching in prompt/generated text; zero overhead, always available, 1.2-1.8× speedup on repetitive workloads
- Medusa — multiple decoding heads for parallel token prediction
- Acceptance Rate Tuning — optimizing num_speculative_tokens

## Fine-Tuning for Inference
- [[LoRA]] — Low-Rank Adaptation for parameter-efficient fine-tuning; vLLM supports multi-LoRA serving with per-request adapter selection, dynamic loading via API endpoints/resolver plugins (filesystem, S3, HuggingFace Hub), batched multi-adapter inference with Punica/BGMV kernels, and compatibility with quantization/prefix caching/speculative decoding
- DPO — direct preference optimization for alignment
- Reward Modeling — training reward models from human preferences

## Compilation & Kernel Optimization
- [[torch.compile Integration]] — vLLM's use of PyTorch compilation for kernel generation, graph optimization, and CUDA graph enablement
- [[Kernel Fusions]] — 12+ production fusions via custom Inductor passes (AllReduce+RMSNorm, Attention+Quant, AsyncTP, Norm+Quant, etc.), hardware support matrix (SM100/SM90/SM89/SM80/ROCm), token-count sensitivity
- [[Dual Batch Overlap]] — overlapping MoE all-to-all communication with computation via microbatching and thread ping-pong

## Constrained Generation
- [[Structured Outputs]] — Constrained decoding via logits masking to guarantee outputs match a schema (JSON, regex, context-free grammar, choice); backends: xgrammar (default), guidance, outlines, lm-format-enforcer; compilation overhead (1-10s first request) with cached FSM reuse; 5-15% per-token slowdown; guarantees syntax but not semantics

## Memory & Resource Management
- [[Sleep Mode]] — Temporarily release GPU memory (model weights + KV cache) without stopping server; Level 1 (offload weights to CPU, discard KV cache) for same-model resume, Level 2 (discard weights + KV cache) for weight updates; fine-grained wake (weights or KV cache separately); ~90% GPU memory freed; RLHF use case for GPU sharing between inference and training

## Training Integration
- [[RLHF with vLLM]] — Using vLLM as inference backend for RLHF training; integration with 11+ RL frameworks (TRL, OpenRLHF, verl, NeMo-RL, etc.); weight synchronization via NCCL (multi-GPU) or IPC (same-GPU); async RL pipelining (overlap generation and training); sleep mode for GPU sharing
