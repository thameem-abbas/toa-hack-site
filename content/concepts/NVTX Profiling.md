---
title: NVTX Profiling
type: concept
aliases: [NVTX, NVIDIA Tools Extension]
created: 2026-04-25
sources:
  - "nvtx-pytorch-hooks"
---

# NVTX Profiling

NVIDIA Tools Extension (NVTX) — annotation API for marking regions of code with named ranges, enabling GPU profilers (Nsight Systems, Nsight Compute) to display human-readable labels on timeline views.

## How It Works

1. Code pushes an NVTX range with a label (e.g., "attention_layer_0")
2. GPU profiler records wall-clock and GPU activity within that range
3. Timeline view shows labeled ranges aligned with CUDA kernel launches

In vLLM, the `vllm.utils.nvtx_pytorch_hooks` module automates this by registering PyTorch forward hooks that push/pop NVTX ranges around every module's forward pass.

## Key Properties

- **Zero overhead when profiler is detached** — NVTX calls are no-ops without an active Nsight session
- **Hierarchical** — ranges can nest (model → layer → attention → kernel)
- **Per-layer granularity** — identifies which transformer layers are bottlenecks
- **Tensor shape tracking** — vLLM's hooks also log input/output tensor dimensions

## Use Cases in LLM Inference

- Identifying slow layers during prefill vs decode phases
- Validating [[Tensor Parallelism]] balance across GPUs
- Profiling [[Speculative Decoding]] overhead (draft model vs verification)
- Measuring [[Chunked Prefill]] chunk boundaries

## Tools

- **Nsight Systems** (`nsys profile`) — system-wide timeline
- **Nsight Compute** (`ncu`) — per-kernel analysis
- **PyTorch Profiler** — can also consume NVTX markers via `torch.cuda.nvtx`

## Related Concepts

- [[KV Cache]] — profiling reveals KV cache allocation patterns
- [[Continuous Batching]] — NVTX helps visualize batch scheduling decisions
- [[Tensor Parallelism]] — verify even compute distribution
