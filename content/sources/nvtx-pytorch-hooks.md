---
title: "vLLM NVTX PyTorch Hooks"
type: source
source_url: https://docs.vllm.ai/en/stable/api/vllm/utils/nvtx_pytorch_hooks/
fetched: 2026-04-25
---

# vLLM NVTX PyTorch Hooks

API reference for `vllm.utils.nvtx_pytorch_hooks` — forward hook registration for [[NVTX Profiling]] in PyTorch networks.

## Key Takeaways

- **PytHooks** class registers forward pre/post hooks on all modules in a [[vLLM]] model, automatically inserting NVTX range markers for Nsight Systems / Nsight Compute profiling
- Lightweight modules (Identity, Dropout) are skipped to reduce profiling noise
- `layerwise_nvtx_marker_context()` provides a context manager for scoped NVTX markers with tensor shape tracking
- `process_layer_params()` extracts static hyperparameters (kernel size, stride, channels) from Conv/Pool/Linear/BN/Embedding layers for annotation

## Usage Pattern

```python
from vllm.utils.nvtx_pytorch_hooks import PytHooks

hooks = PytHooks()
hooks.register_hooks(model)
# All subsequent forward passes emit NVTX markers visible in Nsight
```

## Relevance

Enables layer-level profiling of vLLM model execution. Useful for identifying bottleneck layers, understanding compute distribution across transformer blocks, and validating that [[Tensor Parallelism]] splits are balanced.

## Related

- [[NVTX Profiling]]
- [[vLLM]]
- [[Tensor Parallelism]]
