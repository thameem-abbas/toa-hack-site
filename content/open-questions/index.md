---
title: Open Questions
type: index
---

# Open Questions

Unresolved problems and active research threads.

- Optimal Speculation Length — how to auto-tune num_speculative_tokens per workload?
- Quantization Quality Floor — at what bit-width do task-specific regressions become unacceptable?
- Disaggregated KV Transfer — what's the practical bandwidth ceiling for prefill→decode KV cache transfer?
- LoRA Adapter Switching Cost — overhead of hot-swapping adapters under load?
- Compound Optimization Interactions — do FP8 + speculative decoding + prefix caching compose linearly?
