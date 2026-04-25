---
title: vLLM Speculative Decoding Documentation
type: source
tags: [vllm, speculative-decoding, optimization]
ingested: 2026-04-25
batch: 9
---

# vLLM Speculative Decoding Documentation

Source documentation for vLLM's speculative decoding feature, covering seven speculation methods, configuration schemas, losslessness guarantees, and performance tuning.

## Source Files

1. `docs/features/speculative_decoding/README.md` — Main overview, method selection guide, configuration schema
2. `docs/features/speculative_decoding/eagle.md` — EAGLE speculator configuration and pre-trained heads
3. `docs/features/speculative_decoding/draft_model.md` — Traditional draft model speculation setup
4. `docs/features/speculative_decoding/mtp.md` — Multi-token prediction for natively trained models
5. `docs/features/speculative_decoding/n_gram.md` — N-gram/prompt lookup speculation

## Key Claims

### Speculative Decoding Overview

1. **Purpose**: Reduce inter-token latency under medium-to-low QPS, memory-bound workloads
2. **Principle**: Draft model proposes K tokens, target model verifies all K in parallel in single forward pass
3. **Losslessness**: Theoretically lossless up to floating-point precision (rejection sampling preserves distribution)
4. **QPS sensitivity**: Low QPS = high gain, high QPS = reduced gain (draft overhead dominates)

### Seven Speculation Methods

| Method | Low QPS Gain | High QPS Gain | Requirements | Notes |
|--------|--------------|---------------|--------------|-------|
| **EAGLE** | High | Medium to high | Pre-trained EAGLE head | Strong general-purpose model-based method |
| **MTP** | High | Medium to high | Native MTP support | Best when target model has MTP heads (DeepSeek-V3, Qwen3) |
| **Draft model** | High | Medium | Separate draft model | Needs draft from same family |
| **PARD** | High | Medium to high | Separate draft model | Low draft model latency (parallel generation) |
| **MLP** | Medium to high | Medium | Compatible MLP speculator | Good when available |
| **N-gram** | Low to medium | Medium | None | Lightweight, zero overhead, always available |
| **Suffix** | Low to medium | Medium | None | Dynamic speculation depth based on prefix matches |

### Method Selection Guidance

**Low QPS (latency-focused)**:
- Use EAGLE, MTP, or draft model (2-3× speedup typical)
- Accept higher GPU memory/compute overhead
- Optimize for per-request latency

**High QPS (throughput-focused)**:
- Use EAGLE, MTP, or n-gram (avoid draft model overhead)
- Prioritize peak-friendly methods (n-gram, suffix)
- Optimize for aggregate throughput

### Configuration Schema (`--speculative-config`)

**Common keys**:
- `method`: Speculation method (`draft_model`, `ngram`, `eagle`, `mtp`, `suffix`, etc.)
- `model`: Draft model or EAGLE head path (not needed for ngram/suffix/mtp)
- `num_speculative_tokens`: Number of tokens to speculate per step
- `draft_tensor_parallel_size`: TP size for draft model
- `parallel_drafting`: Enable parallel draft generation (EAGLE/draft only)
- `rejection_sample_method`: `strict` (default), `probabilistic`, or `synthetic`

**N-gram specific**:
- `prompt_lookup_min`: Minimum n-gram window size (default: 5)
- `prompt_lookup_max`: Maximum n-gram window size (default: 5)

**Suffix decoding specific**:
- `suffix_decoding_max_tree_depth`: Max prefix-match + speculation depth (default: 24)
- `suffix_decoding_max_cached_requests`: Max requests in global suffix tree (default: 10000)
- `suffix_decoding_max_spec_factor`: Cap on speculation length vs prefix-match length (default: 1.0)
- `suffix_decoding_min_token_prob`: Min token probability to speculate (default: 0.1)

### Losslessness Guarantees

**Three levels**:

1. **Theoretical losslessness**: Mathematically equivalent to standard autoregressive sampling (up to FP precision limits)
   - Paper: "Accelerating Large Language Model Decoding with Speculative Sampling" (Leviathan et al., 2023)
   - Rejection sampling preserves target distribution

2. **Algorithmic losslessness**: vLLM implementation validated via tests
   - Rejection sampler convergence test: `tests/samplers/test_rejection_sampler.py`
   - Greedy sampling equality test: `tests/spec_decode/e2e/` (greedy spec == greedy baseline)

3. **Practical non-determinism**: Outputs may vary due to:
   - Floating-point precision differences (kernel fusion, batch size changes)
   - Non-deterministic GPU operations (atomicAdd, etc.)
   - Logprob instability (vLLM does not guarantee stable logprobs across runs)

**Mitigation**: See FAQ "Can the output of a prompt vary across runs in vLLM?"

### EAGLE Details

- **Architecture**: Lightweight prediction heads trained on target model's hidden states
- **Variants**: EAGLE (single-path), EAGLE-2 (tree-based), EAGLE-3 (latest)
- **Pre-trained heads**: [RedHatAI/speculator-models](https://huggingface.co/collections/RedHatAI/speculator-models), [yuhuili/models](https://huggingface.co/yuhuili/models?search=eagle)
- **vLLM <0.7.0**: Requires [conversion script](https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d)

**Example**:
```python
speculative_config={
    "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
    "draft_tensor_parallel_size": 1,
    "num_speculative_tokens": 2,
    "method": "eagle",
}
```

### MTP Details

- **Architecture**: Target model natively trained to predict multiple tokens per forward pass
- **No draft model needed**: Uses model's own MTP heads (e.g., DeepSeek-V3, Qwen3, MiMo)
- **Simplest setup**: Just enable via `method=mtp`

**Example**:
```python
speculative_config={
    "method": "mtp",
    "num_speculative_tokens": 1,
}
```

### Draft Model Details

- **Classic approach**: Smaller draft model generates K tokens, target verifies
- **Best practice**: Use draft from same family (Qwen3-0.6B for Qwen3-8B)
- **Deprecated config**: Old `--speculative-model` flag replaced by `--speculative-config`

**Example**:
```python
speculative_config={
    "model": "Qwen/Qwen3-0.6B",
    "num_speculative_tokens": 5,
    "method": "draft_model",
}
```

### N-gram Details

- **Heuristic**: Match n-grams in prompt/generated text, propose historical continuations
- **Zero overhead**: No draft model, no GPU memory, no extra compute
- **Best for**: Repetitive text (code, structured data, templates)
- **Community reference**: [joao_gante thread](https://x.com/joao_gante/status/1747322413006643259)

**Example**:
```python
speculative_config={
    "method": "ngram",
    "num_speculative_tokens": 5,
    "prompt_lookup_max": 4,
}
```

### Known Limitations

1. **Pipeline parallelism incompatibility**: PP not composable with speculative decoding (vLLM ≤0.15.0)
2. **Draft model support**: Not available in vLLM ≤0.10.0
3. **QPS sensitivity**: High QPS reduces or eliminates speedup (draft overhead dominates)

### Performance Benchmarking

**Scripts**:
- `examples/offline_inference/spec_decode.py` — Extract per-request acceptance rate, speedup
- `docs/benchmarking/cli.md` — Benchmark CLI guide

**Key metrics**:
- **Acceptance rate**: Fraction of drafted tokens accepted (target: >60% for model-based, >30% for n-gram)
- **Speedup ratio**: E2E latency improvement (target: 1.5-3× depending on method/workload)
- **num_speculative_tokens**: Tuning parameter (typical: 3-8 for model-based, 4-6 for n-gram)

## Wiki Pages Created

- [[Speculative Decoding]] — Core concept page (what, why, when, losslessness, method selection)
- [[EAGLE]] — EAGLE speculator technique (architecture, config, pre-trained heads)
- [[Draft Model Speculation]] — Traditional draft model approach (pairing, config, PARD variant)
- [[Multi-Token Prediction]] — MTP technique (native MTP heads, simplest setup)
- [[N-gram Speculation]] — Heuristic pattern matching (zero overhead, workload-specific gains)

## Cross-References

- **Related concepts**: [[KV Cache]], [[PagedAttention]], [[CUDA Graphs]], [[Continuous Batching]], [[Prefix Caching]]
- **Related techniques**: [[Kernel Fusions]], [[torch.compile Integration]], [[Optimization Levels]]
- **System integration**: [[vLLM Engine]], [[Tensor Parallelism]], [[Data Parallelism]]

## Resources

- **Paper**: [Accelerating Large Language Model Decoding with Speculative Sampling (Leviathan et al., 2023)](https://arxiv.org/pdf/2302.01318)
- **EAGLE paper**: [EAGLE: Lossless Acceleration of LLM Decoding by Feature Extrapolation (Li et al., 2024)](https://arxiv.org/pdf/2401.15077)
- **vLLM Office Hours**: [Intro to Speculators (#40)](https://www.youtube.com/watch?v=2ISAr_JVGLs)
- **Deep dive**: [A Hacker's Guide to Speculative Decoding in vLLM](https://www.youtube.com/watch?v=9wNAgpX6z_4)
- **Design doc**: [Lookahead Scheduling in vLLM](https://docs.google.com/document/d/1Z9TvqzzBPnh5WHcRwjvK2UEeFeq5zMZb5mFE8jR0HCs/edit#heading=h.1fjfb0donq5a)
- **Batch expansion**: [Information on batch expansion](https://docs.google.com/document/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA/edit#heading=h.kk7dq05lc6q8)
- **Future work**: [Dynamic speculative decoding](https://github.com/vllm-project/vllm/issues/4565)
- **Speculator training**: [vllm-project/speculators](https://github.com/vllm-project/speculators)

## Key Insights

1. **Method selection is workload-dependent**: Low QPS → use EAGLE/MTP/draft (2-3× speedup), high QPS → use n-gram/suffix (peak-friendly, modest speedup)

2. **Losslessness has three dimensions**: Theoretical (rejection sampling), algorithmic (vLLM tests), practical (FP precision, batch size effects cause non-determinism)

3. **N-gram is the universal baseline**: Zero overhead, always works, good starting point before investing in EAGLE/draft models

4. **MTP is the future**: Simplest setup (if model supports it), high speedup, no separate draft — expect broader adoption in 2026+ model releases

5. **Acceptance rate is a proxy, not the goal**: 50% acceptance with num_spec=8 may outperform 80% acceptance with num_spec=2 — measure E2E latency, not just acceptance
