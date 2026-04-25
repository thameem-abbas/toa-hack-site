---
title: Speculative Decoding
type: concept
tags: [optimization, inference, latency, speculation]
created: 2026-04-25
---

# Speculative Decoding

Speculative decoding is a latency optimization technique that converts sequential token generation into partially parallel verification by having a cheaper draft model propose multiple candidate tokens that a target model then verifies in a single forward pass.

## Core Idea

Traditional autoregressive decoding generates one token at a time, making it inherently sequential and memory-bandwidth-bound during the decode phase. Speculative decoding breaks this bottleneck by:

1. **Draft phase**: A small, fast draft model (or lightweight heuristic) generates K candidate tokens
2. **Verification phase**: The target model processes all K candidates in parallel in a single forward pass
3. **Acceptance phase**: Candidates are accepted or rejected based on the target model's probability distribution

The key insight is that verifying K tokens in parallel is nearly as cheap as generating 1 token (both are memory-bound operations), so if the draft model is sufficiently accurate, we get near-free speculation.

## Theoretical Foundation

Speculative decoding is **theoretically lossless** up to floating-point precision limits. The output distribution is mathematically identical to standard autoregressive sampling when using proper rejection sampling:

- **Rejection sampling**: For each candidate token, accept with probability `min(1, p_target(token) / p_draft(token))`
- **Greedy decoding**: Accept all tokens where `argmax(p_target) == argmax(p_draft)`

vLLM's implementation is algorithmically validated via:
- **Rejection sampler convergence tests**: Verify samples align with target distribution
- **Greedy sampling equality tests**: Confirm output matches non-speculative baseline

However, practical **non-determinism** can arise from:
- **Floating-point precision**: Different kernel fusion/batching may produce slightly different logits
- **Batch size variations**: Non-deterministic GPU operations may affect numerical stability
- **Logprob instability**: vLLM does not guarantee stable token log probabilities across runs

## Performance Characteristics

### When Speculative Decoding Works Best

**High-gain scenarios** (2-3× speedup):
- **Low-to-medium QPS** (query per second): GPU has spare capacity for draft model inference
- **Memory-bound workloads**: Decode phase dominates (long sequences, small batches)
- **High acceptance rate**: Draft model aligns well with target model (same family, similar training data)
- **Long output sequences**: Amortizes draft model overhead across many tokens

**Low-gain or negative scenarios**:
- **High QPS**: GPU fully saturated, draft model adds overhead without capacity for parallel verification
- **Compute-bound workloads**: Prefill dominates, decode phase is small fraction of latency
- **Low acceptance rate**: Poor draft model quality (<30% acceptance) wastes GPU cycles
- **Short output sequences**: Overhead dominates before speculation can amortize

### Key Metrics

- **Acceptance rate**: Fraction of drafted tokens accepted by target model (target: >60%)
- **Speedup ratio**: E2E latency improvement (typical: 1.2-3×)
- **num_speculative_tokens**: Number of tokens drafted per step (typical: 3-8)
- **Draft model overhead**: Additional GPU memory and computation cost

## vLLM Speculation Methods

vLLM supports seven speculation methods with different accuracy/cost tradeoffs:

### Model-Based Methods (High Gain)

#### [[EAGLE]] — Extrapolation Algorithm for Greater Language-model Efficiency
- **How it works**: Lightweight prediction heads trained on target model's hidden states
- **Key advantage**: No separate draft model needed, uses target model's own features
- **Configuration**: `--speculative-config '{"method": "eagle", "model": "path/to/eagle-head"}'`
- **Variants**: EAGLE (single-path), EAGLE-2 (tree-based verification)
- **Speedup**: High gain on both low and high QPS
- **Requirements**: Pre-trained EAGLE head for target model family

#### [[Multi-Token Prediction]] (MTP)
- **How it works**: Models natively trained to predict multiple tokens per forward pass
- **Key advantage**: No external draft model, no training required (if model supports MTP)
- **Configuration**: `--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'`
- **Speedup**: High gain on both low and high QPS
- **Requirements**: Model must have native MTP heads (DeepSeek-V3, Qwen3, MiMo)

#### [[Draft Model Speculation]]
- **How it works**: Smaller model from same family generates candidate tokens
- **Key advantage**: Well-understood, broadly applicable, high acceptance rate
- **Configuration**: `--speculative-config '{"method": "draft_model", "model": "path/to/draft"}'`
- **Speedup**: High gain at low QPS, medium gain at high QPS
- **Requirements**: Separate draft model (e.g., Qwen3-0.6B for Qwen3-8B)

#### Parallel Draft Model (PARD)
- **How it works**: Draft model generates tokens in parallel rather than autoregressively
- **Key advantage**: Lower draft model latency
- **Speedup**: High gain on both low and high QPS

#### MLP Speculator
- **How it works**: Multi-layer perceptron predicts next tokens from target model's hidden states
- **Speedup**: Medium to high gain at low QPS, medium gain at high QPS

### Heuristic Methods (Low-Medium Gain)

#### [[N-gram Speculation]]
- **How it works**: Match n-grams in the prompt/generated text to predict next tokens
- **Key advantage**: Zero overhead, no draft model, works on any model
- **Configuration**: `--speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'`
- **Speedup**: Low to medium gain, medium gain at high QPS
- **Use case**: Repetitive text (code, data formatting, templated responses)

#### Suffix Decoding
- **How it works**: Dynamic speculation depth based on prefix match length in cached requests
- **Key advantage**: No extra draft model, adaptive speculation depth
- **Configuration**: `--speculative-config '{"method": "suffix", "suffix_decoding_max_tree_depth": 24}'`
- **Speedup**: Low to medium gain, medium gain at high QPS
- **Use case**: Similar requests with shared prefixes

## Method Selection Guide

| Method | Low QPS (latency) | High QPS (throughput) | Requirements | Notes |
|--------|-------------------|----------------------|--------------|-------|
| EAGLE | High gain | Medium to high gain | Pre-trained EAGLE head | Best general-purpose model-based method |
| MTP | High gain | Medium to high gain | Native MTP support | Best if model supports it (DeepSeek-V3, Qwen3) |
| Draft model | High gain | Medium gain | Separate draft model | Classic approach, well-understood |
| PARD | High gain | Medium to high gain | Separate draft model | Faster draft model latency |
| MLP | Medium to high gain | Medium gain | Compatible MLP speculator | Good middle ground |
| N-gram | Low to medium gain | Medium gain | None | Lightweight, zero overhead |
| Suffix | Low to medium gain | Medium gain | None | Adaptive, works on cached requests |

**Rule of thumb**:
- **Latency-critical (low QPS)**: Use EAGLE or MTP if available, else draft model
- **Throughput-critical (high QPS)**: Use EAGLE, MTP, or n-gram (avoid draft model overhead)
- **No model support**: Use n-gram or suffix for modest gains
- **Exploratory**: Start with n-gram (zero setup), upgrade to EAGLE/MTP if gains justify training/finding speculator

## Configuration in vLLM

### Command-Line Interface

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
    "num_speculative_tokens": 2,
    "draft_tensor_parallel_size": 1
  }'
```

### Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "method": "draft_model",
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 5,
    },
)
```

### Common Configuration Keys

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `method` | `string` | `None` | Speculation method (draft_model, ngram, eagle, mtp, etc.) |
| `model` | `string` | `None` | Draft model or EAGLE head path (not needed for ngram/suffix/mtp) |
| `num_speculative_tokens` | `integer > 0` | `None` | Number of tokens to speculate per step |
| `draft_tensor_parallel_size` | `integer >= 1` | `None` | TP size for draft model |
| `parallel_drafting` | `boolean` | `false` | Enable parallel draft token generation (EAGLE/draft model only) |
| `rejection_sample_method` | `string` | `strict` | `strict`, `probabilistic`, or `synthetic` |

### Method-Specific Keys

**N-gram**:
- `prompt_lookup_min`: Minimum n-gram window size (default: 5)
- `prompt_lookup_max`: Maximum n-gram window size (default: 5)

**Suffix decoding**:
- `suffix_decoding_max_tree_depth`: Max combined prefix-match and speculation tree depth (default: 24)
- `suffix_decoding_max_cached_requests`: Max requests cached in global suffix tree (default: 10000)
- `suffix_decoding_max_spec_factor`: Cap on speculative length as multiple of prefix-match length (default: 1.0)
- `suffix_decoding_min_token_prob`: Min token probability to speculate (default: 0.1)

## Acceptance Rate Tuning

The acceptance rate is the fraction of drafted tokens accepted by the target model. Higher acceptance means better speedup.

### Factors Affecting Acceptance Rate

1. **Draft model quality**: Better-aligned draft models (same family, similar training data) achieve higher acceptance
2. **num_speculative_tokens**: More speculative tokens → lower per-token acceptance (but may still improve E2E speedup)
3. **Sampling temperature**: Higher temperature → more diverse outputs → lower acceptance
4. **Sequence position**: Early tokens (more context) typically have higher acceptance than late tokens

### Tuning Strategy

1. **Start with conservative speculation**: `num_speculative_tokens=2-3`
2. **Measure acceptance rate**: Use `examples/offline_inference/spec_decode.py` to extract per-request metrics
3. **Increase speculation depth**: If acceptance >60%, try `num_speculative_tokens=5-8`
4. **Monitor E2E speedup**: Acceptance rate is a proxy; measure wall-clock latency
5. **A/B test**: Compare speculative vs non-speculative on production traffic

**Target acceptance rates**:
- **EAGLE/MTP**: 70-90% (well-aligned speculators)
- **Draft model**: 60-80% (same-family draft)
- **N-gram**: 30-60% (depends on text repetitiveness)
- **Suffix**: 40-70% (depends on request similarity)

## Losslessness Guarantees

vLLM's speculative decoding is designed to be **lossless** in three senses:

### 1. Theoretical Losslessness

Speculative decoding with proper rejection sampling is mathematically equivalent to standard autoregressive sampling, up to floating-point precision limits. The Speculative Sampling paper (Leviathan et al., 2023) proves this for the rejection sampling algorithm.

### 2. Algorithmic Losslessness

vLLM's implementation is algorithmically validated via:
- **Rejection sampler convergence tests**: `tests/samplers/test_rejection_sampler.py` verifies samples align with target distribution
- **Greedy sampling equality tests**: `tests/spec_decode/e2e/` confirms greedy speculative output matches greedy baseline

### 3. Practical Non-Determinism

Despite theoretical/algorithmic losslessness, **outputs may vary** due to:
- **Floating-point precision**: Different kernel orderings produce slightly different logits
- **Batch size effects**: Non-deterministic GPU operations (e.g., atomicAdd) cause numerical instability
- **Logprob instability**: vLLM does not guarantee stable token log probabilities across runs

**Mitigation strategies** (see vLLM FAQ):
- Use consistent batch sizes
- Pin random seeds
- Use greedy decoding (temperature=0) for reproducibility
- Accept small variations as inherent to GPU numerics

## Known Limitations

1. **Pipeline parallelism incompatibility**: Speculative decoding does not compose with PP as of vLLM ≤0.15.0
2. **Draft model support**: Draft model speculation not supported in vLLM ≤0.10.0
3. **QPS sensitivity**: High-QPS workloads may see no gain or negative speedup due to draft model overhead
4. **Attention backend constraints**: Some speculation methods require specific attention backends ([[CUDA Graphs]] compatibility)

## Cross-References

- **Related techniques**: [[Kernel Fusions]], [[CUDA Graphs]], [[Prefix Caching]], [[Continuous Batching]]
- **Related concepts**: [[KV Cache]], [[PagedAttention]], [[vLLM Engine]]
- **Specific methods**: [[EAGLE]], [[Multi-Token Prediction]], [[Draft Model Speculation]], [[N-gram Speculation]]
- **Parallelism**: [[Tensor Parallelism]], [[Data Parallelism]]

## Resources

- **Paper**: [Accelerating Large Language Model Decoding with Speculative Sampling (Leviathan et al., 2023)](https://arxiv.org/pdf/2302.01318)
- **EAGLE paper**: [EAGLE: Extrapolation Algorithm for Greater Language-model Efficiency (Li et al., 2024)](https://arxiv.org/pdf/2401.15077)
- **vLLM Office Hours**: [Intro to Speculators (#40)](https://www.youtube.com/watch?v=2ISAr_JVGLs)
- **Deep dive**: [A Hacker's Guide to Speculative Decoding in vLLM](https://www.youtube.com/watch?v=9wNAgpX6z_4)
- **Example script**: `examples/offline_inference/spec_decode.py`
- **Benchmark guide**: `docs/benchmarking/cli.md`

## Training Custom Speculators

For training custom draft models or EAGLE heads, see [vllm-project/speculators](https://github.com/vllm-project/speculators) for seamless integration with vLLM.
