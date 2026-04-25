---
title: N-gram Speculation
type: technique
tags: [speculative-decoding, optimization, latency, heuristic]
created: 2026-04-25
---

# N-gram Speculation

N-gram speculation is a lightweight, heuristic-based speculative decoding method that generates candidate tokens by matching n-grams in the prompt or previously generated text, with zero model overhead.

## Overview

N-gram speculation is the **simplest and most universally applicable** speculative decoding method. It requires no separate draft model, no EAGLE heads, no MTP support — just pattern matching on existing text.

**Key advantages**:
- **Zero overhead**: No additional models, no extra GPU memory, no training
- **Always available**: Works on any model, any workload, any hardware
- **Easy to enable**: Single line in speculative_config
- **No alignment issues**: Deterministic pattern matching (no probabilistic acceptance)
- **Peak-friendly**: Does not consume GPU capacity during high QPS

**Speedup**: Low to medium gain at low QPS (latency-focused), medium gain at high QPS (throughput-focused)

## How N-gram Speculation Works

### Algorithm

1. **Pattern matching**: Look for repeated n-grams in the prompt + generated tokens
2. **Candidate generation**: When an n-gram match is found, propose the tokens that historically followed that n-gram
3. **Verification**: Target model verifies all proposed tokens in parallel
4. **Acceptance**: Accept tokens where target model agrees with the n-gram prediction

### Example Walkthrough

```
Prompt: "To be or not to be, that is the question. To be or not"

Current position: After "To be or not"

N-gram lookup (n=4):
  - Search for previous occurrences of "To be or not"
  - Found at start: "To be or not to be"
  - Extract continuation: ["to", "be"]

Draft candidates: ["to", "be"]

Target model verifies: 
  - "to" → ACCEPT (matches target model's prediction)
  - "be" → ACCEPT (matches target model's prediction)

Output: ["to", "be"] (100% acceptance for this repetitive case)
```

### Window Size (prompt_lookup_min/max)

The n-gram window size controls how many tokens to match:

- **Small window (n=2-3)**: More matches, less specific, lower acceptance rate
- **Large window (n=5-8)**: Fewer matches, more specific, higher acceptance rate when matched
- **Dynamic window**: vLLM tries multiple window sizes (min to max) to find best match

**Example**:
```python
speculative_config={
    "prompt_lookup_min": 2,  # Try 2-gram, 3-gram, 4-gram, 5-gram
    "prompt_lookup_max": 5,
}
```

### Prompt Lookup vs Generated Text Lookup

vLLM's n-gram speculation searches:
1. **Prompt text**: Original user input (static)
2. **Generated text**: Tokens produced so far (dynamic, grows with each step)

**Key insight**: Repetitive prompts (e.g., code templates, structured data) benefit most from prompt lookup. Repetitive generation patterns (e.g., lists, enumerations) benefit from generated text lookup.

## Configuration in vLLM

### Python API — Offline Mode

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    },
)

outputs = llm.generate(prompts, sampling_params)
```

### Command-Line Interface — Online Serving

```bash
vllm serve Qwen/Qwen3-8B \
  --speculative-config '{
    "method": "ngram",
    "num_speculative_tokens": 5,
    "prompt_lookup_min": 2,
    "prompt_lookup_max": 5
  }'
```

### Configuration Parameters

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `method` | `string` | — | Set to `"ngram"` |
| `num_speculative_tokens` | `integer` | — | Max tokens to speculate per step (typical: 4-6) |
| `prompt_lookup_min` | `integer >= 1` | `5` | Minimum n-gram window size |
| `prompt_lookup_max` | `integer >= 1` | `5` | Maximum n-gram window size |

**Default behavior**: If both `prompt_lookup_min` and `prompt_lookup_max` are omitted, both default to `5` (5-gram only).

**Dynamic window**: Set different min/max to try multiple window sizes:
```python
# Try 3-gram, 4-gram, 5-gram, 6-gram
"prompt_lookup_min": 3,
"prompt_lookup_max": 6,
```

## When to Use N-gram Speculation

### Best Use Cases

✅ **Use n-gram speculation when**:
- **Exploratory**: You want to try speculative decoding with zero setup/risk
- **No model support**: Target model has no EAGLE heads or MTP support
- **Repetitive workloads**: Code generation, data formatting, templated responses, lists
- **Peak-friendly**: High QPS workloads where draft model overhead is unacceptable
- **Memory-constrained**: Cannot afford to load a draft model or EAGLE head

✅ **N-gram excels at**:
- **Code generation**: Function signatures, boilerplate, imports, docstrings repeat
- **Structured output**: JSON/YAML/XML generation with repeated keys/tags
- **Lists and enumerations**: "1. First item\n2. Second item\n3. Third item..."
- **Templates**: Customer service responses, email drafts, form filling

### When to Consider Alternatives

❌ **Consider alternatives when**:
- **High speedup required**: N-gram only gives 1.2-1.8× speedup; use [[EAGLE]] or [[Multi-Token Prediction]] for 2-3×
- **Non-repetitive text**: Creative writing, diverse conversations → low n-gram match rate
- **Model support available**: If model has EAGLE heads or MTP, use those for higher gains
- **Low QPS with spare GPU capacity**: Draft model or EAGLE may give better speedup

### N-gram as a Baseline

**Best practice**: Start with n-gram speculation as a **baseline**:
1. Enable n-gram with minimal config
2. Measure speedup on your workload
3. If gains are insufficient, upgrade to [[EAGLE]] or [[Draft Model Speculation]]
4. If gains are good enough (1.5-2×), keep n-gram for simplicity

## Performance Tuning

### Optimal num_speculative_tokens

Start with **4-6 tokens** and tune based on match rate:

```python
# Conservative (high match rate)
"num_speculative_tokens": 3  # Shorter sequences, easier to match

# Moderate (balanced)
"num_speculative_tokens": 5  # Typical default

# Aggressive (lower match rate, may still improve E2E)
"num_speculative_tokens": 8  # Longer sequences, harder to match
```

**Key insight**: Unlike model-based speculation, n-gram acceptance is **all-or-nothing** per match. If a 5-gram match is found, all 5 tokens are highly likely to be accepted. If no match, 0 tokens speculated.

### Window Size Tuning

**Small window (2-4)**:
- More matches (easier to find 2-grams than 5-grams)
- Less specific (lower acceptance rate per match)
- Use for highly repetitive text (code, templates)

**Large window (5-8)**:
- Fewer matches (harder to find 8-grams)
- More specific (higher acceptance rate per match)
- Use for less repetitive but still structured text

**Dynamic window** (recommended):
```python
"prompt_lookup_min": 3,
"prompt_lookup_max": 6,
```
vLLM tries all window sizes from min to max, uses longest match found.

### Workload-Specific Tuning

**Code generation**:
```python
"num_speculative_tokens": 6,
"prompt_lookup_min": 3,
"prompt_lookup_max": 6,
```
Code has high repetition (imports, function signatures, docstrings).

**Structured data (JSON/YAML)**:
```python
"num_speculative_tokens": 5,
"prompt_lookup_min": 2,
"prompt_lookup_max": 4,
```
Keys/tags repeat, but values vary.

**General conversation**:
```python
"num_speculative_tokens": 3,
"prompt_lookup_min": 4,
"prompt_lookup_max": 5,
```
Lower repetition, conservative speculation.

## Expected Speedups

Typical speedups measured on vLLM benchmarks:

| Workload | QPS | Speedup Range | Notes |
|----------|-----|---------------|-------|
| Code generation | Low (1-10) | 1.5-2.2× | High repetition (imports, boilerplate) |
| Structured output (JSON) | Low (5-20) | 1.3-1.8× | Keys repeat, values vary |
| General chatbot | Low (1-10) | 1.1-1.4× | Low repetition |
| Code generation | High (100+) | 1.4-1.9× | Peak-friendly (no draft overhead) |

**Factors affecting speedup**:
- **Text repetitiveness**: More repetition → higher match rate → higher speedup
- **Window size**: Larger windows → higher acceptance when matched, but fewer matches
- **Sequence length**: Longer sequences → more opportunities for matches
- **Prompt structure**: Structured prompts (templates, code) → higher speedup

### Acceptance Rate Characteristics

N-gram acceptance is **bimodal**:
- **Match found**: 80-100% acceptance (if 5-gram matches, high confidence)
- **No match**: 0% acceptance (no speculation)

**Average acceptance rate**: 30-60% across all steps (mix of perfect matches and no matches).

This differs from model-based speculation (60-80% acceptance, more consistent).

## Integration with Other Optimizations

N-gram speculation composes well with all vLLM optimizations:

✅ **Compatible**:
- [[Prefix Caching]]: N-gram searches in cached prefixes + generated tokens
- [[CUDA Graphs]]: N-gram matching is CPU-side, does not interfere with graphs
- [[Kernel Fusions]]: Target model verification benefits from fusions
- [[Tensor Parallelism]]: N-gram matching is TP-agnostic
- [[Pipeline Parallelism]]: Works (though PP + speculation generally incompatible in vLLM ≤0.15.0)
- [[Continuous Batching]]: N-gram works at any batch size
- [[Chunked Prefill]]: N-gram only helps decode phase

✅ **Peak-friendly**: Unlike draft models, n-gram does **not** consume GPU capacity during high QPS.

## Debugging and Monitoring

### Measuring Match Rate

Use vLLM's offline inference script:

```bash
python examples/offline_inference/spec_decode.py \
  --model Qwen/Qwen3-8B \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'
```

Output includes:
- Per-request match rate (fraction of steps with n-gram matches)
- Average tokens accepted per match
- E2E latency improvement

### Common Issues

**Very low speedup (<1.1×)**:
- Workload is not repetitive (creative writing, diverse conversations)
- Window size too large (no matches found)
- **Solution**: Reduce `prompt_lookup_min` to 2-3, increase `num_speculative_tokens`

**No speedup or slight slowdown**:
- N-gram matching overhead dominates (very short sequences)
- **Solution**: Disable n-gram for this workload, or increase sequence length

**Inconsistent speedup**:
- Some requests have high repetition, others have none
- **Solution**: This is expected; n-gram speedup varies by request

### Logging

Enable debug logging to see n-gram matches:

```bash
vllm serve <model> \
  --speculative-config '{"method":"ngram",...}' \
  --log-level DEBUG
```

Look for:
```
DEBUG: N-gram match found: window_size=5, tokens=['to', 'be', 'or', 'not', 'to']
DEBUG: No n-gram match found for current position
```

## Comparison to Other Methods

| Method | Acceptance Rate | Memory Overhead | Setup Complexity | Speedup (Low QPS) | Peak-Friendly |
|--------|-----------------|-----------------|------------------|-------------------|---------------|
| [[Draft Model Speculation]] | 60-80% | High (2 models) | Medium | 2.0-3.0× | ❌ No |
| [[EAGLE]] | 70-90% | Low (EAGLE head) | Medium | 1.8-2.5× | ❌ No |
| [[Multi-Token Prediction]] | 75-85% | None (native) | Low | 2.0-2.8× | ⚠️ Moderate |
| **N-gram** | 30-60% | None | Low | 1.2-1.8× | ✅ Yes |

**N-gram strengths**:
- Zero overhead (no models, no memory, no GPU cycles)
- Always available (any model, any workload)
- Peak-friendly (does not compete for GPU capacity)
- Easy to enable (single config line)

**N-gram weaknesses**:
- Lower speedup (1.2-1.8× vs 2-3× for model-based methods)
- Workload-dependent (requires repetitive text)
- Bimodal acceptance (perfect match or nothing)

## Advanced: Combining N-gram with Other Methods

N-gram can be used as a **fallback** in hybrid strategies (not currently supported in vLLM, but conceptually useful):

1. **Try model-based speculation** (EAGLE/MTP) first
2. **If no model support**, fall back to n-gram
3. **Result**: Always get some speculation benefit

This is useful for production systems serving diverse models.

## Cross-References

- **Parent concept**: [[Speculative Decoding]]
- **Alternative methods**: [[EAGLE]], [[Multi-Token Prediction]], [[Draft Model Speculation]]
- **Related optimizations**: [[CUDA Graphs]], [[Prefix Caching]], [[Kernel Fusions]]
- **System integration**: [[vLLM Engine]], [[Continuous Batching]]

## Resources

- **vLLM docs**: `docs/features/speculative_decoding/n_gram.md`
- **Example script**: `examples/offline_inference/spec_decode.py`
- **Community thread**: [joao_gante/status/1747322413006643259](https://x.com/joao_gante/status/1747322413006643259)
- **Blog post**: "Prompt Lookup Decoding" (original n-gram speculation technique)
