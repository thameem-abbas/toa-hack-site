---
title: Draft Model Speculation
type: technique
tags: [speculative-decoding, optimization, latency]
created: 2026-04-25
---

# Draft Model Speculation

Draft model speculation is the classic speculative decoding approach where a smaller, faster draft model generates K candidate tokens that a larger target model then verifies in parallel in a single forward pass.

## Overview

Draft model speculation is the **original and most well-understood** speculative decoding method. It uses a separate, smaller model (typically from the same family) to quickly generate candidate tokens that the target model validates.

**Key advantages**:
- High acceptance rate when draft/target models are well-aligned
- Well-studied, predictable behavior
- Works across any model family (no special training required)
- Broad HuggingFace model availability (e.g., Qwen3-0.6B + Qwen3-8B)

**Speedup**: High gain at low QPS (latency-focused), medium gain at high QPS (throughput-focused)

## How Draft Model Speculation Works

### Basic Algorithm

1. **Draft phase**: Draft model autoregressively generates K candidate tokens
   - Input: Current context (prompt + generated tokens so far)
   - Output: K proposed tokens with their probabilities
2. **Verification phase**: Target model processes all K candidates in parallel
   - Input: Context + all K draft tokens concatenated
   - Output: Target model's probabilities for all K+1 positions
3. **Acceptance phase**: Rejection sampling determines how many tokens to accept
   - For each position i, accept with probability `min(1, p_target[i] / p_draft[i])`
   - Stop at first rejection, discard remaining candidates
4. **Correction phase**: If rejected, sample from adjusted distribution `max(0, p_target - p_draft)`

### Example Walkthrough

```
Context: "The capital of France is"

Draft model generates: ["Paris", ",", "located", "in", "Europe"]  (K=5)

Target model verifies all 5 in parallel:
  Position 0: "Paris"   → p_target=0.95, p_draft=0.90 → ACCEPT (ratio=1.05 > rand)
  Position 1: ","       → p_target=0.88, p_draft=0.85 → ACCEPT (ratio=1.03 > rand)
  Position 2: "located" → p_target=0.45, p_draft=0.70 → REJECT (ratio=0.64 < rand)

Final output: ["Paris", ","] + resample position 2 from adjusted distribution
Tokens accepted: 2 out of 5 (40% acceptance rate)
```

### Parallel Draft Model (PARD)

**PARD** is a variant where the draft model generates tokens **in parallel** rather than autoregressively:

- Traditional draft: Token i depends on tokens 0..i-1 (sequential)
- PARD draft: All K tokens generated simultaneously (parallel)
- **Benefit**: Lower draft model latency (K forward passes → 1 forward pass)
- **Trade-off**: May have slightly lower acceptance rate (less context per token)

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
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 5,
        "method": "draft_model",
    },
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### Command-Line Interface — Online Serving

```bash
vllm serve Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --speculative-config '{
      "model": "Qwen/Qwen3-0.6B",
      "num_speculative_tokens": 5,
      "method": "draft_model"
    }'
```

Client code remains unchanged (uses standard OpenAI API).

### Configuration Parameters

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `method` | `string` | — | Set to `"draft_model"` |
| `model` | `string` | — | Path to draft model on HuggingFace or local filesystem |
| `num_speculative_tokens` | `integer` | — | Number of tokens to speculate per step (typical: 3-8) |
| `draft_tensor_parallel_size` | `integer` | `None` | TP size for draft model (usually 1) |
| `max_model_len` | `integer` | `None` | Maximum context length for draft model (defaults to target) |
| `parallel_drafting` | `boolean` | `false` | Enable PARD (parallel draft token generation) |
| `rejection_sample_method` | `string` | `"strict"` | `strict`, `probabilistic`, or `synthetic` |

### Deprecated Configuration (vLLM ≤0.10.0)

⚠️ **Old style** (no longer recommended):
```bash
vllm serve Qwen/Qwen3-8B \
  --speculative-model Qwen/Qwen3-0.6B \
  --num-speculative-tokens 5
```

**New style** (vLLM ≥0.11.0):
```bash
vllm serve Qwen/Qwen3-8B \
  --speculative-config '{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":5}'
```

## Choosing a Draft Model

### Model Family Alignment

**Best practice**: Use a smaller model from the **same family** as the target model.

✅ **Good pairings**:
- Qwen3-0.6B → Qwen3-8B (same architecture, tokenizer, training data)
- Llama-3-1B → Llama-3-8B
- Mistral-7B → Mixtral-8x7B (MoE variant)

❌ **Poor pairings**:
- GPT-2-125M → Llama-3-8B (different tokenizer, training data, architecture)
- Qwen-0.6B → Mistral-7B (different families)

### Size Ratio

**Rule of thumb**: Draft model should be **5-15× smaller** than target model.

- **Too small** (e.g., 125M for 70B target): Low acceptance rate, wasted verification
- **Too large** (e.g., 7B for 13B target): Draft overhead dominates, little speedup
- **Sweet spot**: 0.6B-1B draft for 7B-13B target, 3B draft for 30B-70B target

### HuggingFace Model Availability

Common draft model families:

| Target Model Family | Recommended Draft Models |
|---------------------|--------------------------|
| Qwen3 | Qwen3-0.6B, Qwen3-1.8B |
| Llama 3 | Llama-3-1B (community), TinyLlama-1.1B |
| Mistral | Mistral-0.5B (if available), Qwen-0.6B (cross-family, lower acceptance) |
| DeepSeek | DeepSeek-1B (same family) |
| CodeLlama | CodeLlama-7B → CodeLlama-34B |

## When to Use Draft Model Speculation

### Best Use Cases

✅ **Use draft model speculation when**:
- No pre-trained EAGLE/MTP heads available for your target model
- You need predictable, well-understood behavior
- High-quality draft model exists in same family
- Latency is more important than peak throughput
- Willing to manage two models (target + draft)

✅ **Draft model excels at**:
- Low-to-medium QPS workloads (latency-sensitive)
- Long output sequences (amortizes overhead)
- Greedy or low-temperature sampling (high acceptance rate)

### When to Consider Alternatives

❌ **Consider alternatives when**:
- Target model has pre-trained EAGLE heads → use [[EAGLE]]
- Model has native MTP support (DeepSeek-V3, Qwen3) → use [[Multi-Token Prediction]]
- Very high QPS with GPU saturation → use [[N-gram Speculation]] (zero overhead)
- No good draft model in same family → use [[N-gram Speculation]] or heuristic methods
- GPU memory is constrained → use [[EAGLE]] (smaller speculator)

## Performance Tuning

### Optimal num_speculative_tokens

Start with **4-6 tokens** and tune based on acceptance rate and E2E latency:

```python
# Conservative (high acceptance rate)
"num_speculative_tokens": 3  # 70-80% acceptance typical

# Moderate (balanced)
"num_speculative_tokens": 5  # 60-70% acceptance

# Aggressive (lower acceptance, may still improve E2E)
"num_speculative_tokens": 8  # 45-60% acceptance
```

**Key insight**: Even with 50% acceptance rate, speculating 6 tokens may give 3× speedup if verification is cheap.

### Draft Model Tensor Parallelism

Draft models are typically small, so **TP=1 is optimal**:

```python
speculative_config={
    "draft_tensor_parallel_size": 1,  # Single GPU for draft
}
```

Only increase TP if:
- Draft model is unusually large (>3B parameters)
- Target model uses high TP (e.g., TP=8), minor load balancing benefit

### Parallel Drafting (PARD)

Enable PARD to reduce draft model latency:

```python
speculative_config={
    "parallel_drafting": True,  # Generate all K tokens in parallel
}
```

**Trade-off**: Lower draft latency, but may reduce acceptance rate (less context per token). Test on your workload.

### Rejection Sampling Method

```python
speculative_config={
    "rejection_sample_method": "strict",  # Default: mathematically correct
}
```

Options:
- `"strict"`: Standard rejection sampling (lossless, default)
- `"probabilistic"`: Probabilistic acceptance (faster, may deviate slightly)
- `"synthetic"`: Synthetic acceptance rate (for testing/simulation)

## Expected Speedups

Typical speedups measured on vLLM benchmarks:

| Workload | QPS | Speedup Range | Notes |
|----------|-----|---------------|-------|
| Chatbot (Qwen3-0.6B→8B) | Low (1-10) | 2.0-3.0× | num_spec=5, greedy sampling |
| Code generation | Low (1-5) | 2.2-3.2× | High repetition, num_spec=6 |
| Q&A (short responses) | Low (5-20) | 1.5-2.2× | num_spec=4 |
| High-throughput | High (100+) | 0.9-1.3× | Draft overhead dominates, limited benefit |

**Factors affecting speedup**:
- **Acceptance rate**: Higher acceptance → higher speedup (target >60%)
- **Sequence length**: Longer sequences → better amortization → higher speedup
- **Batch size**: Smaller batches → more spare GPU capacity → higher speedup
- **Draft model quality**: Same-family draft → higher acceptance

## Integration with Other Optimizations

Draft model speculation composes with most vLLM optimizations:

✅ **Compatible**:
- [[Prefix Caching]]: Both target and draft benefit from shared prefix reuse
- [[CUDA Graphs]]: Verification can be graph-captured
- [[Kernel Fusions]]: Both models benefit from fused kernels
- [[Tensor Parallelism]]: Both target and draft support TP

❌ **Incompatible**:
- [[Pipeline Parallelism]]: PP not supported with speculative decoding (vLLM ≤0.15.0)

⚠️ **Trade-offs**:
- [[Continuous Batching]]: High batch sizes reduce draft benefit (GPU saturated)
- [[Chunked Prefill]]: Draft only helps decode phase, not prefill
- Memory pressure: Two models consume more GPU memory than EAGLE/MTP

## Debugging and Monitoring

### Measuring Acceptance Rate

Use vLLM's offline inference script:

```bash
python examples/offline_inference/spec_decode.py \
  --model Qwen/Qwen3-8B \
  --speculative-config '{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":5}'
```

Output includes:
- Per-request acceptance rate
- Average tokens accepted per step
- E2E latency improvement
- Draft model overhead

### Common Issues

**Low acceptance rate (<50%)**:
- Draft model not from same family as target
- Draft model too small or poorly aligned
- High sampling temperature reducing alignment
- num_speculative_tokens too high

**No speedup despite high acceptance**:
- GPU saturated from high QPS (draft overhead dominates)
- Sequences too short to amortize draft model cost
- Draft model too large (overhead > verification savings)

**Out of memory**:
- Both target and draft loaded simultaneously
- Reduce draft_tensor_parallel_size to 1
- Use smaller draft model
- Consider [[EAGLE]] (smaller speculator) or [[N-gram Speculation]] (no model)

**Draft model loading errors**:
- Ensure draft model architecture is supported by vLLM
- Check HuggingFace model card for compatibility
- Verify model path is correct

## Comparison to Other Methods

| Method | Acceptance Rate | Memory Overhead | Setup Complexity | Speedup (Low QPS) |
|--------|-----------------|-----------------|------------------|-------------------|
| Draft model | 60-80% | High (2 models) | Medium (find draft) | 2.0-3.0× |
| [[EAGLE]] | 70-90% | Low (EAGLE head) | Medium (find head) | 1.8-2.5× |
| [[Multi-Token Prediction]] | 75-85% | None (native) | Low (if supported) | 2.0-2.8× |
| [[N-gram Speculation]] | 30-60% | None | Low (always works) | 1.2-1.8× |

**Draft model strengths**:
- High acceptance rate with good draft model
- Works on any model family
- Predictable, well-understood

**Draft model weaknesses**:
- Higher memory overhead (two full models)
- Medium setup complexity (finding/testing draft models)
- Limited high-QPS benefit (draft overhead)

## Cross-References

- **Parent concept**: [[Speculative Decoding]]
- **Alternative methods**: [[EAGLE]], [[Multi-Token Prediction]], [[N-gram Speculation]]
- **Related optimizations**: [[CUDA Graphs]], [[Prefix Caching]], [[Kernel Fusions]]
- **System integration**: [[vLLM Engine]], [[Continuous Batching]]

## Resources

- **Paper**: [Accelerating Large Language Model Decoding with Speculative Sampling (Leviathan et al., 2023)](https://arxiv.org/pdf/2302.01318)
- **vLLM docs**: `docs/features/speculative_decoding/draft_model.md`
- **Example script**: `examples/offline_inference/spec_decode.py`
- **Benchmark guide**: `docs/benchmarking/cli.md`
- **HuggingFace models**: Search for `{family}-{small-size}` (e.g., `Qwen3-0.6B`, `Llama-3-1B`)
