---
title: EAGLE
type: technique
tags: [speculative-decoding, optimization, latency]
created: 2026-04-25
---

# EAGLE — Extrapolation Algorithm for Greater Language-model Efficiency

EAGLE is a speculative decoding method that uses lightweight prediction heads trained on the target model's hidden states to generate draft tokens without requiring a separate draft model.

## Overview

EAGLE (and its successor EAGLE-2) provides model-based speculative decoding with **no separate draft model**. Instead, EAGLE learns to predict future tokens by extrapolating from the target model's intermediate activations.

**Key advantages**:
- No need to load/manage a separate draft model
- Lower memory footprint than traditional draft model speculation
- High acceptance rate due to direct training on target model's features
- Works across model families with pre-trained EAGLE heads

**Speedup**: High gain at low QPS (latency-focused), medium to high gain at high QPS (throughput-focused)

## How EAGLE Works

### Architecture

1. **Target model forward pass**: Run target model on current context
2. **Feature extraction**: Extract hidden states from target model's intermediate layers
3. **EAGLE head prediction**: Lightweight MLP/transformer heads predict next K tokens from hidden states
4. **Verification**: Target model verifies all K candidates in parallel in single forward pass
5. **Acceptance**: Use rejection sampling to accept/reject candidates

The EAGLE head is typically:
- 1-3 transformer layers or MLP blocks
- Trained on target model's hidden state → next token mapping
- Much smaller than a full draft model (few hundred MB vs several GB)

### EAGLE vs EAGLE-2

**EAGLE (single-path)**:
- Generates a linear sequence of K candidate tokens
- Simple, fast, well-supported

**EAGLE-2 (tree-based)**:
- Generates a tree of candidate continuations (multiple branches per step)
- Higher acceptance rate through exploring multiple hypotheses
- More complex verification logic

**EAGLE-3**:
- Latest variant with improved architecture
- Example: `RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3`

## Configuration in vLLM

### Python API

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 2,
        "method": "eagle",
    },
)

outputs = llm.generate(prompts, sampling_params)
```

### EAGLE-3 Example

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=2,
    speculative_config={
        "model": "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        "draft_tensor_parallel_size": 2,
        "num_speculative_tokens": 2,
        "method": "eagle3",
    },
)
```

### Command-Line Interface

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --tensor-parallel-size 4 \
  --speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
    "num_speculative_tokens": 2,
    "draft_tensor_parallel_size": 1
  }'
```

### Configuration Parameters

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `method` | `string` | — | Set to `"eagle"` or `"eagle3"` |
| `model` | `string` | — | Path to pre-trained EAGLE head on HuggingFace |
| `num_speculative_tokens` | `integer` | — | Number of tokens to speculate per step (typical: 2-4) |
| `draft_tensor_parallel_size` | `integer` | `None` | TP size for EAGLE head (typically 1-2) |
| `parallel_drafting` | `boolean` | `false` | Enable parallel draft generation (EAGLE-2/3) |

## Pre-Trained EAGLE Heads

Multiple organizations provide pre-trained EAGLE heads on HuggingFace:

### RedHatAI Collection
- [RedHatAI/speculator-models](https://huggingface.co/collections/RedHatAI/speculator-models)
- EAGLE-3 heads for Llama 3.1, Qwen, Mistral families
- Optimized for production use

### yuhuili Collection
- [yuhuili/models (EAGLE)](https://huggingface.co/yuhuili/models?search=eagle)
- Original EAGLE heads for Llama 2/3, Vicuna, CodeLlama
- Reference implementations from EAGLE paper authors

### Compatibility Notes

- **vLLM ≥0.7.0**: Use EAGLE heads directly from HuggingFace
- **vLLM <0.7.0**: Use [conversion script](https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d) to modify checkpoint format

## When to Use EAGLE

### Best Use Cases

✅ **Use EAGLE when**:
- You want high-quality speculative decoding without managing a separate draft model
- Target model has pre-trained EAGLE heads available
- Latency is more important than peak throughput
- GPU memory is constrained (EAGLE head <<< full draft model)

✅ **EAGLE excels at**:
- Low-to-medium QPS workloads (latency-sensitive)
- Memory-bound decode phases
- Models with good EAGLE head coverage (Llama, Qwen, Mistral families)

### When to Consider Alternatives

❌ **Consider alternatives when**:
- No pre-trained EAGLE head exists for your model → use [[Draft Model Speculation]] or [[N-gram Speculation]]
- Model has native MTP support (DeepSeek-V3, Qwen3) → use [[Multi-Token Prediction]]
- Very high QPS with GPU saturation → use [[N-gram Speculation]] (zero overhead)
- Extremely tight latency SLO → measure E2E; EAGLE adds small overhead vs baseline

## Performance Tuning

### Optimal num_speculative_tokens

Start with **2-3 tokens** and increase based on acceptance rate:

```python
# Conservative (high acceptance rate)
"num_speculative_tokens": 2  # 75-85% acceptance typical

# Moderate (balanced)
"num_speculative_tokens": 3  # 65-75% acceptance

# Aggressive (lower acceptance, may still improve E2E)
"num_speculative_tokens": 4  # 50-65% acceptance
```

**Rule of thumb**: Increase until acceptance rate drops below 60%, then back off one step.

### Tensor Parallelism for EAGLE Head

EAGLE heads are small, so **TP=1 is usually optimal**:

```python
speculative_config={
    "draft_tensor_parallel_size": 1,  # Default: run on single GPU
}
```

Only increase TP if:
- EAGLE head is unusually large (>1B parameters)
- Target model uses TP=8+ (minor load balancing benefit)

### Parallel Drafting

EAGLE-2/3 support parallel tree-based drafting:

```python
speculative_config={
    "method": "eagle3",
    "parallel_drafting": True,  # Enable tree-based speculation
}
```

**Trade-off**: Higher acceptance rate, but more complex verification logic. Test on your workload.

## Expected Speedups

Typical speedups measured on vLLM benchmarks:

| Workload | QPS | Speedup Range | Notes |
|----------|-----|---------------|-------|
| Chatbot (Llama-3-8B) | Low (1-10) | 1.8-2.5× | EAGLE-LLaMA3, num_spec=3 |
| Code generation | Low (1-5) | 2.0-2.8× | CodeLlama, num_spec=4 |
| Q&A (Qwen-8B) | Medium (10-50) | 1.5-2.0× | EAGLE-Qwen, num_spec=2 |
| High-throughput | High (100+) | 1.2-1.6× | Limited by GPU saturation |

**Factors affecting speedup**:
- **Sequence length**: Longer sequences → better amortization → higher speedup
- **Batch size**: Smaller batches → more spare GPU capacity → higher speedup
- **Sampling temperature**: Lower temperature → higher acceptance → higher speedup

## Integration with Other Optimizations

EAGLE composes well with most vLLM optimizations:

✅ **Compatible**:
- [[Prefix Caching]]: Shared prefixes reused, EAGLE only runs on unique suffixes
- [[CUDA Graphs]]: EAGLE verification can be graph-captured
- [[Kernel Fusions]]: EAGLE benefits from fused attention/norm kernels
- [[Tensor Parallelism]]: Both target and EAGLE head support TP

❌ **Incompatible**:
- [[Pipeline Parallelism]]: PP not supported with speculative decoding (vLLM ≤0.15.0)

⚠️ **Trade-offs**:
- [[Continuous Batching]]: High batch sizes reduce EAGLE benefit (GPU saturated)
- [[Chunked Prefill]]: EAGLE only helps decode phase, not prefill

## Debugging and Monitoring

### Measuring Acceptance Rate

Use vLLM's offline inference script to extract per-request metrics:

```bash
python examples/offline_inference/spec_decode.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --speculative-config '{"method":"eagle","model":"yuhuili/EAGLE-LLaMA3-Instruct-8B","num_speculative_tokens":3}'
```

Output includes:
- Per-request acceptance rate
- Average tokens accepted per step
- E2E latency improvement

### Common Issues

**Low acceptance rate (<50%)**:
- EAGLE head may not match target model variant (base vs instruct)
- num_speculative_tokens too high
- High sampling temperature reducing alignment

**No speedup despite high acceptance**:
- GPU saturated from high QPS
- EAGLE head overhead dominates (check draft_tensor_parallel_size)
- Sequences too short to amortize overhead

**Out of memory**:
- Reduce draft_tensor_parallel_size to 1
- Reduce num_speculative_tokens
- Use smaller EAGLE head variant

## Training Custom EAGLE Heads

For training EAGLE heads on new model families:

1. Clone [vllm-project/speculators](https://github.com/vllm-project/speculators)
2. Collect target model hidden states on representative dataset
3. Train lightweight prediction head (1-3 layers)
4. Export to HuggingFace format compatible with vLLM
5. Validate acceptance rate on held-out data

See speculators repo for detailed training scripts.

## Cross-References

- **Parent concept**: [[Speculative Decoding]]
- **Alternative methods**: [[Multi-Token Prediction]], [[Draft Model Speculation]], [[N-gram Speculation]]
- **Related optimizations**: [[CUDA Graphs]], [[Prefix Caching]], [[Kernel Fusions]]
- **System integration**: [[vLLM Engine]], [[Continuous Batching]]

## Resources

- **Paper**: [EAGLE: Lossless Acceleration of LLM Decoding by Feature Extrapolation (Li et al., 2024)](https://arxiv.org/pdf/2401.15077)
- **Pre-trained heads**: [RedHatAI/speculator-models](https://huggingface.co/collections/RedHatAI/speculator-models), [yuhuili/models](https://huggingface.co/yuhuili/models?search=eagle)
- **Training code**: [vllm-project/speculators](https://github.com/vllm-project/speculators)
- **Example**: `examples/offline_inference/spec_decode.py`
- **Conversion script** (vLLM <0.7.0): [Gist](https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d)
