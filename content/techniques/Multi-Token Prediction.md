---
title: Multi-Token Prediction
type: technique
tags: [speculative-decoding, optimization, latency, mtp]
created: 2026-04-25
---

# Multi-Token Prediction (MTP)

Multi-Token Prediction (MTP) is a speculative decoding method where the target model is natively trained to predict multiple tokens per forward pass, eliminating the need for a separate draft model or speculator.

## Overview

MTP is the **simplest form of speculative decoding** when the model supports it: the model itself generates multiple candidate tokens in parallel during each forward pass.

**Key advantages**:
- **Zero overhead**: No separate draft model, EAGLE head, or heuristic logic
- **Native integration**: Uses model's own MTP heads (built-in during training)
- **High acceptance rate**: Model predicts its own next tokens (no alignment mismatch)
- **Minimal configuration**: Just enable MTP in speculative_config

**Speedup**: High gain at low QPS (latency-focused), medium to high gain at high QPS (throughput-focused)

## How MTP Works

### Architecture

MTP models include **additional prediction heads** during training:

1. **Standard head**: Predicts token at position i from hidden state at i-1 (traditional autoregressive)
2. **MTP head 1**: Predicts token at position i+1 from hidden state at i-1
3. **MTP head 2**: Predicts token at position i+2 from hidden state at i-1
4. ...and so on for K MTP heads

At inference time:
1. Run model forward pass on current context
2. Standard head predicts token at position i
3. MTP heads predict tokens at positions i+1, i+2, ..., i+K in parallel
4. Verify all K+1 predictions (standard + MTP) using standard rejection sampling

### Training Procedure

MTP models are trained with **multi-token prediction loss**:

```
Loss = L_standard(token_i | context) + 
       λ_1 * L_MTP1(token_{i+1} | context) +
       λ_2 * L_MTP2(token_{i+2} | context) +
       ...
```

The MTP heads learn to predict future tokens from the same hidden state, sharing most weights with the main model (only the prediction heads differ).

### Difference from Draft Model Speculation

| Aspect | Draft Model Speculation | MTP |
|--------|------------------------|-----|
| **Training** | Standard autoregressive | Multi-token prediction loss |
| **Inference** | Two models (target + draft) | Single model with MTP heads |
| **Memory** | 2× model size | 1× model size + small MTP heads |
| **Acceptance rate** | 60-80% (alignment dependent) | 75-85% (self-prediction) |
| **Setup** | Find/load draft model | Enable flag |
| **Availability** | Any model (if draft exists) | Only MTP-trained models |

## Configuration in vLLM

### Python API — Offline Mode

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="XiaomiMiMo/MiMo-7B-Base",
    tensor_parallel_size=1,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
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
vllm serve XiaomiMiMo/MiMo-7B-Base \
    --tensor-parallel-size 1 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

### Configuration Parameters

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `method` | `string` | — | Set to `"mtp"` |
| `num_speculative_tokens` | `integer` | — | Number of MTP heads to use (typical: 1-3) |

**Note**: Unlike draft model speculation, you do **not** need to specify a `model` parameter (the target model itself provides MTP).

## Models with Native MTP Support

### Currently Supported in vLLM

| Model Family | Example Model | MTP Heads | Notes |
|--------------|---------------|-----------|-------|
| **DeepSeek-V3** | `deepseek-ai/DeepSeek-V3` | Multiple | 671B MoE, production-grade MTP |
| **Qwen3** | `Qwen/Qwen3-*` | Varies by checkpoint | Some Qwen3 variants have MTP |
| **MiMo** | `XiaomiMiMo/MiMo-7B-Base` | 1-2 | Xiaomi's MTP-trained models |

**Checking MTP support**: Look for `mtp_heads` or `multi_token_prediction` in model config.json on HuggingFace.

### Availability Trend

MTP is an **emerging architecture**:
- **2023-2024**: Few models (research prototypes)
- **2025**: DeepSeek-V3, Qwen3 adopt MTP for production
- **2026+**: Expected to become standard for frontier models (lower training cost for high inference speedup)

If your target model does not support MTP, use [[EAGLE]] or [[Draft Model Speculation]].

## When to Use MTP

### Best Use Cases

✅ **Use MTP when**:
- Your target model natively supports MTP (DeepSeek-V3, Qwen3, MiMo)
- You want the simplest possible speculative decoding setup
- GPU memory is constrained (no separate draft model)
- Latency is more important than peak throughput

✅ **MTP excels at**:
- Low-to-medium QPS workloads (latency-sensitive)
- Memory-bound decode phases
- Any workload where the model supports MTP (no alignment issues)

### When to Consider Alternatives

❌ **Consider alternatives when**:
- Model does not support MTP → use [[EAGLE]] or [[Draft Model Speculation]]
- Very high QPS with GPU saturation → use [[N-gram Speculation]] (truly zero overhead)
- Need to tune draft model separately → use [[Draft Model Speculation]]

## Performance Tuning

### Optimal num_speculative_tokens

MTP models typically have **1-3 MTP heads**. Use all available heads:

```python
# If model has 1 MTP head
"num_speculative_tokens": 1  # Use all available

# If model has 3 MTP heads
"num_speculative_tokens": 3  # Use all available
```

**Key insight**: Unlike draft model speculation, you **cannot exceed** the number of MTP heads built into the model. Check model documentation for the exact number.

**Acceptance rate per head**:
- Head 1 (i+1): 75-85% acceptance (close prediction)
- Head 2 (i+2): 65-75% acceptance (farther prediction)
- Head 3 (i+3): 55-65% acceptance (even farther)

### Start Conservative

If unsure, start with `num_speculative_tokens=1` and increase:

```python
# Conservative: Use only first MTP head
"num_speculative_tokens": 1  # Highest acceptance rate

# Moderate: Use first two MTP heads
"num_speculative_tokens": 2  # Balanced

# Aggressive: Use all available MTP heads
"num_speculative_tokens": 3  # Maximum speculation depth
```

Monitor E2E latency; more heads ≠ always better (depends on acceptance rate).

## Expected Speedups

Typical speedups measured on MTP-enabled models:

| Workload | QPS | Speedup Range | Notes |
|----------|-----|---------------|-------|
| Chatbot (DeepSeek-V3) | Low (1-10) | 2.0-2.8× | num_spec=2-3, greedy sampling |
| Code generation | Low (1-5) | 2.2-3.0× | High repetition, num_spec=3 |
| Q&A (Qwen3-MTP) | Medium (10-50) | 1.6-2.2× | num_spec=1-2 |
| High-throughput | High (100+) | 1.2-1.6× | Limited by GPU saturation |

**Factors affecting speedup**:
- **Number of MTP heads**: More heads → higher potential speedup (if acceptance rate remains high)
- **Sequence length**: Longer sequences → better amortization → higher speedup
- **Batch size**: Smaller batches → more spare GPU capacity → higher speedup
- **Sampling temperature**: Lower temperature → higher acceptance → higher speedup

## Integration with Other Optimizations

MTP composes well with most vLLM optimizations:

✅ **Compatible**:
- [[Prefix Caching]]: MTP only runs on unique suffixes, prefixes reused
- [[CUDA Graphs]]: MTP verification can be graph-captured
- [[Kernel Fusions]]: MTP benefits from fused attention/norm kernels
- [[Tensor Parallelism]]: MTP heads support TP
- [[Mixture of Experts]]: DeepSeek-V3 combines MTP + MoE

❌ **Incompatible**:
- [[Pipeline Parallelism]]: PP not supported with speculative decoding (vLLM ≤0.15.0)

⚠️ **Trade-offs**:
- [[Continuous Batching]]: High batch sizes reduce MTP benefit (GPU saturated)
- [[Chunked Prefill]]: MTP only helps decode phase, not prefill

## Debugging and Monitoring

### Verifying MTP is Enabled

Check vLLM logs at startup:

```
INFO: Speculative decoding enabled with method=mtp, num_speculative_tokens=2
INFO: Model has 3 MTP heads available
```

### Measuring Acceptance Rate

Use vLLM's offline inference script:

```bash
python examples/offline_inference/spec_decode.py \
  --model XiaomiMiMo/MiMo-7B-Base \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Output includes:
- Per-request acceptance rate per MTP head
- Average tokens accepted per step
- E2E latency improvement

### Common Issues

**Model does not support MTP**:
- Error: `Model does not have MTP heads`
- **Solution**: Use [[EAGLE]] or [[Draft Model Speculation]] instead

**Low acceptance rate (<60%)**:
- num_speculative_tokens exceeds reliable MTP head count
- High sampling temperature reducing alignment
- **Solution**: Reduce num_speculative_tokens to 1 or lower temperature

**No speedup despite high acceptance**:
- GPU saturated from high QPS
- Sequences too short to amortize MTP overhead
- **Solution**: Test on lower QPS or longer sequences

**Out of memory**:
- MTP heads increase memory footprint (small but nonzero)
- **Solution**: Reduce batch size or tensor parallel size

## Comparison to Other Methods

| Method | Acceptance Rate | Memory Overhead | Setup Complexity | Speedup (Low QPS) |
|--------|-----------------|-----------------|------------------|-------------------|
| [[Draft Model Speculation]] | 60-80% | High (2 models) | Medium (find draft) | 2.0-3.0× |
| [[EAGLE]] | 70-90% | Low (EAGLE head) | Medium (find head) | 1.8-2.5× |
| **MTP** | 75-85% | None (native) | Low (if supported) | 2.0-2.8× |
| [[N-gram Speculation]] | 30-60% | None | Low (always works) | 1.2-1.8× |

**MTP strengths**:
- Simplest setup (if model supports MTP)
- No separate model/head to manage
- High acceptance rate (self-prediction)

**MTP weaknesses**:
- Limited model availability (only MTP-trained models)
- Cannot tune draft model separately
- Fixed number of MTP heads (cannot exceed model's built-in count)

## Training MTP Models

For organizations training custom models, MTP can be added during training:

1. **Modify loss function**: Add multi-token prediction terms
2. **Add MTP heads**: Lightweight prediction heads for positions i+1, i+2, ...
3. **Joint training**: Train standard + MTP heads simultaneously
4. **Export model**: Standard HuggingFace checkpoint with `mtp_heads` config

**Training cost**: MTP adds ~10-20% training overhead (more forward passes per sample) but provides 2-3× inference speedup.

**Resources**:
- DeepSeek-V3 paper (MTP architecture details)
- Meta's multi-token prediction research (original MTP paper)

## Cross-References

- **Parent concept**: [[Speculative Decoding]]
- **Alternative methods**: [[EAGLE]], [[Draft Model Speculation]], [[N-gram Speculation]]
- **Related optimizations**: [[CUDA Graphs]], [[Prefix Caching]], [[Kernel Fusions]]
- **System integration**: [[vLLM Engine]], [[Continuous Batching]]
- **Model architectures**: [[Mixture of Experts]] (DeepSeek-V3 combines MTP + MoE)

## Resources

- **vLLM docs**: `docs/features/speculative_decoding/mtp.md`
- **Example script**: `examples/offline_inference/spec_decode.py`
- **Models**: [XiaomiMiMo/MiMo-7B-Base](https://huggingface.co/XiaomiMiMo/MiMo-7B-Base), DeepSeek-V3, Qwen3 (check model card)
- **Paper**: Multi-token prediction research (Meta, DeepSeek)
