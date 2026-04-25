---
title: INT8 W8A8
type: technique
created: 2026-04-25
---

# INT8 W8A8

INT8 W8A8 quantization reduces model weights and activations from FP16/BF16 precision to 8-bit integers (INT8), achieving 2× memory reduction with minimal accuracy loss. This method is particularly useful for NVIDIA Turing and Ampere GPUs that lack native FP8 support, and for deployments requiring integer-only arithmetic on CPUs.

## Overview

INT8 W8A8 quantizes both weights and activations to 8-bit signed integers, using calibration data to determine optimal quantization scales for activations. The technique combines:
- **Static per-channel weight quantization**: Scales computed once during quantization
- **Dynamic per-token activation quantization**: Scales computed during each forward pass
- **SmoothQuant activation smoothing**: Mitigates outliers by balancing activation ranges

## Hardware Support

| Platform | Support | Compute Capability | Notes |
|----------|---------|-------------------|-------|
| NVIDIA Turing | ✅ | SM 7.5 | INT8 Tensor Cores |
| NVIDIA Ampere | ✅ | SM 8.0/8.6 | INT8 Tensor Cores |
| NVIDIA Ada | ✅ | SM 8.9 | INT8 Tensor Cores (FP8 preferred) |
| NVIDIA Hopper | ✅ | SM 9.0 | INT8 Tensor Cores (FP8 preferred) |
| NVIDIA Blackwell | ❌ | SM 10.0 | Not supported — use [[FP8 Quantization]] |
| AMD MI300+ | ❌ | — | Not supported — use [[FP8 Quantization]] |
| Intel GPU | ❌ | — | Not supported |
| x86 CPU | ✅ | — | CPU-based INT8 kernels |

**Key limitation**: INT8 is not supported on Blackwell (RTX 6000 Blackwell, SM 10.0+). Use [[FP8 Quantization]] for Blackwell deployments.

## Quantization Scheme

### Weight Quantization (Static)
- **Precision**: INT8 (-128 to 127)
- **Granularity**: Per-channel (one scale per output channel)
- **Method**: GPTQ (Generalized Post-Training Quantization) with calibration data
- **Scale storage**: FP16 scales stored alongside INT8 weights

### Activation Quantization (Dynamic)
- **Precision**: INT8 (-128 to 127)
- **Granularity**: Per-token (one scale per token in the sequence)
- **Method**: Dynamic quantization during forward pass
- **Smoothing**: SmoothQuant with `smoothing_strength` parameter

## SmoothQuant

SmoothQuant is a key technique for INT8 activation quantization. It addresses the challenge of activation outliers by smoothing activation ranges.

### Problem
Activations often have outliers (e.g., 10× larger than typical values), making uniform quantization inaccurate.

### Solution
SmoothQuant migrates the quantization difficulty from activations to weights by:
1. Computing per-channel smoothing factors: `s = max(|X|)^α / max(|W|)^(1-α)`
2. Transforming: `Y = (X / s) @ (W * s)`
3. Quantizing smoother activations `X / s` and adjusted weights `W * s`

### Smoothing Strength Parameter
```python
SmoothQuantModifier(smoothing_strength=0.8)
```

- **α = 0.8** (default): Balanced smoothing
- **α = 0.5**: Equal smoothing between activations and weights
- **α = 1.0**: All smoothing applied to weights (no activation smoothing)

**Best practice**: Start with α = 0.8; increase if accuracy is poor.

## Quantization Workflow

### Prerequisites
```bash
pip install llmcompressor
pip install vllm "lm-eval[api]>=0.4.11"
```

### Step 1: Load Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### Step 2: Prepare Calibration Data

INT8 W8A8 **requires calibration data** to estimate activation scales.

```python
from datasets import load_dataset

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

# Preprocess: apply chat template
def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

# Tokenize
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False
    )
ds = ds.map(tokenize, remove_columns=ds.column_names)
```

**Calibration data selection**:
- Use data representative of deployment workload
- For instruction-tuned models: `ultrachat_200k` or similar chat datasets
- For fine-tuned models: Sample from your training data
- Typical: 512 samples, 2048 max tokens

### Step 3: Apply Quantization

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

# Configure quantization recipe
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
```

**Recipe explanation**:
1. `SmoothQuantModifier`: Smooths activations before quantization
2. `GPTQModifier`: Applies INT8 quantization to all `Linear` layers (except `lm_head`)

### Step 4: Save Quantized Model

```python
SAVE_DIR = "Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

### Step 5: Deploy in vLLM

```python
from vllm import LLM

llm = LLM("./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token")
result = llm.generate("Hello, my name is")
print(result[0].outputs[0].text)
```

## Accuracy Evaluation

Quantized models are sensitive to evaluation configuration.

```bash
lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

**Critical**: Include `add_bos_token=true` — quantized models can be sensitive to the presence of the BOS token.

## Performance Metrics

### Memory Savings
- **Model weights**: 2× reduction (FP16 → INT8)
- **Activations**: No reduction (dynamic quantization)
- **Total**: ~2× end-to-end memory reduction

### Throughput
- **Speedup**: Variable (depends on model size, batch size, sequence length)
- **Turing/Ampere**: Best platform for INT8 (no FP8 support)
- **Ada/Hopper**: [[FP8 Quantization]] preferred (higher throughput)

### Accuracy
- **With SmoothQuant**: Typically within 1-2% of FP16 baseline
- **Without SmoothQuant**: May see 3-5% accuracy drop due to activation outliers

## Best Practices

### 1. Calibration Data
- **Start with 512 samples**: Good balance between calibration quality and time
- **Increase if accuracy drops**: Try 1024 or 2048 samples
- **Match deployment data**: Use chat template, sequence length distribution similar to production

### 2. Sequence Length
- **Start with 2048 tokens**: Captures typical context lengths
- **Adjust based on use case**: Shorter for chat, longer for document analysis

### 3. Chat Template
- **Use model's template**: Apply the same chat/instruction template used during model training
- **Example for Llama**:
  ```python
  tokenizer.apply_chat_template(example["messages"], tokenize=False)
  ```

### 4. Fine-Tuned Models
- **Use training data for calibration**: Sample from the same distribution as fine-tuning data
- **Maintain template consistency**: Apply the same preprocessing as during fine-tuning

### 5. Hyperparameter Tuning
- **smoothing_strength**: Start at 0.8; increase to 0.9 if accuracy is poor
- **num_calibration_samples**: Start at 512; increase if accuracy degrades

## Combining INT8 with Other Optimizations

### INT8 + Quantized KV Cache
```python
llm = LLM(
    "./Meta-Llama-3-8B-Instruct-W8A8",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True
)
```

**Benefit**: Additional ~50% KV cache memory reduction

### INT8 + Prefix Caching
```python
llm = LLM(
    "./Meta-Llama-3-8B-Instruct-W8A8",
    enable_prefix_caching=True
)
```

**Benefit**: Reduced TTFT for requests with shared prefixes

### INT8 + Kernel Fusions
Enable CUDA graphs for kernel fusion benefits:
```python
llm = LLM(
    "./Meta-Llama-3-8B-Instruct-W8A8",
    enforce_eager=False  # Enable CUDA graphs (default in V1)
)
```

## Troubleshooting

### Issue: Accuracy significantly degraded
**Possible causes**:
- Insufficient calibration samples
- Calibration data doesn't match deployment distribution
- Activation outliers not smoothed effectively

**Solutions**:
1. Increase `num_calibration_samples` from 512 to 1024
2. Increase `smoothing_strength` from 0.8 to 0.9
3. Use calibration data more representative of deployment
4. Check for accuracy on multiple tasks (some models quantize better than others)

### Issue: Blackwell GPU incompatibility
**Error**: INT8 not supported on compute capability >= 10.0

**Solution**: Use [[FP8 Quantization]] instead for Blackwell GPUs:
```python
llm = LLM(model_id, quantization="fp8")
```

### Issue: Out of memory during calibration
**Possible causes**:
- Too many calibration samples
- Sequence length too long

**Solutions**:
1. Reduce `MAX_SEQUENCE_LENGTH` from 2048 to 1024
2. Reduce `NUM_CALIBRATION_SAMPLES` from 512 to 256
3. Use `device_map="auto"` for model loading

### Issue: Slow inference
**Possible causes**:
- INT8 kernel overhead on small batch sizes
- Dynamic activation quantization overhead

**Solutions**:
1. Increase batch size for better kernel utilization
2. Consider [[FP8 Quantization]] on Ada/Hopper for better throughput
3. Use [[INT4 W4A16]] for lower QPS workloads

## Comparison with Other Quantization Methods

| Method | Precision | Memory Reduction | Throughput | Accuracy | Calibration Required | Hardware |
|--------|-----------|------------------|------------|----------|---------------------|----------|
| INT8 W8A8 | 8-bit | 2× | Moderate | Good with SmoothQuant | Yes | Turing, Ampere, CPU |
| [[FP8 Quantization]] | 8-bit | 2× | High | Excellent | No | Ada, Hopper, MI300+ |
| [[INT4 W4A16]] | 4-bit | 4× | Low (low QPS) | Moderate | Yes | Ampere+ |

**When to use INT8 W8A8**:
- Deploying on Turing or Ampere GPUs (no FP8 support)
- CPU inference with x86 INT8 instructions
- Need integer-only arithmetic for hardware/compliance reasons
- Good accuracy required (better than INT4)

## Pre-Quantized Models

NeuralMagic provides pre-quantized INT8 models:

```python
from vllm import LLM
llm = LLM("neuralmagic/Meta-Llama-3-8B-Instruct-INT8")
```

**Collection**: https://huggingface.co/collections/neuralmagic/int8-llms-for-vllm-668ec32c049dca0369816415

## Limitations

- **Blackwell incompatibility**: Not supported on SM 10.0+
- **Calibration overhead**: Requires representative dataset and offline quantization step
- **Lower throughput than FP8**: On Ada/Hopper, FP8 provides better performance
- **Activation outliers**: Models with extreme outliers may have accuracy loss despite SmoothQuant

## Cross-References

### Related Techniques
- [[Quantization]] — Overview of quantization methods
- [[FP8 Quantization]] — Preferred method for Ada/Hopper/MI300+
- [[INT4 W4A16]] — Higher memory savings with 4-bit quantization
- [[Quantized KV Cache]] — Orthogonal KV cache memory optimization

### Related Concepts
- [[KV Cache]] — Key-value cache in attention
- [[Kernel Fusions]] — Kernel optimizations

### Related Tools
- [[llm-compressor]] — Quantization toolkit
- [[vLLM]] — Inference engine

### Related Architectures
- [[CustomOp System]] — Platform-specific INT8 kernel dispatch

## References

- Source: [[vllm-quantization]]
- SmoothQuant paper: https://arxiv.org/abs/2211.10438
- NeuralMagic INT8 collection: https://huggingface.co/collections/neuralmagic/int8-llms-for-vllm-668ec32c049dca0369816415
- llm-compressor GitHub: https://github.com/vllm-project/llm-compressor
