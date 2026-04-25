---
title: LoRA (Low-Rank Adaptation)
type: technique
tags: [fine-tuning, inference, multi-model-serving]
related: "Prefix Caching, vLLM Engine, Quantization, Mixture of Experts"
---

# LoRA (Low-Rank Adaptation)

## Overview

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that enables efficient model adaptation by adding small trainable low-rank matrices to frozen base model weights. vLLM supports serving multiple LoRA adapters simultaneously from a single base model with minimal overhead, making it ideal for multi-tenant deployments where each user or task requires a specialized model variant.

## What is LoRA?

LoRA decomposes weight updates into low-rank matrices: for a pre-trained weight matrix W, instead of fine-tuning W directly, LoRA keeps W frozen and adds a trainable low-rank decomposition:

```
W' = W + BA
```

where B is d×r and A is r×k (r << min(d,k) is the rank). This reduces trainable parameters dramatically (e.g., rank 16 for a 4096×4096 layer adds only ~130K parameters vs 16M for full fine-tuning).

## vLLM LoRA Serving Architecture

### Per-Request LoRA Loading

vLLM enables dynamic LoRA adapter selection on a per-request basis via `LoRARequest`:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", enable_lora=True)

outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("sql_adapter", 1, sql_lora_path),
)
```

**Key parameters:**
- Name: human-readable identifier
- LoRA ID: globally unique integer ID for the adapter
- Path: local or HuggingFace Hub path to adapter weights

### Multi-LoRA Batching

vLLM can batch requests with different LoRA adapters in a single forward pass using the Punica/BGMV kernel architecture:

- **Base model weights** remain frozen in GPU memory
- **LoRA adapters** are loaded on-demand and cached
- **Batched execution** applies different LoRA weights to different sequences in the same batch

This enables efficient multi-tenant serving where each request can specify a different adapter without sacrificing throughput.

## Supported Layers

vLLM applies LoRA to models that implement the `SupportsLoRA` interface. Supported layer types:

1. **Linear layers** (QKV projections, FFN layers): Most common target
2. **Embedding layers**: Token and position embeddings
3. **MoE expert layers**: Per-expert LoRA adapters

Use `--lora-target-modules` to restrict LoRA application to specific modules (e.g., `o_proj qkv_proj down_proj`).

## Configuration Flags

### Server Startup

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-lora \
    --lora-modules sql-lora=jeeejeee/llama32-3b-text2sql-spider \
    --max-loras 4 \
    --max-lora-rank 64 \
    --max-cpu-loras 8
```

**Critical parameters:**

- `--enable-lora`: Enable LoRA serving (required)
- `--max-loras N`: Maximum concurrent GPU-resident LoRA adapters (default: 1)
- `--max-lora-rank R`: Maximum rank across all adapters; **must match actual max rank** to avoid wasting memory
- `--max-cpu-loras`: Number of adapters cached in CPU memory
- `--lora-target-modules`: Restrict LoRA to specific module suffixes (performance tuning)

### LoRA Module Format

**Old format (backward-compatible):**
```bash
--lora-modules sql-lora=jeeejeee/llama32-3b-text2sql-spider
```

**New JSON format (with base model lineage):**
```bash
--lora-modules '{"name": "sql-lora", "path": "jeeejeee/llama32-3b-text2sql-spider", "base_model_name": "meta-llama/Llama-3.2-3B-Instruct"}'
```

The new format enables the `/v1/models` endpoint to show parent-child relationships between base models and LoRA adapters.

## Dynamic LoRA Loading

### Security Warning

Dynamic LoRA updates allow loading/unloading adapters at runtime without server restart. **This is insecure** — only use in isolated, fully trusted environments.

### API Endpoints

Enable via `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`:

**Load adapter:**
```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter",
    "lora_path": "/path/to/sql-lora-adapter"
}'
```

**Unload adapter:**
```bash
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
-H "Content-Type: application/json" \
-d '{"lora_name": "sql_adapter"}'
```

**In-place reload** (for asynchronous RL workflows):
```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "my-adapter",
    "lora_path": "/path/to/adapter/v2",
    "load_inplace": true
}'
```

### LoRA Resolver Plugins

LoRA resolvers enable automatic adapter discovery from storage backends (filesystem, S3, HuggingFace Hub).

**Environment setup:**
```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export VLLM_PLUGINS=lora_filesystem_resolver
export VLLM_LORA_RESOLVER_CACHE_DIR=/path/to/lora/adapters
```

**Built-in resolvers:**

1. **`lora_filesystem_resolver`**: Load from local directory
   - Expects `adapter_config.json` + `adapter_model.bin` in subdirectories
   - On first request for adapter `foobar`, checks `$CACHE_DIR/foobar/`

2. **`lora_hf_hub_resolver`**: Download from HuggingFace Hub
   - Set `VLLM_LORA_RESOLVER_HF_REPO_LIST=user/repo1,user/repo2`
   - Request `my/repo/subpath` downloads from `my/repo` at `subpath`

**Custom S3 resolver example:**
```python
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
from vllm.lora.request import LoRARequest

class S3LoRAResolver(LoRAResolver):
    async def resolve_lora(self, base_model_name, lora_name):
        # Download from S3 to local cache
        await self.s3._get(s3_path, local_path, recursive=True)
        return LoRARequest(lora_name, abs(hash(lora_name)), local_path)

LoRAResolverRegistry.register_resolver("s3_resolver", S3LoRAResolver())
```

## Performance Characteristics

### Multi-LoRA Batching Performance

vLLM uses the **Punica kernel** (NVIDIA) or **BGMV kernel** (AMD) for batched LoRA inference:

- **Memory overhead:** Minimal (only adapter weights, typically <100MB per adapter)
- **Compute overhead:** ~5-10% vs base model alone (rank 64, batch size 32)
- **Throughput:** Batching different LoRAs in a single forward pass maintains near-base-model throughput

**Scaling limits:**
- `--max-loras` limited by GPU memory (each adapter rank R adds O(R × model_dim) memory)
- Typical configurations: 4-8 concurrent adapters on 24GB GPU, 16-32 on 80GB

### LoRA + Other Optimizations

LoRA is **compatible** with most vLLM optimizations:

- [[Prefix Caching]]: LoRA-specific KV cache entries are cached separately
- [[Quantization]]: LoRA adapters can be applied on top of quantized base models (FP8/INT8/INT4)
- [[Speculative Decoding]]: Draft model can use LoRA adapters
- [[CUDA Graphs]]: LoRA operations are graph-capturable

**Known limitation:**
- [[Mixture of Experts]]: LoRA on MoE expert layers is supported, but per-expert LoRA memory scales with expert count

## Multimodal LoRA Support (Experimental)

vLLM experimentally supports LoRA for **vision tower and connector** layers in multimodal models. See [PR #26674](https://github.com/vllm-project/vllm/pull/26674) for rationale.

**Supported models:** Limited (check [Issue #31479](https://github.com/vllm-project/vllm/issues/31479) for status)

### Default Multimodal LoRAs

Some multimodal models (e.g., Granite Speech, Phi-4-multimodal) ship with LoRA adapters that should always be applied when a specific modality is present. vLLM supports automatic LoRA application via `default_mm_loras`:

```python
llm = LLM(
    model="ibm-granite/granite-speech-3.3-2b",
    enable_lora=True,
    max_lora_rank=64,
    default_mm_loras={"audio": "ibm-granite/granite-speech-3.3-2b"},
)
```

**Constraint:** Only one LoRA per prompt; if multiple modalities are present, none are applied.

**Server usage:**
```bash
vllm serve ibm-granite/granite-speech-3.3-2b \
    --enable-lora \
    --default-mm-loras '{"audio":"ibm-granite/granite-speech-3.3-2b"}' \
    --max-lora-rank 64
```

## Best Practices

### Configuring `max_lora_rank`

**Critical:** Set `--max-lora-rank` to the **exact maximum rank** of your adapters.

- Too low: Adapters with higher rank will fail to load
- Too high: Wastes memory (LoRA buffers pre-allocated to max rank)

Example: Adapters with ranks [16, 32, 64] → use `--max-lora-rank 64`, not 256.

```bash
# Good: matches actual maximum rank
vllm serve model --enable-lora --max-lora-rank 64

# Bad: unnecessarily high, wastes ~4× memory
vllm serve model --enable-lora --max-lora-rank 256
```

### Restricting Target Modules

Use `--lora-target-modules` to apply LoRA only to specific layers (performance tuning when only attention layers need adaptation):

```bash
# Apply LoRA only to output projection
vllm serve model --enable-lora --lora-target-modules o_proj

# Multiple modules
vllm serve model --enable-lora --lora-target-modules o_proj qkv_proj down_proj
```

**Default:** If not specified, LoRA applies to all supported modules.

### Directory Structure for Filesystem Resolver

```
/path/to/lora/adapters/
├── adapter1/
│   ├── adapter_config.json  # Required: PEFT config
│   ├── adapter_model.bin    # Required: LoRA weights
│   └── tokenizer files      # Optional
└── adapter2/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

**Required `adapter_config.json` fields:**
```json
{
  "peft_type": "LORA",
  "base_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct",
  "r": 64,
  "lora_alpha": 128,
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
  "bias": "none"
}
```

## Troubleshooting

### Common Issues

1. **"LoRA rank exceeds maximum"**
   - Adapter's `r` value in `adapter_config.json` exceeds `--max-lora-rank`
   - Fix: Increase `--max-lora-rank` or use lower-rank adapter

2. **"LoRA adapter not found"** (filesystem resolver)
   - Directory name must match requested model name
   - Ensure `adapter_config.json` and `adapter_model.bin` exist

3. **"Invalid adapter configuration"**
   - `peft_type` must be `"LORA"`
   - `base_model_name_or_path` should match base model (soft check)

4. **High memory usage**
   - Check `--max-lora-rank` is not set too high
   - Reduce `--max-loras` (number of concurrent GPU-resident adapters)

### Debug Logging

```bash
export VLLM_LOGGING_LEVEL=DEBUG
vllm serve model --enable-lora ...
```

## Cross-References

- [[Prefix Caching]] — LoRA-specific KV cache entries
- [[vLLM Engine]] — Per-request LoRA dispatch via scheduler
- [[Quantization]] — LoRA on top of quantized base models
- [[Mixture of Experts]] — LoRA for MoE expert layers
- Multi-Modal Models — LoRA for vision tower/connector

## Further Reading

- [Original LoRA paper](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [vLLM LoRA feature docs](https://docs.vllm.ai/en/latest/features/lora.html)
- [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547)
