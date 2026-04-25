---
title: OpenAI-Compatible Server
type: architecture
tags: [vllm, api, http-server, openai]
created: 2026-04-25
---

# OpenAI-Compatible Server

vLLM's HTTP server implementing OpenAI-compatible APIs for production LLM serving.

## Overview

vLLM provides a FastAPI-based HTTP server that implements OpenAI's API specifications, allowing drop-in replacement for OpenAI API clients while leveraging vLLM's high-throughput inference engine. The server exposes both standard OpenAI endpoints and vLLM-specific extensions.

## Architecture

### Server Stack
- **Framework**: FastAPI for async HTTP handling
- **Client compatibility**: Official OpenAI Python client, curl, any HTTP client
- **Process model**: Integrates with [[vLLM Engine]] via [[V1 Architecture]] multi-process design
- **Default port**: 8000 (`http://localhost:8000/v1`)

### Launch Command
```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
  --dtype auto \
  --api-key token-abc123
```

## Supported Endpoints

### OpenAI Standard APIs

**Text Generation:**
- `/v1/completions` — Completions API for text generation models
  - **Note**: `suffix` parameter not supported
- `/v1/responses` — Responses API for text generation models  
- `/v1/chat/completions` — Chat Completions API with chat template support
  - Requires model with chat template (Jinja2 format)
  - `user` parameter ignored
  - `parallel_tool_calls`: false (≤1 call), true (≥1 call, model-dependent)

**Embeddings:**
- `/v1/embeddings` — Embeddings API for embedding models
  - See Multi-Modal Models for multimodal embeddings

**Audio Processing:**
- `/v1/audio/transcriptions` — Transcriptions API (Automatic Speech Recognition)
  - Requires `pip install vllm[audio]`
  - Supported formats: FLAC, MP3, MP4, MPEG, MPGA, M4A, OGG, WAV, WEBM
  - Max file size: `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` (default 25 MB)
- `/v1/audio/translations` — Translation API (ASR → English)
  - 55 non-English languages → English
  - **Note**: whisper-large-v3-turbo does not support translation
- `/v1/realtime` — Realtime API (WebSocket streaming transcription)
  - PCM16 audio, 16kHz, mono, base64-encoded
  - Bidirectional events: `session.created`, `input_audio_buffer.append`, `transcription.delta`, `transcription.done`

### vLLM Custom APIs

**Tokenization:**
- `/tokenize` — Encode text to token IDs (`tokenizer.encode()`)
- `/detokenize` — Decode token IDs to text (`tokenizer.decode()`)

**Scoring & Ranking:**
- `/score`, `/v1/score` — Score API for cross-encoder/bi-encoder/late-interaction models
- `/generative_scoring` — Generative Scoring API for CausalLM models
  - Computes next-token probabilities for specified `label_token_ids`
  - Example: query "Is this the capital of France?", items ["Paris", "London"], labels [yes_token_id, no_token_id]
  - Score = softmax-normalized probability of first label token
- `/rerank`, `/v1/rerank`, `/v2/rerank` — Rerank API
  - Compatible with Jina AI v1 and Cohere v1/v2 APIs

**Embeddings Extensions:**
- `/v2/embed` — Cohere Embed API compatibility
  - Works with any embedding model including multimodal

**Pooling & Classification:**
- `/pooling` — Pooling API for pooling models
- `/classify` — Classification API for classification models

## Chat Template System

### Template Requirement
Chat models must include a Jinja2 chat template in tokenizer configuration to encode roles/messages/tokens.

### Manual Template Override
```bash
vllm serve <model> --chat-template ./path-to-chat-template.jinja
```

### Content Format Detection
vLLM auto-detects chat template content format:
- **`"string"`**: `"Hello world"` — traditional string content
- **`"openai"`**: `[{"type": "text", "text": "Hello world!"}]` — OpenAI schema with type field

Override with `--chat-template-content-format` if auto-detection fails.

### Multimodal Chat
OpenAI spec now supports multimodal messages:
```python
completion = client.chat.completions.create(
    model="Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "..."}},
        ],
    }],
)
```

## Extra Parameters

### vLLM-Specific Parameters
vLLM supports parameters beyond OpenAI spec (e.g., `top_k`, `structured_outputs`):
```python
completion = client.chat.completions.create(
    model="Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "top_k": 50,
        "structured_outputs": {"choice": ["positive", "negative"]},
    },
)
```

### Sampling Parameters
See protocol definitions for full lists:
- **Completions**: `vllm/entrypoints/openai/completion/protocol.py`
- **Chat**: `vllm/entrypoints/openai/chat_completion/protocol.py`
- **Responses**: `vllm/entrypoints/openai/responses/protocol.py`
- **Transcriptions**: `vllm/entrypoints/openai/speech_to_text/protocol.py`

### HTTP Headers
- `X-Request-Id`: Custom request ID tracking
  - Enable with `--enable-request-id-headers`
  - Access via `completion._request_id`

## Configuration

### Generation Config Override
By default, server applies `generation_config.json` from HuggingFace model repo:
```bash
# Disable HF generation_config.json, use vLLM defaults
vllm serve <model> --generation-config vllm
```

### Offline Documentation
Enable offline FastAPI `/docs` endpoint in air-gapped environments:
```bash
vllm serve <model> --enable-offline-docs
```

## Integration with Ray Serve

Ray Serve LLM extends vLLM with production features:
- **Auto-scaling**: Dynamic replica scaling based on load
- **Load balancing**: Request distribution across replicas
- **Back-pressure**: Queue management and circuit breakers
- **Observability**: Ray dashboards and metrics

See `examples/online_serving/ray_serve_deepseek.py` for multi-node DeepSeek R1 deployment.

## Cross-References

- [[vLLM Engine]] — Underlying inference engine
- [[V1 Architecture]] — Multi-process architecture for API/engine separation
- [[LoRA]] — Per-request LoRA adapter via `/v1/load_lora_adapter`, `/v1/unload_lora_adapter`
- [[Structured Outputs]] — Constrained decoding via `structured_outputs` parameter
- Multi-Modal Models — Vision/audio multimodal inputs via Chat API
- [[Quantization]] — FP8/INT8/INT4 quantized model serving
- [[Speculative Decoding]] — EAGLE/MTP/draft methods for latency reduction

## Performance Considerations

### Throughput
- [[Continuous Batching]] enabled by default
- [[Chunked Prefill]] for mixed prefill/decode batching
- [[KV Cache]] memory management via PagedAttention

### Latency
- TTFT: Time to first token (prefill latency)
- ITL: Inter-token latency (decode latency)
- See [[vLLM Benchmarking]] for measurement tools

### Memory
- KV cache GPU memory controlled by `--gpu-memory-utilization` (default 0.9)
- Model memory reduced via [[Tensor Parallelism]] (multi-GPU sharding)
- See [[Optimization Levels]] for startup time vs performance trade-offs

## Example: Multi-Tenant Serving

Combine OpenAI-Compatible Server features for production multi-tenancy:
```python
# Tenant-specific LoRA + structured outputs + tool calling
completion = client.chat.completions.create(
    model="Meta-Llama-3-70B-Instruct",
    messages=[{"role": "user", "content": "Generate SQL query"}],
    extra_body={
        "lora_request": {"lora_name": "tenant_42_sql", "lora_int_id": 42},
        "structured_outputs": {"grammar": "sql_grammar.ebnf"},
    },
    tools=[{"type": "function", "function": {"name": "execute_query", ...}}],
)
```

## Known Limitations

- **Suffix parameter**: Not supported in `/v1/completions`
- **`image_url.detail`**: Not supported in Chat API
- **Beam search**: Inefficient for ASR models (encoder/decoder cache work in progress)
- **`verbose_json` no_speech_prob**: Not yet supported in transcriptions response

## See Also

- Official OpenAI API docs: [Completions](https://platform.openai.com/docs/api-reference/completions), [Chat](https://platform.openai.com/docs/api-reference/chat)
- vLLM examples: `examples/basic/online_serving/openai_*_client.py`
- GuideLLM for production benchmarking
