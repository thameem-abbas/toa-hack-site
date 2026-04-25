---
title: vLLM Deployment & Benchmarking
type: source
source_files:
  - /tmp/vllm/docs/serving/openai_compatible_server.md
  - /tmp/vllm/docs/benchmarking/cli.md
  - /tmp/vllm/docs/configuration/optimization.md
  - /tmp/vllm/docs/deployment/integrations/llm-d.md
  - /tmp/vllm/docs/configuration/conserving_memory.md
fetched: 2026-04-25
tags: [vllm, deployment, benchmarking, optimization]
---

# vLLM Deployment & Benchmarking

Source documentation covering vLLM's production deployment features, benchmarking tools, optimization strategies, and memory conservation techniques.

## Sources

1. **`docs/serving/openai_compatible_server.md`** — OpenAI-compatible HTTP server implementation
2. **`docs/benchmarking/cli.md`** — Built-in benchmarking CLI tools
3. **`docs/configuration/optimization.md`** — Optimization and tuning guide
4. **`docs/deployment/integrations/llm-d.md`** — llm-d Kubernetes integration
5. **`docs/configuration/conserving_memory.md`** — Memory conservation strategies

## Key Insights

### OpenAI-Compatible Server

vLLM's FastAPI-based HTTP server provides drop-in OpenAI API compatibility while exposing vLLM-specific extensions:

**Standard OpenAI APIs:**
- `/v1/completions`, `/v1/responses`, `/v1/chat/completions` (text generation)
- `/v1/embeddings` (embeddings)
- `/v1/audio/transcriptions`, `/v1/audio/translations` (ASR)
- `/v1/realtime` (WebSocket streaming ASR)

**vLLM Custom APIs:**
- `/tokenize`, `/detokenize` (tokenization)
- `/score`, `/v1/score` (cross-encoder/bi-encoder scoring)
- `/generative_scoring` (CausalLM next-token probability scoring)
- `/rerank`, `/v1/rerank`, `/v2/rerank` (Jina/Cohere rerank compatibility)
- `/pooling`, `/classify` (pooling/classification models)
- `/v2/embed` (Cohere Embed API compatibility)

**Extra Parameters:**
- `extra_body` for vLLM-specific params (`top_k`, `structured_outputs`, etc.)
- `X-Request-Id` header support via `--enable-request-id-headers`

**Chat Template System:**
- Jinja2 templates for role/message encoding
- Auto-detection of content format (string vs OpenAI schema)
- Manual override via `--chat-template` flag

**Configuration:**
- `--generation-config vllm` to disable HF generation_config.json
- `--enable-offline-docs` for air-gapped FastAPI docs
- Integration with Ray Serve LLM for autoscaling/load balancing

### Benchmarking CLI

vLLM provides three benchmark commands:

**`vllm bench serve`** (online serving):
- Metrics: TTFT, TPOT, ITL, E2E latency, request/token throughput
- 15+ datasets: ShareGPT, ShareGPT4V, BurstGPT, Spec Bench, SPEED-Bench, VisionArena, InstructCoder, ASR datasets, synthetic (random, random-mm)
- Load patterns: `--request-rate` (inf/finite), `--burstiness` (Gamma distribution shape), `--max-concurrency` (backpressure limit)
- Visualization: `--plot-timeline` (interactive HTML), `--plot-dataset-stats` (PNG)

**`vllm bench throughput`** (offline):
- Direct engine benchmark without HTTP overhead
- Same datasets as online serving
- Metrics: requests/sec, total/output tokens/sec

**`vllm bench mm-processor`** (multimodal profiling):
- Per-stage latency: hashing, cache lookup, HF processor, merging, encoder forward pass
- Percentile reporting: `--metric-percentiles 50,90,95,99`
- JSON output: `--output-json results.json`

**Load Pattern Recommendations:**
- Maximum throughput: `--request-rate=inf --max-concurrency=<limit>` (most common)
- Realistic testing: `--burstiness=1.0 --request-rate=5-20`
- Stress testing: `--burstiness=0.1-0.5 --request-rate=20-100`
- Latency profiling: `--burstiness=2.0-5.0 --request-rate=1-10`

**Speculative Decoding Benchmarks:**
- Spec Bench: 12 categories (writing, coding, summarization, etc.)
- SPEED-Bench: NVIDIA benchmark with qualitative + 5 throughput splits (1k/2k/8k/16k/32k ISL)

### Optimization & Tuning

**Optimization Levels** (`-O0` to `-O3`):
- `-O0`: No optimizations, fastest startup, lowest performance
- `-O1`: Fast compilation, PIECEWISE cudagraphs
- `-O2`: Default, additional fusions, FULL_AND_PIECEWISE cudagraphs
- `-O3`: Aggressive (currently equal to `-O2`, future experimental optimizations)

**Preemption:**
- KV cache space exhaustion → preempt requests (recompute when space available)
- Default mode: `RECOMPUTE` (lower overhead than `SWAP` in V1)
- Mitigation: Increase `gpu_memory_utilization`, decrease `max_num_seqs`/`max_num_batched_tokens`, increase `tensor_parallel_size`/`pipeline_parallel_size`

**Chunked Prefill** (enabled by default):
- Decode prioritization → better ITL
- Compute/memory-bound co-location → better GPU utilization
- Tuning: smaller `max_num_batched_tokens` (better ITL), larger (better TTFT)
- Recommended: `max_num_batched_tokens > 8192` for small models on large GPUs

**Parallelism Strategies:**
- [[Tensor Parallelism]]: Shard layers across GPUs (reduce memory per GPU for more KV cache)
- [[Pipeline Parallelism]]: Distribute layers sequentially (deep/narrow models, cross-node)
- [[Expert Parallelism]]: MoE-specific (enable via `--enable-expert-parallel`)
- [[Data Parallelism]]: Replicate model (throughput scaling, `--data-parallel-size`)
- [[Context Parallelism]]: Shard KV cache along sequence dimension (decode CP for TP_size > num_kv_heads)

**NUMA Binding** (multi-socket GPUs):
- `--numa-bind` auto-detects GPU-to-NUMA mapping
- `--numa-bind-nodes` explicit NUMA node indices
- `--numa-bind-cpus` explicit CPU pinning (PCT/high-frequency cores)
- Requires `VLLM_WORKER_MULTIPROC_METHOD=spawn` via Python API

**Batch-level DP for Multimodal Encoders:**
- `mm_encoder_tp_mode="data"` shards input data (not weights) via TP
- 10% throughput/TTFT improvement (TP=8), 40% for Conv3D ops
- Minor memory increase (encoder weights replicated per TP rank)
- Supported models: dots_ocr, GLM-4.1V, InternVL, Kimi-VL, Llama4, MiniCPM-V, Qwen2-VL, Step3

**Input Processing:**
- API server scale-out: `--api-server-count 4` (parallel preprocessing)
- Adjust `VLLM_MEDIA_LOADING_THREAD_COUNT` to avoid CPU exhaustion
- Disables multimodal IPC caching (requires 1:1 API:engine)

**Multi-Modal Caching:**
- Processor caching: Auto-enabled in `BaseMultiModalProcessor`
- IPC caching: Auto-enabled for 1:1 API:engine (key-replicated or shared memory)
- `mm_processor_cache_type="shm"` for shared memory (multi-worker efficiency)
- `mm_processor_cache_gb` controls cache size (default 4 GiB)

**CPU Resources for GPU Deployments:**
- Minimum: `2 + N` physical cores (1 API server, 1 engine core, N GPU workers)
- Data parallel: `A + DP + N + (1 if DP > 1)` (A=API count, DP=data parallel size, N=GPUs)
- Hyperthreading: `2 × (2 + N)` vCPUs minimum
- Example: DP=4, TP=2, 8 GPUs → 4 API + 4 engines + 8 workers + 1 coordinator = 17 processes

**Attention Backend Selection:**
- Automatic priority-ordered selection (Blackwell: FlashInfer→FlashAttention→Triton)
- Manual override: `--attention-backend <name>`
- See [[Attention Backends]] for 13+ standard, 15+ MLA backends

### llm-d Integration

**Kubernetes-native Disaggregated Serving:**
- Prefill pool: Compute-bound (TTFT-optimized)
- Decode pool: Memory-bound (ITL-optimized)
- KV cache routing: Automatic via connectors (P2pNccl, Nixl, LMCache, Mooncake)

**Autoscaling:**
- Per-pool HPA (prefill: queue depth, decode: active generations)
- Independent scaling ratios (e.g., 2:1 decode:prefill)

**Deployment:**
- Direct: llm-d operator CRD (`LLMInferenceService`)
- KServe: `LLMInferenceService` with `modelFormat: vllm`

### Memory Conservation

**Tensor Parallelism:**
- Memory reduction: `1 / tp_size` (e.g., TP=4 → 4× reduction)
- `CUDA_VISIBLE_DEVICES` to control GPU selection

**Quantization:**
- Static: Pre-quantized models from HF Hub (Red Hat AI collection)
- Dynamic: `--quantization <method>` (FP8/INT8/INT4/AWQ/GPTQ)

**Context Length & Batch Size:**
- `--max-model-len <tokens>` limit context window
- `--max-num-seqs <num>` limit concurrent sequences

**CUDA Graphs Reduction:**
- `compilation_config.cudagraph_capture_sizes=[1,2,4,8,16]` (fewer sizes)
- `--enforce-eager` disable CUDA graphs entirely

**Cache Size Adjustment:**
- Multimodal cache: `--mm-processor-cache-gb <gb>` (default 4 GiB)
- CPU KV cache: `VLLM_CPU_KVCACHE_SPACE=<gb>` (CPU backend only)

**Multi-Modal Input Limits:**
- `--limit-mm-per-prompt '{"image": 3, "video": 1}'` cap items per prompt
- Disable modalities: `--limit-mm-per-prompt '{"video": 0}'` (images only)
- Text-only: `--limit-mm-per-prompt '{"image": 0}'` (no multimodal)

**Configurable Size Hints:**
- Image: `{"count": 5, "width": 512, "height": 512}`
- Video: `{"count": 1, "num_frames": 32, "width": 640, "height": 640}`
- Audio: `{"count": 1, "length": <samples>}`
- Affects profiling dummy inputs, not runtime processing

**Multi-Modal Processor Arguments:**
- Qwen2-VL: `--mm-processor-kwargs '{"max_pixels": 589824}'` (default 1280×28×28)
- InternVL: `--mm-processor-kwargs '{"max_dynamic_patch": 4}'` (default 12)

## Claims

1. **OpenAI API compatibility**: vLLM server is drop-in compatible with OpenAI Python client
2. **Extra parameters**: vLLM supports parameters beyond OpenAI spec via `extra_body`
3. **Chunked prefill by default**: V1 enables chunked prefill whenever possible for better ITL/throughput
4. **Optimization levels**: -O0 to -O3 trade startup time for runtime performance
5. **Preemption mode**: V1 defaults to RECOMPUTE (lower overhead than SWAP)
6. **NUMA binding**: Auto-detects GPU-to-NUMA mapping on multi-socket nodes
7. **Multimodal TP mode**: Batch-level DP (`mm_encoder_tp_mode="data"`) provides 10-40% improvement
8. **CPU requirements**: Minimum `2 + N` physical cores for N GPUs (often underprovisioned)
9. **Load patterns**: Maximum throughput pattern (`--request-rate=inf --max-concurrency=<limit>`) is most common
10. **Speculative decoding benchmarks**: Spec Bench (12 categories), SPEED-Bench (qualitative + 5 throughput splits)

## Contradictions

None detected. Documentation is consistent with previous batches on parallelism, quantization, and speculative decoding.

## Pages Created

- [[OpenAI-Compatible Server]] — Architecture page for HTTP API server
- [[vLLM Benchmarking]] — Benchmark page for CLI tools
- [[llm-d]] — Tool page for Kubernetes-native serving
- [[vllm-deployment]] — This source summary

## Cross-References

- [[vLLM Engine]] — Underlying inference engine
- [[V1 Architecture]] — Multi-process design (API/engine separation)
- [[Optimization Levels]] — -O0 to -O3 tuning
- [[Tensor Parallelism]], [[Pipeline Parallelism]], [[Data Parallelism]], [[Context Parallelism]], [[Expert Parallelism]]
- [[Chunked Prefill]] — Prefill/decode co-location
- [[Prefix Caching]] — Automatic hash-based caching
- [[LoRA]] — Multi-LoRA serving
- [[Structured Outputs]] — Grammar/regex/JSON schema constraints
- [[Speculative Decoding]] — EAGLE/MTP/draft/n-gram methods
- [[Quantization]] — FP8/INT8/INT4 compression
- Multi-Modal Models — Vision/audio preprocessing
- [[Attention Backends]] — Pluggable attention implementations
- [[CUDA Graphs]] — Graph capture modes
- GuideLLM — Production benchmarking framework
- Ray Serve LLM — Alternative autoscaling/load balancing
- [[Disaggregated Serving]] — Prefill/decode separation
- [[KV Cache Transfer]] — KV routing connectors

## See Also

- vLLM docs: `/tmp/vllm/docs/`
- OpenAI API reference: [platform.openai.com/docs/api-reference](https://platform.openai.com/docs/api-reference)
- llm-d docs: [llm-d.ai/docs/guide](https://llm-d.ai/docs/guide)
- KServe LLMInferenceService: [kserve.github.io/website/docs/.../llmisvc-overview](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/llmisvc-overview)
