---
title: vLLM Benchmarking
type: benchmark
tags: [vllm, benchmarking, metrics, performance]
created: 2026-04-25
---

# vLLM Benchmarking

Built-in CLI tools for measuring vLLM inference performance across latency, throughput, and quality dimensions.

## Overview

vLLM provides three primary benchmarking commands via the `vllm bench` CLI:
- **`vllm bench serve`** — Online serving benchmark (client → HTTP server)
- **`vllm bench throughput`** — Offline throughput benchmark (direct engine)
- **`vllm bench mm-processor`** — Multimodal processor pipeline profiling

For production workloads, [[GuideLLM]] is recommended (live progress, automatic reports, flexible workload patterns).

## Key Metrics

### Latency Metrics
- **TTFT** (Time to First Token): Prefill latency, ms
- **TPOT** (Time per Output Token): Per-token decode latency excluding first token, ms
- **ITL** (Inter-Token Latency): Per-token decode latency including communication overhead, ms
- **E2E Latency**: Total request completion time, ms

### Throughput Metrics
- **Request throughput**: requests/sec
- **Output token throughput**: tokens/sec (generated tokens only)
- **Total token throughput**: tokens/sec (prompt + generated)

### Quality Metrics
- **Acceptance rate**: Percentage of accepted tokens (speculative decoding)
- **Perplexity**: Model quality (via lm-eval, separate tool)

## Online Serving Benchmark (`vllm bench serve`)

### Basic Usage
```bash
# Start server
vllm serve NousResearch/Hermes-3-Llama-3.1-8B

# Run benchmark
vllm bench serve \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10
```

### Example Output
```text
============ Serving Benchmark Result ============
Successful requests:                     10
Benchmark duration (s):                  5.78
Total input tokens:                      1369
Total generated tokens:                  2212
Request throughput (req/s):              1.73
Output token throughput (tok/s):         382.89
Total token throughput (tok/s):          619.85
---------------Time to First Token----------------
Mean TTFT (ms):                          71.54
Median TTFT (ms):                        73.88
P99 TTFT (ms):                           79.49
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.91
Median TPOT (ms):                        7.96
P99 TPOT (ms):                           8.03
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.74
Median ITL (ms):                         7.70
P99 ITL (ms):                            8.39
==================================================
```

### Supported Datasets

| Dataset | Type | Online | Offline | Source |
|---------|------|--------|---------|--------|
| ShareGPT | Text | ✅ | ✅ | `wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json` |
| ShareGPT4V | Image | ✅ | ✅ | `wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json` |
| ShareGPT4Video | Video | ✅ | ✅ | `git clone https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video` |
| BurstGPT | Text | ✅ | ✅ | `wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv` |
| Spec Bench | Text | ✅ | ✅ | Speculative decoding evaluation (12 categories) |
| SPEED-Bench | Text | ✅ | ✅ | NVIDIA speculative decoding benchmark (qualitative + 5 throughput splits) |
| VisionArena | Image | ✅ | ✅ | `lmarena-ai/VisionArena-Chat` (HF dataset) |
| InstructCoder | Code | ✅ | ✅ | `likaixin/InstructCoder` (HF dataset) |
| ASR datasets | Audio | ✅ | ✅ | `openslr/librispeech_asr`, `facebook/voxpopuli`, etc. |
| Random | Synthetic | ✅ | ✅ | `--dataset-name random` |
| RandomMM | Synthetic MM | ✅ | ✅ | `--dataset-name random-mm` |
| Custom | User JSONL | ✅ | ✅ | `--dataset-name custom --dataset-path data.jsonl` |
| Custom MM | User MM JSONL | ✅ | ✅ | `--dataset-name custom_mm --dataset-path mm_data.jsonl` |

### Load Pattern Configuration

Three parameters control request generation and concurrency:

**`--request-rate`**: Requests per second
- `inf` (default): Maximum throughput testing (send all requests immediately)
- Finite value: Controlled load simulation using Gamma distribution

**`--burstiness`**: Traffic variability (Gamma distribution shape parameter, > 0)
- `0.1-0.5`: Bursty traffic (stress testing), CV ≈ 3.16 at 0.1
- `1.0`: Natural Poisson traffic (realistic simulation), CV = 1.0
- `2.0-5.0`: Uniform traffic (controlled load), CV ≈ 0.45 at 5.0
- Only effective when `--request-rate` is finite

**`--max-concurrency`**: Concurrent outstanding requests limit
- `None` (default): Unlimited concurrency
- Integer: Simulates load balancer/API gateway constraints

**Load Pattern Recommendations:**

| Use Case | Burstiness | Request Rate | Max Concurrency | Description |
|----------|-----------|--------------|-----------------|-------------|
| Maximum Throughput | N/A | Infinite | Limited | Most common: simulates load balancer limits with unlimited demand |
| Realistic Testing | 1.0 | Moderate (5-20) | Infinite | Natural Poisson traffic for baseline performance |
| Stress Testing | 0.1-0.5 | High (20-100) | Infinite | Challenging burst patterns |
| Latency Profiling | 2.0-5.0 | Low (1-10) | Infinite | Uniform load for consistent timing |
| Capacity Planning | 1.0 | Variable | Limited | Test resource limits with realistic constraints |
| SLA Validation | 1.0 | Target rate | SLA limit | Production-like constraints |

**KV Cache Capacity Guidance:**
```text
GPU KV cache size: 15,728,640 tokens
Maximum concurrency for 8,192 tokens per request: 1920
```
- Capacity planning: Set `--max-concurrency` to 80-90% of reported maximum
- SLA validation: Use reported maximum as SLA limit

### Ramp-Up Request Rate

Gradually increase request rate over benchmark duration:
```bash
vllm bench serve \
  --ramp-up-strategy linear \
  --ramp-up-start-rps 5 \
  --ramp-up-end-rps 50 \
  ...
```

Strategies:
- `linear`: Linear increase from start to end RPS
- `exponential`: Exponential increase

### Sampling Parameters

Pass OpenAI-compatible sampling params:
```bash
vllm bench serve \
  --top-k 10 \
  --top-p 0.9 \
  --temperature 0.5 \
  --num-prompts 10 \
  ...
```

### Results Visualization

**Interactive Timeline** (HTML output):
```bash
vllm bench serve \
  --plot-timeline \
  --timeline-itl-thresholds 2,5 \
  --save-result \
  ...
```

**Dataset Statistics** (PNG output):
```bash
vllm bench serve \
  --plot-dataset-stats \
  --save-result \
  ...
```

### Speculative Decoding Benchmarks

**InstructCoder with N-gram:**
```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 5, "prompt_lookup_min": 2}'

vllm bench serve \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset-name hf \
  --dataset-path likaixin/InstructCoder \
  --num-prompts 2048
```

**Spec Bench (12 categories):**
```bash
# All categories
vllm bench serve \
  --dataset-name spec_bench \
  --dataset-path question.jsonl \
  --num-prompts -1

# Single category
vllm bench serve \
  --dataset-name spec_bench \
  --dataset-path question.jsonl \
  --spec-bench-category summarization \
  --num-prompts -1
```

**SPEED-Bench (qualitative + throughput splits):**
```bash
# Download dataset
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py | python3 -

# Qualitative split
vllm bench serve \
  --dataset-name speed_bench \
  --dataset-path data/speed_bench \
  --num-prompts -1

# Throughput split (2k ISL)
vllm bench serve \
  --dataset-name speed_bench \
  --speed-bench-dataset-subset throughput_2k \
  --dataset-path data/speed_bench \
  --num-prompts -1
```

## Offline Throughput Benchmark (`vllm bench throughput`)

Direct engine benchmark without HTTP server overhead:
```bash
vllm bench throughput \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset-name sonnet \
  --dataset-path vllm/benchmarks/sonnet.txt \
  --num-prompts 10
```

### Example Output
```text
Throughput: 7.15 requests/s, 4656.00 total tokens/s, 1072.15 output tokens/s
Total num prompt tokens:  5014
Total num output tokens:  1500
```

### Synthetic Multimodal (random-mm)

Generate synthetic multimodal inputs without external datasets:
```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name random-mm \
  --num-prompts 100 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-mm-base-items-per-request 2 \
  --random-mm-limit-mm-per-prompt '{"image": 3, "video": 0}' \
  --random-mm-bucket-config '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}'
```

## Multimodal Processor Benchmark (`vllm bench mm-processor`)

Per-stage latency profiling for multimodal preprocessing pipeline:
```bash
vllm bench mm-processor \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset-name random-mm \
  --num-prompts 50 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-mm-base-items-per-request 2 \
  --metric-percentiles 50,90,95,99 \
  --output-json results.json
```

### Measured Stages

| Stage | Description |
|-------|-------------|
| `get_mm_hashes_secs` | Multimodal input hashing |
| `get_cache_missing_items_secs` | Processor cache lookup |
| `apply_hf_processor_secs` | HuggingFace processor |
| `merge_mm_kwargs_secs` | Multimodal kwargs merging |
| `apply_prompt_updates_secs` | Prompt token updates |
| `preprocessor_total_secs` | Total preprocessing time |
| `encoder_forward_secs` | Encoder model forward pass |
| `num_encoder_calls` | Encoder invocation count |

## Specialized Benchmarks

### Structured Outputs
```bash
vllm serve NousResearch/Hermes-3-Llama-3.1-8B

# JSON schema
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset json \
  --structured-output-ratio 1.0 \
  --request-rate 10 \
  --num-prompts 1000
```

See [[Structured Outputs]] for grammar/regex/choice benchmarks.

### Long Document QA (Prefix Caching)
```bash
python3 benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-documents 16 \
  --document-length 2000 \
  --output-len 50 \
  --repeat-count 5
```

Repeat modes: `random` (shuffle), `tile` (repeat list), `interleave` (repeat each).

### Prefix Caching Efficiency
```bash
python3 benchmarks/benchmark_prefix_caching.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --enable-prefix-caching \
  --num-prompts 20 \
  --repeat-count 5 \
  --input-length-range 128:256
```

### Hashing Performance
```bash
# Micro-benchmark (per-call latency)
python benchmarks/benchmark_hash.py --iterations 20000 --seed 42

# End-to-end block hashing (throughput)
python benchmarks/benchmark_prefix_block_hash.py --num-blocks 20000 --block-size 32 --trials 5
```

Algorithms: `sha256`, `sha256_cbor`, `xxhash`, `xxhash_cbor` (requires `uv pip install xxhash cbor2`).

### Embedding Benchmarks

**Text embeddings:**
```bash
vllm serve jinaai/jina-embeddings-v3 --trust-remote-code

vllm bench serve \
  --model jinaai/jina-embeddings-v3 \
  --backend openai-embeddings \
  --endpoint /v1/embeddings \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
```

**Multimodal embeddings (CLIP):**
```bash
vllm serve openai/clip-vit-base-patch32

vllm bench serve \
  --model openai/clip-vit-base-patch32 \
  --backend openai-embeddings-clip \
  --endpoint /v1/embeddings \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat
```

### Reranker Benchmarks
```bash
vllm serve BAAI/bge-reranker-v2-m3

vllm bench serve \
  --model BAAI/bge-reranker-v2-m3 \
  --backend vllm-rerank \
  --endpoint /v1/rerank \
  --dataset-name random-rerank \
  --tokenizer BAAI/bge-reranker-v2-m3 \
  --random-input-len 512 \
  --num-prompts 10 \
  --random-batch-size 5
```

Creates `num_prompts / random_batch_size` requests with `random_batch_size` documents each.

## Cross-References

- [[GuideLLM]] — Production-grade benchmarking framework (recommended)
- [[vLLM Metrics]] — Prometheus metrics for live monitoring
- [[Optimization Levels]] — -O0 to -O3 performance tuning
- [[Speculative Decoding]] — EAGLE/MTP/n-gram acceptance rate measurements
- [[Chunked Prefill]] — Tuning `max_num_batched_tokens` for TTFT/ITL balance
- [[Prefix Caching]] — Measuring cache hit rates and TTFT reduction
- [[LoRA]] — Multi-LoRA serving benchmarks
- [[Structured Outputs]] — Grammar/regex/JSON schema overhead measurement

## Performance Analysis Tips

### Identifying Bottlenecks
1. **Low GPU utilization** → CPU underprovisioning (add cores, see [[Optimization Levels]])
2. **High TTFT, low ITL** → Prefill-bound (increase `max_num_batched_tokens`)
3. **Low TTFT, high ITL** → Decode-bound (reduce `max_num_batched_tokens`)
4. **Frequent preemptions** → Increase `gpu_memory_utilization` or reduce `max_num_seqs`

### Throughput Optimization
- Use `--request-rate=inf --max-concurrency=<limit>` for capacity planning
- Measure KV cache size at startup for concurrency guidance
- Profile with `--plot-timeline` to visualize request completion patterns

### Latency Optimization
- Use speculative decoding ([[EAGLE]], [[Multi-Token Prediction]], [[N-gram Speculation]])
- Enable [[Chunked Prefill]] for mixed workloads
- Tune `--burstiness=2.0-5.0` for uniform load (latency profiling)
- Avoid high QPS (reduces speculative decoding effectiveness)

## See Also

- vLLM benchmarking docs: `/tmp/vllm/docs/benchmarking/cli.md`
- Benchmark scripts: `benchmarks/benchmark_*.py`
- Example datasets: `benchmarks/sonnet.txt`, HuggingFace Hub
