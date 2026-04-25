---
title: vLLM Metrics
type: architecture
created: 2026-04-25
tags: [observability, prometheus, metrics, monitoring]
---

# vLLM Metrics

vLLM exposes a comprehensive Prometheus-compatible metrics system for production observability and capacity planning. The V1 engine collects metrics at both server and request levels to enable SLO tracking, autoscaling, and performance debugging.

## Overview

Metrics in vLLM are categorized into:

1. **Server-level metrics**: Global state and performance (Gauges/Counters) - e.g., number of running requests, GPU cache usage
2. **Request-level metrics**: Per-request characteristics (Histograms) - e.g., TTFT, inter-token latency, E2E latency

The mental model: server-level metrics help explain the values of request-level metrics.

## Key Metric Categories

### System Metrics

**Throughput:**
- `vllm:prompt_tokens_total` (Counter) - Total prompt tokens processed
- `vllm:generation_tokens_total` (Counter) - Total tokens generated
- `vllm:request_success_total` (Counter) - Finished requests by finish reason (stop, length, abort)

**Request State:**
- `vllm:num_requests_running` (Gauge) - Requests in execution batches
- `vllm:num_requests_waiting` (Gauge) - Requests queued
- `vllm:num_requests_swapped` (Gauge, deprecated) - Preempted requests (V0 only)

### Request Metrics

**Latency Histograms:**
- `vllm:time_to_first_token_seconds` - TTFT latency
- `vllm:inter_token_latency_seconds` - Time per output token (TPOT)
- `vllm:e2e_request_latency_seconds` - End-to-end request latency
- `vllm:request_prefill_time_seconds` - Prefill phase duration
- `vllm:request_decode_time_seconds` - Decode phase duration
- `vllm:request_queue_time_seconds` - Queue wait time

**Token Counts:**
- `vllm:request_prompt_tokens` (Histogram) - Input prompt lengths
- `vllm:request_generation_tokens` (Histogram) - Generation lengths

### KV Cache Metrics

**Usage:**
- `vllm:kv_cache_usage_perc` (Gauge) - Fraction of used KV cache blocks (0-1)
- `vllm:cpu_cache_usage_perc` (Gauge, deprecated) - CPU swap space usage (V0 only)

**Prefix Cache:**
- `vllm:prefix_cache_queries` (Counter) - Total prefix cache queries
- `vllm:prefix_cache_hits` (Counter) - Prefix cache hits

Prefix cache hit rate over 5 minutes:
```promql
rate(vllm:prefix_cache_hits[5m]) / rate(vllm:prefix_cache_queries[5m])
```

**Cache Residency (sampled):**
- `vllm:kv_block_lifetime_seconds` (Histogram) - Block lifetime (allocation → eviction)
- `vllm:kv_block_idle_before_evict_seconds` (Histogram) - Idle time before eviction
- `vllm:kv_block_reuse_gap_seconds` (Histogram) - Time between consecutive accesses

Enable with `--kv-cache-metrics-sample`.

### Speculative Decoding Metrics

**Acceptance Rate:**
- `vllm:spec_decode_draft_acceptance_rate` (Gauge, legacy)
- `vllm:spec_decode_efficiency` (Gauge, legacy)
- `vllm:spec_decode_num_accepted_tokens` (Counter)
- `vllm:spec_decode_num_draft_tokens` (Counter)
- `vllm:spec_decode_num_emitted_tokens` (Counter)

Note: Acceptance rate should be derived from counters (accepted/draft) for time-series flexibility.

### LoRA Metrics

- `vllm:lora_requests_info` (Gauge) - Per-adapter running/waiting request counts

Labels: `running_lora_adapters`, `waiting_lora_adapters`, `max_lora`

Warning: Adapter counts encoded as comma-separated strings; should be refactored to use proper labels.

## Access

### Prometheus Endpoint

```bash
curl http://0.0.0.0:8000/metrics
```

All metrics use `vllm:` prefix and include `model_name` label.

Example output:
```text
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="meta-llama/Llama-3.1-8B-Instruct"} 8.0

# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.02",model_name="..."} 13.0
vllm:time_to_first_token_seconds_bucket{le="0.04",model_name="..."} 97.0
vllm:time_to_first_token_seconds_count{model_name="..."} 140.0
```

### Logging

`LoggingStatLogger` outputs INFO logs every 5 seconds:
- Running/waiting request count
- GPU cache usage percentage
- Prompt tokens/sec (5s window)
- Generation tokens/sec (5s window)
- Prefix cache hit rate (1k most recent queries)

## Metrics Architecture

### V1 Design Principles

1. **Minimal engine core overhead**: Metrics collection in API server process, not engine core
2. **Monotonic time for intervals**: Use `time.monotonic()` to avoid NTP clock skew
3. **Same-process timestamp comparison**: Monotonic clocks differ across processes

### Event Timeline

Engine core records timestamps for per-request events:

- `QUEUED` - Request received by engine core
- `SCHEDULED` - First scheduled for execution
- `PREEMPTED` - Put back in waiting queue (to free KV cache)
- `NEW_TOKENS` - Output generated (single timestamp per iteration in `EngineCoreOutputs`)

Calculated intervals:
- **Queue interval**: `QUEUED` → most recent `SCHEDULED`
- **Prefill interval**: most recent `SCHEDULED` → first `NEW_TOKENS`
- **Decode interval**: first `NEW_TOKENS` → last `NEW_TOKENS` (after most recent `SCHEDULED`)
- **Inference interval**: most recent `SCHEDULED` → last `NEW_TOKENS`
- **Inter-token interval**: successive `NEW_TOKENS`

TTFT calculated from frontend `arrival_time` (tokenization start) to first token, accounting for input processing.

### Preemption Handling

- **Decode preemption**: affects inter-token, decode, and inference intervals
- **Prefill preemption**: affects TTFT and prefill intervals

### Frontend Stats Collection

AsyncLLM.output_handler_loop processes each `EngineCoreOutputs` and collects:
- New tokens generated (per iteration)
- Prompt tokens processed (completed prefills)
- Queue intervals (newly scheduled requests)
- Prefill intervals (completed prefills)
- Inter-token intervals (all requests in iteration)
- TTFT (completed prefills, relative to frontend arrival_time)

For finished requests:
- Inference and decode intervals
- E2E latency (arrival_time → final token)

### Multi-process Mode

Metrics collected in API server process. Multiprocess mode only used when `--api-server-count > 1`.

Python/process metrics (GC, memory, CPU, FDs) unavailable in multiprocess mode.

## HTTP Metrics

`prometheus_fastapi_instrumentator` tracks HTTP request metrics:

```bash
$ curl http://0.0.0.0:8000/metrics | grep -P '^http_'
http_requests_total{handler="/v1/completions",method="POST",status="2xx"} 201.0
http_request_size_bytes_count{handler="/v1/completions"} 201.0
http_response_size_bytes_count{handler="/v1/completions"} 201.0
http_request_duration_seconds_count{handler="/v1/completions",method="POST"} 201.0
```

## Grafana Dashboard

vLLM provides [reference Grafana dashboard](https://github.com/vllm-project/vllm/tree/main/examples/observability/prometheus_grafana) highlighting critical metrics:

**Latency:**
- TTFT (p50, p95, p99)
- Inter-token latency (TPOT)
- E2E request latency
- Queue time
- Prefill/decode time

**System:**
- Running/waiting/swapped request counts
- KV cache usage percentage
- Prompt tokens processed
- Generation tokens produced
- Request success by finish reason

**Token Distribution:**
- Request prompt token histogram
- Request generation token histogram
- Max generation tokens per sequence group

## Configuration Metadata

`vllm:cache_config_info` (Gauge = 1.0) exposes cache config as labels:

```text
vllm:cache_config_info{
  block_size="16",
  cache_dtype="auto",
  enable_prefix_caching="False",
  gpu_memory_utilization="0.9",
  ...
} 1.0
```

Uses `multiprocess_mode="mostrecent"` (Info metrics unsupported in multiprocess mode).

## Deprecated Metrics

### Removed in V1

- `vllm:num_requests_swapped` - V0 KV cache swapping to CPU (replaced by prefix caching)
- `vllm:cpu_cache_usage_perc` - CPU swap space (V0 only)

Rationale: Prefix caching (zero overhead, on by default) eliminates need for CPU swapping. Beam search moved out of core.

### Pending Deprecation

- `vllm:time_in_queue_requests` - Duplicate of `vllm:request_queue_time_seconds` (Grafana uses latter)
- `vllm:avg_prompt_throughput_toks_per_s` - Removed, users derive from counters

## Future Metrics

### Parallel Sampling (`n > 1`)

Planned for [PR #10980](https://github.com/vllm-project/vllm/pull/10980):
- `vllm:request_params_n` (Histogram) - Value of `n` parameter
- `vllm:request_max_num_generation_tokens` (Histogram) - Max output length across sequences in group

### OpenTelemetry Integration

Detailed timing metrics (enabled with `--collect-detailed-traces`):
- `vllm:model_forward_time_milliseconds` (Histogram) - Model forward pass time
- `vllm:model_execute_time_milliseconds` (Histogram) - Full execute time (forward + sync + sampling)

Considered separately from core metrics due to performance overhead.

## Autoscaling Use Cases

Metrics enable Kubernetes HPA and custom autoscaling:

**Saturation Detection:**
- Monitor when request rate causes queue time > inter-token latency
- Track inflection point: max concurrency before latency spikes

**Prometheus Queries:**
```promql
# Average TTFT over 5 minutes
rate(vllm:time_to_first_token_seconds_sum[5m]) / rate(vllm:time_to_first_token_seconds_count[5m])

# KV cache saturation (> 80%)
vllm:kv_cache_usage_perc > 0.8

# Request throughput (requests/sec)
rate(vllm:request_success_total[1m])
```

See [Kubernetes Serving WG proposals](https://github.com/kubernetes/community/tree/master/wg-serving) for standardization efforts.

## Best Practices

### Metric Naming

1. Avoid colons (reserved for recording rules) - vLLM uses `vllm:` prefix contrary to convention
2. End with units (`_seconds`, `_total`, `_perc`)
3. Counters: Use `_total` suffix (OpenMetrics compatibility)

### Deprecation Policy

When deprecating metrics:
1. Add prominent notice in `/metrics` help string
2. Document in release notes and user-facing docs
3. Consider escape hatch CLI flag (show hidden metrics)
4. Coordinate with known downstream users (e.g., Kubernetes Gateway API)

See [vLLM deprecation policy](https://github.com/vllm-project/vllm/blob/main/docs/contributing/deprecation_policy.md).

### Adding New Metrics

Considerations:
1. Difficult to remove once added (user dependency)
2. Performance impact when enabled (must work in production)
3. Maintenance burden over time

Evaluate necessity against OpenTelemetry Semantic Conventions for Gen AI.

## Implementation

### Prometheus Client Library

- Initially `aioprometheus` ([PR #1890](https://github.com/vllm-project/vllm/pull/1890))
- Switched to `prometheus_client` ([PR #2730](https://github.com/vllm-project/vllm/pull/2730))
- HTTP middleware via `prometheus_fastapi_instrumentator` ([PR #15657](https://github.com/vllm-project/vllm/pull/15657))

### Key PRs

Metrics design and implementation:
- [Issue #3616](https://github.com/vllm-project/vllm/issues/3616) - "Even Better Observability" roadmap
- [Issue #10582](https://github.com/vllm-project/vllm/issues/10582) - V1 metrics implementation tracking
- [PR #2316](https://github.com/vllm-project/vllm/pull/2316) - Grafana dashboard
- [PR #7279](https://github.com/vllm-project/vllm/pull/7279) - Multiprocess metrics
- [PR #17546](https://github.com/vllm-project/vllm/pull/17546) - API server metrics collection

## Cross-References

- [[vLLM Engine]] — Core scheduler and execution engine that emits metrics
- [[V1 Architecture]] — AsyncLLM.output_handler_loop metrics collection
- [[Prefix Caching]] — Prefix cache hit/miss metrics
- [[Speculative Decoding]] — Draft acceptance metrics
- [[KV Cache]] — Cache usage and residency metrics
- [[LoRA]] — Per-adapter request tracking
- [[Data Parallelism]] — Internal load balancer metric aggregation
- [[NVTX Profiling]] — Complementary GPU profiling via NVTX markers

## Related

- **OpenTelemetry**: Distributed tracing (separate from metrics aggregation)
- **Grafana**: Visualization and alerting on Prometheus time-series
- **Kubernetes HPA**: Autoscaling based on custom vLLM metrics
