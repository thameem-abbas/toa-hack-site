---
title: llm-d
type: tool
tags: [kubernetes, orchestration, disaggregated-serving, llm-inference]
created: 2026-04-25
---

# llm-d

Kubernetes-native distributed inference serving stack for large generative AI models at scale.

## Overview

llm-d (LLM Daemon) is a Kubernetes-native distributed inference serving system that provides well-lit paths for deploying state-of-the-art open-source LLMs at scale. It integrates tightly with [[vLLM]] as the serving backend and extends it with Kubernetes-native orchestration, autoscaling, and disaggregated serving capabilities.

**Project**: [llm-d/llm-d](https://github.com/llm-d/llm-d)  
**Documentation**: [llm-d.ai/docs/guide](https://llm-d.ai/docs/guide)

## Key Features

### Disaggregated Serving Architecture
llm-d implements [[Disaggregated Serving]] patterns at the Kubernetes orchestration layer:
- **Prefill pools**: Optimized for TTFT (compute-bound, high GPU compute)
- **Decode pools**: Optimized for ITL (memory-bound, high throughput)
- **KV cache routing**: Automatic KV cache transfer between pools via [[KV Cache Transfer]] connectors

### Autoscaling
- **Horizontal Pod Autoscaler (HPA)**: Scale pods based on QPS, GPU utilization, queue depth
- **Per-pool scaling**: Independent scaling of prefill and decode pools
- **Custom metrics**: Prometheus integration for vLLM-specific metrics

### Hardware Acceleration Support
llm-d helps achieve "fastest time to SOTA performance" across:
- NVIDIA GPUs (H100, A100, L40S, etc.)
- AMD GPUs (MI300, MI250)
- Intel Gaudi accelerators
- Custom accelerators via vLLM OOT plugins

## Architecture

### Components

**llm-d Operator**: Kubernetes operator managing LLM inference workloads
- Custom Resource Definitions (CRDs) for LLMInferenceService
- Reconciliation loop for desired state management
- Integration with vLLM serving pods

**vLLM Backend**: High-throughput inference engine
- OpenAI-compatible API via [[OpenAI-Compatible Server]]
- [[PagedAttention]] for KV cache memory management
- [[Continuous Batching]] for request scheduling

**KServe Integration**: Model serving framework
- LLMInferenceService CRD for declarative model deployment
- Canary rollouts and A/B testing
- Traffic routing and load balancing

### Deployment Topology

```
┌──────────────────────────────────────────────┐
│  Kubernetes Cluster (llm-d)                  │
│                                              │
│  ┌─────────────┐      ┌─────────────┐       │
│  │ Prefill     │ KV → │ Decode      │       │
│  │ Pool        │ xfer │ Pool        │       │
│  │ (8×H100)    │──────│ (16×H100)   │       │
│  └─────────────┘      └─────────────┘       │
│         ↑                     ↓              │
│         │                     │              │
│    ┌────┴─────────────────────┴────┐         │
│    │  llm-d Operator               │         │
│    │  (KV routing, autoscaling)    │         │
│    └────────────────────────────────┘         │
│         ↑                                    │
│         │                                    │
│    ┌────┴────────────────┐                   │
│    │ External LB         │                   │
│    │ (nginx/Istio)       │                   │
│    └─────────────────────┘                   │
└──────────────────────────────────────────────┘
         ↑
         │ HTTP/gRPC
         │
    Client Applications
```

## vLLM Integration

### Direct Usage
```bash
# Install llm-d operator
kubectl apply -f https://llm-d.ai/manifests/operator.yaml

# Deploy LLM via llm-d
kubectl apply -f - <<EOF
apiVersion: llm-d.ai/v1alpha1
kind: LLMInferenceService
metadata:
  name: llama3-70b
spec:
  modelId: meta-llama/Llama-3.3-70B-Instruct
  backend: vllm
  tensorParallelism: 8
  disaggregated:
    prefillPoolSize: 8
    decodePoolSize: 16
    kvCacheConnector: P2pNcclConnector
EOF
```

### KServe LLMInferenceService
```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
metadata:
  name: deepseek-r1
spec:
  predictor:
    model:
      modelFormat: vllm
      args:
        - --model=deepseek-ai/DeepSeek-R1
        - --tensor-parallel-size=16
        - --pipeline-parallel-size=2
```

See [KServe LLMInferenceService docs](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/llmisvc-overview).

## Disaggregated Serving Strategies

### Prefill/Decode Separation
llm-d orchestrates separation of prefill (compute-bound) and decode (memory-bound) workloads:
- **Prefill pool**: Smaller pool with high compute (fewer GPUs, high FLOPS)
- **Decode pool**: Larger pool with high memory bandwidth (more GPUs, more KV cache capacity)
- **KV cache transfer**: Automatic routing via connector (P2pNccl, Nixl, LMCache, Mooncake)

### Independent Scaling
- Prefill pool scales based on request queue depth (TTFT target)
- Decode pool scales based on active generation count (ITL target)
- Separate HPA policies for each pool

### Cost Optimization
- **Low QPS**: Single pool (no disaggregation overhead)
- **Medium QPS**: Disaggregated with small prefill pool (2:1 or 4:1 decode:prefill ratio)
- **High QPS**: Disaggregated with auto-scaling (dynamic pool sizing)

## Comparison with vLLM Native Deployment

| Feature | vLLM Native | llm-d + vLLM |
|---------|-------------|--------------|
| **Deployment** | Manual YAML/Helm | Declarative CRD |
| **Autoscaling** | External HPA | Built-in per-pool |
| **Disaggregation** | Manual setup | Automatic orchestration |
| **KV transfer** | Manual connector config | Automatic routing |
| **Multi-tenancy** | External LB | Built-in routing |
| **Hardware support** | Manual GPU selection | Auto-detection |
| **Canary rollouts** | External (Argo/Flagger) | KServe integration |

## Performance Characteristics

### Latency
- **TTFT**: 10-30% reduction via prefill pool optimization (dedicated compute)
- **ITL**: 5-15% reduction via decode pool optimization (dedicated memory bandwidth)
- **KV transfer overhead**: 1-5ms per request (negligible for >100 token generations)

### Throughput
- **xPyD pattern** (disaggregated prefill/decode): No throughput improvement, latency-focused
- **Autoscaling**: 2-5× throughput improvement under variable load (elastic scaling)

### Resource Utilization
- **Prefill pool**: 80-95% GPU compute utilization (compute-bound)
- **Decode pool**: 60-80% GPU memory bandwidth utilization (memory-bound)
- **Overall efficiency**: 15-25% improvement over monolithic deployment (specialized workload placement)

## Configuration

### Prefill Pool Tuning
```yaml
spec:
  prefillPool:
    replicas: 4
    gpuType: nvidia.com/h100
    gpusPerPod: 2
    maxNumBatchedTokens: 32768  # Large for high TTFT throughput
```

### Decode Pool Tuning
```yaml
spec:
  decodePool:
    replicas: 8
    gpuType: nvidia.com/h100
    gpusPerPod: 2
    maxNumBatchedTokens: 8192   # Smaller for low ITL
    gpuMemoryUtilization: 0.95  # Maximize KV cache capacity
```

### KV Cache Connector Selection
```yaml
spec:
  disaggregated:
    kvCacheConnector: NixlConnector  # Options: P2pNcclConnector, NixlConnector, LMCache, Mooncake
```

See [[KV Cache Transfer]] for connector comparison.

## Cross-References

- [[vLLM]] — Underlying serving engine
- [[Disaggregated Serving]] — Prefill/decode separation concept
- [[KV Cache Transfer]] — KV cache routing between pools
- [[OpenAI-Compatible Server]] — HTTP API exposed by vLLM backend
- [[Tensor Parallelism]] — Multi-GPU model sharding
- [[Data Parallelism]] — Multi-replica scaling (external LB pattern)
- [[Continuous Batching]] — Request batching within vLLM
- [[PagedAttention]] — KV cache memory management

## Known Limitations

- **Single-node prefill/decode only**: Cross-node KV transfer under development
- **KV cache connector compatibility**: Some connectors (Mooncake, FlexKV) require specific hardware/drivers
- **Autoscaling cold start**: 30-120s pod startup time (GPU initialization)
- **Cost overhead**: Disaggregation adds KV transfer latency (1-5ms per request)

## Example: DeepSeek-R1 Deployment

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: LLMInferenceService
metadata:
  name: deepseek-r1
spec:
  modelId: deepseek-ai/DeepSeek-R1
  backend: vllm
  # 671B MoE model, 16 experts per GPU in FP8
  tensorParallelism: 16
  quantization: fp8
  disaggregated:
    enabled: true
    prefillPoolSize: 4   # 4×(16 GPUs) = 64 GPUs for prefill
    decodePoolSize: 8    # 8×(16 GPUs) = 128 GPUs for decode
    kvCacheConnector: NixlConnector
  autoscaling:
    enabled: true
    minReplicas: 1
    maxReplicas: 10
    targetQueueDepth: 50  # Scale when queue depth exceeds 50
```

See `examples/online_serving/ray_serve_deepseek.py` for Ray Serve alternative.

## See Also

- Official docs: [llm-d.ai/docs/guide](https://llm-d.ai/docs/guide)
- GitHub: [llm-d/llm-d](https://github.com/llm-d/llm-d)
- KServe LLMInferenceService: [kserve.github.io/website/docs/.../llmisvc-overview](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/llmisvc-overview)
- [[Ray Serve LLM]] for alternative Kubernetes-native serving
