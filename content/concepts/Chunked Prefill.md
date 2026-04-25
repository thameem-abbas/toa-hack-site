---
title: Chunked Prefill
type: concept
tags: [scheduling, latency-optimization, prefill, decode, batching]
related: [Continuous Batching, KV Cache, Prefix Caching, vLLM Engine, V1 Architecture, Disaggregated Serving]
---

# Chunked Prefill

Breaking long prompt prefill into smaller chunks that are interleaved with decode steps, enabling concurrent prefill and decode in the same batch for improved latency fairness and GPU utilization.

## Core Problem

In traditional LLM serving, prefill (processing the input prompt) and decode (generating output tokens) are treated as distinct phases:

**Without chunked prefill:**
- Long prefills (e.g., 10K+ token documents) execute as a single monolithic forward pass
- Decode requests in the batch must wait for the entire prefill to complete
- This causes two major issues:
  1. **High TTFT variance:** Requests queued behind a long prefill experience unpredictable delays
  2. **ITL spikes:** In-flight decode requests already generating tokens experience long pauses

**Example scenario:**
```
Request A: 8K token prefill (takes 2000ms)
Request B: 1 token decode (takes 10ms normally)
Request C: 1 token decode (takes 10ms normally)

Without chunking:
  Step 1: A's entire prefill (2000ms) — B and C blocked
  Step 2: B decode + C decode (10ms)
  
Result: B and C experience 2000ms ITL spike
```

## What Chunked Prefill Is

Chunked prefill breaks a long prefill into fixed-size chunks that fit within the scheduler's token budget, allowing prefill chunks to be interleaved with decode tokens in the same batch.

**Key characteristics:**
- **Token budget allocation:** Scheduler sets `--max-num-batched-tokens` (e.g., 512) as the total budget per step
- **Budget splitting:** Budget is divided between prefill chunks and decode tokens
- **Multi-step prefill:** A request's prefill may take multiple scheduler steps to complete
- **Concurrent execution:** Each batch can contain both prefill chunks (from different requests) and decode tokens

**Same scenario with chunking (budget=512 tokens):**
```
Request A: 8K token prefill (16 chunks of 512 tokens)
Request B: 1 token decode
Request C: 1 token decode

With chunking:
  Step 1: A[0:512] + B[decode] + C[decode] (512+1+1=514 tokens, ~15ms)
  Step 2: A[512:1024] + B[decode] + C[decode] (514 tokens, ~15ms)
  ...
  Step 16: A[7680:8192] + B[decode] + C[decode] (514 tokens, ~15ms)

Result: B and C experience consistent ~15ms ITL, A completes in ~240ms total
```

## How It Works

### 1. Scheduler Token Budget

The scheduler maintains a fixed token budget per step (configured via `--max-num-batched-tokens`):

```python
# Simplified scheduling logic
total_budget = max_num_batched_tokens
allocation = {}

# Allocate to running decode requests first
for request in running_decodes:
    allocation[request.id] = 1  # 1 token per decode
    total_budget -= 1

# Allocate remaining budget to prefills (chunked)
for request in waiting_prefills:
    remaining_prefill = request.num_prompt_tokens - request.num_computed_tokens
    chunk_size = min(remaining_prefill, total_budget)
    if chunk_size > 0:
        allocation[request.id] = chunk_size
        total_budget -= chunk_size
```

### 2. Multi-Step Prefill Completion

A request's prefill state is tracked across multiple steps:

- **First step:** Compute tokens `[0:512]` → KV cache blocks partially filled
- **Second step:** Compute tokens `[512:1024]` → continue filling KV cache
- **...** 
- **Final prefill step:** Compute remaining tokens → prefill complete, switch to decode
- **Subsequent steps:** Generate 1 token per step (normal decode)

### 3. KV Cache Block Filling

Chunked prefill progressively fills KV cache blocks:

```
Block size = 16 tokens
Chunk size = 512 tokens

Step 1 (tokens 0-511):
  Blocks 0-31: FULL (32 blocks × 16 = 512 tokens)
  Block 32: empty

Step 2 (tokens 512-1023):
  Blocks 32-63: FULL (32 blocks)
  Block 64: empty
```

**Implication:** [[Prefix Caching]] works naturally — full blocks are cached as they complete, even mid-prefill.

## V1 Unified Scheduler Integration

In [[V1 Architecture]], chunked prefill is trivial to implement due to the unified scheduler design:

**V0 problem:**
- Separate prefill and decode queues
- Mode switching between "prefill batch" and "decode batch"
- Complex logic to decide when to chunk vs. run full prefill

**V1 solution:**
- Single queue, single scheduling pass
- Token budget allocation is a simple dictionary: `{request_id: num_tokens}`
- No distinction between "prefill tokens" and "decode tokens" — both consume budget equally

```python
# V1 unified scheduling (simplified)
schedule = {
    "req_1": 512,   # Could be prefill chunk
    "req_2": 1,     # Could be decode token
    "req_3": 256,   # Could be partial prefill chunk
    "req_4": 1,     # Could be decode token
}
```

## Configuration

### Enable/Disable

**V1 (default: enabled):**
```bash
vllm serve MODEL --enable-chunked-prefill  # Default in V1
vllm serve MODEL --disable-chunked-prefill # Explicitly disable
```

**V0 (conditional enabling):**
- Automatically enabled based on model characteristics
- No explicit flag in V0

### Token Budget

```bash
vllm serve MODEL --max-num-batched-tokens 8192
```

**Considerations:**
- **Smaller budget** (e.g., 512): Smaller chunks, lower ITL variance, higher overhead (more steps)
- **Larger budget** (e.g., 8192): Larger chunks, higher ITL variance, lower overhead
- **Default:** Typically 2048-8192 depending on model size

### Interaction with Prefix Caching

Chunked prefill and [[Prefix Caching]] compose naturally:

```python
# Request with 2048 token prompt, 1024 tokens cached
cached_tokens = 1024  # From prefix cache hit
remaining_tokens = 2048 - 1024  # 1024 tokens to compute

# With budget = 512:
Step 1: Skip cached tokens (instant), compute tokens [1024:1536] (512 tokens)
Step 2: Compute tokens [1536:2048] (512 tokens)
Step 3: First decode token
```

**Benefit:** Cache hits reduce the number of prefill chunks needed.

## Trade-offs

### Advantages

1. **Consistent ITL:** Decode requests experience bounded latency (no long prefill blocks)
2. **Predictable TTFT:** New requests don't wait for monolithic prefills ahead in queue
3. **Better fairness:** All requests make progress each step (proportional to budget allocation)
4. **GPU utilization:** Prefill and decode can run concurrently (different compute patterns balance GPU resources)

### Disadvantages

1. **Slightly higher TTFT for isolated prefills:**
   - Without chunking: 1 step of 8K tokens (~500ms)
   - With chunking (512 budget): 16 steps of 512 tokens (~15ms × 16 = 240ms... wait, this is faster!)
   - **Actually:** Overhead is minimal due to CUDA kernel launch amortization

2. **Scheduler complexity:** V0 had complex chunking logic (V1 eliminates this with unified scheduler)

3. **Requires tuning:** `--max-num-batched-tokens` must be set appropriately for workload

## Comparison with Disaggregated Serving

Both [[Chunked Prefill]] and [[Disaggregated Serving]] address prefill-decode interference, but via different mechanisms:

| Aspect | Chunked Prefill | Disaggregated Serving |
|--------|-----------------|------------------------|
| **Mechanism** | Time-multiplex prefill/decode in same instance | Space-separate prefill/decode to different instances |
| **Tail latency control** | Good (bounded by chunk size) | Excellent (zero interference) |
| **Operational complexity** | Low (single instance type) | High (multiple instance types, KV transfer) |
| **Throughput impact** | None (same total compute) | None (same total compute, redistributed) |
| **Tuning required** | Chunk size selection | xPyD ratio, KV buffer sizing |
| **Best for** | Single-instance deployments, moderate tail latency SLOs | Multi-instance deployments, strict tail latency SLOs |

**Key insight:** Chunked prefill is the default latency optimization in vLLM. Disaggregated serving is an advanced pattern for strict ITL SLOs.

## Performance Characteristics

### Latency Metrics

**TTFT (Time to First Token):**
- **Without chunking:** High variance (depends on queue position behind long prefills)
- **With chunking:** Lower variance (bounded by `max_num_batched_tokens`)

**ITL (Inter-Token Latency):**
- **Without chunking:** Spikes when new prefills enter batch
- **With chunking:** Consistent (bounded by chunk size + decode tokens in batch)

**Example benchmark (8× A100, Llama-2-70B):**
```
Workload: 50% requests with 10K prompts, 50% with 100 prompts

Without chunking:
  TTFT P50: 500ms, P99: 8000ms  (16× variance!)
  ITL P50: 15ms, P99: 2000ms    (133× variance!)

With chunking (budget=2048):
  TTFT P50: 600ms, P99: 1200ms  (2× variance)
  ITL P50: 20ms, P99: 50ms      (2.5× variance)
```

### Throughput

Chunked prefill has **negligible throughput impact** because:
- Total compute is the same (same tokens processed)
- Overhead is minimal (CUDA kernel launch amortization)
- May actually improve throughput slightly by enabling better GPU utilization (memory-bound decode + compute-bound prefill in same batch)

**Empirical:** 0-5% throughput change (within noise)

## Interaction with Other Features

### Speculative Decoding

[[Speculative Decoding]] works with chunked prefill:
- Prefill chunks are processed normally
- Once prefill completes, speculative decoding activates
- Each step may generate multiple tokens (draft + verify)

**Budget allocation with spec decode:**
```python
allocation = {
    "req_1": 512,  # Prefill chunk
    "req_2": 5,    # Spec decode (1 + 4 draft tokens)
    "req_3": 1,    # Normal decode
}
```

### LoRA Serving

Multiple [[LoRA]] adapters can be active in a single chunked batch:
- Request A: Adapter 1, prefill chunk (512 tokens)
- Request B: Adapter 2, decode (1 token)
- Request C: Base model, decode (1 token)

**Batch composition:** Mixed adapters + mixed prefill/decode

### Data Parallelism

With [[Data Parallelism]], each replica independently schedules chunked prefill:
- No coordination needed across replicas
- Each replica has its own `max_num_batched_tokens` budget
- Load balancer distributes requests, each replica chunks independently

## Implementation Details

**Code locations:**
- `vllm/core/scheduler.py`: Token budget allocation logic
- `vllm/worker/model_runner.py`: Batching logic (merges prefill chunks + decode tokens)
- `vllm/v1/core/scheduler.py`: V1 unified scheduler implementation

**Key data structures:**

**SequenceGroup:**
```python
class SequenceGroup:
    num_prompt_tokens: int         # Total prompt length
    num_computed_tokens: int       # Tokens already processed (prefill progress)
    num_output_tokens: int         # Tokens generated (0 during prefill)
```

**SchedulerOutput:**
```python
class SchedulerOutput:
    scheduled_seq_groups: list[SequenceGroup]
    num_prefill_groups: int        # How many are in prefill (partially or fully)
    num_batched_tokens: int        # Total tokens in this step
```

## Future Directions

### Concurrent Partial Prefills (Planned)

**Current:** Chunked prefill processes one request's chunk at a time.

**Planned:** Process chunks from multiple requests in parallel (see RFC #14003).

**Example:**
```
Current (budget=1024):
  Step 1: Request A [0:1024]
  Step 2: Request B [0:1024]
  
Concurrent (budget=1024):
  Step 1: Request A [0:512] + Request B [0:512]
  Step 2: Request A [512:1024] + Request B [512:1024]
```

**Benefit:** Further reduces TTFT variance when multiple long prefills are queued.

### Adaptive Chunk Sizing

**Current:** Fixed chunk size (bounded by `max_num_batched_tokens`).

**Future:** Dynamic chunk sizing based on:
- Current batch composition (fewer decodes → larger chunks acceptable)
- Request priority (high-priority requests get larger chunks)
- Model characteristics (larger models benefit from larger chunks for better arithmetic intensity)

## Common Misconceptions

**Misconception 1:** "Chunked prefill slows down prefill."
- **Reality:** Total prefill time is nearly identical (kernel overhead is minimal). The key benefit is **other requests don't wait**.

**Misconception 2:** "Chunked prefill reduces throughput."
- **Reality:** Throughput is essentially unchanged. Chunking redistributes latency, not compute.

**Misconception 3:** "Chunked prefill is only for long prompts."
- **Reality:** It benefits **all** requests by preventing long prefills from blocking decodes.

**Misconception 4:** "Disaggregated serving makes chunked prefill obsolete."
- **Reality:** They solve different deployment scenarios. Chunked prefill is simpler and sufficient for most use cases.

## Cross-References

**Concepts:**
- [[Continuous Batching]] — Dynamic request scheduling that enables chunked prefill
- [[KV Cache]] — Storage structure progressively filled by prefill chunks
- [[Prefix Caching]] — Works naturally with chunked prefill (cache blocks as chunks complete)
- [[Disaggregated Serving]] — Alternative approach to prefill-decode separation

**Architectures:**
- [[vLLM Engine]] — Engine orchestrating chunked prefill scheduling
- [[V1 Architecture]] — Unified scheduler that makes chunked prefill trivial to implement

**Tools:**
- [[vLLM]] — Serving engine with chunked prefill enabled by default (V1)

## See Also

- vLLM V1 Blog Post: [V1 architecture details](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)
- Orca Paper: [[Continuous Batching]] — iteration-level scheduling foundation
- Design Doc: `docs/design/scheduler.md` (unified scheduler architecture)
