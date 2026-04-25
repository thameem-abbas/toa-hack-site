---
title: Continuous Batching
type: concept
tags: [scheduling, batching, optimization, throughput]
related: [Chunked Prefill, PagedAttention, vLLM Engine, V1 Architecture, KV Cache]
---

# Continuous Batching

Dynamic request scheduling where new requests are added to the batch as soon as existing requests finish, rather than waiting for the entire batch to complete. Also known as **iteration-level scheduling**.

## Core Problem: Static Batching Inefficiency

Traditional LLM serving uses **static batching**:

1. **Wait** until batch is full (e.g., 8 requests)
2. **Process** all requests together until the **longest** request completes
3. **Wait** for next batch to fill
4. **Repeat**

**Example scenario with static batching:**
```
Batch size = 4
Request A: 10 output tokens
Request B: 100 output tokens  ← longest
Request C: 20 output tokens
Request D: 15 output tokens

Timeline (static batching):
  Step 1-10:   All 4 requests active (A finishes at step 10, GPU continues)
  Step 11-15:  B, C, D active (C and D finish, GPU continues)
  Step 16-100: Only B active (GPU wasted on 75% padding!)
  Step 101:    Batch completes, next batch can start

GPU utilization: 40/100 = 40% (60% wasted on padding)
```

**Problems:**
- **Padding waste:** Shorter requests finish early but GPU waits for longest
- **Batching delay:** New requests wait for entire current batch to complete
- **Low throughput:** GPU cycles wasted on empty slots

## What Continuous Batching Is

Continuous batching dynamically adds/removes requests from the batch **after every iteration** (forward pass):

- **No padding:** Each request generates tokens at its own rate
- **Immediate removal:** Request removed as soon as it finishes (no waiting)
- **Immediate addition:** New requests added as soon as batch has capacity
- **Iteration-level scheduling:** Scheduler runs after each forward pass

**Same scenario with continuous batching:**
```
Batch capacity = 4

Step 1:    A, B, C, D active (batch size 4)
Step 10:   A finishes → remove A, add E → B, C, D, E active (batch size 4)
Step 15:   C, D finish → remove C, D, add F, G → B, E, F, G active (batch size 4)
Step 30:   E finishes → remove E, add H → B, F, G, H active (batch size 4)
...
Step 100:  B finishes

GPU utilization: ~100% (no padding waste)
```

**Benefits:**
- **Higher throughput:** 2-10× improvement over static batching (depending on output length variance)
- **Lower latency:** New requests start immediately when batch has capacity
- **No padding:** GPU always processing useful tokens

## Iteration-Level Scheduling

The key innovation in continuous batching is **iteration-level scheduling** (vs. batch-level scheduling):

**Static batching (batch-level):**
```python
while True:
    batch = wait_for_full_batch()        # Wait for N requests
    while not batch.all_finished():      # Wait for ALL to finish
        outputs = model(batch)
        batch.update(outputs)
    return_results(batch)
```

**Continuous batching (iteration-level):**
```python
batch = []
while True:
    # Add new requests up to capacity
    while len(batch) < max_batch_size and has_waiting_requests():
        batch.append(get_next_request())
    
    # Execute one iteration
    outputs = model(batch)
    
    # Remove finished requests
    batch = [req for req in batch if not req.finished()]
```

**Key difference:** Scheduling decision happens **every iteration**, not every batch.

## Request Lifecycle

In continuous batching, requests flow through states:

```
┌─────────┐
│ WAITING │  ← New request arrives, added to queue
└────┬────┘
     │ Scheduler adds to batch (when capacity available)
     ↓
┌─────────┐
│ RUNNING │  ← Request in active batch, generating tokens
└────┬────┘
     │ EOS token generated OR max_tokens reached
     ↓
┌──────────┐
│ FINISHED │  ← Request removed from batch, result returned
└──────────┘
```

**Scheduler actions each iteration:**
1. **Remove:** Finished requests (EOS, max_tokens, stop sequences, errors)
2. **Add:** Waiting requests (up to `max_batch_size` or token budget)
3. **Preempt (optional):** Pause low-priority requests to make room for high-priority
4. **Resume (optional):** Unpause preempted requests when capacity available

## vLLM Implementation

### V0 Scheduler

**Two-queue design:**
- **Prefill queue:** Requests waiting for first token (TTFT)
- **Decode queue:** Requests generating subsequent tokens (ITL)

**Scheduling priority:**
1. Running decode requests (minimize ITL variance)
2. New prefill requests (minimize queue time)

**Limitations:**
- Mode switching between "prefill batch" and "decode batch"
- Complex logic to balance TTFT vs. ITL
- Difficult to combine with [[Chunked Prefill]]

### V1 Unified Scheduler

**Single-queue design:**
- All requests in one priority queue (FCFS or priority-based)
- Token budget allocation: `{request_id: num_tokens}`
- No distinction between prefill and decode (see [[V1 Architecture#Unified Scheduler]])

**Scheduling algorithm (simplified):**
```python
def schedule_iteration():
    budget = max_num_batched_tokens
    allocation = {}
    
    # Priority 1: Running requests (decode)
    for request in running_requests:
        if request.is_decoding:
            allocation[request.id] = 1  # 1 token per decode
            budget -= 1
    
    # Priority 2: New requests (prefill, possibly chunked)
    for request in waiting_requests:
        if budget <= 0:
            break
        
        if request.is_prefill:
            chunk_size = min(request.remaining_prefill_tokens, budget)
            allocation[request.id] = chunk_size
            budget -= chunk_size
    
    return allocation
```

**Benefits:**
- Simpler code (no mode switching)
- Natural [[Chunked Prefill]] support
- Easier to add features (spec decode, prefix caching)

## Contrast with Static Batching

| Aspect | Static Batching | Continuous Batching |
|--------|-----------------|---------------------|
| **Batch lifetime** | Until longest request finishes | One iteration (forward pass) |
| **Request addition** | Wait for full batch | Add when capacity available |
| **Request removal** | All finish together | Remove individually as they finish |
| **Padding** | Pad to max length | No padding (variable-length sequences) |
| **GPU utilization** | Low (50-80% typical) | High (90-100% typical) |
| **Throughput** | Baseline | 2-10× higher |
| **Complexity** | Simple | Moderate (requires efficient scheduling) |

## Enabling Technology: PagedAttention

Continuous batching requires **efficient memory management** because:
- Batch composition changes every iteration
- Variable-length sequences (no pre-allocation to max length)
- Frequent request additions/removals

**Pre-PagedAttention challenges:**
1. **Pre-allocation:** Allocate max length for each request → massive waste
2. **Defragmentation:** Removing requests creates memory holes
3. **Reallocation:** Adding requests requires finding contiguous memory

**[[PagedAttention]] solution:**
- Block-based KV cache allocation (like OS virtual memory)
- Non-contiguous memory (blocks scattered in physical memory)
- Fast allocation/deallocation (blocks from free pool)
- Zero defragmentation (blocks reused immediately)

**Impact:** PagedAttention enables continuous batching to be **memory-efficient** (no waste) and **fast** (no memory reallocation overhead).

## Performance Characteristics

### Throughput Gains

Continuous batching throughput depends on **output length variance**:

**Low variance (all requests similar length):**
- Static batching: 100 requests/sec
- Continuous batching: 120 requests/sec
- **Gain:** 20% (low variance → less padding waste)

**High variance (10× difference in lengths):**
- Static batching: 50 requests/sec
- Continuous batching: 300 requests/sec
- **Gain:** 6× (high variance → massive padding waste in static)

**Typical real-world gain:** 2-4× for mixed workloads

### Latency Impact

**TTFT (Time to First Token):**
- **Static:** Wait for batch to fill, then wait for prefill
- **Continuous:** Start as soon as capacity available
- **Improvement:** 50-90% reduction in queue time

**ITL (Inter-Token Latency):**
- **Static:** Consistent (all requests same batch)
- **Continuous:** Slightly higher (batch composition changes)
- **Trade-off:** +10-20% ITL variance for +2-4× throughput

### GPU Utilization

**Static batching:**
- Early iterations: 100% (full batch)
- Late iterations: 25% (only longest request remains)
- **Average:** 60-70%

**Continuous batching:**
- All iterations: 90-100% (batch kept full)
- **Average:** 95%+

## Interaction with Other Features

### Chunked Prefill

[[Chunked Prefill]] extends continuous batching to **prefill phase**:

**Without chunked prefill:**
- Continuous batching only during decode (prefill is monolithic)
- Long prefills block decode requests

**With chunked prefill:**
- Prefill broken into chunks, interleaved with decode
- Both prefill and decode benefit from continuous batching

**Unified view:** Both are forms of iteration-level scheduling.

### Prefix Caching

[[Prefix Caching]] works naturally with continuous batching:

1. New request arrives with cached prefix
2. Scheduler adds request to batch (skips cached tokens)
3. Request executes only uncached portion
4. Request removed when finished

**No special handling needed** — caching just reduces tokens to compute.

### Speculative Decoding

[[Speculative Decoding]] changes token generation rate but not batching logic:

**Without spec decode:**
- Each iteration: 1 token per request

**With spec decode:**
- Each iteration: 1-5 tokens per request (depending on acceptance rate)

**Continuous batching still works:** Scheduler removes requests when they reach EOS/max_tokens, regardless of how many tokens per iteration.

### Data Parallelism

With [[Data Parallelism]], each replica independently runs continuous batching:

- **No coordination:** Each replica has its own batch
- **Load balancing:** Requests distributed across replicas by proxy/router
- **Independent scheduling:** Each replica's scheduler operates autonomously

## Orca: The Original Paper

Continuous batching was introduced by **Orca** (Yu et al., 2022):

**Key contributions:**
1. **Iteration-level scheduling:** Add/remove requests per iteration (vs. batch-level)
2. **Selective batching:** Prefill and decode in separate batches (vLLM V0 adopted this)
3. **Performance analysis:** Showed 2-10× throughput gains

**vLLM extensions:**
- [[PagedAttention]]: Efficient memory management for continuous batching
- Unified scheduler (V1): No prefill/decode separation
- [[Chunked Prefill]]: Iteration-level scheduling for prefill phase

**Paper reference:** "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022)

## Implementation Details

### Scheduler Core Loop

**Simplified vLLM V1 scheduler:**

```python
class Scheduler:
    def __init__(self, max_num_seqs, max_num_batched_tokens):
        self.waiting_queue = []
        self.running_requests = []
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
    
    def schedule(self):
        # 1. Remove finished requests
        self.running_requests = [
            req for req in self.running_requests 
            if not req.is_finished()
        ]
        
        # 2. Allocate budget to running requests (decode)
        budget = self.max_num_batched_tokens
        allocation = {}
        for req in self.running_requests:
            allocation[req.id] = 1  # 1 token per decode
            budget -= 1
        
        # 3. Add new requests from waiting queue (prefill)
        while (len(self.running_requests) < self.max_num_seqs and 
               self.waiting_queue and 
               budget > 0):
            req = self.waiting_queue.pop(0)
            chunk = min(req.num_remaining_tokens, budget)
            allocation[req.id] = chunk
            budget -= chunk
            self.running_requests.append(req)
        
        return allocation
```

### Request State Machine

```
┌──────────────┐
│   WAITING    │ ← Request in queue, not yet scheduled
└──────┬───────┘
       │ Scheduler adds to batch
       ↓
┌──────────────┐
│   RUNNING    │ ← Request generating tokens
│  (prefill)   │
└──────┬───────┘
       │ Prefill completes
       ↓
┌──────────────┐
│   RUNNING    │ ← Request generating output tokens
│   (decode)   │
└──────┬───────┘
       │ EOS or max_tokens
       ↓
┌──────────────┐
│   FINISHED   │ ← Request complete, result returned
└──────────────┘
```

**Additional states (optional):**
- **PREEMPTED:** Paused to free resources for higher-priority request
- **SWAPPED:** KV cache offloaded to CPU (V0 only, removed in V1)

## Common Misconceptions

**Misconception 1:** "Continuous batching means batch size is always changing."
- **Reality:** Batch size can stay constant (e.g., always 32 requests) by adding requests as others finish.

**Misconception 2:** "Continuous batching reduces latency."
- **Clarification:** It reduces **queue latency** (TTFT) but may slightly increase **ITL variance**. Primary benefit is **throughput**.

**Misconception 3:** "Continuous batching only works for decode."
- **Reality:** With [[Chunked Prefill]], continuous batching applies to prefill too.

**Misconception 4:** "Continuous batching requires complex scheduling."
- **Reality:** V1's unified scheduler makes it straightforward (single queue, token budget allocation).

## Trade-offs and Limitations

### Advantages

- **High throughput:** 2-10× over static batching
- **High GPU utilization:** 90-100% vs. 60-70%
- **Low queue latency:** Requests start immediately when capacity available
- **No padding waste:** Each request generates at its own rate

### Disadvantages

- **Scheduler overhead:** Scheduling decision every iteration (mitigated by efficient implementation)
- **ITL variance:** Batch composition changes → slight latency variance (mitigated by [[Chunked Prefill]])
- **Complexity:** More complex than static batching (but abstracted away in vLLM)

### When to Use

**Always use continuous batching for production LLM serving:**
- Default in all modern serving systems (vLLM, TGI, TensorRT-LLM)
- No downside in real-world workloads
- Only avoid if you have strict ITL SLO and low throughput needs (rare)

## Future Directions

### Priority-Based Scheduling

**Current:** FCFS (first-come, first-served) within continuous batching.

**Planned:** Priority-based scheduling:
- High-priority requests get tokens first
- Low-priority requests preempted if needed
- SLO-aware scheduling (allocate budget to meet latency targets)

### Preemption Policies

**Current:** Requests run to completion (or max_tokens).

**Future:** Smart preemption:
- Preempt long-running requests to serve short bursts
- Resume preempted requests when load decreases
- Credit-based fairness (preempted requests get priority later)

### Adaptive Batching

**Current:** Fixed `max_num_seqs` and `max_num_batched_tokens`.

**Future:** Dynamic adjustment:
- Increase batch size when GPU memory underutilized
- Decrease batch size when latency SLO at risk
- Workload-adaptive tuning

## Cross-References

**Concepts:**
- [[Chunked Prefill]] — Iteration-level scheduling for prefill phase
- [[PagedAttention]] — Efficient memory management enabling continuous batching
- [[KV Cache]] — Storage structure managed by continuous batching scheduler
- [[Prefix Caching]] — Cache reuse works naturally with continuous batching

**Architectures:**
- [[vLLM Engine]] — Engine implementing continuous batching scheduler
- [[V1 Architecture]] — Unified scheduler design for continuous batching

**Tools:**
- [[vLLM]] — Serving system with continuous batching by default

## See Also

- Orca Paper: "A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022)
- vLLM Paper: [[Efficient Memory Management for Large Language Model Serving with PagedAttention]]
- Design Doc: `docs/design/scheduler.md` (V1 unified scheduler)
- Blog Post: [vLLM V1 Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)
