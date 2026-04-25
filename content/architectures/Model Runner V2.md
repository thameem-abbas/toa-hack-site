---
title: Model Runner V2
type: architecture
created: 2026-04-25
tags: [vllm, model-runner, v2, performance, async]
---

# Model Runner V2

vLLM's Model Runner V2 (MRV2) is a ground-up redesign of the model execution engine, addressing fundamental design flaws in V1 and incorporating lessons learned about sampling techniques, CUDA features, and async scheduling.

## Overview

MRV2 reimplements model execution from first principles with nine key improvements:

1. **Persistent batch decoupling** — separate state tensors from per-step inputs
2. **Async-first design** — CUDA stream execution with no CPU synchronization
3. **Async barrier elimination** — temporary pinned copies prevent race conditions
4. **StagedWriteTensor** — incremental GPU tensor updates for block tables
5. **GPU-native input preparation** — Triton kernels for metadata derivation
6. **Triton-native sampler** — Gumbel sampling, efficient top-k logprobs
7. **Modularity** — dedicated files per feature, `InputBatch` class
8. **No dummy_run abuse** — separate paths for profiling/capture/warmup
9. **Explicit CUDA graph management** — `CUDAGraphManager` for lifecycle control

While not yet feature-complete compared to V1, MRV2 provides cleaner abstractions and better async behavior.

## 1. Persistent Batch Decoupling

### V1 Problem: Coupled State and Inputs

V1's persistent batch optimization maintains state tensors (block tables, temperature values) across steps to avoid full reconstruction. However, V1 uses these persistent tensors **directly as model/sampler inputs**, imposing strict layout requirements.

**Consequences:**
- Complex tensor reordering when requests join/finish (instead of simple row insertion/removal)
- `CachedRequestState` backup copies needed (rows can be overwritten while requests active)
- Difficult bookkeeping under async scheduling

### MRV2 Solution: Decouple State from Inputs

MRV2 separates persistent state from per-step inputs:

1. **Pre-allocate** fixed-size tensor with `max_num_reqs` rows (1024 default)
2. **Permanent row assignment** for each request's active lifetime (until finish/preemption)
3. **Treat preemption as completion** — on resume, re-add as fresh state
4. **Gather inputs from state** — attention backend determines request ordering, MRV2 gathers input tensors from persistent state

**Benefits:**
- No need for `CachedRequestState`
- Simpler bookkeeping (no complex reordering)
- GPU-parallel gather (low overhead, state tensors in GPU memory)

**Example:**
```python
# V1: persistent state used directly as input
self.block_table_tensor[req_idx] = new_blocks  # complex reordering required

# MRV2: persistent state gathered into per-step input
self.persistent_state[req_idx] = new_blocks
input_block_table = gather(self.persistent_state, request_ordering)  # GPU kernel
```

## 2. Async-First Design

### Background

vLLM's async scheduling prepares inputs for step `N+1` while GPU executes step `N`, overlapping CPU and GPU work. V1 was retrofitted for async with hacks; MRV2 assumes async from the start.

### Core Principle

Model execution loop is a **CUDA stream with no CPU synchronization points**. CPU entrypoints queue work onto the stream.

**Timeline:**
```
CPU: [Prepare step N] [Prepare step N+1] [Prepare step N+2]
GPU:                  [Execute step N]    [Execute step N+1]
```

**Requirements:**
- No explicit sync: `torch.accelerator.synchronize()`
- No implicit sync: unpinned `.to("cuda")`
- All CPU operations non-blocking

## 3. Removing Async Barrier

### V1 Problem: Async Barrier Protects Shared Buffers

V1 uses async barrier to prevent race conditions when CPU and GPU concurrently access shared pinned memory.

**Example (V1 unsafe code):**
```python
class ModelRunner:
    def __init__(self, ...):
        # Pinned buffer shared between CPU and GPU
        self.states = torch.zeros(
            max_num_reqs, dtype=torch.int32, device="cpu", pin_memory=True
        )

    def execute_step(self, ...):
        self.states[req_idx] = new_req.data  # CPU write
        states = self.states.to("cuda", non_blocking=True)  # GPU async read
        # RACE: CPU may modify self.states while GPU still reading
```

**V1 solution:** Async barrier around critical sections

**Drawbacks:**
1. Bug-prone (easy to miss protected buffers)
2. Inflexible organization (all CPU work must stay inside barrier)
3. Less overlap due to synchronization

### MRV2 Solution: Eliminate the Race

Separate persistent CPU state from copied tensor via **temporary pinned copies**:

```python
class ModelRunner:
    def __init__(self, ...):
        # Not pinned (CPU-only state)
        self.states = torch.zeros(
            max_num_reqs, dtype=torch.int32, device="cpu", pin_memory=False
        )

    def execute_step(self, ...):
        self.states[req_idx] = new_req.data  # CPU write to unpinned buffer
        tmp_states = self.states.pin_memory()  # Allocate temporary pinned copy
        states = tmp_states.to("cuda", non_blocking=True)  # GPU reads from tmp
        # NO RACE: CPU writes to self.states, GPU reads from tmp_states
```

**Benefits:**
- Zero synchronization overhead
- CPU and GPU operate on separate memory
- Simpler code organization

## 4. StagedWriteTensor

### Problem: Full CPU-GPU Copies for Large Tensors

Block tables are large (e.g., 1024 requests × 1000 blocks) but only a few rows change per step. Full CPU-to-GPU copy each step is wasteful.

### Solution: Incremental GPU Updates

`StagedWriteTensor` keeps base tensor on GPU and applies incremental diffs:

1. Keep base tensor on GPU
2. Stage diffs on CPU
3. Pack diffs into contiguous buffers
4. Copy packed diffs to GPU
5. Launch single kernel to apply diffs

**Usage:**
```python
# Initialize state on GPU
state = StagedWriteTensor(size=(1024, 1000), dtype=torch.int32, device="cuda")

# Write [3, 1, 2] into row 2, starting at index 3
state.stage_write(row=2, start=3, value=[3, 1, 2])

# Write [-1, -2, -5] into row 0, starting at index 1
state.stage_write(row=0, start=1, value=[-1, -2, -5])

# Apply staged changes (pack diffs, copy to GPU, apply via kernel)
state.apply_write()
```

**Benefits:**
- Supports ragged updates (variable-length writes per row)
- No CPU-GPU synchronization
- Minimal kernel launches (one per apply)
- Especially useful for block tables and `num_computed_tokens`

## 5. GPU-Native Input Metadata Preparation

### Triton Kernels for Input Derivation

MRV2 uses Triton kernels to prepare inputs (`input_ids`, `positions`, `query_start_loc`, `seq_lens`) instead of CPU loops.

**Benefits:**
1. **Better async behavior** — GPU can derive values (e.g., speculative decoding draft tokens) that CPU may not know yet
2. **Lower CPU overhead** — input prep is cheap on GPU, avoids Python bottlenecks

**Example:**
```python
# V1: CPU prepares input_ids
input_ids = []
for req in batch:
    input_ids.append(req.next_token_id)
input_ids_tensor = torch.tensor(input_ids, device="cuda")  # CPU loop + transfer

# MRV2: GPU prepares input_ids via Triton kernel
prepare_input_ids_kernel(persistent_state, output=input_ids_tensor)  # GPU kernel
```

### Universal Virtual Addressing (UVA)

MRV2 uses UVA to let GPU kernels access large CPU-resident tensors (e.g., `prefill_token_ids`) directly without duplicating into GPU memory.

**Benefits:**
- Reduces GPU memory pressure
- No explicit CPU-GPU transfer for read-only data
- Transparent access via virtual addressing

## 6. Triton-Native Sampler

MRV2 reimplements sampling in Triton for better numeric/memory control.

### Gumbel Sampling Kernel

**Innovation:** Avoid explicit softmax materialization, use stateless in-kernel RNG from seed input.

**Benefits:**
- Lower peak memory (no full-vocabulary softmax)
- Deterministic RNG (seed-based, reproducible)
- Faster sampling (fused computation)

### Efficient Top-K Logprobs

**V1 approach:** Materialize full-vocabulary logprobs before top-k
```python
# V1: compute logprobs for all vocab tokens
logprobs = F.log_softmax(logits, dim=-1)  # (batch, vocab_size)
topk_logprobs = torch.topk(logprobs, k=k)  # then select top-k
```

**MRV2 approach:** Identify top-k tokens from logits first, compute logprobs only for selected tokens
```python
# MRV2: top-k on logits, then compute logprobs only for selected tokens
topk_logits = torch.topk(logits, k=k)  # (batch, k)
topk_logprobs = F.log_softmax(topk_logits, dim=-1)  # (batch, k) instead of (batch, vocab_size)
```

**Benefits:**
- Reduced peak GPU memory (k << vocab_size)
- Faster (less computation)

### Memory-Efficient Prompt Logprobs

MRV2 supports finer-grained chunking, including **chunking inside a single prompt**, to avoid memory spikes on long prompts.

**V1 limitation:** Chunk at prompt boundaries only
**MRV2 improvement:** Chunk within prompts for very long sequences

### Better Speculative Decoding Compatibility

**V1 approach:** Expand per-request sampling states to match per-logit shapes (complex for multi-token prediction)

**MRV2 approach:** Use indirection (`idx_mapping`) inside kernels to map each logits vector to the right request state

**Benefits:**
- Simpler support for complex sampling parameters
- Better integration with [[Speculative Decoding]] (multiple draft tokens per request)
- Easier [[Logits Processors]] integration

## 7. Modularity

### V1 Problem: Entangled gpu_model_runner.py

V1's `gpu_model_runner.py` is large and entangled, mixing multiple feature concerns.

### MRV2 Solution: Dedicated Files and InputBatch Class

MRV2 splits feature logic across dedicated files:
- `mrope_utils.py` — MRoPE (multi-resolution RoPE) utilities
- `penalties.py` — sampling penalties (repetition, frequency, presence)
- ... many others

**InputBatch class:**
Consolidates model inputs into single data structure, reduces direct model-runner attribute coupling.

**Benefits:**
- Easier code navigation
- Better testability (isolated feature modules)
- Clearer abstraction boundaries

## 8. No Abuse of dummy_run

### V1 Problem: dummy_run Handles Too Much

V1's `dummy_run` method handles:
- Initial memory profiling and `torch.compile`
- CUDA graph capture
- Warmups
- Empty DP forward passes for EP+DP

**Consequence:** Complex logic, divergence between `execute_model` and `dummy_run` behavior (source of bugs)

### MRV2 Solution: Separate Paths

1. `execute_model` supports dummy runs without affecting state (flag-based)
2. `dummy_run` delegates to `execute_model` for profiling, warmup, empty DP forward passes
3. CUDA graph capture uses **separate dedicated path** (see next section)

**Benefits:**
- Reduced complexity
- No behavior divergence
- Clearer responsibilities

## 9. Explicit CUDA Graph Management

### V1 Problem: Implicit Graph Handling

V1's CUDA graph handling is implicit and hard to reason about (graph lifecycle hidden in model runner logic).

### MRV2 Solution: CUDAGraphManager

MRV2 uses `CUDAGraphManager` that **explicitly** captures and launches full CUDA graphs through standard PyTorch APIs.

**Features:**
- Clear graph lifecycle (creation, capture, launch)
- Easy to understand execution mode decisions
- Extensible (e.g., capture multiple draft-model forward passes into one CUDA graph)

**Example:**
```python
# MRV2: explicit graph management
graph_manager = CUDAGraphManager()
graph = graph_manager.capture(model_forward_fn, inputs)
output = graph_manager.launch(graph, inputs)
```

**Benefits:**
- Better reasoning about graph execution
- Easier debugging (explicit capture/launch points)
- More flexible (e.g., [[Speculative Decoding]] multi-step graphs)

## Development Philosophy

MRV2 changes must meet a **higher code quality bar**:

- Features reconsidered from first principles (not quick V1 ports)
- Preserve modularity and clean abstraction boundaries
- More upfront design iteration acceptable

**Guideline:** As feature gaps with V1 are filled, features should be reconsidered in the MRV2 design context instead of quickly porting V1 behavior.

## Limitations and Future Work

### Current Limitations

- **Not feature-complete** — some V1 features still missing (document does not specify which)
- **Not rigorously tested** — less battle-tested than V1
- **Open design decisions** — some aspects still evolving

### Migration Path

MRV2 coexists with V1 in vLLM codebase. Users can select model runner version via configuration (mechanism not specified in document).

## Cross-References

- [[vLLM Engine]] — MRV2 as alternative model runner implementation
- [[V1 Architecture]] — comparison with V1 design choices and limitations
- [[CUDA Graphs]] — explicit graph management via `CUDAGraphManager`
- [[Speculative Decoding]] — better compatibility via `idx_mapping` in sampler, multi-step graph capture
- [[Logits Processors]] — Triton-native sampler integration with logits processor pipeline
- [[Attention Backends]] — backend integration in MRV2 architecture
- [[Kernel Fusions]] — MRV2's modular design enables better fusion opportunities
- [[torch.compile Integration]] — MRV2 interaction with torch.compile and Inductor

## See Also

- [[Optimization Levels]] — MRV2 optimization level selection
- [[Disaggregated Serving]] — MRV2 in prefill vs decode instances
- [[Continuous Batching]] — persistent batch management in MRV2
