---
title: Logits Processors
type: architecture
created: 2026-04-25
tags: [vllm, logits-processors, sampling, batch-processing]
---

# Logits Processors

vLLM's logits processor architecture enables batch-granularity transformation of model output logits to steer next-token probability distributions, with stateful request tracking and efficient batch update synchronization.

## Overview

Logits processors adjust the next-token probability distribution by transforming raw model output logits before softmax. In vLLM, processors operate at **batch granularity** on `(num_requests) x (vocab_size)` tensors and maintain per-request metadata synchronized with the persistent batch state.

**Note:** The logits processor API is still evolving and may change in the near future.

## Core Architecture

### Batch-Granularity Processing

Each engine step:
1. Model outputs `(num_requests) x (vocab_size)` logits tensor
2. For each loaded logits processor:
   - Transform rows corresponding to requests that enable the processor
   - Leave other rows unmodified
3. Pass transformed logits to softmax → sampling

**Example:**
```
Input logits:    [req0_logits, req1_logits, req2_logits]  # (3, vocab_size)
Processor apply: [transform(req0), req1_logits, transform(req2)]  # req1 disabled
Output logits:   Transformed tensor passed to softmax
```

### Stateful Processors

Logits processors maintain metadata about batch requests:
- Per-request configuration (e.g., bias values, min-p threshold)
- Per-request state (e.g., generated tokens for repetition penalty)
- Mapping between batch indices and request metadata

State must be synchronized with persistent batch changes (add/remove/reorder requests).

## Two-Phase Execution

### Phase 1: Update State

**When:** Beginning of each engine step, after persistent batch reorganization

**Purpose:** Synchronize logits processor internal state with batch changes

**Process:**
```python
# Pseudocode: gpu_input_batch.py
def refresh_metadata(self):
    batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
    for logit_proc in self.logitsprocs.all:
        logit_proc.update_state(batch_update)
```

**BatchUpdate data structure:**
- `batch_size`: current number of requests
- `removed`: list of removed request indices
- `added`: list of (index, SamplingParams, prompt_token_ids, output_token_ids) tuples
- `moved`: list of (src_idx, dst_idx, directionality) tuples

### Phase 2: Apply Transformation

**When:** After model inference, during sampling

**Purpose:** Transform logits based on processor logic

**Process:**
```python
# Pseudocode: sampler.py
def forward(self, logits, sampling_metadata):
    # Apply non-argmax-invariant processors (always)
    for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
        logits = processor.apply(logits)
    
    # Skip argmax-invariant processors if all requests use greedy sampling
    if not all_greedy_sampling:
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)
    
    return sample(logits, ...)
```

## Argmax-Invariant vs Non-Argmax-Invariant

### Argmax-Invariant Processors

**Definition:** Processors that **never modify the argmax** (token ID with highest logit value)

**Examples:**
- **Min-P:** Masks low-probability tokens (doesn't change which token has max logit)
- **Top-K:** Masks tokens outside top-k (argmax always in top-k)

**Optimization:** Can be **skipped for greedy sampling** (greedy always picks argmax)

**Batch-level skip condition:** All requests in batch use greedy sampling (no batch-level skip if even one request uses non-greedy)

### Non-Argmax-Invariant Processors

**Definition:** Processors that **may modify the argmax**

**Examples:**
- **Forced EOS:** Masks all tokens except EOS after N steps (may mask previous argmax)
- **Logit bias:** Adds bias to specific tokens (may change argmax)

**Behavior:** Always applied, cannot be skipped for greedy sampling

## LogitsProcessor Base Class

### Required Methods

```python
class LogitsProcessor(ABC):
    @abstractmethod
    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                is_pin_memory: bool) -> None:
        """Initialize processor with engine config and device info."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Transform (num_requests) x (vocab_size) logits tensor.
        Can modify in-place (more memory-efficient) or out-of-place.
        """
        raise NotImplementedError

    @abstractmethod
    def is_argmax_invariant(self) -> bool:
        """Return True if processor never changes argmax token ID.
        Evaluated once at startup.
        """
        raise NotImplementedError

    @abstractmethod
    def update_state(self, batch_update: BatchUpdate | None) -> None:
        """Update internal state based on batch changes.
        batch_update is None if no batch changes (only new output tokens).
        """
        raise NotImplementedError

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        """Validate SamplingParams at request submission time.
        Raise ValueError for invalid arguments.
        """
        return None
```

### Constructor Parameters

- **`vllm_config`:** Engine configuration (model, parallelism, memory)
- **`device`:** Hardware accelerator device (e.g., `cuda:0`)
- **`is_pin_memory`:** Whether pinned memory is available (for CPU-GPU transfer optimization)

## BatchUpdate Data Structure

### Fields

```python
@dataclass(frozen=True)
class BatchUpdate:
    batch_size: int  # Current number of requests in batch
    removed: Sequence[RemovedRequest]  # List of removed indices
    added: Sequence[AddedRequest]  # List of (idx, params, prompt_toks, output_toks)
    moved: Sequence[MovedRequest]  # List of (src, dst, directionality)
```

**AddedRequest:** `(index, SamplingParams, prompt_token_ids, output_token_ids)`
- `output_token_ids` is a **reference** to request's running output list (grows each step)
- Processors can retain reference to track generated tokens

**MovedRequest:** `(src_index, dst_index, MoveDirectionality)`
- `UNIDIRECTIONAL`: one-way move (src → dst, src becomes empty slot)
- `SWAP`: two-way swap (src ↔ dst)

**RemovedRequest:** `int` (index of removed request)

### Processing Order

**Processors must process batch updates in this order:**

1. **Removes** — discard state for removed requests
2. **Adds** — initialize state for new requests
3. **Moves** — reorder internal state to match batch reordering

**Index semantics:**
- Add operations use indices **at the time of Add** (before Moves)
- Move operations applied in order (subsequent Moves see results of previous Moves)

### Batch Update Operations

#### Remove

**Operation:** Remove request at index `i` without replacement, leave empty slot

**Example:**
```
Batch: [A, B, C]
Remove @ i=1

Result: [A, x, C]  # B discarded, empty slot at index 1
```

**Processor action:** Discard state for request at index 1

#### Add

**Operation:** Add new request at index `i` (or replace existing request)

**Replace existing:**
```
Batch: [A, B, C]
Add D @ i=1

Result: [A, D, C]  # D replaces B, discard B's state
```

**Extend batch:**
```
Batch: [A, B, C]
Add D @ i=3

Result: [A, B, C, D]  # D extends batch
```

**Processor action:** Initialize state for new request at index

#### Move

**UNIDIRECTIONAL (one-way):**
```
Batch: [A, x, C, D]
Move UNIDIRECTIONAL: 3 → 1

Result: [A, D, C, x]  # D moves to 1, leaves empty slot at 3
```

**UNIDIRECTIONAL with replace:**
```
Batch: [A, B, C, D]
Move UNIDIRECTIONAL: 3 → 1

Result: [A, D, C, x]  # D moves to 1, discards B, leaves empty slot at 3
```

**SWAP (two-way):**
```
Batch: [A, B, C, D]
Move SWAP: 3 ↔ 1

Result: [A, D, C, B]  # B and D swap positions
```

**Processor action:** Reorder internal state arrays/dicts to match new indices

### Batch Update Examples

#### Example 1: Fewer New Requests Than Finished

**Scenario:** 1 new request, 2 finished requests

```
Initial batch: [A, B, C, D]  (batch_size=4)
New requests: E
Finished requests: A, C

Step 1: Add E @ 0 (replace A)
[E, B, C, D]  (batch_size=4)

Step 2: Remove @ 2 (remove C without replacement)
[E, B, x, D]  (batch_size=4, empty slot at 2)

Step 3: Condense batch (Move UNIDIRECTIONAL: 3 → 2) + shrink
[E, B, D]  (batch_size=3, empty slot moved outside batch)

Step 4: Attention backend optimization (Move SWAP: 0 ↔ 1)
[B, E, D]  (batch_size=3)

BatchUpdate:
* removed: [2]  # C removed without replacement
* added: [(0, E's params, E's prompt toks, E's output toks)]
* moved: [(3, 2, UNIDIRECTIONAL), (0, 1, SWAP)]
```

#### Example 2: More New Requests Than Finished

**Scenario:** 2 new requests, 1 finished request

```
Initial batch: [A, B, C, D]  (batch_size=4)
New requests: E, F
Finished requests: C

Step 1: Add E @ 2 (replace C)
[A, B, E, D]  (batch_size=4)

Step 2: Add F @ 4 (extend batch)
[A, B, E, D, F]  (batch_size=5)

Step 3: Attention backend optimization (Move SWAP: 0 ↔ 1)
[B, A, E, D, F]  (batch_size=5)

BatchUpdate:
* removed: []  # No requests removed without replacement
* added: [(2, E's params, ...), (4, F's params, ...)]
* moved: [(0, 1, SWAP)]
```

## Built-In Logits Processors

### Current Processors (New API)

Using the new programming model:
- **Min-P** — mask tokens below min probability threshold (argmax-invariant)
- **Logit bias** — add per-token bias values (non-argmax-invariant)
- **Min-tokens** — enforce minimum generation length before EOS (non-argmax-invariant)

### Legacy Processors (To Be Refactored)

Hard-coded in sampler, awaiting migration to new API:
- Allowed token IDs
- Bad words
- Repetition penalty
- Frequency penalty
- Presence penalty
- Temperature
- Top-K
- Top-P

## Best Practices for Writing Processors

### 1. Efficient Batch-Granularity Operations

**Goal:** Minimize per-request overhead via vectorized operations

**Dense representation (all requests):**
```python
# Good: vectorized operation on entire batch
def apply(self, logits: torch.Tensor) -> torch.Tensor:
    # self.min_p_values: (batch_size,) tensor
    # logits: (batch_size, vocab_size) tensor
    probs = F.softmax(logits, dim=-1)
    mask = probs < self.min_p_values.unsqueeze(1)
    logits[mask] = -float('inf')
    return logits
```

**Sparse representation (enabled requests only):**
```python
# Good for infrequently-used processors
def apply(self, logits: torch.Tensor) -> torch.Tensor:
    # self.enabled_indices: list of indices where processor is enabled
    if not self.enabled_indices:
        return logits  # short-circuit if no requests enable processor
    
    for idx in self.enabled_indices:
        logits[idx] = transform(logits[idx], self.config[idx])
    return logits
```

### 2. Define Per-Request Configurability

**Determine:**
1. What fields to add to `SamplingParams` (if built-in processor)
2. How to disable processor per-request (e.g., `min_p=None`, `logit_bias={}`)
3. Implement `validate_params()` to reject invalid configurations

**Example:**
```python
class MinPProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.min_p_values = torch.zeros(vllm_config.max_num_reqs, device=device)
    
    def update_state(self, batch_update):
        if batch_update is None:
            return  # early exit if no batch changes
        
        for idx, params, _, _ in batch_update.added:
            # params.min_p is None if disabled, float otherwise
            self.min_p_values[idx] = params.min_p if params.min_p else 0.0
    
    @classmethod
    def validate_params(cls, sampling_params):
        if sampling_params.min_p is not None:
            if not (0.0 <= sampling_params.min_p <= 1.0):
                raise ValueError(f"min_p must be in [0, 1], got {sampling_params.min_p}")
```

### 3. Short-Circuit at Batch Level

**Goal:** Skip entire `apply()` if no requests enable processor

**Example:**
```python
def apply(self, logits: torch.Tensor) -> torch.Tensor:
    # Check if all requests have processor disabled
    if torch.all(self.min_p_values == 0.0):
        return logits  # short-circuit (no transformation)
    
    # Otherwise, apply transformation
    ...
```

### 4. Discard Finished Request State

**Ensure `update_state()` discards state for:**
- Requests replaced by Add (Add at existing index)
- Requests subject to Remove

**Example:**
```python
def update_state(self, batch_update):
    if batch_update is None:
        return
    
    # Process removes (discard state)
    for idx in batch_update.removed:
        self.state[idx] = None  # or delete from dict
    
    # Process adds (initialize or replace)
    for idx, params, prompt_toks, output_toks in batch_update.added:
        self.state[idx] = initialize_state(params)
    
    # Process moves (reorder state)
    for src, dst, directionality in batch_update.moved:
        if directionality == MoveDirectionality.SWAP:
            self.state[src], self.state[dst] = self.state[dst], self.state[src]
        else:  # UNIDIRECTIONAL
            self.state[dst] = self.state[src]
            self.state[src] = None
```

### 5. Argmax Invariance Declaration

**Hard-coded if consistent:**
```python
def is_argmax_invariant(self) -> bool:
    return True  # Min-P never changes argmax
```

**Programmatic if depends on config:**
```python
def is_argmax_invariant(self) -> bool:
    # Logit bias is argmax-invariant only if all biases are 0
    return torch.all(self.bias_values == 0.0)
```

## Custom Logits Processors

vLLM supports user-provided custom logits processors. See `docs/features/custom_logitsprocs.md` for integration guide.

## Cross-References

- [[vLLM Engine]] — persistent batch state synchronization with logits processors
- [[V1 Architecture]] — logits processor lifecycle in V1 engine
- [[Model Runner V2]] — Triton-native sampler integration with logits processors
- [[Structured Outputs]] — logits processors for guided generation (e.g., JSON schema enforcement)
- [[Speculative Decoding]] — logits processors in multi-token verification path
- [[Continuous Batching]] — batch update operations when adding/removing requests

## See Also

- [[torch.compile Integration]] — logits processor kernel compilation
- [[Optimization Levels]] — logits processor optimization in different modes
