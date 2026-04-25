---
title: Plugin System
type: architecture
created: 2026-04-25
tags: [vllm, plugins, extensibility, oot-hardware]
---

# Plugin System

vLLM's plugin system enables extensibility without modifying the core codebase, supporting custom models, out-of-tree (OOT) hardware platforms, multimodal processing, and logging via standard Python entry points.

## Overview

Plugins are user-registered code executed by vLLM across all processes (API server, scheduler, workers) in distributed inference. Four plugin types supported:

1. **General plugins** — custom model architectures
2. **Platform plugins** — OOT hardware support (NPUs, accelerators)
3. **IO processor plugins** — multimodal pre/post-processing
4. **Stat logger plugins** — custom telemetry and logging

Plugins use standard Python `entry_points` mechanism for discovery and loading.

## Plugin Discovery and Loading

### Entry Points Mechanism

vLLM discovers plugins via Python package metadata (`setup.py` or `pyproject.toml`).

**Example plugin registration:**
```python
# setup.py
from setuptools import setup

setup(
    name='vllm_add_dummy_model',
    version='0.1',
    packages=['vllm_add_dummy_model'],
    entry_points={
        'vllm.general_plugins': [
            "register_dummy_model = vllm_add_dummy_model:register"
        ]
    }
)

# vllm_add_dummy_model/__init__.py
def register():
    from vllm import ModelRegistry
    
    if "MyLlava" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "MyLlava",
            "vllm_add_dummy_model.my_llava:MyLlava",
        )
```

### Plugin Structure

Every plugin has three components:

1. **Plugin group** — entry point group name (e.g., `vllm.general_plugins`)
2. **Plugin name** — identifier in `entry_points` dict (e.g., `register_dummy_model`)
3. **Plugin value** — fully qualified name of register function (e.g., `vllm_add_dummy_model:register`)

### Loading Mechanism

**Function:** `vllm.plugins.load_plugins_by_group(group_name)`

**When called:** During vLLM initialization in every process (API server, scheduler, GPU workers)

**Why multi-process:** Distributed inference spawns multiple processes (TP, PP, DP workers); each must load plugins

### Plugin Filtering

**Environment variable:** `VLLM_PLUGINS`

**Usage:** Load only specific plugins by name
```bash
VLLM_PLUGINS=register_dummy_model,my_custom_logger vllm serve <model>
```

## Plugin Types

### 1. General Plugins

**Group name:** `vllm.general_plugins`

**Primary use case:** Register custom, out-of-tree model architectures

**Registration pattern:**
```python
def register():
    from vllm import ModelRegistry
    
    ModelRegistry.register_model(
        "CustomModelArch",  # architecture name
        "my_plugin.custom_model:CustomModel",  # fully qualified class name
    )
```

**Example:** [bart-plugin](https://github.com/vllm-project/bart-plugin) adds support for `BartForConditionalGeneration`

**When to use:** Adding new model architectures not in vLLM core (e.g., proprietary models, experimental architectures)

### 2. Platform Plugins

**Group name:** `vllm.platform_plugins`

**Primary use case:** Register OOT hardware platforms (NPUs, custom accelerators)

**Registration pattern:**
```python
def register():
    # Return None if platform not supported in current environment
    # Return fully qualified platform class name if supported
    
    if check_hardware_available():
        return "my_plugin.my_platform:MyPlatform"
    else:
        return None
```

**Official OOT platform plugins:**
- **vllm-ascend** — Huawei Ascend NPU support
- **vllm-gaudi** — Intel Gaudi accelerator support
- **vllm-neuron** — AWS Neuron (Inferentia/Trainium) support
- **vllm-kunlun** — Baidu Kunlun accelerator support

**See "Platform Plugin Implementation Guide" below for details.**

### 3. IO Processor Plugins

**Group name:** `vllm.io_processor_plugins`

**Primary use case:** Custom pre/post-processing for pooling models (e.g., embedding models, rerankers)

**Registration pattern:**
```python
def register():
    # Return fully qualified IOProcessor class name
    return "my_plugin.my_io_processor:MyIOProcessor"
```

**When to use:** Custom input processing (tokenization, feature extraction) or output processing (pooling, normalization) for multimodal or embedding models

### 4. Stat Logger Plugins

**Group name:** `vllm.stat_logger_plugins`

**Primary use case:** Custom telemetry, metrics, and logging

**Registration pattern:**
```python
# Entry point should be a class that subclasses StatLoggerBase
# (not a function)

# setup.py
entry_points={
    'vllm.stat_logger_plugins': [
        "my_logger = my_plugin.my_logger:MyStatLogger"
    ]
}

# my_plugin/my_logger.py
from vllm.stat_logger import StatLoggerBase

class MyStatLogger(StatLoggerBase):
    def log(self, stats):
        # Send stats to external monitoring system
        ...
```

**When to use:** Integrating vLLM with custom monitoring/observability platforms (Prometheus, Datadog, custom backends)

## Platform Plugin Implementation Guide

Platform plugins enable OOT hardware support without modifying vLLM core. Five components required:

### 1. Platform Class

**Inheritance:** `vllm.platforms.interface.Platform`

**Required properties/methods:**

**`_enum` (property):**
```python
from vllm.platforms.interface import PlatformEnum

@property
def _enum(self) -> PlatformEnum:
    return PlatformEnum.OOT  # out-of-tree platform
```

**`device_type` (property):**
```python
@property
def device_type(self) -> str:
    return "my_device"  # PyTorch device type (e.g., "cuda", "cpu", "xpu")
```

**`device_name` (property):**
```python
@property
def device_name(self) -> str:
    return "my_device"  # Usually same as device_type, used for logging
```

**`check_and_update_config(vllm_config)` (method):**

Most critical method — called very early in vLLM initialization.

**Responsibilities:**
- Update vLLM configuration (block size, graph mode, etc.)
- **Set `worker_cls`** to tell vLLM which worker class to use

**Example:**
```python
def check_and_update_config(self, vllm_config):
    # Update block size for hardware constraints
    vllm_config.cache_config.block_size = 64
    
    # Disable CUDA graphs if not supported
    vllm_config.compilation_config.graph_mode = None
    
    # **Critical: Set worker class**
    from my_plugin.my_worker import MyDummyWorker
    vllm_config.worker_cls = MyDummyWorker
```

**`get_attn_backend_cls()` (method):**
```python
def get_attn_backend_cls(self) -> str:
    return "my_plugin.my_attention:MyDummyAttention"  # fully qualified name
```

**`get_device_communicator_cls()` (method):**
```python
def get_device_communicator_cls(self) -> str:
    return "my_plugin.my_communicator:MyDummyDeviceCommunicator"
```

### 2. Worker Class

**Inheritance:** `vllm.v1.worker.worker_base.WorkerBase`

**Required methods (basic inference):**

**`init_device()`:**
```python
def init_device(self):
    # Initialize hardware device, allocate resources
    self.device = torch.device("my_device:0")
```

**`initialize_cache(cache_config)`:**
```python
def initialize_cache(self, cache_config):
    # Set up KV cache configuration
    self.cache_config = cache_config
```

**`load_model()`:**
```python
def load_model(self):
    # Load model weights to device
    self.model = load_model_to_device(self.model_config, self.device)
```

**`get_kv_cache_spec()`:**
```python
def get_kv_cache_spec(self):
    # Generate KV cache spec for the model
    return create_kv_cache_spec(self.model_config, self.cache_config)
```

**`determine_available_memory()`:**
```python
def determine_available_memory(self):
    # Profile peak memory usage to determine KV cache capacity
    # Run dummy forward pass, measure memory, return available GPU memory
    return available_memory_bytes
```

**`initialize_from_config(kv_cache_config)`:**
```python
def initialize_from_config(self, kv_cache_config):
    # Allocate device KV cache with specified config
    self.kv_cache = allocate_kv_cache(kv_cache_config, self.device)
```

**`execute_model(model_input)`:**
```python
def execute_model(self, model_input):
    # Run model inference step (called every step)
    with torch.no_grad():
        output = self.model(model_input)
    return output
```

**Optional methods (advanced features):**

- **Sleep mode:** `sleep()`, `wakeup()`
- **Graph mode:** `compile_or_warm_up_model()`
- **Speculative decoding:** `take_draft_token_ids()`
- **LoRA:** `add_lora()`, `remove_lora()`, `list_loras()`, `pin_lora()`
- **Data parallelism:** `execute_dummy_batch()`

### 3. Attention Backend Class

**Inheritance:** `vllm.v1.attention.backend.AttentionBackend`

**Purpose:** Implement attention computation for custom hardware

**Examples:** See `vllm.v1.attention.backends` for reference implementations (FlashAttention, FlashInfer, Triton, etc.)

**Required methods:** (not fully specified in document, refer to `AttentionBackend` base class)

### 4. Device Communicator Class

**Inheritance:** `vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase`

**Purpose:** Implement collective communication ops (all-reduce, all-gather, etc.) for distributed inference

**Required ops:** all-reduce, all-gather, send, recv, broadcast, barrier

**Example:**
```python
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

class MyDummyDeviceCommunicator(DeviceCommunicatorBase):
    def all_reduce(self, tensor):
        # Implement all-reduce for custom hardware
        ...
    
    def all_gather(self, tensor, dim):
        # Implement all-gather
        ...
```

### 5. Custom Ops

**Three categories:**

#### A. Communicator Ops

Device communicator ops (all-reduce, all-gather, etc.) — implemented in DeviceCommunicator class above

#### B. Common Ops

Register via [[CustomOp System]] for platform-specific ops:
```python
from vllm.model_executor.custom_op import CustomOp

@CustomOp.register_oot("my_platform")
def my_custom_matmul(a, b):
    # Platform-specific optimized matmul
    ...
```

**Op categories:** Attention, Activation, MM-Conv, Embedding, Linear, Logits, Mamba, MoE, Norm, Quantization, RoPE, Encoder

#### C. CSRC Ops (C++ Extensions)

C++ ops registered as torch custom ops:
```cpp
// my_plugin/csrc/my_ops.cpp
TORCH_LIBRARY(my_plugin, m) {
    m.def("my_custom_op", &my_custom_op_impl);
}
```

**Note:** Triton ops do not support custom way currently.

### 6. Optional: Other Pluggable Modules

- **LoRA support** — custom LoRA adapter loading/switching
- **Graph backend** — custom CUDA graph or equivalent for hardware
- **Quantization** — hardware-specific quantization kernels
- **Mamba attention backend** — for Mamba/SSM models

## Plugin Guidelines

### Re-entrancy Requirement

Plugin functions must be **re-entrant** (can be called multiple times without issues).

**Reason:** Plugins may be loaded multiple times across different processes in distributed inference.

**Example:**
```python
def register():
    from vllm import ModelRegistry
    
    # Check if already registered (re-entrant pattern)
    if "MyModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("MyModel", "my_plugin.my_model:MyModel")
```

### Compatibility Guarantee

**vLLM guarantees:**
- Documented plugin interfaces (e.g., `ModelRegistry.register_model`) always available
- Stable API for plugin registration mechanisms

**Plugin developer responsibility:**
- Ensure plugin code compatible with target vLLM version
- Watch for deprecation warnings and upgrade accordingly

### Deprecation Announcements

**Deprecated interfaces:**
- `use_v1` parameter in `Platform.get_attn_backend_cls` — removed in v0.13.0
- `_Backend` in `vllm.attention` — removed in v0.13.0 (use `vllm.v1.attention.backends.registry.register_backend` instead)
- `seed_everything` platform interface — removed in v0.16.0 (use `vllm.utils.torch_utils.set_random_seed`)
- `prompt` in `Platform.validate_request` — removed in v0.18.0

## Platform Plugin Project Structure

**Example:** `vllm_add_dummy_platform`

```
vllm_add_dummy_platform/
├── vllm_add_dummy_platform/
│   ├── __init__.py                    # register() function
│   ├── my_dummy_platform.py           # Platform class
│   ├── my_dummy_worker.py             # Worker class
│   ├── my_dummy_attention.py          # AttentionBackend class
│   ├── my_dummy_device_communicator.py # DeviceCommunicator class
│   ├── my_dummy_custom_ops.py         # CustomOps
├── setup.py                            # entry_points registration
```

**setup.py:**
```python
setup(
    name="vllm_add_dummy_platform",
    version="0.1.0",
    packages=["vllm_add_dummy_platform"],
    entry_points={
        "vllm.platform_plugins": [
            "my_dummy_platform = vllm_add_dummy_platform:register"
        ]
    },
)
```

**`__init__.py`:**
```python
def register():
    # Return None if platform not supported
    # Return fully qualified Platform class name if supported
    try:
        import my_hardware_lib  # Check hardware availability
        return "vllm_add_dummy_platform.my_dummy_platform:MyDummyPlatform"
    except ImportError:
        return None
```

## Cross-References

- [[CustomOp System]] — custom ops for OOT hardware (communicator, common, csrc ops)
- [[vLLM Engine]] — plugin loading during engine initialization
- [[Attention Backends]] — OOT attention backend registration via platform plugins
- [[Multi-Modal Models]] — IO processor plugins for multimodal pre/post-processing
- [[V1 Architecture]] — plugin integration in V1 multi-process architecture
- [[Model Runner V2]] — platform-specific optimizations in MRV2

## See Also

- [[torch.compile Integration]] — plugin compatibility with torch.compile
- [[CUDA Graphs]] — platform plugin graph mode support
- [[Tensor Parallelism]] — device communicator for distributed inference
- [[Expert Parallelism]] — MoE all-to-all communication via device communicator
