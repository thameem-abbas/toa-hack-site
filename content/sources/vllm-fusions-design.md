---
title: vLLM Kernel Fusions, CustomOp System, and Compilation Debugging
type: source-summary
created: 2026-04-25
sources:
  - .raw/articles/vllm-fusions-2026-04-25.md
  - .raw/articles/vllm-custom-op-2026-04-25.md
  - .raw/articles/vllm-debug-compile-2026-04-25.md
---

# vLLM Kernel Fusions, CustomOp System, and Compilation Debugging

Source summary for vLLM design docs on kernel fusions, custom operation dispatch, and torch.compile debugging.

## Key Takeaways

### Kernel Fusions (`fusions.md`)

vLLM applies 12+ kernel/operator fusions via custom torch.compile Inductor passes to reduce memory bandwidth and kernel launch overhead:

1. **AllReduce + RMSNorm** (`fuse_allreduce_rms`) — Hopper/Blackwell only, 5-20% speedup at low token counts, TP > 1
2. **Attention + Quant** (`fuse_attn_quant`) — eliminates full-precision attention output write, 3-7% speedup, requires full graph visibility
3. **RoPE + KV-Cache** (`fuse_rope_kvcache`) — ROCm/AITER only, 2-4% speedup
4. **Sequence Parallelism** (`enable_sp`) — prerequisite for AsyncTP, transforms AllReduce → ReduceScatter + AllGather
5. **AsyncTP GEMM + Collective** (`fuse_gemm_comms`) — 7-10% speedup at high token counts, overlaps compute and communication
6. **RMSNorm + Quant** (`fuse_norm_quant`) — 1-4% speedup, conditional on custom kernel usage (Inductor's native fusion is faster on NVIDIA)
7. **SiLU+Mul + Quant** (`fuse_act_quant`) — 1-4% speedup, gate-up projection fusion
8. **QK Norm + RoPE** (`enable_qk_norm_rope_fusion`) — for Qwen-style models, 2-3% speedup
9. **MLA Dual RMSNorm** (`fuse_mla_dual_rms_norm`) — ROCm/AITER only, DeepSeek-V3/Kimi-K2 specific
10. **MiniMax QK Norm** (`fuse_minimax_qk_norm`) — MiniMax M2 specific, TP-aware Q/K normalization
11. **RMSNorm + Padding** (`fuse_act_padding`) — ROCm/AITER only, for GPT-OSS models

**Architecture Support Matrix:**
- SM100 (Blackwell): FP16/BF16, FP8 static, NVFP4 (most fusions)
- SM90 (Hopper): FP16/BF16, FP8 static (primary target for AsyncTP/SP)
- SM89 (Ada): Limited support (no AllReduce fusion)
- SM80 (Ampere): Minimal support
- ROCm: AITER-specific fusions (RoPE+KV, MLA, padding)

**Configuration:**
- Fusions exposed via `PassConfig` fields (nested in `CompilationConfig`)
- Controlled by optimization levels (-O0 to -O3)
- CLI flags: `vllm serve -O2 -cc.pass_config.fuse_allreduce_rms=False`
- User-set flags override optimization-level defaults

**Key constraints:**
- Some fusions require full graph visibility (Inductor partition or `splitting_ops=[]`)
- Token-count sensitive: low-token fusions (AllReduce+RMS, RoPE+KV), high-token fusions (AsyncTP), or always-on (Norm+Quant)
- Hardware-specific: FlashInfer for AllReduce fusion, AITER for ROCm fusions

### CustomOp System (`custom_op.md`)

Abstract class for dispatching operations to platform-specific backends and enabling OOT plugin registration.

**Dispatch flow:**
1. If enabled via `--compilation_config.custom_ops '["+op_name"]'` → dispatch by platform:
   - CPU → `forward_cpu()`
   - CUDA → `forward_cuda()`
   - ROCm → `forward_hip()` (fallback to `forward_cuda()`)
   - XPU → `forward_xpu()`
   - TPU → `forward_tpu()`
   - OOT → `forward_oot()`
2. If disabled (or no specialized impl) → `forward_native()` (PyTorch-native fallback)

**Default enable/disable logic:**
- If `backend == "inductor"` and `mode != NONE` → append `none` (disable custom ops, let Inductor fuse)
- Otherwise → append `all` (enable custom ops)
- Exception: Multi-modal models enforce-enable `MMEncoderAttention` and `ApplyRotaryEmb` for ViT performance

**12 CustomOp categories:**
1. Attention (MLA, MM encoder attention)
2. Activation (silu_and_mul, gelu variants, fatrelu, etc.)
3. MM-Conv (conv2d, conv3d)
4. Embedding (vocab_parallel_embedding, parallel_lm_head)
5. Linear (row/column/replicated parallel linear)
6. Logits Processor
7. Mamba (mamba_mixer, mamba_mixer2, short_conv)
8. MoE (fused_moe, modular_fused_moe, grouped_topk)
9. Norm (rms_norm, rms_norm_gated, gemma_rms_norm)
10. Quantization (quant_fp8)
11. RoPE (rotary_embedding, dual_chunk_rotary_embedding, apply_rotary_emb)
12. Encoder (qwen2_decoder, mm_encoder_attn, rel_pos_attention)

**OOT Plugin Registration:**
- Hardware vendors extend vLLM ops and register via `@CustomOp.register_oot("OpName")`
- vLLM replaces base class with OOT class at instantiation time
- Enables device-specific kernels (Ascend NPU, Intel Gaudi, AWS Neuron, Huawei Kunlun, etc.) without modifying vLLM core

**Fine-grained control:**
- `--compilation_config.custom_ops '["all"]'` — enable all
- `--compilation_config.custom_ops '["none"]'` — disable all
- `--compilation_config.custom_ops '["all,-op1"]'` — enable all except op1
- `--compilation_config.custom_ops '["none,+op1,+op2"]'` — only enable op1 and op2

### Debugging vLLM-compile (`debug_vllm_compile.md`)

Four-stage compilation pipeline: **TorchDynamo graph capture** → **vLLM graph splitting/specialization** → **TorchInductor compilation (with custom passes)** → **vLLM compile cache** → **CUDAGraphs**.

**Debugging tools:**
- **tlparse** (`TORCH_TRACE=~/trace_dir vllm serve`) — visualize compilation stages, fused kernels, constraints
- **Flag table:**
  - `--enforce-eager` → turn off torch.compile + CUDA graphs
  - `-cc.mode=0` → turn off torch.compile only
  - `-cc.cudagraph_mode=NONE` → turn off CUDA graphs only
  - `-cc.backend=eager` → turn off TorchInductor (but keep Dynamo)

**Common issues:**

1. **Graph breaks** (TorchDynamo)
   - vLLM requires full-graph capture; Dynamo errors if feature unsupported
   - Workaround: rewrite code or file PyTorch bug

2. **Dynamic shape constraints**
   - vLLM requires graph dynamic on batch size (num_tokens)
   - Code like `if data.size[0] % 128 == 0: ...` creates batch-size guards
   - Diagnose: tlparse `compilation_metrics` shows symbolic constraints
   - Fix: avoid branching on num_tokens OR wrap in custom op

3. **Constraint violations / dynamic-shape guards**
   - vLLM assumes all guards are safe to drop
   - If violated → `ConstraintViolationError` or silent incorrectness
   - Debug with stricter modes: `-cc.dynamic_shapes_config.type=unbacked` or `backed_size_oblivious`
   - Print guards: `TORCH_LOGS=+dynamic vllm serve ...`

4. **Inductor bugs**
   - Rare: incorrect Triton kernel generation
   - Manifests as silent incorrectness or CUDA illegal memory access
   - Inductor runtime assertions (disabled by default < torch 2.12 for perf):
     - Enable: `VLLM_LOGGING_LEVEL=DEBUG` or `-cc.inductor_compile_config='{"size_asserts": true}'`
   - Disable Inductor: `-cc.backend=eager`
   - Editable code: `VLLM_COMPILE_CACHE_SAVE_FORMAT=unpacked` (allows breakpoints in generated Triton)

5. **vLLM compile cache issues**
   - Layer on top of torch.compile's cache, not always correct
   - Disable: `VLLM_DISABLE_COMPILE_CACHE=1`
   - Clear: `rm -rf ~/.cache/vllm` and `rm -rf /tmp/torchinductor_$(whoami)`
   - Cache key computed from config flags + model name; bugs usually mean missing factor

6. **CUDAGraphs issues**
   - CUDAGraph captures CUDA kernels into replay buffer (same memory regions)
   - Restrictions: need buffer copy for new data, CPU work not captured
   - Unsafe when used incorrectly
   - Turn off: `-cc.cudagraph_mode=NONE`

**Resources:**
- [Blog: Introduction to vLLM-torch.compile](https://blog.vllm.ai/2025/08/20/torch-compile.html)
- [vLLM Office Hours #26](https://www.youtube.com/live/xLyxc7hxCJc?si=Xulo9pe53C6ywf0V&t=561)
- [PyTorch Conference 2025 talk](https://youtu.be/1wV1ESbGrVQ?si=s1GqymUfwiwOrDTg&t=725)

## Related Pages

- [[Kernel Fusions]] — technique page
- [[CustomOp System]] — architecture page
- [[torch.compile Integration]] — compilation pipeline
- [[CUDA Graphs]] — kernel capture and replay
- [[Optimization Levels]] — fusion presets
- [[Quantization]] — FP8/NVFP4 integration with fusions
- [[Tensor Parallelism]] — AllReduce fusion, AsyncTP
- [[Plugin System]] — OOT device plugin mechanism

## Cross-References

### From Kernel Fusions
- **AllReduce + RMSNorm** requires [[Tensor Parallelism]] (TP > 1), FlashInfer backend
- **Attention + Quant** integrates with [[Quantization]] (FP8 static, NVFP4)
- **Sequence Parallelism** prerequisite for AsyncTP, transforms TP collectives
- **AsyncTP** overlaps [[Tensor Parallelism]] GEMMs with communication
- All fusions controlled by [[Optimization Levels]] and [[torch.compile Integration]]

### From CustomOp System
- Dispatch mechanism enables [[Plugin System]] OOT device support
- Integration with [[torch.compile Integration]] (custom ops vs Inductor fusion)
- Custom ops for [[Quantization]] (quant_fp8), [[RoPE]], [[MoE]], [[Attention]]

### From Debug Guide
- Debugging [[torch.compile Integration]] via tlparse, flags, dynamic shape modes
- [[CUDA Graphs]] troubleshooting (buffer management, CPU work exclusion)
- Inductor pass pipeline ties to [[Kernel Fusions]]

## Quotes

> "Speedup depends heavily on the exact model, batch size, and hardware. If tuning performance by hand, always benchmark your exact use-case with and without the fusion to verify the impact."

> "On NVIDIA, Inductor actually generates a faster fused kernel than our custom CUDA kernel. Hence, this fusion is only enabled when either `rms_norm` or `quant_fp8` is using a custom kernel."

> "Sequence Parallelism itself does not directly improve performance; it is a prerequisite for the AsyncTP pass (`fuse_gemm_comms`)."

> "`CustomOp` can enable these hardware manufacturers to seamlessly replace vLLM's operations with their deep-optimized kernels for specific devices at runtime, by just registering an OOT `CustomOp` and implementing the `forward_oot()` method."

> "vLLM assumes that all guards added by torch.compile are safe to drop and will not constrain the compiled graph to specific input shapes. When this assumption is violated, it can cause issues that users need to debug."

> "Most notably, vLLM-compile is NOT torch.compile, it is a custom compiler built using internal PyTorch Compile APIs."
