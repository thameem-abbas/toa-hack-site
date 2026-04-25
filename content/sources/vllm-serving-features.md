---
title: vLLM Serving Features (LoRA, Structured Outputs, Tool Calling)
type: source
created: 2026-04-25
source_urls:
  - file:///tmp/vllm/docs/features/lora.md
  - file:///tmp/vllm/docs/design/lora_resolver_plugins.md
  - file:///tmp/vllm/docs/features/structured_outputs.md
  - file:///tmp/vllm/docs/features/tool_calling.md
tags: [vllm, lora, structured-outputs, tool-calling, serving]
---

# vLLM Serving Features

## Overview

This source bundle covers three advanced vLLM serving features: LoRA adapter serving, structured outputs (constrained decoding), and tool calling. These enable production deployments with multi-tenant LoRA serving, guaranteed JSON/grammar outputs, and LLM function calling.

## LoRA Adapter Serving

**Source:** `docs/features/lora.md`, `docs/design/lora_resolver_plugins.md`

### Key Claims

1. **Per-request LoRA selection:** vLLM enables serving multiple LoRA adapters from a single base model with per-request adapter specification via `LoRARequest(name, id, path)`

2. **Multi-LoRA batching:** Requests with different LoRA adapters can be batched in a single forward pass using Punica/BGMV kernels, maintaining near-base-model throughput

3. **Dynamic loading:** LoRA adapters can be loaded/unloaded at runtime via API endpoints (`/v1/load_lora_adapter`, `/v1/unload_lora_adapter`) or resolver plugins (filesystem, S3, HuggingFace Hub)

4. **Supported layers:** Linear layers (QKV/FFN), embedding layers, MoE expert layers; configurable via `--lora-target-modules`

5. **Configuration constraints:**
   - `--max-lora-rank` must match actual max rank (too high wastes memory, too low fails)
   - `--max-loras` controls concurrent GPU-resident adapters (4-8 on 24GB GPU, 16-32 on 80GB)
   - Memory overhead: ~5-10% compute overhead for rank 64 at batch size 32

6. **Compatibility:** Works with [[Prefix Caching]], [[Quantization]] (FP8/INT8/INT4), [[Speculative Decoding]], [[CUDA Graphs]]; experimental support for multimodal tower/connector

7. **LoRA resolver plugins:** Three built-in resolvers:
   - `lora_filesystem_resolver`: Local directory with `adapter_config.json` + `adapter_model.bin`
   - `lora_hf_hub_resolver`: Download from HuggingFace Hub repositories
   - Custom resolvers: Implement `LoRAResolver` interface (S3, GCS, etc.)

8. **Multimodal default LoRAs:** Models like Granite Speech ship with LoRA adapters that auto-apply when specific modalities are present (via `default_mm_loras={"audio": model_id}`)

### Implementation Details

**LoRARequest structure:**
```python
LoRARequest(
    lora_name="sql_adapter",  # Human-readable name
    lora_int_id=1,            # Globally unique ID
    lora_path="/path/to/adapter"  # Local or HF path
)
```

**Server startup:**
```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-lora \
    --lora-modules sql-lora=jeeejeee/llama32-3b-text2sql-spider \
    --max-loras 4 \
    --max-lora-rank 64
```

**Directory structure for filesystem resolver:**
```
/path/to/lora/adapters/
├── adapter1/
│   ├── adapter_config.json  # {"peft_type": "LORA", "r": 64, ...}
│   └── adapter_model.bin
└── adapter2/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

### Cross-References

- [[LoRA]] — Full technique page
- [[vLLM Engine]] — Per-request LoRA dispatch
- [[Prefix Caching]] — LoRA-specific KV cache entries
- [[Quantization]] — LoRA on quantized base models
- [[Mixture of Experts]] — LoRA for MoE expert layers

---

## Structured Outputs

**Source:** `docs/features/structured_outputs.md`

### Key Claims

1. **Constrained decoding via logits masking:** vLLM guarantees outputs match a schema (JSON, regex, grammar, choice) by masking invalid tokens at each decoding step using a compiled FSM

2. **Backends:** Four backends (selected via `--structured-outputs-config.backend`):
   - `xgrammar` (default): MLC-AI's high-performance grammar engine
   - `guidance`: Microsoft's llguidance library
   - `outlines`: Legacy Outlines library (deprecated)
   - `lm-format-enforcer`: Python regex-based backend

3. **Constraint types:**
   - **Choice:** Output is exactly one of fixed strings (e.g., `["positive", "negative"]`)
   - **Regex:** Match a pattern (email, phone number, ID); syntax depends on backend (Rust vs Python)
   - **JSON:** Enforce JSON schema via Pydantic models or raw schema
   - **Grammar:** Context-free EBNF grammars (SQL, code, custom DSLs)
   - **Structural tag:** JSON within XML-like tags in mixed text/structured output

4. **Performance:**
   - First request: 1-10 seconds compilation overhead (complex schemas)
   - Subsequent requests: Near-zero overhead (compiled FSM cached)
   - Per-token overhead: 5-15% slowdown vs unconstrained generation
   - Memory: 1-50 MB per unique schema

5. **OpenAI API compatibility:**
   - Deprecated fields (v0.12.0): `guided_json`, `guided_regex`, `guided_choice` → now `{"structured_outputs": {...}}`
   - Beta parsing API: `client.beta.chat.completions.parse()` returns Pydantic objects directly

6. **Reasoning integration:** Works with reasoning models (DeepSeek-R1, Qwen3-Coder) for constrained reasoning outputs; Qwen3-Coder requires `--structured-outputs-config.enable_in_reasoning=True`

7. **Limitations:**
   - Guarantees **syntax** correctness, not **semantic** quality (model can generate nonsensical but valid JSON)
   - Complex schemas have high first-request latency
   - Regex dialect differs by backend (Rust vs Python)

### Implementation Details

**Choice constraint:**
```python
extra_body={"structured_outputs": {"choice": ["positive", "negative"]}}
```

**JSON schema with Pydantic:**
```python
class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType  # Enum

response_format={
    "type": "json_schema",
    "json_schema": {
        "name": "car-description",
        "schema": CarDescription.model_json_schema()
    }
}
```

**EBNF grammar (simplified SQL):**
```python
simplified_sql_grammar = """
    root ::= select_statement
    select_statement ::= "SELECT " column " from " table
    column ::= "col_1 " | "col_2 "
    table ::= "table_1 " | "table_2 "
"""
extra_body={"structured_outputs": {"grammar": simplified_sql_grammar}}
```

**Offline inference:**
```python
from vllm.sampling_params import StructuredOutputsParams

structured_outputs_params = StructuredOutputsParams(choice=["Positive", "Negative"])
sampling_params = SamplingParams(structured_outputs=structured_outputs_params)
```

### Cross-References

- [[Structured Outputs]] — Full technique page
- [[vLLM Engine]] — Logits processors for masking
- [[V1 Architecture]] — Grammar compilation in model executor
- [[Tool Calling]] — Uses structured outputs for function arguments

---

## Tool Calling

**Source:** `docs/features/tool_calling.md`

### Key Claims

1. **Tool choice modes:** vLLM supports four modes:
   - **Named function:** Specify exact function via `tool_choice={"type": "function", "function": {"name": "get_weather"}}`
   - **`required`:** Model must generate ≥1 tool calls from provided list (guaranteed via structured outputs)
   - **`auto`:** Model generates tool calls freely; parser extracts from raw text (arguments may be malformed)
   - **`none`:** No tool calls, text-only response

2. **Constrained vs unconstrained decoding:**
   - Named function + `required`: Schema-constrained (via structured outputs backend), guaranteed valid JSON
   - `auto`: No schema constraint, parser extracts tool calls from raw text, arguments may violate schema
   - **Strict mode NOT implemented:** OpenAI's `strict=true` field is accepted but ignored (tracking: #15526, #16313)

3. **Auto tool choice requires:**
   - `--enable-auto-tool-choice` (mandatory)
   - `--tool-call-parser {name}` (model-specific parser)
   - `--chat-template {path}` (optional, for tool-role messages)

4. **Supported parsers:** 20+ model-specific parsers:
   - **Hermes:** NousResearch Hermes-2-Pro/Theta/3 series
   - **Mistral:** Mistral-7B-Instruct-v0.3+; struggles with parallel calls
   - **Llama:** Llama 3.1/3.2/4 (JSON-based via `llama3_json`, pythonic via `pythonic`/`llama4_pythonic`)
   - **IBM Granite:** Granite 3.0/3.1/4.0, Granite-20b-functioncalling
   - **InternLM, Jamba, xLAM, Qwen, MiniMax, DeepSeek-V3/V3.1, OpenAI OSS, Kimi-K2, Hunyuan, LongCat, GLM-4.5/4.7, FunctionGemma, Qwen3-Coder, Olmo3, Gigachat3, Pythonic models**

5. **Pythonic tool calling:** Models output Python lists instead of JSON: `[get_weather(city='SF'), get_weather(city='NYC')]`
   - Advantages: Inherent parallel call support, no JSON schema ambiguity
   - Limitations: Can't mix text + tool calls in same generation; Llama 3.2 small models struggle

6. **Custom parsers:** Users can implement `ToolParser` interface and register via `--tool-parser-plugin`:
   ```python
   class ExampleToolParser(ToolParser):
       def extract_tool_calls(self, model_output, request):
           # Parse raw text into tool calls
           return ExtractedToolCallInformation(...)
   
   ToolParserManager.register_lazy_module("example", "path", "ExampleToolParser")
   ```

7. **Model-specific issues:**
   - Mistral 7B: Struggles with parallel calls, requires 9-digit tool call IDs (vLLM uses longer IDs → custom template needed)
   - Llama 3: No parallel calls (Llama 4 supports them)
   - Llama 3.2 small models: Frequently fail to format tool calls correctly

### Implementation Details

**Quickstart (Llama 3.1):**
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template examples/tool_chat_template_llama3.1_json.jinja
```

**Request with tool_choice="auto":**
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "unit"]
        }
    }
}]

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Weather in SF?"}],
    tools=tools,
    tool_choice="auto",
)

tool_call = response.choices[0].message.tool_calls[0].function
# tool_call.name = "get_weather"
# tool_call.arguments = '{"location": "San Francisco, CA", "unit": "fahrenheit"}'
```

**Named function (constrained):**
```python
tool_choice={"type": "function", "function": {"name": "get_weather"}}
# Guaranteed valid JSON arguments via structured outputs
```

### Cross-References

- [[Structured Outputs]] — Backend for constrained tool arguments
- [[vLLM Engine]] — Tool call extraction in completion pipeline

---

## Combined Insights

### Multi-Tenant Serving Pattern

vLLM's LoRA + structured outputs + tool calling enables sophisticated multi-tenant deployments:

1. **LoRA adapters** per tenant/task (SQL, code, translation)
2. **Structured outputs** guarantee valid API inputs/outputs
3. **Tool calling** enables agentic workflows with tenant-specific tools

**Example:** Multi-tenant SQL assistant:
- Base model: Llama 3.1 70B
- LoRA adapters: `tenant-a-sql-lora`, `tenant-b-sql-lora` (fine-tuned on different schemas)
- Structured output: SQL grammar (guarantees valid syntax)
- Tool calling: `execute_query(sql: str)` function

### Performance Compounding

These features compose efficiently:

- **LoRA + structured outputs:** 5-10% LoRA overhead + 5-15% grammar overhead ≈ 10-25% total (additive)
- **LoRA + tool calling:** Tool parsing happens post-generation (no interaction)
- **Structured outputs + tool calling:** Tool arguments use structured outputs backend (eliminates malformed JSON)

### Security Considerations

1. **LoRA dynamic loading:** Insecure (arbitrary code execution risk) → only use in isolated environments
2. **Structured outputs:** Safe (only constrains generation)
3. **Tool calling:** Caller responsibility to validate function arguments and sanitize inputs

## Related Pages

- [[LoRA]] — Low-rank adapter serving technique
- [[Structured Outputs]] — Constrained decoding technique
- [[vLLM Engine]] — Request routing and batch composition
- [[Prefix Caching]] — Interaction with LoRA caching
- [[Quantization]] — LoRA on quantized models
