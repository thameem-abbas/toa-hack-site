---
title: Structured Outputs
type: technique
tags: [constrained-decoding, json, grammar, regex]
related: "vLLM Engine, V1 Architecture, Tool Calling"
---

# Structured Outputs

## Overview

Structured outputs constrain LLM generation to match a predefined schema (JSON, regex pattern, context-free grammar, or fixed choices). vLLM implements this via **logits masking** based on a grammar state machine, guaranteeing outputs parse correctly. This is critical for production systems that integrate LLM output into downstream APIs or databases.

## What are Structured Outputs?

Standard LLM generation is unconstrained: the model can produce any text. Structured outputs enforce constraints during decoding by **masking invalid tokens** at each step, ensuring the output conforms to a schema.

**Example:** Generate a JSON object with fields `{"name": str, "age": int}`:
- Without constraints: Model might generate `{"name": "Alice", "age": "twenty-five"}` (invalid int)
- With constraints: Model is forced to generate valid integers for `age`

## How It Works

### Grammar State Machine

vLLM compiles the schema (JSON/regex/grammar) into a **finite state machine (FSM)**:

1. **Initialization:** FSM starts in the root state
2. **Token generation:** For each decoding step, FSM determines which tokens are valid
3. **Logits masking:** Invalid tokens get -inf logits, forcing the model to choose a valid token
4. **State transition:** FSM transitions to the next state based on the generated token

**Key insight:** This guarantees syntactic correctness but not semantic quality. The model can still generate nonsensical but valid JSON.

### Backends

vLLM supports multiple structured output backends (selected via `--structured-outputs-config.backend`):

1. **`xgrammar`** (default): MLC-AI's high-performance grammar engine
   - Fast compilation, efficient FSM
   - Rust-style regex syntax
   
2. **`guidance`** (llguidance): Microsoft's guidance library
   - Rich grammar features
   - Rust-style regex syntax

3. **`outlines`** (legacy): Outlines library
   - Rust-style regex syntax
   - Deprecated in favor of xgrammar

4. **`lm-format-enforcer`**: Python regex-based backend
   - Python `re` module syntax
   - Slower than xgrammar/guidance

**Default:** `auto` mode selects backend based on request type.

## Supported Constraint Types

### 1. Choice (Simplest)

Force output to be exactly one of a fixed set of strings:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")

completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Classify: vLLM is wonderful!"}],
    extra_body={"structured_outputs": {"choice": ["positive", "negative"]}},
)
# Output: "positive"
```

**Use case:** Classification tasks with fixed labels.

### 2. Regex

Match a regular expression pattern:

```python
completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Generate email for Alan Turing at Enigma"}],
    extra_body={"structured_outputs": {"regex": r"\w+@\w+\.com\n"}},
)
# Output: "alan.turing@enigma.com\n"
```

**Regex syntax:** Depends on backend (xgrammar/guidance/outlines use Rust regex, lm-format-enforcer uses Python `re`).

**Use case:** Emails, phone numbers, IDs, structured text patterns.

### 3. JSON Schema

Most important constraint type: enforce a JSON schema.

**Method 1: Pydantic model (recommended)**

```python
from pydantic import BaseModel
from enum import Enum

class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType

completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Most iconic 90s car?"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "car-description",
            "schema": CarDescription.model_json_schema()
        },
    },
)
# Output: {"brand": "Honda", "model": "Civic", "car_type": "sedan"}
```

**Method 2: Direct JSON Schema**

Provide a raw JSON Schema dict instead of `CarDescription.model_json_schema()`.

**Pro tip:** Include schema description in prompt for better semantic quality:
```
"Generate a JSON with brand (string), model (string), and car_type (sedan/SUV/truck/coupe)"
```

### 4. Context-Free Grammar (EBNF)

Most powerful but hardest to use: define a custom language using EBNF notation.

**Example: Simplified SQL**

```python
simplified_sql_grammar = """
    root ::= select_statement
    select_statement ::= "SELECT " column " from " table " where " condition
    column ::= "col_1 " | "col_2 "
    table ::= "table_1 " | "table_2 "
    condition ::= column "= " number
    number ::= "1 " | "2 "
"""

completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Query for username from users table"}],
    extra_body={"structured_outputs": {"grammar": simplified_sql_grammar}},
)
# Output: "SELECT col_1 from table_1 where col_1 = 1 "
```

**Use case:** Domain-specific languages (SQL, code, config files).

### 5. Structural Tag

Generate JSON within specific XML-like tags in the output text:

```python
extra_body={"structured_outputs": {"structural_tag": {"json_schema": schema, "tag": "output"}}}
```

**Output:**
```
Here's the result:
<output>{"name": "Alice", "age": 30}</output>
That's the answer!
```

**Use case:** Mixed natural language + structured data responses.

## OpenAI API Compatibility

### Deprecated Fields (Removed in v0.12.0)

Old API:
```python
guided_json=schema  # DEPRECATED
guided_regex=pattern  # DEPRECATED
guided_choice=["a", "b"]  # DEPRECATED
```

New API:
```python
extra_body={"structured_outputs": {"json": schema}}
extra_body={"structured_outputs": {"regex": pattern}}
extra_body={"structured_outputs": {"choice": ["a", "b"]}}
```

### Experimental: OpenAI Beta Parsing API

The `openai` client library (v1.54.4+) provides a beta wrapper for automatic Pydantic parsing:

```python
completion = client.beta.chat.completions.parse(
    model=model,
    messages=[{"role": "user", "content": "My name is Cameron, I'm 28."}],
    response_format=Info,  # Pydantic model directly
)

message = completion.choices[0].message
assert message.parsed  # Already a Pydantic object
print(message.parsed.name)  # "Cameron"
print(message.parsed.age)  # 28
```

**Key difference:** Returns `message.parsed` as a Pydantic object instead of raw JSON string.

## Offline Inference

Use `StructuredOutputsParams` inside `SamplingParams`:

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")

structured_outputs_params = StructuredOutputsParams(choice=["Positive", "Negative"])
sampling_params = SamplingParams(structured_outputs=structured_outputs_params)

outputs = llm.generate(
    prompts="Classify: vLLM is wonderful!",
    sampling_params=sampling_params,
)
# Output: "Positive"
```

**Available parameters:** `json`, `regex`, `choice`, `grammar`, `structural_tag`, `whitespace_pattern`

## Integration with Reasoning Models

Structured outputs work with reasoning-enabled models (DeepSeek-R1, Qwen3-Coder with reasoning):

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --reasoning-parser deepseek_r1
```

**Example: JSON output with reasoning**

```python
completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Generate a random person"}],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "people", "schema": People.model_json_schema()}
    },
)
print("reasoning:", completion.choices[0].message.reasoning)
print("content:", completion.choices[0].message.content)
```

**Output:**
- `reasoning`: Chain-of-thought trace
- `content`: Valid JSON matching schema

**Qwen3-Coder note:** Structured outputs may be disabled in reasoning mode by default (v0.11.2+). Enable with:
```bash
--structured-outputs-config.enable_in_reasoning=True
```

## Performance Considerations

### Grammar Compilation Overhead

**First request:** High latency (1-10 seconds) for complex schemas
- Grammar is compiled into an FSM
- FSM compilation is CPU-intensive

**Subsequent requests:** Near-zero overhead
- Compiled FSM is cached
- Same schema reuses cached FSM

**Recommendation:** Pre-warm the cache during server startup by sending a dummy request for each schema.

### Token Generation Overhead

**Per-token overhead:** 5-15% slowdown vs unconstrained generation
- Logits masking adds small compute cost
- FSM state transitions are fast (hash table lookups)

**Throughput impact:** Minimal at high batch sizes (amortized across batch)

### Memory Overhead

**Per-schema:** 1-50 MB depending on complexity
- Simple schemas (regex, choice): <1 MB
- Complex JSON schemas: 10-50 MB
- EBNF grammars: Varies widely

**Total memory:** O(unique_schemas) — shared across all requests using the same schema

## Configuration

### Server Flags

```bash
vllm serve model \
    --structured-outputs-config.backend xgrammar \
    --structured-outputs-config.enable_in_reasoning True
```

**Backend options:**
- `auto` (default): Auto-select based on request
- `xgrammar`: MLC-AI backend (recommended)
- `guidance`: llguidance backend
- `outlines`: Legacy Outlines backend
- `lm-format-enforcer`: Python regex backend

### Backend-Specific Options

See `vllm serve --help` for full list. Example:
```bash
--structured-outputs-config.xgrammar_max_threads 4
```

## Limitations and Gotchas

### 1. Syntax vs Semantics

Structured outputs guarantee **syntactic correctness** but not **semantic quality**.

**Example:** JSON schema with `{"age": int}`:
- Valid: `{"age": 9999}` (syntactically correct)
- Invalid: `{"age": "twenty"}` (blocked by grammar)
- Nonsense: `{"age": -500}` (valid int, but semantically wrong)

**Mitigation:** Use prompting to guide semantic quality ("age should be between 0 and 120").

### 2. Compilation Time

Complex schemas (deeply nested JSON, large grammars) can take **10+ seconds** to compile on first use.

**Mitigation:** Pre-warm cache during server startup.

### 3. Regex Dialect Differences

- xgrammar/guidance/outlines: **Rust regex** (no lookbehind, different syntax)
- lm-format-enforcer: **Python `re`** module

**Recommendation:** Test regex patterns with the selected backend.

### 4. Model Quality Dependence

Even with constraints, model quality matters:
- Weak models may repeatedly hit "dead ends" in the FSM (backtracking)
- Strong models generate more natural outputs that align with the grammar

**Best models for structured outputs:** GPT-4, Llama 3.1 70B+, Qwen 2.5 72B+

## Cross-References

- [[vLLM Engine]] — Logits processors integrate structured output masking
- [[V1 Architecture]] — Grammar compilation happens in model executor
- [[Tool Calling]] — Uses structured outputs for function arguments

## Further Reading

- [xgrammar documentation](https://github.com/mlc-ai/xgrammar)
- [llguidance documentation](https://github.com/guidance-ai/llguidance)
- [vLLM structured outputs docs](https://docs.vllm.ai/en/latest/features/structured_outputs.html)
