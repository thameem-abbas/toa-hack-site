---
title: Reasoning Models
type: concept
created: 2026-04-25
tags: [reasoning, chain-of-thought, deepseek-r1, qwen3, thinking-budget]
---

# Reasoning Models

Reasoning models are large language models designed to generate internal reasoning steps (chain-of-thought) before producing a final answer. These models output both a `reasoning` field (thinking process) and a `content` field (final conclusion), enabling transparency into the model's decision-making process.

## What Are Reasoning Models?

Traditional LLMs generate answers directly from prompts. Reasoning models insert an explicit thinking phase:

1. **Thinking phase**: Model generates internal reasoning tokens (e.g., `<think>...</think>`)
2. **Answer phase**: Model generates final response based on reasoning

This two-phase approach improves accuracy on complex reasoning tasks (math, logic, code) at the cost of additional latency and token consumption.

## Supported Models in vLLM

vLLM supports 11+ reasoning model families via pluggable `ReasoningParser` system:

| Model Series | Parser Name | Thinking Default | Structured Outputs | Tool Calling |
|--------------|-------------|------------------|-------------------|--------------|
| DeepSeek R1 series | `deepseek_r1` | On | JSON, regex | No |
| DeepSeek-V3.1 | `deepseek_v3` | Off (enable via `thinking=True`) | JSON, regex | No (yes in non-thinking mode) |
| QwQ-32B | `deepseek_r1` | On | JSON, regex | Yes |
| Qwen3 series | `qwen3` | On (disable via `enable_thinking=False`) | JSON, regex | Yes |
| IBM Granite 3.2 | `granite` | Off (enable via `thinking=True`) | No | No |
| ERNIE-4.5-VL | `ernie45` | On | JSON, regex | No |
| ERNIE-4.5-21B-A3B-Thinking | `ernie45` | On | JSON, regex | Yes |
| GLM-4.5 | `glm45` | On | JSON, regex | Yes |
| Holo2 | `holo2` | On (disable via `thinking=False`) | JSON, regex | Yes |
| Hunyuan A13B | `hunyuan_a13b` | On | JSON, regex | Yes |
| MiniMax-M2 | `minimax_m2_append_think` | On | JSON, regex | Yes |

## Usage

### Server Setup

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --reasoning-parser deepseek_r1
```

### Request (Python)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    messages=[{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
)

print("reasoning:", response.choices[0].message.reasoning)
print("content:", response.choices[0].message.content)
```

### Streaming

Reasoning content streams in `delta.reasoning` field:

```json
{
  "choices": [{
    "delta": {
      "role": "assistant",
      "reasoning": "First, I need to compare...",
    },
    "finish_reason": null
  }]
}
```

OpenAI Python client supports extra attributes; use `hasattr(delta, "reasoning")` to check.

## Thinking Budget Control

Some models support limiting reasoning token count to control latency/cost:

### Models with Thinking Budget

- Qwen3 series
- DeepSeek-R1/R1-Distill
- NVIDIA Nemotron3

### Configuration

```bash
vllm serve Qwen/Qwen3-0.6B \
    --reasoning-parser qwen3 \
    --reasoning-config '{
      "reasoning_start_str": "<think>",
      "reasoning_end_str": "I have to give the solution now.</think>"
    }'
```

Per-request budget:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "9.11 and 9.8, which is greater?"}],
    "thinking_token_budget": 10
  }'
```

### Behavior

Token counting starts from `reasoning_start_str`. When count reaches `thinking_token_budget`, vLLM forces the model to emit `reasoning_end_str`, terminating the reasoning block.

**Natural termination phrase**: Set `reasoning_end_str` to include a transition, e.g.:
```json
{
  "reasoning_end_str": "I have to give the solution based on the reasoning directly now.</think>"
}
```

This makes budget exhaustion less abrupt.

If `thinking_token_budget` is unspecified, reasoning length is limited only by `max_tokens`.

## Server-Level Defaults

Set default `chat_template_kwargs` to control thinking behavior without per-request specification:

**Disable thinking (Qwen3 default: on):**
```bash
vllm serve Qwen/Qwen3-8B \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

**Enable thinking (Granite 3.2 default: off):**
```bash
vllm serve ibm-granite/granite-3.2-2b-instruct \
    --reasoning-parser granite \
    --default-chat-template-kwargs '{"thinking": true}'
```

**Request-level override** always takes precedence:
```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}}  # Overrides server default
)
```

## Interleaved Thinking Mode

Some models (e.g., Qwen3) support interleaved thinking: reasoning tokens appear throughout the response, not just at the beginning.

Example output:
```text
reasoning: "The user is comparing 9.11 and 9.8. Let me convert to decimals: 9.11 vs 9.80."
content: "9.8 is greater."
reasoning: "Wait, I should double-check: 9.11 = nine point one one, not nine hundred eleven."
content: "Actually, 9.11 is greater than 9.8."
```

Parsers handle this via stateful `extract_reasoning_streaming()` method.

## Structured Outputs Integration

Reasoning models support constrained decoding via [[Structured Outputs]]:

- Structured output engine (xgrammar) skips reasoning content
- Constraints apply only to `content` field
- Reasoner class provides `is_reasoning_end()` to detect `</think>` token

Example: Force JSON schema in final answer while allowing free-form reasoning.

## Tool Calling Integration

When reasoning + tool calling both enabled:

1. Reasoning content available in `response.choices[0].message.reasoning`
2. Tool calls parsed from `content` field only (not from reasoning)
3. Model can explain tool selection in reasoning tokens

Example:
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {...}
    }
}]

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
    tools=tools
)

print(response.choices[0].message.reasoning)  # "User wants weather info, I'll call get_weather..."
print(response.choices[0].message.tool_calls[0].function.name)  # "get_weather"
```

## How It Works

### ReasoningParser

ReasoningParser extracts reasoning content from model output:

**For streaming:**
```python
def extract_reasoning_streaming(
    previous_text: str,
    current_text: str,
    delta_text: str,
    previous_token_ids: Sequence[int],
    current_token_ids: Sequence[int],
    delta_token_ids: Sequence[int],
) -> DeltaMessage | None:
    # Parse incremental output and extract reasoning delta
    ...
```

**For non-streaming:**
```python
def extract_reasoning(
    model_output: str,
    request: ChatCompletionRequest,
) -> tuple[str | None, str | None]:
    # Parse complete output and return (reasoning, content)
    ...
```

### Reasoner (for Structured Outputs)

Reasoner detects reasoning boundaries for structured output engines:

```python
@dataclass
class DeepSeekReasoner(Reasoner):
    start_token_id: int  # <think> token ID
    end_token_id: int    # </think> token ID

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self.end_token_id in input_ids

    def is_reasoning_end_streaming(self, input_ids: list[int], delta_ids: list[int]) -> bool:
        return self.end_token_id in delta_ids
```

xgrammar uses `end_token_id` to skip structured output while in reasoning block.

## Adding New Reasoning Models

Implement `ReasoningParser` and register:

```python
from vllm.reasoning import ReasoningParser, ReasoningParserManager

class MyParser(ReasoningParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def extract_reasoning_streaming(...):
        # Parse incremental output
        ...

    def extract_reasoning(...):
        # Parse complete output
        ...

ReasoningParserManager.register_lazy_module(
    name="my_model",
    module_path="vllm.reasoning.my_parser",
    class_name="MyParser",
)
```

Then:
```bash
vllm serve <model> --reasoning-parser my_model
```

For structured output support, also implement `Reasoner` class.

## Limitations

- Reasoning content only available for `/v1/chat/completions` endpoint (not `/v1/completions`)
- Some models don't support tool calling in thinking mode (e.g., DeepSeek-V3.1)
- Thinking budget enforcement requires explicit `reasoning_config`

## Performance Considerations

**Latency:**
- 2-10× increase in TTFT (depends on reasoning token count)
- Thinking budget controls max overhead

**Throughput:**
- Reasoning tokens consume KV cache
- Batch size may decrease due to longer sequences

**Cost:**
- Reasoning tokens count toward input/output token usage
- Set budgets to control expense

**Quality:**
- Improved accuracy on reasoning-heavy tasks (math, code, logic)
- May over-explain for simple queries

## Use Cases

**Strong fit:**
- Mathematical reasoning (GSM8K, MATH)
- Code generation with explanation
- Multi-step logical inference
- Scientific problem-solving

**Weak fit:**
- Simple factual queries (reasoning overhead unnecessary)
- High-throughput serving (latency-sensitive)

## Migration Note

`reasoning_content` → `reasoning` (breaking change in recent versions)

Old:
```python
response.choices[0].message.reasoning_content
```

New:
```python
response.choices[0].message.reasoning
```

## Cross-References

- [[Structured Outputs]] — Constrained decoding on final answer (skips reasoning)
- [[vLLM Engine]] — Reasoning parser integration in chat completion endpoint
- [[Speculative Decoding]] — Reasoning models incompatible with draft model speculation (reasoning tokens unpredictable)
- [[Multi-Token Prediction]] — Some reasoning models support MTP natively (e.g., DeepSeek-V3)
- [[Logits Processors]] — Reasoning boundary detection via logits masking
- [[Quantization]] — Reasoning models can be quantized (FP8/INT8/INT4)

## Related

- **Chain-of-Thought Prompting**: User-provided reasoning prompts (all models)
- **Reasoning Models**: Model-native reasoning tokens (this page)
- **Tool Calling**: Function calling with reasoning explanations
