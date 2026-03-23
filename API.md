# OctoHub API Reference

Base URL: `http://127.0.0.1:8080` (configurable via `octohub.toml`)

## Authentication

If `api_key` is set in config, all requests require an `Authorization` header:

```
Authorization: Bearer <your-api-key>
```

If no `api_key` is configured, authentication is disabled.

---

## POST /v1/completions

Create a model response. Supports multi-turn conversations via response chaining, system instructions, and function calling.

### Request

```json
{
  "model": "string",
  "input": "string | InputItem[]",
  "instructions": "string | null",
  "previous_completion_id": "string | null",
  "temperature": 1.0,
  "max_output_tokens": 0,
  "tools": "ToolDefinition[] | null"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | string | ✅ | — | Model identifier. Use `"provider:model"` format (e.g. `"openai:gpt-4o"`) or a mapped name from config. |
| `input` | string \| array | ✅ | — | Input content. A plain string becomes a user message. An array allows typed items (see below). |
| `instructions` | string | — | `null` | System instructions prepended to the conversation. |
| `previous_completion_id` | string | — | `null` | Chain to a previous completion for multi-turn conversations. OctoHub reconstructs the full history automatically. |
| `temperature` | float | — | `1.0` | Sampling temperature (0.0–2.0). |
| `max_output_tokens` | integer | — | `0` | Maximum tokens in the response. `0` = provider default. |
| `tools` | array | — | `null` | Function definitions for tool/function calling. |

#### Input Items

When `input` is an array, each element is a tagged object:

**Message:**
```json
{"type": "message", "role": "user", "content": "What is Rust?"}
```

**Function call output** (tool result, requires `previous_completion_id`):
```json
{"type": "function_call_output", "call_id": "call_abc123", "output": "72°F sunny"}
```

#### Tool Definition

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string"}
    }
  }
}
```

### Response

```json
{
  "id": "cmpl_<uuid>",
  "object": "completion",
  "output": [OutputItem],
  "usage": Usage,
  "created_at": 1700000000
}
```

#### Output Items

**Message:**
```json
{
  "type": "message",
  "id": "msg_<uuid>",
  "role": "assistant",
  "content": [
    {"type": "output_text", "text": "Hello!"}
  ]
}
```

**Function call:**
```json
{
  "type": "function_call",
  "id": "fc_<uuid>",
  "call_id": "call_abc123",
  "name": "get_weather",
  "arguments": "{\"location\":\"NYC\"}"
}
```

#### Usage

```json
{
  "input_tokens": 10,
  "output_tokens": 5,
  "total_tokens": 15,
  "cache_read_tokens": null,
  "cache_write_tokens": null,
  "cost": 0.0001,
  "request_time_ms": 500
}
```

`cache_read_tokens`, `cache_write_tokens`, `cost`, and `request_time_ms` are omitted when `null`.

### Examples

#### Simple text completion

```bash
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o",
    "input": "Explain Rust in one sentence."
  }'
```

#### With system instructions

```bash
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o",
    "input": "What should I learn first?",
    "instructions": "You are a programming tutor. Be concise.",
    "temperature": 0.7
  }'
```

#### Multi-turn conversation

```bash
# First turn
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o",
    "input": "What is the capital of France?"
  }'
# Response: {"id": "cmpl_abc123", ...}

# Second turn — chains automatically
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o",
    "input": "And what is its population?",
    "previous_completion_id": "cmpl_abc123"
  }'
```

#### Function calling

```bash
# Step 1: Ask with tools
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o",
    "input": "What is the weather in NYC?",
    "tools": [
      {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    ]
  }'
# Response includes: {"type": "function_call", "call_id": "call_xyz", "name": "get_weather", ...}

# Step 2: Send tool result back
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o",
    "input": [
      {"type": "function_call_output", "call_id": "call_xyz", "output": "72°F, sunny"}
    ],
    "previous_completion_id": "cmpl_from_step1"
  }'
```

#### Using a mapped model name

With this config:
```toml
[models]
"my-model" = ["openai:gpt-4o", "anthropic:claude-sonnet-4-20250514"]
```

```bash
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "input": "Hello!"
  }'
```

OctoHub randomly picks one provider from the list for load balancing.

---

## POST /v1/embeddings

Generate vector embeddings for text input.

### Request

```json
{
  "model": "string",
  "input": "string | string[]"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | string | ✅ | — | Model identifier. Use `"provider:model"` format (e.g. `"voyage:voyage-3.5"`) or a mapped name from config. |
| `input` | string \| string[] | ✅ | — | Text(s) to embed. A single string or an array of strings. |

### Response

Returns the embedding vector(s) directly:

**Single input:**
```json
[0.0023, -0.0091, 0.0234, ...]
```

**Batch input:**
```json
[
  [0.0023, -0.0091, ...],
  [0.0156, 0.0089, ...],
  [0.0034, -0.0123, ...]
]
```

### Examples

#### Single text

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voyage:voyage-3.5",
    "input": "The quick brown fox jumps over the lazy dog."
  }'
```

#### Batch embedding

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:text-embedding-3-small",
    "input": [
      "First document to embed",
      "Second document to embed",
      "Third document to embed"
    ]
  }'
```

#### Using a mapped embedding model

With this config:
```toml
[embedding_models]
"my-embeddings" = ["voyage:voyage-3.5"]
```

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-embeddings",
    "input": "Embed this text"
  }'
```

---

## GET /health

Health check endpoint.

### Response

```json
{"status": "ok"}
```

```bash
curl http://127.0.0.1:8080/health
```

---

## Supported Providers

### Completions (`/v1/completions`)

Use `"provider:model"` format. Available providers depend on octolib configuration and environment variables (API keys).

| Provider | Format Example |
|---|---|
| OpenAI | `openai:gpt-4o` |
| Anthropic | `anthropic:claude-sonnet-4-20250514` |
| Google Vertex | `google:gemini-2.0-flash` |
| DeepSeek | `deepseek:deepseek-chat` |
| Minimax | `minimax:minimax-m2.7` |
| Ollama | `ollama:llama3` |
| OpenRouter | `openrouter:meta-llama/llama-3-70b` |

### Embeddings (`/v1/embeddings`)

| Provider | Format Example |
|---|---|
| Voyage | `voyage:voyage-3.5` |
| OpenAI | `openai:text-embedding-3-small` |
| Jina | `jina:jina-embeddings-v3` |
| Google | `google:gemini-embedding-001` |
| OpenRouter | `openrouter:openai/text-embedding-3-small` |
| OctoHub | `octohub:my-model` (proxy to another OctoHub instance) |

---

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "message": "Description of what went wrong",
    "type": "invalid_request_error"
  }
}
```

| HTTP Status | Meaning |
|---|---|
| 400 | Bad request (invalid model, malformed input, unknown completion ID) |
| 401 | Unauthorized (missing or invalid API key) |
| 404 | Unknown endpoint |
| 500 | Internal error (provider failure, storage error) |
