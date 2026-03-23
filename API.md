# OctoHub API Reference

Base URL: `http://127.0.0.1:8080` (configurable via `octohub.toml`)

## Authentication

OctoHub has two authentication layers, both controlled by the `api_key` setting in `octohub.toml`.

**If `api_key` is not set** (default): the server starts without authentication. All client endpoints are open, admin endpoints are disabled. A warning is printed at startup.

**If `api_key` is set**: full authentication is enforced on both layers.

### Client endpoints (`/v1/completions`, `/v1/embeddings`)

Authenticated with a **client API key** issued via the admin API. Pass it as a Bearer token:

```
Authorization: Bearer <client-api-key>
```

### Admin endpoints (`/v1/admin/*`)

Authenticated with the **master key** set in `octohub.toml` (`api_key`):

```
Authorization: Bearer <master-key>
```

All requests to admin endpoints without a valid master key return `401 Unauthorized`. Admin endpoints are unavailable when no master key is configured.

---

## Error Format

All errors return JSON:

```json
{
  "error": {
    "message": "Description of what went wrong",
    "type": "invalid_request_error"
  }
}
```

---

## Client Endpoints

### POST /v1/completions

Create a model response. Supports multi-turn conversations via response chaining, system instructions, and function calling.

#### Request

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

#### Response

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

#### Examples

##### Simple text completion

```bash
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer <client-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o",
    "input": "Explain Rust in one sentence."
  }'
```

##### With system instructions

```bash
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer <client-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o",
    "input": "What should I learn first?",
    "instructions": "You are a programming tutor. Be concise.",
    "temperature": 0.7
  }'
```

##### Multi-turn conversation

```bash
# First turn
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer <client-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o",
    "input": "What is the capital of France?"
  }'
# Response: {"id": "cmpl_abc123", ...}

# Second turn — chains automatically
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer <client-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o",
    "input": "And what is its population?",
    "previous_completion_id": "cmpl_abc123"
  }'
```

##### Function calling

```bash
# Step 1: Ask with tools
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer <client-api-key>" \
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
  -H "Authorization: Bearer <client-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o",
    "input": [
      {"type": "function_call_output", "call_id": "call_xyz", "output": "72°F, sunny"}
    ],
    "previous_completion_id": "cmpl_from_step1"
  }'
```

##### Using a mapped model name

With this config:
```toml
[models]
"my-model" = ["openai:gpt-4o", "anthropic:claude-sonnet-4-20250514"]
```

```bash
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer <client-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "input": "Hello!"
  }'
```

OctoHub randomly picks one provider from the list for load balancing.

---

### POST /v1/embeddings

Generate vector embeddings for text input.

#### Request

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

#### Response

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

#### Examples

##### Single text

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Authorization: Bearer <client-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voyage:voyage-3.5",
    "input": "The quick brown fox jumps over the lazy dog."
  }'
```

##### Batch embedding

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Authorization: Bearer <client-api-key>" \
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

##### Using a mapped embedding model

With this config:
```toml
[embedding_models]
"my-embeddings" = ["voyage:voyage-3.5"]
```

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Authorization: Bearer <client-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-embeddings",
    "input": "Embed this text"
  }'
```

---

### GET /health

Health check endpoint. No authentication required.

#### Response

```json
{"status": "ok"}
```

```bash
curl http://127.0.0.1:8080/health
```

---

## Admin Endpoints

All admin endpoints require the master key from `octohub.toml`:

```
Authorization: Bearer <master-key>
```

---

### POST /v1/admin/keys

Create a new client API key.

#### Request

```json
{
  "name": "string"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | ✅ | Human-readable label for this key (e.g. `"mobile-app"`, `"ci-pipeline"`). |

#### Response `201 Created`

```json
{
  "id": 1,
  "name": "mobile-app",
  "key": "abc123...xyz",
  "key_hint": "...xyz",
  "status": "active",
  "created_at": 1700000000
}
```

> **Important**: The `key` field is only returned on creation. Store it securely — it cannot be retrieved again.

| Field | Description |
|---|---|
| `id` | Numeric key ID. Used in admin queries to filter by key. |
| `name` | Label as provided. |
| `key` | Full 43-character base64url key. Only shown once. |
| `key_hint` | Last 4 characters prefixed with `...` for identification. |
| `status` | `"active"` or `"revoked"`. |
| `created_at` | Unix timestamp. |

#### Example

```bash
curl -X POST http://127.0.0.1:8080/v1/admin/keys \
  -H "Authorization: Bearer <master-key>" \
  -H "Content-Type: application/json" \
  -d '{"name": "mobile-app"}'
```

---

### GET /v1/admin/keys

List all API keys. The full key value is never returned — only the hint.

#### Response `200 OK`

```json
{
  "data": [
    {
      "id": 1,
      "name": "mobile-app",
      "key_hint": "...xyz",
      "status": "active",
      "created_at": 1700000000
    },
    {
      "id": 2,
      "name": "ci-pipeline",
      "key_hint": "...abc",
      "status": "revoked",
      "created_at": 1700001000
    }
  ]
}
```

#### Example

```bash
curl http://127.0.0.1:8080/v1/admin/keys \
  -H "Authorization: Bearer <master-key>"
```

---

### GET /v1/admin/keys/:id

Get a single API key by its numeric ID.

#### Response `200 OK`

```json
{
  "id": 1,
  "name": "mobile-app",
  "key_hint": "...xyz",
  "status": "active",
  "created_at": 1700000000
}
```

Returns `404` if the key does not exist.

#### Example

```bash
curl http://127.0.0.1:8080/v1/admin/keys/1 \
  -H "Authorization: Bearer <master-key>"
```

---

### POST /v1/admin/keys/:id/revoke

Revoke an API key. Revoked keys are rejected on all client endpoints immediately. Keys cannot be deleted — only revoked — because usage records are linked to the key ID.

#### Response `200 OK`

```json
{"status": "revoked"}
```

Returns `404` if the key does not exist.

#### Example

```bash
curl -X POST http://127.0.0.1:8080/v1/admin/keys/1/revoke \
  -H "Authorization: Bearer <master-key>"
```

---

### GET /v1/admin/usage

Aggregated usage statistics. Optionally grouped by time bucket and filtered by key.

#### Query Parameters

| Parameter | Type | Description |
|---|---|---|
| `key_id` | string | Comma-separated key IDs to filter by (e.g. `key_id=1,3`). Omit for all keys. |
| `bucket` | string | Time grouping: `hour`, `day`, `week`, `month`. Omit for a single total row per key. |
| `since` | integer | Unix timestamp — only include records after this time. |
| `until` | integer | Unix timestamp — only include records before this time. |

#### Response `200 OK`

```json
{
  "data": [
    {
      "period": 1700000000,
      "key_id": 1,
      "key_name": "mobile-app",
      "completions_count": 42,
      "embeddings_count": 10,
      "total_input_tokens": 8500,
      "total_output_tokens": 3200
    }
  ]
}
```

| Field | Description |
|---|---|
| `period` | Start of the time bucket as unix timestamp. `null` when no bucket is specified (total). |
| `key_id` | API key ID. |
| `key_name` | API key name at time of query. |
| `completions_count` | Number of completion requests in this period. |
| `embeddings_count` | Number of embedding requests in this period. |
| `total_input_tokens` | Sum of input tokens across all requests. |
| `total_output_tokens` | Sum of output tokens across completion requests. |

#### Examples

```bash
# Total usage across all keys
curl "http://127.0.0.1:8080/v1/admin/usage" \
  -H "Authorization: Bearer <master-key>"

# Daily breakdown for key 1 in the last week
curl "http://127.0.0.1:8080/v1/admin/usage?key_id=1&bucket=day&since=1699395200" \
  -H "Authorization: Bearer <master-key>"

# Hourly breakdown for keys 1 and 3
curl "http://127.0.0.1:8080/v1/admin/usage?key_id=1,3&bucket=hour" \
  -H "Authorization: Bearer <master-key>"
```

---

### GET /v1/admin/completions

List raw completion records with full input/output.

#### Query Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `key_id` | string | all | Comma-separated key IDs to filter by. |
| `since` | integer | — | Unix timestamp lower bound. |
| `until` | integer | — | Unix timestamp upper bound. |
| `limit` | integer | `100` | Max records to return. |
| `offset` | integer | `0` | Pagination offset. |

#### Response `200 OK`

```json
{
  "data": [
    {
      "id": "cmpl_<uuid>",
      "api_key_id": 1,
      "session_id": "<uuid>",
      "input_model": "my-model",
      "resolved_model": "gpt-4o",
      "provider": "openai",
      "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "cost": 0.0001,
        "request_time_ms": 320
      },
      "input": [...],
      "output": [...],
      "created_at": 1700000000
    }
  ]
}
```

#### Example

```bash
# Last 50 completions for key 2
curl "http://127.0.0.1:8080/v1/admin/completions?key_id=2&limit=50" \
  -H "Authorization: Bearer <master-key>"
```

---

### GET /v1/admin/embeddings

List raw embedding records.

#### Query Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `key_id` | string | all | Comma-separated key IDs to filter by. |
| `since` | integer | — | Unix timestamp lower bound. |
| `until` | integer | — | Unix timestamp upper bound. |
| `limit` | integer | `100` | Max records to return. |
| `offset` | integer | `0` | Pagination offset. |

#### Response `200 OK`

```json
{
  "data": [
    {
      "id": "emb_<uuid>",
      "api_key_id": 1,
      "input_model": "my-embeddings",
      "resolved_model": "voyage-3.5",
      "provider": "voyage",
      "usage": {
        "input_tokens": 8,
        "total_tokens": 8,
        "request_time_ms": 120
      },
      "input": ["text to embed"],
      "created_at": 1700000000
    }
  ]
}
```

#### Example

```bash
curl "http://127.0.0.1:8080/v1/admin/embeddings?key_id=1&limit=20&offset=40" \
  -H "Authorization: Bearer <master-key>"
```

---

## Supported Providers

### Completions (`/v1/completions`)

Use `"provider:model"` format. Available providers depend on octolib configuration and environment variables (API keys).

| Provider | Format Example |
|---|---|
| OpenAI | `openai:gpt-4o` |
| Anthropic | `anthropic:claude-sonnet-4-20250514` |
| Google | `google:gemini-2.0-flash` |
| DeepSeek | `deepseek:deepseek-chat` |
| MiniMax | `minimax:minimax-m2.7` |
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

---

## Quick Start

```bash
# 1. Start the server (api_key must be set in octohub.toml)
./octohub

# 2. Create a client API key
curl -X POST http://127.0.0.1:8080/v1/admin/keys \
  -H "Authorization: Bearer <master-key>" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app"}'
# → {"id": 1, "key": "abc...xyz", ...}  ← save this key!

# 3. Use the client key for completions
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer abc...xyz" \
  -H "Content-Type: application/json" \
  -d '{"model": "openai:gpt-4o", "input": "Hello!"}'

# 4. Check usage
curl "http://127.0.0.1:8080/v1/admin/usage?key_id=1&bucket=day" \
  -H "Authorization: Bearer <master-key>"
```

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
| `400` | Bad request (missing/invalid fields, bad model name) |
| `401` | Missing or invalid API key |
| `404` | Unknown endpoint |
| `500` | Internal error (provider failure, storage error) |
