# 05 — Client API

This is the public client-facing surface. Every endpoint here requires a
**client** API key from the `api_keys` table — see
[04 — Authentication](./04-authentication.md) for how to obtain one.

Sources: [`src/api/handler.rs`](../../src/api/handler.rs),
[`src/api/types.rs`](../../src/api/types.rs),
[`src/proxy/engine.rs`](../../src/proxy/engine.rs).

## Endpoints

| Method | Path | Style | Handler | Auth |
|---|---|---|---|---|
| `POST` | `/v1/completions` | OctoHub Responses API | `handle_create_completion` at `src/api/handler.rs:44` | client key |
| `GET` | `/v1/completions` | HTTP/1.1 WebSocket upgrade; sequential buffered completions | `handle_upgrade` in `src/api/websocket.rs` | client key |
| `POST` | `/v1/chat/completions` | Classic OpenAI Chat Completions | `handle_chat_completion` at `src/api/handler.rs:284` | client key |
| `POST` | `/v1/embeddings` | OpenAI-compatible | `handle_create_embedding` at `src/api/handler.rs:146` | client key |
| `GET`  | `/health` | — | `handle_health` at `src/api/handler.rs:141` | none |

## Which completion endpoint to use

**`/v1/completions`** — OctoHub's native Responses API. Used by Octomind.
Supports multi-turn chains via `previous_completion_id`, reasoning-block
replay for thinking models, and a richer output structure (`output[]` array
with typed items). Choose this when building against OctoHub directly.
`POST` remains unchanged. Machine clients can also use an authenticated
HTTP/1.1 `GET` WebSocket upgrade on the same path for sequential requests;
see [WebSocket transport](#get-v1completions--websocket).

**`/v1/chat/completions`** — Classic OpenAI Chat Completions. A drop-in
replacement for `api.openai.com/v1/chat/completions`. Use this when pointing
an existing OpenAI-compatible client (Python `openai` SDK, LangChain,
LiteLLM, any third-party tool) at OctoHub — no code changes on the client
side, just swap the `base_url`.

Both endpoints share the **same engine, same storage, same auth, same
metrics**. The completion `id` returned is the same DB row either way. The
only difference is the wire format going in and coming out.

> **Streaming:** `/v1/chat/completions` accepts `"stream": true` and emits
> buffered SSE with text and indexed tool-call deltas. The upstream call finishes
> first; this is not upstream token streaming and does not reduce first-token latency.
> Set `"stream_options": {"include_usage": true}` to receive a final usage-only
> chunk (`choices: []`) before `[DONE]`. Other chunks have `usage: null` when
> requested; without this option no usage chunk is emitted.

All client responses include an `X-Request-Id` header
(`src/main.rs:198`). Use it to correlate against server logs and
metrics.

## `GET /health`

No auth. Returns `200 OK` with `{"status":"ok"}` if the process is up.
This is a process-liveness probe — it does **not** check that the
database is reachable, the configured providers are reachable, or that
the master key is set. If you want readiness, hit `/v1/admin/keys` with
the master key; a 200 means auth + DB both work.

Use it as your `livenessProbe` in Kubernetes. For a deeper check, send a
`/v1/embeddings` request with a 1-token input against a cheap model.

## `POST /v1/completions`

OpenAI Responses API-compatible. Request body deserialized into
`CreateCompletionRequest` at `src/api/types.rs:5`–`48`.

### Request schema

```jsonc
{
  "model": "minimax-m2.7",     // required — alias from [models] or "provider:model"
  "input": "Say hi",           // required — string OR array of input items
  "instructions": "Be brief.", // optional — system prompt; string or array of parts
  "previous_completion_id": "...",  // optional — for multi-turn chain replay
  "temperature": 1.0,          // optional, default 1.0
  "top_p": 1.0,                // optional, default 1.0 — proxied through
  "max_output_tokens": 0,      // optional, 0 = provider default
  "reasoning_effort": "high",  // optional — "low"|"medium"|"high"|"xhigh"|"max"
  "text": {                    // optional — structured output
    "format": { "type": "json_object" }
    // OR
    "format": { "type": "json_schema", "schema": { ... }, "strict": true }
  },
  "tools": [                   // optional — function calling (flat shape)
    {
      "type": "function",
      "name": "get_weather",
      "description": "...",
      "parameters": { /* JSON schema */ }
    }
  ]
}
```

### `input` shapes

`Input` is an untagged enum (`src/api/types.rs:87`–`94`):

- **String** — `"input": "hello"` becomes a single user message.
- **Array of items** — `"input": [{"type": "message", "role": "user", "content": "..."}]`.
  Each item is `InputItem` (`src/api/types.rs:99`–`122`), an internally
  tagged enum keyed on `type`, with exactly **four** variants:
  - `message` — a conversation turn; `content` is a string or an array
    of content parts (`input_text`, `input_image`, `input_video`).
    Images and videos are parts **inside** a message, not standalone
    input items.
  - `function_call_output` — a tool result (`call_id` + `output`).
  - `function_call` — a prior assistant tool call replayed by the client.
  - `reasoning` — a prior assistant thinking block replayed by the client.

The authoritative list is the `InputItem` enum in `src/api/types.rs:99`.

### Multi-turn chains (`previous_completion_id`)

Set `"previous_completion_id": "<id from a prior response>"` to continue
a conversation. OctoHub loads the prior completion from storage, replays
its full exchange (user + assistant messages, including any reasoning
blocks), and pre-pends them to the new request before calling the
provider. The handler that walks the chain is at
`src/proxy/engine.rs:79`; reasoning-block replay is at
`src/proxy/engine.rs:248`.

Practical effects:

- **Reasoning models** (DeepSeek, o1, etc.) keep their chain-of-thought
  context across turns — without this, follow-up turns look like
  fresh-start questions and the model loses its train of thought.
- **The chain always replays from the database**, not from the
  client's memory. A client that lost the conversation history can
  recover it by passing the last `id`.
- **An unknown ID starts a fresh conversation** using only the inline
  input. The new completion is stored without a parent link.
- **A chain that cannot be loaded fails the request** before any upstream
  provider call, so the model never answers with stored history silently cut.
- **The chain is loaded once per request.**

### `text.format` — structured output

Two variants (`src/api/types.rs:75`–`84`):

- **`json_object`** — the model is told to emit a JSON object; no
  schema enforcement.
- **`json_schema`** — the model is told to emit JSON conforming to the
  provided schema; `strict: true` enables strict adherence when the
  upstream provider supports it.

A `json_schema` request is routed only to the model's lanes whose provider
guarantees the schema (octolib's `enforces_response_schema`), in their
configured order. Lanes that could only approximate it — a forced tool
call or `json_object` mode — are skipped. When no lane of the model
qualifies, the request fails with 400 before any upstream call.
`json_object` requests use every lane.

### Response shape

A successful response is the upstream provider's `Completion` (from
`octolib`), serialized to JSON. The proxy is a **pass-through** for the
response body — it does not reshape what providers return. Fields you
can rely on:

- `id` — pass this back as `previous_completion_id` for the next turn
- `model` — the resolved model name
- `provider` — the resolved provider name
- `output` — the model's output (text, tool calls, reasoning, etc.)
- `usage.input_tokens`, `usage.output_tokens` — surfaced in metrics and
  logs
- `finish_reason` — why the model stopped (`stop`, `length`,
  `tool_calls`, …), when the provider reports it (omitted otherwise)
- `structured_output` — parsed JSON, present only when the request
  attached a `json_schema` and the provider returned conforming output
- `created_at` — Unix seconds

The response does **not** echo `previous_completion_id`
(`CreateCompletionResponse`, `src/api/types.rs:240`); the chain link you
sent is recorded in storage, not returned in the body.

### Error responses

The HTTP status code is computed by `classify_engine_error` at
`src/api/handler.rs:233`:

| Condition | Status | Body |
|---|---|---|
| `MODEL_FORBIDDEN_MARKER` in the error | `403 Forbidden` | The model name + "is not permitted for this API key" |
| Unknown model, resolution failure, invalid request | `400 Bad Request` | The error text |
| Anything else (upstream failure, internal error) | `500 Internal Server Error` | The full `anyhow` chain via `{:#}` formatting |
| Missing/invalid `Authorization` header | `401 Unauthorized` | `"Missing API key"` or `"Invalid or revoked API key"` |
| `serde_json` parse failure on the body | `400 Bad Request` | `"Invalid request JSON: …"` |
| Request body over 16 MiB (`MAX_REQUEST_BODY_BYTES`) | `413 Payload Too Large` | `"Request body exceeds 16 MiB"` — rejected before any upstream call |

### A note on retries

OctoHub does **not** retry upstream failures on the client's behalf.
When a provider is down, you get a 5xx back, and you decide whether to
retry. This is deliberate — silent retries can blow through token
budgets and make a hung provider look like a slow but successful call.
Use your SDK's retry policy.

If you want OctoHub to load-balance across providers, that's the
`[models]` config — see [03 — Configuration](./03-configuration.md).
The list is sampled **once per request**, randomly.

## `GET /v1/completions` — WebSocket

### Handshake and authentication

Upgrade `/v1/completions` over **HTTP/1.1**, using `ws://` or `wss://` as
appropriate for the deployment. Include `Authorization: Bearer <client-key>`
in the upgrade request. A successful handshake returns HTTP `101 Switching
Protocols`; the existing `POST /v1/completions` request/response contract is
unchanged. HTTP/2 WebSocket upgrade is not supported.

This is a machine-to-machine bearer endpoint. **Any `Origin` header is
rejected with HTTP `403`**, even for an otherwise valid upgrade; it is not a
browser WebSocket API. Invalid or malformed upgrades are checked before
authentication and return HTTP `400`. A valid handshake without a bearer
token returns HTTP `401`; invalid or revoked client keys also use the
existing `401` authentication errors. The implementation is
[`src/api/websocket.rs`](../src/api/websocket.rs).

The client key is authenticated both at the handshake and **for every
request message**, so revocation, allowed-model edits, and owner edits take
effect on subsequent requests without reconnecting. Each request reads a
fresh configuration snapshot instead of pinning the connection to its
handshake configuration. The handshake's `X-Model-Purpose` and attribution
headers (`X-Title`, `HTTP-Referer`, `X-OpenRouter-Categories`) are carried
into each completion request; they are not additional JSON request fields.

### Messages and final replies

Send a **text message** containing the existing `CreateCompletionRequest`
JSON object, exactly as for `POST`, with no outer request envelope:

```json
{"model":"minimax-m2.7","input":"Say hi"}
```

The shared completion engine supplies the same routing, chain replay,
limits, storage, response body, and error mapping as `POST`. A completion
is buffered: there are no token deltas, partial responses, or upstream
streaming, and no promise of faster completion or earlier first-token
delivery. Each request receives **one final text message** (shape shown as
JSONC):

```jsonc
{
  "status": 200,
  "headers": {
    "content-type": "application/json",
    "x-request-id": "01JAZ6Y6D6YAW9W7G9NRJHY7GK"
  },
  "body": { /* existing CreateCompletionResponse JSON object */ }
}
```

- `status` is the corresponding HTTP status: `200` on success, the existing
  error status otherwise.
- `body` is the existing completion response or error JSON **object**, not a
  JSON-encoded string. Existing error discriminators are unchanged,
  including `error.type = "modality_not_supported"`.
- Header names are **lowercase**, with **string** values. `content-type`
  is `application/json`; `x-request-id` is a **fresh ULID for each reply**,
  not the handshake's request ID. A valid client-supplied `X-Request-Id`
  is used only for the HTTP handshake. A rate-window `429` preserves the
  existing `Retry-After` header as `headers["retry-after"]` when supplied
  by the shared handler; its seconds value is a string.

### Sequencing and transport limits

There is **one in-flight request per connection**. The client must wait
for the final reply before sending its next request. Pipelining closes the
socket with `1008` (policy violation); requests are not accumulated in an
application backlog. Use native WebSocket **Ping/Pong** for transport
liveness; control frames remain supported while a completion is pending.

| Condition | Result |
|---|---|
| Incoming frame or complete request message exceeds **16 MiB** | Close `1009` (message too big) |
| Single outgoing serialized envelope (status, headers, and body together) exceeds **64 MiB** | Close `1009` (message too big) |
| Binary message | Close `1003` (unsupported data) |
| Malformed WebSocket protocol frame (for example, unmasked client text) | Close `1002` (protocol error) |
| Another request before the previous reply | Close `1008` (policy violation), no backlog |
| Malformed HTTP upgrade | HTTP `400`, before upgrade |
| Any `Origin` header | HTTP `403`, before upgrade |

Acceptance through the first headers, including HTTP protocol detection,
is bounded to **10 seconds**. Every HTTP/1 request has a 10-second
header-read limit, including non-WebSocket routes. Handshake authentication,
HTTP upgrade completion (`OnUpgrade`), and transport send/flush/close
operations each have fixed **10-second** bounds. These are transport bounds, not a
10-second completion deadline; existing owner/provider queue and upstream
timeouts still govern completion work. There are no new WebSocket timeout
configuration fields.

### Disconnect and shutdown

When the caller closes the socket, the local request future is cancelled
and its permits are released. Cancellation **cannot roll back** an
already-issued provider call or a running blocking database write.

The existing `server.shutdown_timeout_secs` bounds graceful drain.
Idle sockets close with `1001` (going away). A busy socket lets its current
request finish within that bound, sends its reply, then closes with `1001`.
No new requests are accepted during drain, and overdue work is aborted
when the drain bound expires.

The HTTP `GET` handshake has its own request metrics and tracing. Each
completion message is observed as `POST /v1/completions` through the
shared engine and existing error/metric helpers; see
[07 — Observability](07-observability.md#websocket-completions).

## `POST /v1/chat/completions`

Classic OpenAI Chat Completions. Point any OpenAI-compatible client here
by setting `base_url = "http://127.0.0.1:8080/v1"`. Internally this is
identical to `/v1/completions` — same engine, same DB row, same auth.

### Request schema

```jsonc
{
  "model": "minimax-m2.7",
  "messages": [
    {"role": "system", "content": "Be concise."},
    {"role": "user",   "content": "What is Rust?"}
  ],
  "temperature": 1.0,      // optional, default 1.0
  "top_p": 1.0,            // optional, default 1.0
  "max_tokens": 512,       // optional, 0/null = provider default
  "stream": false,         // optional — true emits buffered SSE
  "stream_options": {"include_usage": true}, // optional — usage-only final chunk
  "tools": [...],          // optional — classic nested-function shape
  "tool_choice": "auto"    // optional — accepted and ignored
}
```

Message conversion (transparent to the caller):

| Classic role | Becomes |
|---|---|
| `system` | `instructions` (multiple system messages joined with `\n`) |
| `user` / `assistant` (text) | `message` input item |
| `assistant` + `tool_calls` | one `function_call` input item per call |
| `tool` | `function_call_output` (matched by `tool_call_id`) |

### Response schema

```jsonc
{
  "id": "cmpl_01JXXXXXXXXXX",   // DB completion id
  "object": "chat.completion",
  "created": 1749300000,
  "model": "minimax:minimax-m2.7",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Rust is a systems language…",
      "tool_calls": null          // populated when the model calls a tool
    },
    "finish_reason": "stop"       // always a string; "stop" fallback
  }],
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 9,
    "total_tokens": 27
  }
}
```

Reasoning blocks from thinking models are not included (invisible to
classic clients by design).

### Python SDK example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="<client-api-key>",
)
resp = client.chat.completions.create(
    model="minimax-m2.7",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

## `POST /v1/embeddings`

Schema: `CreateEmbeddingRequest` at `src/api/types.rs:333`.
Handler: `handle_create_embedding` at `src/api/handler.rs:146`.

```jsonc
{
  "model": "voyage",          // required — alias from [embedding_models] or "provider:model"
  "input": "Some text to embed"  // string OR array of strings
}
```

Or batched:

```jsonc
{
  "model": "voyage",
  "input": ["text 1", "text 2", "text 3"]
}
```

The model is resolved through `[embedding_models]` (not `[models]`).
Aliases defined in `[models]` are **not** available for embedding calls
and vice versa — they're separate maps (`src/config.rs:66` and `:69`).

### Response

The body is **just the raw embedding vector(s)** — there is no envelope,
no `model`/`provider`/`usage` fields. `CreateEmbeddingResponse` is an
untagged enum (`src/api/types.rs:390`) whose shape mirrors the input:

- **Single string input** → one flat array of floats:

  ```json
  [0.012, -0.044, 0.087, ...]
  ```

- **Array input** → an array of arrays, one per input string, in order:

  ```json
  [[0.012, -0.044, ...], [0.031, 0.057, ...], ...]
  ```

Token usage is **not** in the response body. It is approximated
internally (`input_chars / 4`, `src/proxy/engine.rs:433`), recorded into
the `octohub_embedding_tokens_total{direction="in"}` counter, and stored
on the embedding row for the admin API — but the client receives only the
vectors. Embeddings have no "output" token side.

### Errors

Same classification rules as completions. A 403 here means the
embedding model name is not in the key's `allowed_models` list.

## `POST /v1/evaluations`

Schema: `CreateEvaluationRequest` in `src/api/types.rs`.
Handler: `handle_create_evaluation` in `src/api/handler.rs`.

```jsonc
{
  "model": "jev",                       // required — alias from [evaluation_models] or "provider:model"
  "state": "Help! My payouts have been failing for 3 days.",  // string, object or array
  "questions": {
    "is_urgent":   {"type": "noul",   "instructions": "Does this convey urgency?"},
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "Payments, refunds", "technical": "Bugs, outages"}},
    "frustration": {"type": "score",  "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}
```

`"model": "auto"` first resolves to an `[evaluation_models]` alias: the key
owner's stored `evaluation` entry, then `[auto].evaluation` (never
`default`, which names a chat model); none is a 400. The resolved alias
must also be in the key's `allowed_models`, and `input_model` keeps `auto`.

The model is resolved through `[evaluation_models]`, with the same
rotation, rate-window admission and optional provider failover as
embeddings. The request goes to octolib's `EvaluationProvider` for the
chosen candidate; there is no job to poll.

### Response

```jsonc
{
  "id": "eval_9f1c...",
  "model": "jev-1.13.0",               // the versioned model that answered
  "answers": {
    "is_urgent":   {"type": "noul", "noul": 0.95},
    "department":  {"type": "choice", "choice": "billing",
                    "probabilities": {"billing": 0.88, "technical": 0.12}, "confidence": 0.82},
    "frustration": {"type": "score", "score": 1.03,
                    "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                    "probabilities": {"0": 0.0, "1": 0.97, "2": 0.03}, "confidence": 0.95}
  },
  "usage": {"input_tokens": 414, "output_tokens": 73, "cost": 0.000017388}
}
```

`usage.cost` is octolib's published rate applied to the tokens (Jev:
$0.042 per 1M input, output free); it is `null` when no rate is known.
Tokens are counted against the provider's windows and recorded in
`octohub_evaluation_tokens_total` in both directions.

### Errors

Same classification rules as completions. octolib's typed evaluation
errors map directly: a request the provider or octolib rejected → 400,
upstream rate limit → 429 with `Retry-After`, upstream authentication or
permission failure → 401/403, anything else upstream → 500.

## Examples

### Plain text completion

```bash
curl -sX POST http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m2.7",
    "input": "Explain vector databases in one sentence."
  }'
```

### Multi-turn with chain

```bash
# Turn 1 — capture the id from the response
RESP=$(curl -sX POST http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"minimax-m2.7","input":"My name is Alice."}')
ID=$(echo "$RESP" | jq -r .id)

# Turn 2 — pass the id back, OctoHub replays turn 1 automatically
curl -sX POST http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\":\"minimax-m2.7\",
    \"input\":\"What's my name?\",
    \"previous_completion_id\":\"$ID\"
  }"
```

### Batched embeddings

```bash
curl -sX POST http://127.0.0.1:8080/v1/embeddings \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voyage",
    "input": ["first document", "second document", "third document"]
  }'
```

### Structured output

```bash
curl -sX POST http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m2.7",
    "input": "Extract: name, age from \"Jane is 29 years old\"",
    "text": {
      "format": {
        "type": "json_schema",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "name": { "type": "string" },
            "age":  { "type": "integer" }
          },
          "required": ["name", "age"]
        }
      }
    }
  }'
```

### Tool calling

Pass `tools` as a flat array of function definitions
(`{"type":"function","name":…,"description":…,"parameters":…}`). The
model's reply carries `function_call` items in `output`; execute them and
post each result back as an `input` item of type `function_call_output`
(`{"type":"function_call_output","call_id":…,"output":…}`), usually
alongside `previous_completion_id` to continue the chain.

## Request lifecycle at a glance

```
client request
  │
  ├─ X-Request-Id extracted (passthrough or fresh ULID)            src/main.rs:183
  ├─ HTTP/1 or HTTP/2 detected, keep-alive off                     src/main.rs:138
  ├─ in_flight gauge incremented                                   src/metrics.rs:187
  ├─ route label classified                                        src/main.rs:163
  ├─ span opened: req_id, method, route, path, remote, ...
  │
  ├─ auth: authenticate_client → 401 or ApiKey                     src/auth.rs:22
  ├─ allow-list: ensure_model_allowed → 403 or continue            src/proxy/engine.rs:781
  ├─ body parsed as CreateCompletionRequest                        src/api/handler.rs:76
  ├─ chain replay (if previous_completion_id)                      src/proxy/engine.rs:79
  ├─ provider permit acquired (blocks if at concurrency limit)     src/proxy/limiter.rs:49
  ├─ upstream call (octolib)
  ├─ record metric: octohub_completions_total, _tokens_total       src/metrics.rs:203
  ├─ persist completion row (input, output, usage, provider, ...)
  └─ response: provider output verbatim + X-Request-Id
```


---

## Media endpoints

`POST /v1/images/generations`, `POST /v1/videos`, `POST /v1/audio/speech`,
`POST /v1/audio/transcriptions`, plus `GET /v1/media/{id}`,
`POST /v1/media/{id}/cancel` and `GET /v1/media/models`.

They authenticate with the same client keys and honour the same `allowed_models`
allow-list and owner concurrency budget as the endpoints above. Long jobs answer
`202 Accepted` with a job id you poll.

See [11 — Media](./11-media.md) for the full contract.
