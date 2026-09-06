# 11 — Media

OctoHub proxies image generation, video generation, speech synthesis and
transcription behind the same API keys, quotas and usage logging as
`/v1/completions`. Providers: `elevenlabs`, `fal`, `openrouter`, `replicate`,
`runway`.

For the design rationale behind these choices, see
[SPEC-media.md](./SPEC-media.md).

---

## Configuration

Media aliases work exactly like `[models]` — an alias mapped to a list of
`provider:model` mirrors, picked in rotation, with fallthrough when one is rate
limited:

```toml
[media_models]
"flux" = ["fal:fal-ai/flux/dev", "replicate:black-forest-labs/flux-1.1-pro"]
"veo"  = ["openrouter:google/veo-3.1"]
"tts"  = ["elevenlabs:eleven_flash_v2_5"]
```

A literal `provider:model` in a request bypasses the map, as it does elsewhere.

Concurrency and rate windows reuse `[providers.<name>]` unchanged — the media
adapters are keyed by the same provider names:

```toml
[providers.fal]
concurrency = 8
requests_per_minute = 60
```

`tokens_per_minute` / `tokens_per_day` do not apply: media providers report no
tokens, so those windows are never fed.

Optional transport knobs:

```toml
[media]
max_source_bytes = 20971520       # 20 MiB cap on decoded inline uploads
max_response_bytes = 104857600    # 100 MiB
polling_interval_secs = 2
submit_timeout_secs = 120
```

Wait deadlines reuse `server.upstream_timeout_secs`.

To point an adapter at a self-hosted or gateway-fronted endpoint:

```toml
[media_providers.fal]
api_base = "https://fal.internal.example"
```

This is exported into the adapter's own environment variable at startup, so
there is exactly **one custom endpoint per adapter** per deployment. An
already-set environment variable wins.

Bad configuration fails at boot, not on the first paid request: an unknown
provider, a malformed `provider:model`, an empty mirror list, or an alias that
collides with `[models]` / `[embedding_models]` all refuse to start.

---

## Provider credentials

Upstream keys are read from the server's environment by octolib, exactly as for
the LLM providers — OctoHub never accepts them from a client.

| Provider | Variable |
|---|---|
| `elevenlabs` | `ELEVENLABS_API_KEY` |
| `fal` | `FAL_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `replicate` | `REPLICATE_API_KEY` |
| `runway` | `RUNWAY_API_KEY` |

A missing key surfaces as `500 configuration_error` on the first request that
routes to that provider, not at startup — a deployment can legitimately
configure a provider it has no key for yet.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/images/generations` | generate / edit / inpaint / vary an image |
| POST | `/v1/videos` | text-to-video, image-to-video, extend, edit |
| POST | `/v1/audio/speech` | text to speech |
| POST | `/v1/audio/transcriptions` | speech to text |
| GET | `/v1/media/{id}` | fetch or advance a job |
| POST | `/v1/media/{id}/cancel` | best-effort remote cancellation |
| GET | `/v1/media/models` | capabilities, parameter schema, price |

All take a client key, exactly like `/v1/completions`:

```bash
curl -sX POST http://127.0.0.1:8080/v1/images/generations \
  -H "Authorization: Bearer <client-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"flux","prompt":"a red panda astronaut","count":2,"size":"1024x1024"}'
```

**Not OpenAI-identical:** every endpoint takes JSON, never
`multipart/form-data`. Binary inputs are objects (below). Image edit, inpaint
and variation are the `mode` field on `/v1/images/generations` rather than
separate paths.

### Request fields

Shared by all four:

```jsonc
{
  "model": "flux",                    // alias, or "provider:model"
  "wait": true,                       // false returns 202 immediately
  "provider_options": { "fal": { … } },
  "unsupported_parameters": "error"   // "error" (default) | "warn_and_drop"
}
```

Task-specific fields map one-to-one onto octolib's request structs:

```jsonc
// /v1/images/generations
{ "prompt": "…", "mode": "generate|edit|inpaint|variation",
  "source_images": [Source], "mask": Source,
  "count": 2, "seed": 42, "size": "1024x1024",
  "negative_prompt": "…", "output_format": "png|jpeg|webp|svg" }

// /v1/videos
{ "prompt": "…", "mode": "text_to_video|image_to_video|reference_to_video|extend|edit",
  "first_frame": Source, "last_frame": Source,
  "reference_images": [Source], "source_video": Source,
  "count": 1, "seed": 42, "duration_secs": 8.0, "size": "16:9",
  "negative_prompt": "…", "output_format": "mp4|webm|mov" }

// /v1/audio/speech
{ "input": "…", "voice": "alloy", "language": "en", "instructions": "…",
  "speed": 1.0,
  "output": { "format": "mp3", "sample_rate_hz": 24000, "channels": 1 } }

// /v1/audio/transcriptions
{ "audio": Source, "language": "en", "prompt": "…",
  "timestamp_granularities": ["segment", "word"] }
```

`size` accepts `"1024x1024"` (exact dimensions) or `"16:9"` (aspect ratio).
Anything else is a 400.

### Binary inputs

```jsonc
{ "type": "url",    "url": "https://…", "media_type": "image/png" }
{ "type": "base64", "data": "iVBORw0…", "media_type": "image/png" }
```

`file`, `provider_file` and `object_storage` are rejected: a client-supplied
path would read the *server's* filesystem. Decoded base64 is checked against
`media.max_source_bytes` before anything reaches a provider, and inline payloads
are never written to the database — the stored request keeps the shape and the
byte count, not the bytes.

### Provider-native parameters

Portable fields (`prompt`, `count`, `seed`, `size`, `duration_secs`,
`negative_prompt`, `output_format`) behave the same everywhere. Anything else
goes under the provider's own namespace and is passed through verbatim:

```jsonc
"provider_options": {
  "fal": {
    "input": { "num_inference_steps": 28, "guidance_scale": 3.5 },
    "field_map": { "prompt": "text" },
    "webhook": "https://my-app.example/hooks/fal"
  }
}
```

`field_map` remaps a portable name onto whatever the endpoint actually calls it,
so portable `prompt` keeps working against an endpoint whose field is `text`.

Send every namespace at once when an alias spans providers — only the winning
candidate's namespace is forwarded, and the rest are dropped:

```jsonc
"provider_options": { "fal": {…}, "replicate": {…} }
```

`unsupported_parameters` decides what happens when the adapter cannot honour a
parameter: `"error"` fails before money is spent (correct for production);
`"warn_and_drop"` drops it and returns a warning (useful when an alias fans out
across providers with uneven support).

`provider_options.<provider>.cost_estimate` is rejected — pricing is resolved
server-side.

---

## Responses

One envelope for every task, terminal or not:

```jsonc
{
  "id": "med_01J…",
  "object": "media",
  "task": "text_to_image",
  "status": "succeeded",
  "model": "fal-ai/flux/dev",
  "provider": "fal",
  "progress": 1.0,
  "artifacts": [
    { "kind": "image", "media_type": "image/png",
      "source": { "type": "url", "value": "https://…" },
      "size_bytes": 812345, "expires_at": 1767225600 }
  ],
  "usage": {
    "line_items": [ { "unit": "images", "quantity": 2, "cost": 0.08 } ],
    "provider_reported_cost": 0.08,
    "estimated_cost": null,
    "cost": 0.08,
    "cost_source": "provider",
    "currency": "USD"
  },
  "warnings": [],
  "safety": { "status": "passed", "filtered_count": 0, "reasons": [] },
  "error": null,
  "created_at": 1767225000,
  "completed_at": 1767225042
}
```

Inline artifacts arrive as `{"type": "inline_base64", "value": "…"}`.
Transcriptions add a `result` object with `text`, `language`, `segments` and
`words`.

**`202 Accepted`** means the job is still running — same body, `status` of
`queued` or `running`, empty `artifacts`. You get it when `wait: false`, or when
the inline wait hit `server.upstream_timeout_secs`. A timeout is not a failure:
the job id is live and the work is not lost.

```bash
# submit without waiting
curl -sX POST http://127.0.0.1:8080/v1/videos \
  -H "Authorization: Bearer <client-key>" \
  -d '{"model":"veo","prompt":"a boat at dawn","duration_secs":8,"wait":false}'
# → 202 {"id":"med_01J…","status":"queued"}

# poll until terminal
curl -s http://127.0.0.1:8080/v1/media/med_01J… \
  -H "Authorization: Bearer <client-key>"
# → 200 {"status":"succeeded","artifacts":[…]}
```

Polling is what advances a job — there is no background worker. A job you never
poll stays `queued` and its cost is never recorded. A terminal record is served
from the database with no upstream call.

Records are scoped to the key that created them; another key's id returns 404,
not 403.

---

## Cost

Cost is resolved by octolib and recorded by OctoHub. `usage.cost` is the number
billed and `usage.cost_source` says where it came from:

| `cost_source` | Meaning |
|---|---|
| `provider` | The upstream returned a dollar amount. OpenRouter and Replicate do this. |
| `estimate` | Computed locally from octolib's reference rate table. |
| `unavailable` | Nothing could price it — `cost` is `null` and a `cost_unavailable` warning is attached. |

`null` is not zero. A request nothing could price is recorded as unpriced, never
as free, and shows up in `octohub_media_cost_unknown_total` rather than silently
dragging the spend total down.

Known gaps today: transcription is unpriced on every provider (Scribe bills
input-audio duration, which is not reported back), and OpenRouter's streamed
speech produces no usage at all.

`GET /v1/admin/usage` gains `media_count` and `total_cost` alongside the token
counts. `total_cost` sums completions, embeddings and media. Individual records
— including in-flight jobs — are readable at `GET /v1/admin/media`.

---

## Metrics

```
octohub_media_requests_total{task,model,provider,status}
octohub_media_duration_seconds{task,model,provider}
octohub_media_cost_usd_total{task,model,provider,source}
octohub_media_cost_unknown_total{task,model,provider}
```

`status` is `ok`, `accepted` (202, still running) or `error`. Per-key labels
follow the existing `metrics.per_key` switch.

There is no outstanding-jobs gauge. An accurate one would have to count
non-terminal rows in the database, and an in-process counter would be wrong the
moment a job is polled by a different replica or survives a restart. The
existing `octohub_requests_in_flight{route}` already covers requests currently
blocked on a media job.

---

## Errors

Standard OpenAI envelope, `{"error": {"message", "type"}}`.

| Status | `type` | When |
|---|---|---|
| 400 | `invalid_request_error` | bad JSON, bad `size`/`mode`, oversized upload |
| 400 | `unsupported_parameter` | the adapter cannot honour a parameter and you asked for strict |
| 400 | `media_task_not_supported` | the provider does not serve that model or task |
| 400 | `media_generation_failed` | upstream rejected on content policy or invalid input |
| 401 | — | missing or revoked key |
| 403 | `invalid_request_error` | the key's allow-list does not include this model |
| 404 | `not_found` | no such record for this key |
| 429 | `rate_limit_error` | provider rate window exhausted (`Retry-After` set) |
| 500 | `configuration_error` | the provider's API key is missing from the server environment |
| 502 | `upstream_auth_error`, `upstream_insufficient_credits`, `upstream_error` | upstream refused or misbehaved |

---

## Discovery

`GET /v1/media/models` reports what each configured candidate accepts, including
the adapter's own `provider_options` JSON Schema and the reference price:

```jsonc
{ "object": "list", "data": [ {
    "alias": "flux",
    "provider": "fal",
    "model": "fal-ai/flux/dev",
    "descriptor": { "tasks": ["text_to_image"], "execution": {…}, "parameters": {…},
                    "limits": {…}, "provider_options_schema": {…} },
    "price": { "unit": "compute_seconds", "usd_per_unit": 0.00056, "pattern": "fal-ai" }
} ] }
```

Many capability fields read `unknown` — the adapters genuinely cannot know every
endpoint's schema. That is why `provider_options` exists and why
`unsupported_parameters` is yours to choose per request.
