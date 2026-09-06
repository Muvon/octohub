# SPEC — Multimodal (media) support in OctoHub

Status: **implemented**, on a local `octolib` path dependency
(`octolib = { path = "../octolib" }` in `Cargo.toml`) pending a published
release. Both halves are written: the octolib pricing module described in §7.1,
and the OctoHub layer described everywhere else. See §11 for the file map.

OctoHub today proxies two octolib surfaces: `llm` (`/v1/completions`,
`/v1/chat/completions`) and `embedding` (`/v1/embeddings`). octolib 0.36 ships a
third — `octolib::media` — covering image generation/edit, video generation,
speech synthesis and transcription across `elevenlabs`, `fal`, `openrouter`,
`replicate` and `runway`.

This spec adds that surface to OctoHub with the same guarantees the existing
ones have: multi-tenant keys, per-key attribution, full request/response
logging, alias→provider routing, Prometheus metrics, and per-request cost.

The shape of the OctoHub layer follows from one decision: **anything that is
about models belongs in octolib, and anything that is about tenants belongs
here.** Routing grammar, capability descriptors, job lifecycle and pricing are
octolib's; keys, quotas, persistence and metrics are OctoHub's. That is why the
config below adds no media-specific syntax and why §7 is three paragraphs
instead of a subsystem.

---

## 1. Goals / non-goals

**Goals**

1. Four media tasks behind OctoHub's own API keys, aliased and routed like models.
2. A **usable parameter interface**: portable core parameters that behave the
   same everywhere, an escape hatch that lets any model receive any
   provider-native parameter, and discovery so a caller can find out which is
   which before spending money.
3. **Customizable providers**: per-request provider-native options, and
   per-deployment endpoint overrides in config.
4. **Cost recorded on every request that octolib can price**, including the
   providers that report no dollar amount of their own.
5. Long-running jobs (video, minutes) survive a process restart and are
   resumable — never lose a paid job.

**Non-goals (deliberate, revisit only on demand)**

- OctoHub does not become a blob store. Artifact bytes are never written to the
  database; URL artifacts are passed through, inline artifacts are base64'd into
  the response once and then only their metadata is retained.
- No background job worker. Jobs advance when a client polls (§4.3).
- No multipart/form-data endpoints. All requests are JSON (§3.1 deviation note).
- No streaming TTS (`SpeechSynthesisProvider::stream_speech`) in v1.

---

## 2. What octolib already provides

Understanding what is *already solved* determines how thin the OctoHub layer is.

| octolib concept | Type | What OctoHub does with it |
|---|---|---|
| Task traits | `ImageGenerationProvider`, `VideoGenerationProvider`, `SpeechSynthesisProvider`, `TranscriptionProvider` | Calls `submit_*` / `poll_*` / `cancel_*` directly |
| Provider selection | `MediaProviderFactory::get_*_provider_for_model("fal:fal-ai/flux/dev")` | Same `provider:model` grammar as `[models]` |
| Portable request core | `ImageGenerationRequest`, `VideoGenerationRequest`, `SpeechSynthesisRequest`, `TranscriptionRequest` | Mapped 1:1 from the wire body |
| Provider escape hatch | `ProviderOptions = BTreeMap<String, Value>` namespaced by provider | Taken from the client body, filtered to the winning provider (§5.3) |
| Transport controls | `RequestOptions` (timeouts, retries, byte caps, `unsupported_parameter_policy`) | Filled from server config + a few request fields |
| Async job identity | `JobHandle` — serializable, credential-free, carries `cost_estimate` | Persisted in the `media` table, resumed after restart |
| Job lifecycle | `Operation<T>` + `OperationStatus` (`Queued`/`Running`/`Succeeded`/`Failed`/`Cancelled`/`Expired`) | Persisted verbatim as the row status |
| Capability descriptor | `MediaModelDescriptor` incl. `provider_options_schema` | Served by `GET /v1/media/models` (§4.4) |
| Usage & cost | `MediaUsage { line_items, provider_reported_cost, estimated_cost, currency }` | Stored, aggregated, exported (§7) |
| Contract warnings | `ProviderWarning` (`UnsupportedParameterDropped`, `CostUnavailable`, `PartialOutput`, …) | Returned verbatim in the response |
| Failure classification | `FailureCategory` + `MediaError` | Mapped to HTTP status (§9) |
| Reference pricing | `media::reference_pricing` — provider-scoped, unit-aware rate table | Nothing; adapters apply it. OctoHub only reads `usage.cost` (§7) |

Two octolib behaviours drive design decisions downstream:

- **`shared::provider_options` rejects foreign namespaces.** Passing
  `{"fal": {...}, "replicate": {...}}` to the fal adapter is an error, not a
  no-op. Since one alias can map to several providers, OctoHub must **filter
  `provider_options` down to the selected candidate's namespace** before
  submitting. This is what makes a multi-provider alias usable at all (§5.3).
- **`ArtifactSource::Inline(Vec<u8>)` serialises as a JSON array of integers.**
  Never return octolib's serialisation directly — OctoHub re-encodes inline
  artifacts as base64 on the wire (§3.4).

---

## 3. Wire API

### 3.1 Endpoints

| Method | Path | Task |
|---|---|---|
| POST | `/v1/images/generations` | `text_to_image`, `image_edit`, `inpainting`, `image_variation` |
| POST | `/v1/videos` | `text_to_video`, `image_to_video`, `reference_to_video`, `video_extend`, `video_edit` |
| POST | `/v1/audio/speech` | `text_to_speech` |
| POST | `/v1/audio/transcriptions` | `speech_to_text` |
| GET | `/v1/media/{id}` | poll / fetch a record |
| POST | `/v1/media/{id}/cancel` | request remote cancellation |
| GET | `/v1/media/models` | capability + parameter discovery |

The four create paths are thin wrappers over one internal
`ProxyEngine::process_media(task, request, api_key)`; the route table picks the
task and the request struct. They use OpenAI's familiar paths on purpose —
`/v1/images/generations` in particular is wire-compatible enough that an OpenAI
SDK works unchanged for the common case.

**Deviations from OpenAI, stated once:** `/v1/audio/transcriptions` and image
edit/inpaint take **JSON**, not `multipart/form-data` — binary inputs are given
as a `MediaSource` object (§3.3). Adding multipart is a separate, additive
change and is not required by any OctoHub client today. Image edit/variation do
not get their own paths; they are `mode` on `/v1/images/generations`.

Auth, `X-Request-Id`, `X-Forwarded-For` handling and the master-key/DB-key split
are all unchanged — media endpoints are client endpoints, authenticated exactly
like `/v1/completions`.

### 3.2 Request body

Shared envelope across all four:

```jsonc
{
  "model": "flux",                    // alias from [media_models], or "provider:model"
  "wait": true,                       // default true; false = return 202 immediately
  "provider_options": { "fal": { ... } },
  "unsupported_parameters": "error",  // "error" (default) | "warn_and_drop"

  // ... task-specific portable fields below
}
```

Portable fields per task map 1:1 onto the octolib request structs — same names,
same semantics, so the octolib docs are the reference:

```jsonc
// POST /v1/images/generations
{ "prompt": "...", "mode": "generate|edit|inpaint|variation",
  "source_images": [MediaSource], "mask": MediaSource,
  "count": 2, "seed": 42, "size": "1024x1024" | "16:9",
  "negative_prompt": "...", "output_format": "png|jpeg|webp|svg" }

// POST /v1/videos
{ "prompt": "...", "mode": "text_to_video|image_to_video|reference_to_video|extend|edit",
  "first_frame": MediaSource, "last_frame": MediaSource,
  "reference_images": [MediaSource], "source_video": MediaSource,
  "count": 1, "seed": 42, "duration_secs": 8.0, "size": "16:9",
  "negative_prompt": "...", "output_format": "mp4|webm|mov" }

// POST /v1/audio/speech
{ "input": "...", "voice": "alloy", "language": "en", "instructions": "...",
  "speed": 1.0,
  "output": { "format": "mp3", "sample_rate_hz": 24000, "channels": 1 } }

// POST /v1/audio/transcriptions
{ "audio": MediaSource, "language": "en", "prompt": "...",
  "timestamp_granularities": ["segment", "word"] }
```

`size` accepts both `"1024x1024"` (→ `OutputGeometry::Dimensions`) and `"16:9"`
(→ `OutputGeometry::AspectRatio`), matching `OutputGeometry::as_api_string()` in
reverse. Anything else is a 400.

### 3.3 `MediaSource` on the wire

Serialised exactly as octolib's `MediaSource` (internally tagged `type`), with
one addition and one restriction:

```jsonc
{ "type": "url",     "url": "https://…", "media_type": "image/png" }
{ "type": "base64",  "data": "iVBORw0…",  "media_type": "image/png" }  // → Bytes
```

- `base64` is OctoHub's wire encoding for `MediaSource::Bytes`; octolib
  deliberately has no base64 variant, so the handler decodes it.
- `file`, `provider_file` and `object_storage` are **rejected with 400**. A
  client-supplied local path would read the *server's* filesystem — that is an
  SSRF/LFI class hole, not a feature.
- Decoded size is checked against `RequestOptions::max_source_bytes` before
  submission (config: `media.max_source_bytes`, default 20 MiB).

### 3.4 Response body

One shape for all tasks, terminal or not:

```jsonc
{
  "id": "med_01J…",
  "object": "media",
  "task": "text_to_image",
  "status": "succeeded",              // OperationStatus, verbatim
  "model": "fal-ai/flux/dev",         // resolved model
  "provider": "fal",
  "progress": 1.0,                    // null when the provider reports none
  "artifacts": [
    { "kind": "image", "media_type": "image/png",
      "source": { "type": "url", "value": "https://…" },
      "size_bytes": 812345, "dimensions": { "width": 1024, "height": 1024 },
      "duration_secs": null, "expires_at": 1767225600 }
  ],
  "result": { "text": "…", "language": "en", "segments": [], "words": [] },  // transcription only
  "usage": {
    "line_items": [ { "unit": "images", "quantity": 2, "cost": 0.05,
                      "description": "fal image outputs" } ],
    "provider_reported_cost": null,
    "estimated_cost": 0.05,
    "cost": 0.05,                     // best_available_cost(); the billed number
    "cost_source": "estimate",         // "provider" | "estimate" | "unavailable"
    "currency": "USD"
  },
  "warnings": [ { "code": "cost_unavailable", "message": "…", "parameter": null } ],
  "safety": { "status": "passed", "filtered_count": 0, "reasons": [] },
  "error": null,                      // GenerationFailure when status is failed
  "created_at": 1767225000,
  "completed_at": 1767225042
}
```

Inline artifacts are `{"type": "inline_base64", "value": "<base64>"}` — see the
`Vec<u8>` note in §2. `provider_file` and `object_storage` artifact sources are
passed through as-is; OctoHub does not resolve them.

`202 Accepted` is returned with the same body (`status: "queued" | "running"`,
`artifacts: []`) when `wait: false`, or when `wait: true` but the inline wait hit
`server.upstream_timeout_secs`. **A wait timeout is not an error** — octolib's
`MediaError::WaitTimeout` hands back a resumable handle, so the client gets 202
and a job id, and the paid job is not lost.

---

## 4. Lifecycle

### 4.1 Submit

```
authenticate (existing authenticate_client)
  → ensure_model_allowed(api_key, model)          // unchanged; media aliases are model names
  → acquire_owner_slot(api_key)                   // unchanged; shared with completions
  → resolve candidates from [media_models]
  → pick_admitted(...)                            // reuse: concurrency + RPM/RPD windows
  → build octolib request (portable core + merged provider_options, §5)
  → provider.submit_*(request)
  → PERSIST the row and JobHandle immediately     // before any waiting
  → if wait: poll loop bounded by upstream_timeout_secs
  → update row on every status change
```

Persisting **before** waiting is the point: from the moment money is committed
upstream, a restart can still find and resume the job.

### 4.2 Poll — `GET /v1/media/{id}`

Terminal row → return it as stored, no upstream call. Non-terminal row →
deserialise `JobHandle`, call `provider.poll_*(&handle)`, persist the new state,
return it. Records are scoped to the requesting `api_key_id`; another key's id is
a 404, not a 403.

### 4.3 Why no background worker

Jobs advance only when polled. A client that submits and never polls leaves a row
`queued` forever, and its cost is never recorded.

This is accepted for v1: it removes a scheduler, a worker pool, leader election
across replicas, and a whole class of double-billing bugs, in exchange for a
constraint clients already meet (anyone who wants their video must poll for it).
The upgrade path, when a real deployment needs it: one `tokio` task per process
polling non-terminal rows older than N seconds, guarded by a `SELECT … FOR
UPDATE SKIP LOCKED`-style claim so replicas do not duplicate work.

### 4.4 Discovery — `GET /v1/media/models`

Returns `MediaModelDescriptor` for every configured candidate — the machine-
readable answer to "what can this model actually take?":

```jsonc
{ "data": [ {
    "alias": "flux",
    "provider": "fal", "model": "fal-ai/flux/dev",
    "tasks": ["text_to_image"],
    "input_modalities": ["text", "image"], "output_modalities": ["image"],
    "execution": { "immediate": "unsupported", "persistent_jobs": "supported",
                   "cancellation": "supported", "progress": "unknown", … },
    "parameters": { "count": "unknown", "seed": "unknown", "mask": "unknown", … },
    "limits": { "max_input_bytes": null, "max_outputs": null, "supported_formats": [] },
    "provider_options_schema": { "type": "object", "properties": { … } },
    "price": { "unit": "compute_seconds", "usd_per_unit": 0.00056 }   // octolib reference table, null if unpriced
} ] }
```

`CapabilitySupport::Unknown` is honest and common — the adapters cannot know
every endpoint's schema. That is precisely why the escape hatch in §5 exists, and
why `unsupported_parameters` is a request-level choice rather than a server
policy.

---

## 5. The parameter model

The requirement — *"any model can get any parameters while keeping focus on the
required ones"* — is already octolib's design. OctoHub's job is to expose it
without flattening it.

### 5.1 Two tiers

**Portable core** (§3.2): fields whose meaning is identical on every provider.
Validated by OctoHub, mapped to typed octolib fields, and — critically — the
adapters translate them into each endpoint's native field names. `prompt`,
`count`, `seed`, `size`, `duration_secs`, `output_format`, `negative_prompt`.
This tier is what a client should reach for first, and what the SDK-compatible
path uses.

**Provider-native escape hatch**: `provider_options.<provider>` is passed to the
adapter verbatim, validated against that adapter's own options schema. Nothing is
whitelisted, nothing is silently dropped. For fal, this is where a model's real
input schema lives:

```jsonc
"provider_options": {
  "fal": {
    "input": { "num_inference_steps": 28, "guidance_scale": 3.5, "enable_safety_checker": false },
    "field_map": { "prompt": "text", "source_image": "image_url" },
    "webhook": "https://my-app.example/hooks/fal"
  }
}
```

`field_map` is the piece that makes "any model, any parameter" true in practice:
it remaps a portable semantic name onto whatever the endpoint actually calls it,
so a client can keep using portable `prompt` against an endpoint whose field is
`text`. Supplying the same field both portably and inside `input` is an error
from octolib — surfaced as 400, not resolved by precedence.

### 5.2 Strictness is the caller's choice

`unsupported_parameters` maps to `RequestOptions::unsupported_parameter_policy`:

- `"error"` (default) — a parameter the adapter cannot honour fails the request
  before any money is spent. Correct for production.
- `"warn_and_drop"` — the parameter is dropped and a
  `ProviderWarning { code: "unsupported_parameter_dropped", parameter }` is
  returned. Correct for exploration and for aliases that fan out across providers
  with uneven support.

### 5.3 Merging, and the namespace filter

Effective options for the selected candidate `(provider, model)` are just the
request's own namespace for that provider:

```
request provider_options[provider]
  → ProviderOptions { provider: … }             // single namespace, always
```

There is no config-side options layer: `[media_models]` holds mirrors and
nothing else (§6.1), so what the client sends is what the adapter gets.

Namespaces for *other* providers in the request are **dropped, not rejected** —
octolib's adapter would reject them, and dropping is what makes a multi-provider
alias work: a client sends `{"fal": {...}, "replicate": {...}}` once and the
right half is used whichever candidate wins. A namespace naming no configured
provider at all is a 400 (typo protection).

`cost_estimate` inside `provider_options` is **rejected** with a 400. octolib's
adapters resolve the rate themselves from the reference table, and a caller who
could pass their own rate could declare their own generation free.

---

## 6. Configuration

Media reuses the existing config shapes verbatim. There is no media-specific
routing, pricing or limiting syntax to learn.

### 6.1 `[media_models]` — mirrors, exactly like `[models]`

```toml
[media_models]
"flux" = ["fal:fal-ai/flux/dev", "replicate:black-forest-labs/flux-1.1-pro"]
"veo"  = ["openrouter:google/veo-3.1", "replicate:google/veo-3.1"]
"tts"  = ["elevenlabs:eleven_flash_v2_5"]
```

`HashMap<String, Vec<String>>`, resolved by the same
`Config::candidates_from_map()` the LLM and embedding maps use: random start
index for load spread, ordered fallthrough for failover, and a literal
`provider:model` in the request bypassing the map as one fixed candidate.

Nothing else lives here. Prices come from octolib (§7); per-model provider
tweaks are request-level `provider_options` (§5).

### 6.2 `[providers.<name>]` — unchanged

Already keyed by octolib's provider name, so the media adapters slot in with no
new config:

```toml
[providers.fal]
concurrency = 8
requests_per_minute = 60
requests_per_day = 5000
```

`ProviderLimiter`, `ProviderRateTracker` and `ProviderHealth` are reused as-is.
The one difference is `tokens_per_minute` / `tokens_per_day`: media providers
report no tokens, so those windows are simply never fed and never bind. That is
stated rather than special-cased — `record_tokens` is just not called on the
media path.

### 6.3 `[media]` — server knobs

```toml
[media]
max_source_bytes   = 20971520   # 20 MiB, → RequestOptions::max_source_bytes
max_response_bytes = 104857600  # 100 MiB
polling_interval_secs = 2
submit_timeout_secs   = 120
```

Wait deadline and provider queue wait reuse the existing
`server.upstream_timeout_secs` / `server.provider_queue_timeout_secs`.

### 6.4 Custom / self-hosted providers

```toml
[media_providers.fal]
api_base = "https://fal.internal.example"     # exported as FAL_API_URL at startup
```

Adapter name → endpoint override, keyed by octolib's provider name
(`elevenlabs`, `fal`, `openrouter`, `replicate`, `runway`). Applied by
`std::env::set_var` at startup, the same mechanism `main.rs` already uses for
`OPENROUTER_APP_TITLE`, mapping onto the adapters' existing `FAL_API_URL` /
`OPENROUTER_MEDIA_API_URL` / `REPLICATE_API_URL` / `ELEVENLABS_API_URL` /
`RUNWAY_API_URL` overrides. Zero octolib change.

**Known limitation, stated rather than worked around:** octolib reads the base
URL and API key from process-global env vars, so there can be exactly one custom
endpoint *per adapter* per deployment. Multiple fal endpoints in one process
would need an octolib instance constructor
(`FalMediaProvider::with_endpoint(base, key_env)`); out of scope until a
deployment needs it.

### 6.5 Startup validation (fail fast, like `validate_auto`)

At config load, reject:

1. A candidate whose provider is not in `MediaProviderFactory::supported_providers()`.
2. A candidate string not in `provider:model` form.
3. An alias colliding with a `[models]` or `[embedding_models]` alias.

No pricing validation: pricing is octolib's, and an unpriced model is a warning
on the response, not a config error (§7.3).

---

## 7. Pricing

**Pricing lives in octolib, not here** — same as LLM completions and embeddings.
OctoHub reads `usage.cost` off the result and stores it. That is the whole
integration.

### 7.1 What octolib does (implemented — `octolib::media::reference_pricing`)

The media counterpart of `llm::reference_models`, built to the same pattern:

```rust
pub fn get_reference_pricing(provider: &str, model: &str) -> Option<MediaModelPricing>;
pub fn reference_cost_estimate(
    provider: &str,
    model: &str,
    known_quantity: Option<(UsageUnit, f64)>,
) -> Option<CostEstimate>;
```

A `const REFERENCE_MEDIA_MODELS` table of `(provider, pattern, unit,
usd_per_unit)`, substring-matched through the same
`sanitize_model_name(normalize_model_name(model))` the LLM table uses, with the
same first-match-wins ordering rule and the same reachability test guarding it.

Two things differ from the LLM table, both forced by the domain:

- **Rates carry a unit.** LLM pricing is always per 1M tokens; media bills per
  image, video-second, character or compute-second, so the unit is part of the
  entry and is matched against what the adapter actually observed.
- **Entries are keyed by provider first.** `flux` is per-image on Replicate and
  per-GPU-second on fal; `veo` is reachable through three providers at three
  prices. A bare model-name table would cross-match and misbill.

`known_quantity` is the billable amount already settled at submit — requested
video seconds, input characters. It is attached only when its unit is the one
the model bills on; a unit only the upstream can report (compute-seconds, output
images) stays open so the adapter's existing `rate.quantity.or(<metric>)`
fallback fills it. An unknown quantity stays `None` rather than becoming zero,
so a missing duration never bills as a free video.

Each adapter resolves its rate at submit through one shared helper:

```rust
shared::resolved_cost_estimate(PROVIDER, &request.model, options.cost_estimate, known_quantity)
```

wired at every submit site: fal ×4, Replicate ×4, Runway ×2, ElevenLabs ×2.
OpenRouter is left alone — it reports real cost on the paths that have usage.

A caller-supplied `CostEstimate` always wins; otherwise the table's rate is
used. Because the result is an ordinary `CostEstimate`, it rides into
`JobHandle.cost_estimate` at submit and all the existing poll-time math works
unchanged — and a job resumed after a restart is priced at the rate it was
submitted under, not whatever the table says later.

### 7.2 Cost precedence

1. `usage.provider_reported_cost` — an actual upstream dollar amount. Only
   OpenRouter (`/generation` → `total_cost`) and Replicate (`metrics.cost`)
   ever produce one.
2. `usage.estimated_cost` — `quantity × usd_per_unit`, from either the caller's
   rate or the reference table. **Never** presented as provider-reported.
3. Neither → `WarningCode::CostUnavailable` on the response and `cost` is null.

`usage.cost` in OctoHub's response is `MediaUsage::best_available_cost()`, and
`cost_source` (`"provider" | "estimate" | "unavailable"`) says which, because a
billed number and a computed one are different claims.

### 7.3 Coverage gaps, stated honestly

- Every rate in the seed table is tagged `estimate` and must be verified against
  the provider's published pricing page before it bills a customer — the same
  caveat the embedding table carries.
- **Transcription is unpriced on every provider.** ElevenLabs Scribe bills input
  audio duration, which is not known at submit and is not reported back, so no
  quantity exists to multiply.
- **`Megapixels` is deliberately absent from the table.** No adapter reports
  output dimensions (`MediaArtifact.dimensions` is `None` at every construction
  site), so a per-megapixel rate could never fire and would read as coverage
  that does not exist.
- **OpenRouter is not in the table.** It reports real cost on most paths;
  streamed speech (`stream_speech`) produces no `MediaUsage` at all and is
  unpriced.

### 7.4 What OctoHub does

Stores `usage.cost` on the media row and sums it. `UsageRow` gains:

```rust
pub media_count: u64,
pub total_cost: f64,      // SUM(json_extract(usage, '$.cost'))
```

summed across completions, embeddings and media in one query shape. Completions
already store `usage.cost` and never aggregate it — this closes that gap too.
`GET /v1/admin/usage` returns both new fields.

### 7.5 Metrics

```
octohub_media_requests_total{task,model,provider,status}
octohub_media_duration_seconds{task,model,provider}       # histogram, upstream only
octohub_media_cost_usd{task,model,provider,source}         # gauge, cumulative; source=provider|estimate
octohub_media_cost_unknown_total{task,model,provider}
```

Per-key labels follow the existing `metrics.per_key` switch.

No outstanding-jobs gauge: an accurate one needs a database count of
non-terminal rows, and an in-process counter is wrong across replicas and
restarts. `octohub_requests_in_flight{route}` already covers requests blocked on
a media job.

---

## 8. Storage

One table for both in-flight jobs and finished records — a job *is* the record,
at an earlier status; splitting them would mean a join and a migration between
two tables on every completion.

```sql
CREATE TABLE media (
  id              TEXT PRIMARY KEY,       -- "med_<uuid>"
  api_key_id      INTEGER NOT NULL REFERENCES api_keys(id),
  task            TEXT NOT NULL,          -- MediaTask
  status          TEXT NOT NULL,          -- OperationStatus
  input_model     TEXT NOT NULL,          -- alias as sent by the client
  resolved_model  TEXT NOT NULL,
  provider        TEXT NOT NULL,
  request         TEXT NOT NULL,          -- client body, media bytes redacted
  handle          TEXT,                   -- serialized JobHandle; NULL when terminal
  result          TEXT,                   -- artifacts + task result metadata
  usage           TEXT,                   -- MediaUsage incl. cost + cost_source
  warnings        TEXT,
  error           TEXT,                   -- GenerationFailure
  created_at      INTEGER NOT NULL,
  completed_at    INTEGER
);
CREATE INDEX idx_media_api_key ON media(api_key_id);
CREATE INDEX idx_media_created ON media(created_at);
CREATE INDEX idx_media_status  ON media(status);
```

`request` is stored with binary inputs replaced by
`{"inline_media_omitted": true, "bytes": N, "media_type": "…"}` — mirroring
octolib's own `sanitize_media_value`. Base64 payloads must not reach the
database; they would multiply row size by megabytes per request for no
observability gain.

Trait additions on `Storage` (all three backends):

```rust
fn store_media(&self, record: &StoredMedia) -> Result<()>;
fn update_media(&self, record: &StoredMedia) -> Result<()>;
fn get_media(&self, id: &str, api_key_id: i64) -> Result<Option<StoredMedia>>;
fn list_media(&self, filter: &ListFilter) -> Result<Vec<StoredMedia>>;   // GET /v1/admin/media
```

Table creation follows the existing `CREATE TABLE IF NOT EXISTS` + `ensure_column`
additive-migration pattern; no versioned migration framework.

---

## 9. Errors

`MediaError` → HTTP, following `classify_engine_error`'s existing shape. Error
bodies keep the OpenAI envelope `{"error": {"message", "type"}}`.

| `MediaError` | Status | `error.type` |
|---|---|---|
| `InvalidModelFormat`, `InvalidRequest`, `SourceTooLarge` | 400 | `invalid_request_error` |
| `UnsupportedParameter` | 400 | `unsupported_parameter` |
| `UnsupportedProvider`, `UnsupportedTask` | 400 | `media_task_not_supported` |
| `MissingApiKey` | 500 | `configuration_error` |
| `Authentication`, `Permission` | 502 | `upstream_auth_error` |
| `InsufficientCredits` | 502 | `upstream_insufficient_credits` |
| `RateLimit` | 429 + `Retry-After` | `rate_limit_error` |
| `Api { status }` | 4xx passthrough, else 502 | `upstream_error` |
| `InvalidResponse`, `ResponseTooLarge`, `ArtifactTooLarge` | 502 | `upstream_error` |
| `RemoteFailure(f)` | 502 | `media_generation_failed` (+ `f.category`) |
| `WaitTimeout` | **202**, not an error | — |
| `LocalWaitCancelled` | **202**, not an error | — |
| `WrongJobHandle` | 500 | `internal_error` |

`FailureCategory::ContentPolicy` inside a `RemoteFailure` maps to 400 —
the request is at fault and retrying it upstream is pointless.

Model-not-allowed and owner-concurrency reuse the existing
`MODEL_FORBIDDEN_MARKER` / `OWNER_LIMIT_MARKER` paths unchanged.

---

## 10. Reuse of existing machinery

| Concern | Decision |
|---|---|
| Client auth, key hashing | Unchanged |
| `allowed_models` | Unchanged — media aliases are model names, `is_model_allowed` already covers them |
| Owner concurrency (`acquire_owner_slot`) | Shared budget with completions/embeddings — one tenant, one in-flight budget |
| `[providers.<name>].concurrency` | Applies; a media submit holds a permit for the submit call only, **not** for the whole job (a 4-minute video must not hold a provider permit) |
| `requests_per_minute` / `requests_per_day` | Apply — request-counted, provider-agnostic |
| `tokens_per_minute` / `tokens_per_day` | Do **not** apply to media; most media providers report no tokens. Skipped rather than approximated |
| Failover (`failover_on_error`) | Applies at **submit** only. Once a job is accepted upstream it is never failed over — that would double-bill |
| Provider cooldown / health | Applies, same `is_provider_fault` classification |
| `auto` virtual model | Not extended to media in v1 |
| SIGHUP reload | `[media_models]`/`[media]` reload with the rest; in-flight jobs keep the rate frozen in their `JobHandle` |

---

## 11. Implementation plan

Ordered so each step compiles and is independently testable.

1. **`src/config.rs`** — `media_models: HashMap<String, Vec<String>>` +
   `media_model_candidates()` (a third caller of the existing
   `candidates_from_map`), `MediaConfig` for `[media]`, `media_providers` for
   the endpoint overrides, and `validate_media()` (§6.5).
   Tests: candidate rotation, `provider:model` bypass, alias collision rejected.
2. **`src/storage/`** — `StoredMedia`, four trait methods, table DDL, `UsageRow`
   extension, `get_usage` cost aggregation, across sqlite/mysql/postgres.
3. **`src/api/types.rs`** — request/response structs, `MediaSource` wire codec
   (incl. base64 decode + `file`/`provider_file`/`object_storage` rejection),
   `size` ↔ `OutputGeometry` parsing, inline-artifact base64 encoding.
4. **`src/proxy/media.rs`** *(new)* — `process_media`: candidate selection
   (reusing `pick_admitted`), namespace filter (§5.3), submit → persist →
   optional bounded wait → persist; `poll_media`, `cancel_media`,
   `media_models`. Pricing is octolib's — this layer only reads
   `usage.best_available_cost()`. Errors mapped per §9.
5. **`src/api/handler.rs`** — four create handlers over `process_media`, plus
   poll/cancel/models; `classify_media_error`.
6. **`src/main.rs`** — routes, `classify_route` labels, `[media_providers]` env
   export at startup.
7. **`src/metrics.rs`** — the five media metrics.
8. **Docs** — promote this spec to `doc/11-media.md`, extend
   `doc/03-configuration.md`, `doc/05-api-client.md`, `doc/06-api-admin.md`
   (usage cost fields), and the `doc/README.md` index.

Test focus, minimum bar: config validation; the options merge/filter (a
two-provider alias must submit exactly one namespace); the `MediaSource` codec
(base64 round-trip, oversize rejection, `file` rejection); cost recording
(`cost_source` labels provider-reported vs estimate, and an unpriced result
stores `null` rather than `0`);
wait-timeout → 202-with-resumable-handle; and a poll that transitions a stored
row from `running` to `succeeded` and records cost exactly once.

---

## 12. Open questions

1. **Artifact retention.** URLs from fal/Replicate expire (hours to days).
   OctoHub currently passes them through and stores metadata only. If clients
   need durable artifacts, the answer is an object-storage sink
   (`[media].artifact_store`) rather than database blobs — deliberately not
   specified here.
2. **`auto` for media.** The purpose-routing chain (`X-Model-Purpose` → owner map
   → `[auto]`) could extend to media aliases. Left out until a caller asks.
3. **Per-key media spend caps.** The cost data this spec adds makes a
   `[keys].max_daily_cost_usd` limit possible for the first time. Separate
   feature; noted so the storage shape does not preclude it (it does not — cost
   is on every row, indexed by `api_key_id` and `created_at`).
