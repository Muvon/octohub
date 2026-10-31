# 03 — Configuration

OctoHub's runtime configuration is a single TOML file (default:
`octohub.toml` in the current working directory), with environment
variables overriding individual values.

## The full config file

```toml
# ─── Server ───────────────────────────────────────────────────────────
[server]
host = "127.0.0.1"             # bind address
port = 8080                    # bind port
api_key = "your-master-secret" # master key for /v1/admin/* (empty = admin disabled)
db_url = "sqlite://octohub.db" # database DSN
trust_forwarded_for = false    # honor X-Forwarded-For (only behind a trusted proxy)

# ─── Logging ──────────────────────────────────────────────────────────
[logging]
format = "auto"                # "auto" (default) | "pretty" | "json"
level = "info"                 # overrides RUST_LOG; absent = RUST_LOG or "info"

# ─── Metrics ──────────────────────────────────────────────────────────
[metrics]
enabled = true                 # expose Prometheus endpoint
bind = "127.0.0.1:9090"        # scrape address (separate from main API)
per_key = false                # add api_key_id label to completion/embedding metrics

# ─── Model mappings ───────────────────────────────────────────────────
[models]
"minimax-m2.7" = ["minimax:minimax-m2.7"]
"my-fast"      = ["openai:gpt-5-nano", "groq:llama-3.1-8b-instant"]  # random pick

[embedding_models]
"voyage"        = ["voyage:voyage-4"]

# ─── Per-provider concurrency ────────────────────────────────────────
[providers.ollama]
concurrency = 5                # max in-flight requests to ollama (queued beyond this)

# [providers.openai]
# concurrency = 32             # uncomment to cap
```

The default file ships in the repo root as [`octohub.toml`](../../octohub.toml).

## Top-level keys

| Key | Type | Default | Description |
|---|---|---|---|
| `[server]` | table | see below | Server bind, auth, database |
| `[logging]` | table | `format="auto"`, `level=null` | Log output and filter |
| `[metrics]` | table | `enabled=true`, `bind="127.0.0.1:9090"`, `per_key=false` | Prometheus endpoint |
| `[models]` | table | `{}` | Short model name → list of `provider:model` strings |
| `[embedding_models]` | table | `{}` | Same, for embedding calls |
| `[evaluation_models]` | table | `{}` | Same, for `POST /v1/evaluations` (`typesafe`, `cloudflare`) |
| `[providers.<name>]` | table | none | Per-provider tuning: `concurrency` + rate windows |

## `[server]`

Defined in `src/config.rs:83`. All keys optional except as noted.

| Key | Default | Description |
|---|---|---|
| `host` | `"127.0.0.1"` | Bind address. Use `"0.0.0.0"` to accept on all interfaces. |
| `port` | `8080` | Bind port. |
| `api_key` | `""` | Master key. Empty string = admin endpoints return 401, server logs a warning at startup (`src/main.rs:63`). |
| `db_url` | `"sqlite://octohub.db"` | DSN. Accepts `sqlite://…`, `mysql://…`, `postgres://…`, or a bare path (treated as SQLite). |
| `trust_forwarded_for` | `false` | Honor `Forwarded` / `X-Forwarded-For` for the `remote` log field. **Only enable behind a trusted reverse proxy** — see [08 — Deployment](./08-deployment.md#x-forwarded-for). |
| `provider_queue_timeout_secs` | `60` | Maximum wait for a provider concurrency permit. Exceeding it returns `503`. |
| `owner_queue_timeout_secs` | `30` | Maximum wait for a slot in the key owner's shared budget (`owner_concurrency`). Exceeding it returns `429`. `0` = never queue. |
| `upstream_timeout_secs` | `360` | Maximum duration of the complete provider operation, including octolib retries. Exceeding it returns `504`. |
| `upstream_stream_timeout_secs` | `0` | Maximum silence on a streaming provider call (time to headers, then between chunks); any chunk resets it. Fails a hung lane over fast while `upstream_timeout_secs` stays the total bound. `0` = off. Non-streaming providers ignore it. |
| `failover_on_error` | `false` | Re-route provider-side failures (timeout, connect, 401/403 key or billing rejection, 404, 429, 5xx) to the next candidate provider of the model alias. A 400 never fails over. |
| `provider_error_cooldown_secs` | `0` | After 3 consecutive provider-side failures, deprioritize the provider for this many seconds (used only when no healthy candidate admits). `0` = off. |
| `shutdown_timeout_secs` | `900` | Drain bound on SIGTERM/SIGINT: in-flight requests get this long to finish, then the process exits non-zero. Must be > 0 and ≥ the longest admitted request (`owner_queue + attempts × (provider_queue + upstream)`); the container stop timeout must exceed it. See [08 — Deployment](08-deployment.md#shutdown-and-restarts). |

The legacy alias `db_path` is also accepted and treated as `db_url`
(`src/config.rs:98`).

### WebSocket completion lifecycle

The authenticated HTTP/1.1 `GET /v1/completions` WebSocket transport uses
the same completion configuration as `POST`; there are no new WebSocket
configuration fields. Each request message reads a fresh configuration
snapshot, so a long-lived connection does not pin routing to its handshake
configuration. The key is reauthenticated for every message, applying
revocation and model/owner edits to subsequent requests. Purpose and
attribution headers remain those supplied at the handshake.

The listener bounds acceptance through the first request headers, including
HTTP protocol detection, to **10 seconds**. Every HTTP/1 request also has a
**10-second** header-read limit; these limits apply to HTTP routes as well
as WebSocket handshakes, not to bodies or handler duration. Handshake
authentication, HTTP upgrade completion (`OnUpgrade`), and transport
send/flush/close operations each have fixed **10-second** bounds. They do not
replace the existing owner queue, provider queue, or upstream timeouts. Incoming frames and complete
request messages are capped at **16 MiB**. Each single outgoing serialized
envelope, including status, headers, and body, is capped at **64 MiB**;
exceeding either cap closes with `1009`. The
[client protocol](05-api-client.md#get-v1completions--websocket) describes
close codes and sequencing.

`server.shutdown_timeout_secs` also bounds WebSocket drain on
SIGTERM/SIGINT: idle sockets close with `1001`, busy requests may finish
and send their final reply before closing with `1001`, and overdue work
is aborted at the drain deadline. No new requests are accepted during
drain. Closing a caller's socket cancels its local future and releases
permits, but cannot undo an issued provider call or a running blocking
database write.

## `[logging]`

Defined in `src/config.rs:20`.

| Key | Default | Description |
|---|---|---|
| `format` | `"auto"` | `auto` picks pretty when stdout is a TTY, JSON otherwise. `pretty` and `json` are explicit overrides. |
| `level` | unset | When set, takes precedence over `RUST_LOG`. Otherwise the level comes from `RUST_LOG`, falling back to `"info"`. |

See [07 — Observability](./07-observability.md#logging) for sample output
and the full list of request-span fields.

## `[metrics]`

Defined in `src/config.rs:28`.

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | When false, the metrics endpoint is not started. |
| `bind` | `"127.0.0.1:9090"` | Bind for the Prometheus scrape endpoint. |
| `per_key` | `false` | When true, `octohub_completions_total` and `octohub_embeddings_total` gain an `api_key_id` label. High-cardinality — only enable if your scrape storage can handle it. |

The full list of exposed metrics is in
[07 — Observability](./07-observability.md#metrics).

## `[models]` and `[embedding_models]`

Both are `HashMap<String, Vec<String>>` (`src/config.rs:66`).
Each value is a list of `provider:model` strings. When a request
arrives, OctoHub starts from a random entry and takes the first whose
provider [rate windows](#providersname) admit the request — that's the
simple load-balancing mechanism. Requests continuing a chain
(`previous_completion_id`) prefer the provider that served the previous
turn (keeps provider-side prompt caches warm), falling back to the
others only when its rate windows are exhausted.

If a request's `model` field is *already* in `provider:model` form (it
contains a colon), the config map is bypassed and that exact upstream is
used. See `Config::model_candidates` in `src/config.rs`.

### Examples

Single-provider mapping:

```toml
[models]
"minimax-m2.7" = ["minimax:minimax-m2.7"]
```

Multi-provider with random selection:

```toml
[models]
"fast" = ["openai:gpt-5-nano", "groq:llama-3.1-8b-instant", "anthropic:claude-haiku-4-5"]
```

Bypass the map entirely by calling the API with a `provider:model` model
field. The above config lets clients do either.

### Embedding models

The `[embedding_models]` table works the same way for
`POST /v1/embeddings`. An alias in `[embedding_models]` does not work
for `/v1/completions` and vice versa.

### Evaluation models

The `[evaluation_models]` table works the same way for
`POST /v1/evaluations`. Entries name octolib's evaluation adapters:
`typesafe:jev-latest` (direct, `TYPESAFE_API_KEY`) or
`cloudflare:typesafe/jev` (AI Gateway unified billing, the Cloudflare
key and account id). Aliases must not collide with `[models]`,
`[embedding_models]` or `[media_models]`; unknown providers, malformed
entries and empty lists are rejected at load, like `[media_models]`.

```toml
[evaluation_models]
"jev" = ["typesafe:jev-latest", "cloudflare:typesafe/jev"]

[auto]
default = "glm"      # [auto] needs its chat default either way
evaluation = "jev"   # "model": "auto" on /v1/evaluations
```

`[auto].evaluation` is the one `[auto]` value that names an
`[evaluation_models]` alias instead of a `[models]` one; the load check
enforces both directions.

## `[providers.<name>]`

Defined in `src/config.rs:108`. Keyed by the **lowercase** provider name
as octolib returns it (`ollama`, `openai`, `anthropic`, `minimax`,
`deepseek`, …). Lookup is case-insensitive at runtime
(`src/proxy/limiter.rs:50`).

All keys are optional; **unset or `0` = unlimited**.

| Key | Type | Default | Description |
|---|---|---|---|
| `concurrency` | `u32` | unlimited | Max in-flight requests to this provider. Beyond the limit, callers queue (the HTTP request blocks) until a permit frees up. |
| `requests_per_minute` | `u64` | unlimited | Max requests per fixed 60s window (provider "RPM"). |
| `tokens_per_minute` | `u64` | unlimited | Max tokens per fixed 60s window (provider "TPM"). Counted from provider-reported usage (input + output) after each response. |
| `requests_per_day` | `u64` | unlimited | Max requests per UTC day (provider "RPD"). |
| `tokens_per_day` | `u64` | unlimited | Max tokens per UTC day (provider "TPD"). |

Unlike `concurrency` (which queues), an exhausted rate window **skips**
that provider when the model alias has other candidates, and returns
`429` with a `Retry-After` header when every candidate is exhausted —
you can't park a request until a day window resets. Windows are fixed,
not rolling, and counters live in memory (reset on restart; they do
survive SIGHUP reloads). Configure values below your real account
limits to leave headroom. Grounded per-provider baselines with doc
links ship commented-out in [`octohub.toml`](../../octohub.toml) — see
also [09 — Providers](./09-providers.md#per-provider-tuning).

### When to set this

- **Local providers** (ollama, a Modal-hosted vLLM, a custom `local`
  provider) — you almost always want a cap. They're usually GPU-bound
  and a few concurrent requests can saturate the hardware.
- **Hosted APIs with rate limits** (Anthropic, OpenAI) — set this just
  below the published rate limit to smooth bursts.
- **Unlimited providers** — simply omit the section. The provider runs
  with no throttling.

When unset, the metric `octohub_provider_permits_available` is not
emitted for that provider, and the gauge loop in `src/metrics.rs:164`
silently skips it.

## Database

OctoHub supports three backends. The DSN scheme is detected by
`storage::from_url` at `src/storage/mod.rs:202`.

| Backend | DSN form | Example |
|---|---|---|
| SQLite (default) | `sqlite://path` or bare path | `sqlite://octohub.db` |
| MySQL | `mysql://user:pass@host:port/db` | `mysql://root:secret@localhost:3306/octohub` |
| PostgreSQL | `postgres://user:pass@host:port/db` (also `postgresql://`) | `postgres://postgres:secret@localhost:5432/octohub` |

The schema is created automatically on first connect — see
`src/storage/sqlite.rs:27` for the `CREATE TABLE IF NOT EXISTS`
statements, and the matching `ensure_mysql_schema` / `ensure_pg_schema`
in the respective files.

For MySQL and PostgreSQL, the user needs `CREATE`, `ALTER`, `INSERT`,
`SELECT`, `UPDATE`, and `DELETE` on the target database. The proxy never
runs `DROP` or destructive migrations.

## Environment variables

All environment variables are listed here. Precedence is
**CLI flag > environment > config file > default** for the corresponding
key, with one exception: the env-only fallback when no config file is
loaded (`src/config.rs:190`).

| Variable | Overrides | Notes |
|---|---|---|
| `OCTOHUB_CREDENTIALS_KEY` | Startup only; no TOML field | Base64 of 32 bytes for owner credential encryption; see below. |
| `OCTOHUB_MASTER_KEY` | `[server].api_key` | Master admin key. |
| `OCTOHUB_DB_URL` | `[server].db_url` | Useful in containerized deployments. |
| `OCTOHUB_PROVIDER_QUEUE_TIMEOUT_SECS` | `[server].provider_queue_timeout_secs` | Overrides the provider queue deadline. |
| `OCTOHUB_OWNER_QUEUE_TIMEOUT_SECS` | `[server].owner_queue_timeout_secs` | Overrides the owner queue deadline. |
| `OCTOHUB_UPSTREAM_TIMEOUT_SECS` | `[server].upstream_timeout_secs` | Overrides the complete upstream operation deadline. |
| `OCTOHUB_UPSTREAM_STREAM_TIMEOUT_SECS` | `[server].upstream_stream_timeout_secs` | Overrides the streaming silence limit. |
| `OCTOHUB_FAILOVER_ON_ERROR` | `[server].failover_on_error` | `true`/`1` enables provider failover. |
| `OCTOHUB_PROVIDER_ERROR_COOLDOWN_SECS` | `[server].provider_error_cooldown_secs` | Overrides the provider cooldown duration. |
| `OCTOHUB_SHUTDOWN_TIMEOUT_SECS` | `[server].shutdown_timeout_secs` | Overrides the drain bound. Not whole seconds, or `0`, is a startup error. |
| `OCTOHUB_HOST` | `[server].host` | Overrides the file too. Set but blank is a startup error. |
| `OCTOHUB_PORT` | `[server].port` | Overrides the file too. Not a port number (0-65535) is a startup error. |
| `OCTOHUB_LOG_FORMAT` | `[logging].format` | `auto`, `pretty`, or `json`. |
| `OCTOHUB_LOG_LEVEL` | `[logging].level` | `trace`, `debug`, `info`, `warn`, `error`. Takes precedence over `RUST_LOG`. |
| `OCTOHUB_METRICS_ENABLED` | `[metrics].enabled` | `true`/`false`/`1`/`0`. |
| `OCTOHUB_METRICS_BIND` | `[metrics].bind` | `HOST:PORT`. |
| `RUST_LOG` | log filter (fallback) | Standard `tracing_subscriber::EnvFilter` syntax. |
| `OCTOHUB_URL` | Used by `octohub-admin.sh` | Base URL for the admin script. |
| `OCTOHUB_SERVER_HOST` | Used by `octohub-admin.sh` | Server host (default `127.0.0.1`). |
| `OCTOHUB_SERVER_PORT` | Used by `octohub-admin.sh` | Server port (default `8080`). |

The admin script's env vars are unrelated to the server's own config —
they're convenience defaults for the shell wrapper.

## Full annotated example

```toml
# octohub.toml

[server]
host = "0.0.0.0"
port = 8080
api_key = "lkAbaTWoBs6HLVG2Kf46"   # generate something like: openssl rand -base64 24
db_url = "sqlite://octohub.db"     # default; see [Database] above for MySQL/Postgres
trust_forwarded_for = true         # only if behind nginx/Caddy/Envoy

[logging]
format = "json"      # pipe to journald / vector / fluentbit
level = "info"

[metrics]
enabled = true
bind = "127.0.0.1:9090"
per_key = false

[models]
# Single-provider
"minimax-m2.7" = ["minimax:minimax-m2.7"]

# Multi-provider (random pick; enable [server].failover_on_error to retry
# provider-side failures on the next candidate)
"my-fast" = ["openai:gpt-5-nano", "groq:llama-3.1-8b-instant"]

# Self-hosted with a local model
"local-llama" = ["ollama:llama3.3:70b-instruct-q5_K_M"]

# Primary + fallback instead of a random pick (see 09 — Providers, "Ordered aliases")
"my-primary" = ["anthropic:claude-opus-4-1", "openai:gpt-5"]

[routing.models]
"my-primary" = "ordered"

[embedding_models]
"voyage" = ["voyage:voyage-4"]

[providers.ollama]
concurrency = 4   # GPU is busy; queue the rest

[providers.openai]
concurrency = 30  # stay under the 60-rps published limit
```


---

## `[media_models]`

Media aliases map to lists of `provider:model` mirrors, rotating with fallthrough
unless `[routing.models].<alias> = "ordered"` selects the first available lane
and enables submit failover even with `failover_on_error` off. Aliases without
an override keep rotating regardless of `[routing].default`; accepted jobs
stay on their provider for polling.

```toml
[media_models]
"flux" = ["fal:fal-ai/flux/dev", "replicate:black-forest-labs/flux-1.1-pro"]
"veo"  = ["openrouter:google/veo-3.1"]
"tts"  = ["elevenlabs:eleven_flash_v2_5"]
```

Providers: `elevenlabs`, `fal`, `openrouter`, `replicate`, `runway`. Rate limits
and concurrency come from `[providers.<name>]` unchanged — the media adapters
are keyed by the same names. `tokens_per_minute` / `tokens_per_day` never bind,
because media providers report no tokens.

Pricing is **not** configured here. octolib resolves a rate from its own
reference table and reports it on the response; see
[11 — Media](./11-media.md#cost).

The server refuses to start on an unknown provider, a malformed
`provider:model`, an empty mirror list, or an alias that collides with
`[models]` or `[embedding_models]`.

## `[media]`

Transport knobs for media requests. All optional.

| Key | Default | Meaning |
|---|---|---|
| `max_source_bytes` | `20971520` (20 MiB) | Cap on a decoded inline upload, checked before anything reaches a provider |
| `max_response_bytes` | `104857600` (100 MiB) | Cap on a provider response body |
| `polling_interval_secs` | `2` | Delay between job polls |
| `submit_timeout_secs` | `120` | Deadline for the submit call itself |

The inline wait deadline reuses `server.upstream_timeout_secs`; the provider
permit wait reuses `server.provider_queue_timeout_secs`.

## `[media_providers]`

Point an adapter at a self-hosted or gateway-fronted endpoint:

```toml
[media_providers.fal]
api_base = "https://fal.internal.example"
```

Applied by exporting the adapter's own base-URL environment variable at startup
(`FAL_API_URL`, `OPENROUTER_MEDIA_API_URL`, `REPLICATE_API_URL`,
`ELEVENLABS_API_URL`, `RUNWAY_API_URL`). Because those are process-global, there
is exactly **one custom endpoint per adapter** per deployment. An
already-set environment variable wins over the config.

### Owner credential encryption

`OCTOHUB_CREDENTIALS_KEY` is standard base64 encoding of exactly 32 random bytes.
It is an environment-only startup setting, not a TOML field or a SIGHUP-reloaded
value. Generate it once, for example with `openssl rand -base64 32`, and keep it
in the deployment's secret store. All processes sharing the credential database
must use the same key. Retain it with database backups; changing it does not
reencrypt existing credentials.

- Unset with no credential rows: startup succeeds, BYOK is disabled, credential
  admin routes return 503 `credentials_disabled`, and platform routing continues.
- Unset while any credential rows exist: startup fails with an error naming
  `OCTOHUB_CREDENTIALS_KEY`.
- Invalid base64 or a decoded length other than 32 bytes: startup fails, even if
  the credential table is empty.
- A valid but incorrect encryption key cannot decrypt existing credentials;
  affected completion requests fail rather than using platform credentials.

SQLite, MySQL, and PostgreSQL initialize `owner_credentials` automatically on
startup. Each `(owner, provider)` row stores AES-256-GCM ciphertext as
`base64(nonce || ciphertext-and-tag)` with a fresh random 96-bit nonce. The owner
and provider are authenticated as associated data, preventing ciphertext from
being moved to another owner/provider. Only hints and operational metadata are
returned by the [credential admin API](./06-api-admin.md#owner-provider-credentials-byok).
Each process caches an owner's decrypted credentials, including an empty set, for
up to 60 seconds. Credential admin edits on the same process take effect on the next
request; edits made by another process sharing the database can take up to 60 seconds.
Usage telemetry (`last_used_at`, `last_error`) does not affect routing and does not
invalidate the cache. Decryption failures are not cached.
