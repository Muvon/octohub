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
| `upstream_timeout_secs` | `360` | Maximum duration of the complete provider operation, including octolib retries. Exceeding it returns `504`. |
| `failover_on_error` | `false` | Re-route provider-side failures (timeout, connect, 429, 5xx) to the next candidate provider of the model alias. 4xx never fails over. |
| `provider_error_cooldown_secs` | `0` | After 3 consecutive provider-side failures, deprioritize the provider for this many seconds (used only when no healthy candidate admits). `0` = off. |

The legacy alias `db_path` is also accepted and treated as `db_url`
(`src/config.rs:98`).

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
| `OCTOHUB_MASTER_KEY` | `[server].api_key` | Master admin key. |
| `OCTOHUB_DB_URL` | `[server].db_url` | Useful in containerized deployments. |
| `OCTOHUB_PROVIDER_QUEUE_TIMEOUT_SECS` | `[server].provider_queue_timeout_secs` | Overrides the provider queue deadline. |
| `OCTOHUB_UPSTREAM_TIMEOUT_SECS` | `[server].upstream_timeout_secs` | Overrides the complete upstream operation deadline. |
| `OCTOHUB_FAILOVER_ON_ERROR` | `[server].failover_on_error` | `true`/`1` enables provider failover. |
| `OCTOHUB_PROVIDER_ERROR_COOLDOWN_SECS` | `[server].provider_error_cooldown_secs` | Overrides the provider cooldown duration. |
| `OCTOHUB_HOST` | `[server].host` | Only when no config file is loaded. |
| `OCTOHUB_PORT` | `[server].port` | Only when no config file is loaded. |
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

[embedding_models]
"voyage" = ["voyage:voyage-4"]

[providers.ollama]
concurrency = 4   # GPU is busy; queue the rest

[providers.openai]
concurrency = 30  # stay under the 60-rps published limit
```
