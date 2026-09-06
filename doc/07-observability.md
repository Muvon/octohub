# 07 — Observability

OctoHub ships two complementary observability surfaces: **structured
logs** on stdout, and a **Prometheus** scrape endpoint on a separate
port. They share a request-correlation model — every log line and every
metric carries a request ID you can use to join them.

Sources: [`src/logging.rs`](../../src/logging.rs),
[`src/metrics.rs`](../../src/metrics.rs),
[`src/main.rs`](../../src/main.rs),
[`src/http_util.rs`](../../src/http_util.rs).

## Logging

`logging::init` at `src/logging.rs:5` configures `tracing_subscriber`.
It takes a `[logging]` config (see [03 — Configuration](./03-configuration.md#logging)).

### Log level resolution

Precedence (highest first):

1. `[logging].level` if set
2. `RUST_LOG` env var
3. `"info"` (default)

The level string is fed to `EnvFilter::try_new` — standard
`tracing_subscriber` syntax is supported (`"info,hyper=warn"`,
`"octohub=debug"`, etc.).

### Log format

`[logging].format` (`src/config.rs:8`–`18`):

| Value | Behavior |
|---|---|
| `"auto"` (default) | Pretty if stdout is a TTY, JSON otherwise. |
| `"pretty"` | Compact, single-line, human-readable. ANSI escape codes only when stdout is a TTY (the `use_ansi` check at `src/logging.rs:18` is hardcoded — explicit `pretty` over a pipe still renders plain text). |
| `"json"` | One JSON object per line. `with_current_span(true)` and `flatten_event(true)` — span fields are inlined into the event. |

**The two env overrides** (in `src/config.rs:179`–`194`):

| Variable | Overrides | Allowed values |
|---|---|---|
| `OCTOHUB_LOG_FORMAT` | `[logging].format` | `auto`, `pretty`, `json` |
| `OCTOHUB_LOG_LEVEL` | `[logging].level` | `trace`, `debug`, `info`, `warn`, `error` |

`OCTOHUB_LOG_LEVEL` takes precedence over `RUST_LOG` regardless of
which was set first.

### Sample pretty line (startup)

```
2024-01-15T10:30:01.234Z INFO octohub starting version="0.1.0" bind="127.0.0.1:8080" db="sqlite" admin_auth=true providers="ollama:4,openai:32" models=1 embed_models=1 metrics=true
```

This is the line emitted at `src/main.rs:100`–`110`. Fields:

| Field | Source | Meaning |
|---|---|---|
| `version` | `env!("CARGO_PKG_VERSION")` | Build version |
| `bind` | `[server].host:port` after CLI overrides | Where the API is listening |
| `db` | `src/main.rs:90`–`98` | `sqlite`, `mysql`, or `postgres` |
| `admin_auth` | `!api_key.is_empty()` | Whether the admin API is enabled |
| `providers` | `build_provider_summary` at `src/main.rs:147` | `name:limit,…` for each configured provider; `none` if no `[providers]` section |
| `models` | `config.models.len()` | Number of `[models]` entries |
| `embed_models` | `config.embedding_models.len()` | Number of `[embedding_models]` entries |
| `metrics` | `config.metrics.enabled` | Whether the metrics endpoint is started |

If metrics are enabled, a second startup line follows at `src/main.rs:112`:

```
INFO metrics endpoint listening bind="127.0.0.1:9090"
```

If the master key is empty, this warning fires at `src/main.rs:64`:

```
WARN Master API key is empty (server.api_key). Admin endpoints disabled.
```

### Sample JSON line (request)

```json
{"timestamp":"2024-01-15T10:30:01.234Z","level":"INFO","message":"request completed","req_id":"01HMQGSB3R","method":"POST","route":"/v1/completions","path":"/v1/completions","remote":"10.0.0.1","status":200,"dur_ms":1523,"api_key_id":1,"model":"minimax-m2.7","provider":"minimax","queued_ms":0,"tok_in":56,"tok_out":120}
```

### Request span fields

Every request opens a tracing span in `src/main.rs:208`–`220`. Fields
populated by handlers as the request progresses:

| Field | Set at | Type | Notes |
|---|---|---|---|
| `req_id` | `src/main.rs:183` | string | `X-Request-Id` passthrough (1–64 chars, `[A-Za-z0-9._-]`) or fresh ULID |
| `method` | `src/main.rs:215` | string | `GET`, `POST`, … |
| `route` | `src/main.rs:163` | string | Low-cardinality label: `/v1/completions`, `/v1/embeddings`, `/v1/admin/keys`, `/v1/admin/usage`, `/v1/admin/completions`, `/v1/admin/embeddings`, `/health`, or `other` |
| `path` | `src/main.rs:216` | string | Full request path (high-cardinality) |
| `remote` | `src/http_util.rs:9` | string | Peer address, or `Forwarded` / `X-Forwarded-For` if `trust_forwarded_for = true` |
| `status` | on response | u16 | HTTP status code |
| `dur_ms` | on response | u64 | Total request duration in ms |
| `api_key_id` | `src/api/handler.rs:62` | i64 | Set after client auth succeeds |
| `model` | `src/api/handler.rs:86` | string | The `model` field from the request (before resolution) |
| `provider` | on completion success | string | The resolved provider name |
| `queued_ms` | on completion success | u64 | Time spent waiting for a provider permit (0 if no wait) |
| `tok_in` | on completion success | u64 | Input tokens from the response usage |
| `tok_out` | on completion success | u64 | Output tokens from the response usage |

For embeddings, only `tok_in` is recorded — embeddings have no output
side.

### `X-Request-Id`

Every response carries an `X-Request-Id` header (`src/main.rs:198`).
Behavior:

- If the incoming request has a valid `X-Request-Id` (1–64 chars, only
  `[A-Za-z0-9._-]`), it's echoed back unchanged. The validation is in
  `extract_request_id` at `src/main.rs:183`.
- Otherwise a fresh **ULID** is generated.
- The value is what shows up as `req_id` in the logs and the metric
  label set (when present).

This is the join key. Take a 500 from a client, grab the `X-Request-Id`
from the response, grep your logs.

### `X-Forwarded-For` and `Forwarded` (RFC 7239)

The `remote` field comes from `effective_remote` at
`src/http_util.rs:9`:

- `trust_forwarded_for = false` (default) — always the peer address
- `trust_forwarded_for = true` — try `Forwarded` first (parse
  `for=…`), fall back to `X-Forwarded-For` (first comma-separated entry,
  must parse as `IpAddr`)

**Only enable this behind a trusted reverse proxy.** With it on, clients
can spoof their `remote` value in your logs.

## Metrics

Prometheus-format metrics on a separate HTTP endpoint, configurable via
`[metrics]` (see [03 — Configuration](./03-configuration.md#metrics)).
Source: [`src/metrics.rs`](../../src/metrics.rs).

### Endpoint

`GET /metrics` on the bind address in `[metrics].bind` (default
`127.0.0.1:9090`). Path is hardcoded at `src/metrics.rs:136` — anything
else returns 404 from the same listener. Content-Type:
`text/plain; version=0.0.4`.

### Configuration

```toml
[metrics]
enabled = true             # default true
bind = "127.0.0.1:9090"    # default
per_key = false            # default false; add api_key_id label
```

Env overrides: `OCTOHUB_METRICS_ENABLED` (`true`/`false`/`1`/`0`) and
`OCTOHUB_METRICS_BIND`.

When `enabled = false`, `metrics::init` returns `None` and the listener
is never spawned (`src/main.rs:76`–`81`).

### Exposed metrics

All names use the `octohub_` prefix. Bucket boundaries for histograms
are defined at `src/metrics.rs:17`–`20`.

| Name | Type | Labels | Description |
|---|---|---|---|
| `octohub_build_info`                  | gauge     | `version` | Always 1; identifies the running version |
| `octohub_requests_total`              | counter   | `route`, `method`, `status` | Every HTTP request |
| `octohub_request_duration_seconds`    | histogram | `route`, `method` | Total request duration |
| `octohub_requests_in_flight`          | gauge     | `route` | In-flight requests right now |
| `octohub_completions_total`           | counter   | `model`, `provider`, `status`, `api_key_id`\* | Completion requests |
| `octohub_completion_duration_seconds` | histogram | `model`, `provider` | Upstream call duration |
| `octohub_completion_tokens_total`     | counter   | `model`, `provider`, `direction` (`in`/`out`) | Token counts |
| `octohub_embeddings_total`            | counter   | `model`, `provider`, `status`, `api_key_id`\* | Embedding requests |
| `octohub_embedding_duration_seconds`  | histogram | `model`, `provider` | Upstream call duration |
| `octohub_embedding_tokens_total`      | counter   | `model`, `provider`, `direction` (`in`) | Input tokens |
| `octohub_provider_queue_wait_seconds` | histogram | `provider` | Time waiting for a concurrency permit |
| `octohub_provider_permits_available`  | gauge     | `provider` | Free concurrency permits |
| `octohub_provider_in_flight`          | gauge     | `provider` | Active requests at the provider |

\* `api_key_id` label is **only present when `[metrics].per_key = true`**.
When false, the metric has 3 labels (`model`, `provider`, `status`).

**Status label values** on completions/embeddings:
- `"ok"` — upstream call succeeded
- `"error"` — upstream failed or rejected (the `provider` label becomes
  `"unknown"` in this case — `src/api/handler.rs:126`)

### Provider gauges — how they work

`octohub_provider_permits_available` and `octohub_provider_in_flight`
are populated by a **5-second poll loop** in `provider_gauge_loop` at
`src/metrics.rs:164`. The loop snapshots `ProviderLimiter::snapshot()`
and writes to the gauges. Side effect: when a provider is at zero
utilization, the gauge values lag by up to 5 seconds.

Providers **not configured** in `[providers.<name>]` are absent from
this gauge entirely. There's no entry for "unlimited" — the loop
silently skips them (`src/metrics.rs:168`).

### Prometheus scrape config

```yaml
scrape_configs:
  - job_name: octohub
    static_configs:
      - targets: ['127.0.0.1:9090']
```

### Useful PromQL

P95 request latency by route:

```promql
histogram_quantile(0.95,
  sum(rate(octohub_request_duration_seconds_bucket[5m])) by (route, le)
)
```

Error rate by route:

```promql
sum(rate(octohub_requests_total{status=~"5.."}[5m])) by (route)
  /
sum(rate(octohub_requests_total[5m])) by (route)
```

Provider queue-wait P99 (a leading indicator of saturation):

```promql
histogram_quantile(0.99,
  sum(rate(octohub_provider_queue_wait_seconds_bucket[5m])) by (provider, le)
)
```

Output tokens/sec by model:

```promql
sum(rate(octohub_completion_tokens_total{direction="out"}[5m])) by (model)
```

Completion error rate by model (a key is the failure rate):

```promql
sum(rate(octohub_completions_total{status="error"}[5m])) by (model, provider)
  /
sum(rate(octohub_completions_total[5m])) by (model, provider)
```

## Log/metric correlation

The `req_id` in a log line corresponds to nothing in metrics today —
metrics don't carry it as a label (that would explode cardinality).
Use the `X-Request-Id` response header as the join key from a client
error into logs, and use route/model/status from logs and metrics as
the join key between them.


---

## Media metrics

```
octohub_media_requests_total{task,model,provider,status}
octohub_media_duration_seconds{task,model,provider}
octohub_media_cost_usd{task,model,provider,source}
octohub_media_cost_unknown_total{task,model,provider}
```

`task` is `image`, `video`, `speech` or `transcription`. `status` is `ok`,
`accepted` (202 — the job is still running) or `error`. `source` on the cost
counter is `provider` (the upstream reported a dollar amount) or `estimate`
(computed from octolib's reference rates); requests nothing could price are
counted in `octohub_media_cost_unknown_total` instead, never as $0.

The `/v1/media/{id}` route label is collapsed so per-record ids cannot explode
metric cardinality.
