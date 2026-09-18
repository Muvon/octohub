# OctoHub Development Instructions

## Working Standards

- Keep changes focused; reuse existing patterns and dependencies. Prefer simple functions and explicit data flow over new abstraction layers.
- Validate client input before upstream work. Use `anyhow::Result` and contextual errors internally; preserve typed errors and stable API discriminators at the HTTP boundary.
- Rust changes must remain formatted and pass Clippy with zero warnings. Follow the verification workflow below and honor explicit user restrictions on execution.
- Inspect current source before relying on comments, examples, or design documents. Document implemented behavior separately from intended behavior or known limitations.
- Preserve unrelated working-tree changes. Do not switch the `octolib` dependency to a sibling checkout or update dependencies as a side effect of unrelated work.

## Architecture and Source Map

OctoHub is a Rust 2021 binary using Tokio and Hyper 1. It proxies completions, embeddings, and media through `octolib`, with synchronous database backends behind a shared storage trait.

| Path | Responsibility |
|---|---|
| `src/main.rs` | CLI, startup, HTTP/1 and HTTP/2 listener, route dispatch/classification, request spans and IDs, Unix SIGHUP reload |
| `src/config.rs` | TOML/env loading, defaults, validation, model candidate lists, media endpoint overrides |
| `src/auth.rs` | Independent client-key and admin-master-key authentication |
| `src/api/handler.rs` | Client handlers, request conversion, error/status mapping, buffered chat SSE, metrics |
| `src/api/types.rs` | Completion/chat/embedding wire types and conversions, structured output and multimodal content |
| `src/api/media_types.rs` | Media wire types, source decoding, validation, redaction, response envelope |
| `src/api/admin.rs` | Key/owner management, owner auto maps, usage and stored-record queries, observed model status |
| `src/proxy/engine.rs` | Completion replay, routing/admission, upstream calls, embeddings, persistence |
| `src/proxy/auto.rs` | Purpose-to-alias resolution for virtual `auto` |
| `src/proxy/limiter.rs` | Provider and owner semaphores, request/token windows, provider cooldowns |
| `src/proxy/media.rs` | Media submission, persistence, polling/cancellation, normalization and pricing |
| `src/storage/mod.rs` | `Storage: Send + Sync`, shared records/filters, JSON/key helpers, DSN factory |
| `src/storage/{sqlite,mysql,postgres}.rs` | Schema initialization/upgrades and complete backend implementations |
| `src/logging.rs`, `src/http_util.rs` | Logging setup and effective remote address parsing |
| `src/metrics.rs`, `src/health.rs` | Prometheus instrumentation and in-memory health from actual traffic |

The active `octolib` dependency comes from the registry, with default features disabled and `llm`, `embeddings`, and `media` enabled. The sibling path declaration in `Cargo.toml` is commented out. Treat the active manifest and lockfile as authoritative for dependency versions. Provider adapters, capability/pricing registries, and schema enforcement belong to `octolib`; OctoHub owns HTTP translation, routing, limits, and storage.

## HTTP and Authentication

### Client routes

Every route below requires an active DB key via `Authorization: Bearer <client-key>`:

| Method | Path | Behavior |
|---|---|---|
| POST | `/v1/completions` | Responses-style input/output with `previous_completion_id` chaining |
| POST | `/v1/chat/completions` | Classic chat format converted through the same completion engine |
| POST | `/v1/embeddings` | Embedding proxy |
| POST | `/v1/images/generations` | Image generation/edit modes |
| POST | `/v1/videos` | Video generation |
| POST | `/v1/audio/speech` | Speech synthesis |
| POST | `/v1/audio/transcriptions` | Transcription |
| GET | `/v1/media/models` | Configured media candidates with capabilities/pricing |
| GET | `/v1/media/{id}` | Read a terminal record or advance a pending job |
| POST | `/v1/media/{id}/cancel` | Provider cancellation and record update |

`GET /health` is unauthenticated. `/v1/completions` is the actual Responses-style route; do not assume `/v1/responses` exists. Chat `stream: true` is emulated SSE over a completed, buffered upstream response; it does not provide upstream token streaming or earlier first-token delivery. Media endpoints use JSON requests and a shared JSON result/job envelope, including speech and transcription.

### Auth boundaries and key policy

- Client auth calls `Storage::get_api_key_by_key` and checks `status == "active"`. The database currently stores and matches the raw key, **not a hash**. Creation returns the full generated key; ordinary admin list/get responses expose a hint.
- Admin handlers authenticate against `server.api_key`, overridden by `OCTOHUB_MASTER_KEY`. The master key has no special authority on client routes.
- **Current limitation:** startup warns that an empty master key disables admin endpoints, but `authenticate_admin` compares tokens directly and its tests explicitly accept `Bearer ` against an empty master key. Do not claim unconditional 401 behavior until the implementation changes.
- `allowed_models: None` means unrestricted; `Some([])` means deny all. Matching is exact against the requested alias or `provider:model`. Virtual `auto` checks both the requested `auto` and its resolved alias.
- Keys with the same owner label and a positive `owner_concurrency` share an in-process request budget. Completion, embedding, and media creation hold the owner permit through queueing, upstream work, and storage. Saturation waits up to `OWNER_QUEUE_WAIT` (30 seconds), then returns 429. Missing owner or missing/zero capacity is unlimited.
- Model-list and owner updates change active key metadata in place without rotating credentials. Owner budgets are read from authenticated key rows; resizing swaps semaphores, allowing old permits to drain.
- Media lookups include `api_key_id`; another key's record is returned as absent (404). Preserve that boundary for both poll and cancel.

### Admin routes

`src/main.rs::route_admin` dispatches:

- `POST /v1/admin/keys`, `GET /v1/admin/keys`, `GET /v1/admin/keys/{id}`.
- `POST /v1/admin/keys/{id}/revoke`, `/models`, and `/owner`.
- `GET` and `PUT /v1/admin/owners/{owner}/auto` (empty/null map clears the override).
- `GET /v1/admin/status`, `/usage`, `/completions`, `/embeddings`, and `/media`.

Use existing handler authentication, parsing, filtering, and error helpers when extending these routes.

## Configuration and Reloading

### Loading and precedence

- Pass a file explicitly with `-c PATH` / `--config PATH`. There is **no automatic loading of `./octohub.toml`**. Without a path, startup uses defaults plus environment variables and empty model maps.
- `--bind HOST:PORT` overrides the loaded bind address. `OCTOHUB_HOST` and `OCTOHUB_PORT` apply only in the no-file environment path.
- If a TOML `[server]` table is present, `api_key` is currently a required field during deserialization, even when `OCTOHUB_MASTER_KEY` will override it. Omitting the entire table uses `ServerConfig::default()`.
- Supported sections are `[server]`, `[models]`, `[embedding_models]`, `[media_models]`, `[auto]`, `[providers.<name>]`, `[logging]`, `[metrics]`, `[media]`, and `[media_providers.<name>]`.
- File and no-file modes both accept `OCTOHUB_MASTER_KEY`, `OCTOHUB_DB_URL`, `OCTOHUB_LOG_FORMAT`, `OCTOHUB_LOG_LEVEL`, `OCTOHUB_METRICS_BIND`, `OCTOHUB_METRICS_ENABLED`, `OCTOHUB_PROVIDER_QUEUE_TIMEOUT_SECS`, `OCTOHUB_UPSTREAM_TIMEOUT_SECS`, `OCTOHUB_FAILOVER_ON_ERROR`, and `OCTOHUB_PROVIDER_ERROR_COOLDOWN_SECS`.
- Boolean env overrides for metrics/failover are true only for `true` or `1`. Invalid numeric overrides are ignored. Check `Config::apply_env_overrides` before adding or documenting new overrides.

### Defaults

| Setting | Default |
|---|---|
| `server.host`, `server.port` | `127.0.0.1`, `8080` |
| `server.db_url` | `sqlite://octohub.db` (`db_path` is a legacy field alias) |
| `server.api_key` | Empty in the default server config |
| `server.trust_forwarded_for` | `false` |
| `server.provider_queue_timeout_secs` | `60` |
| `server.upstream_timeout_secs` | `360` |
| `server.failover_on_error` | `false` |
| `server.provider_error_cooldown_secs` | `0` (disabled) |
| `logging.format` | `auto` |
| `metrics.enabled`, `metrics.bind`, `metrics.per_key` | `true`, `127.0.0.1:9090`, `false` |
| `media.max_source_bytes`, `media.max_response_bytes` | 20 MiB, 100 MiB |
| `media.polling_interval_secs`, `media.submit_timeout_secs` | `2`, `120` |

Model aliases are lists of fully qualified `provider:model` candidates in the corresponding model map. A direct `provider:model` bypasses alias lookup. Candidate lists rotate to a random starting position and support fallthrough; resolution is more than a single random pick.

A nonempty `[auto]` enables virtual `auto` for completions. It must contain `default`, every target must be a `[models]` alias, and `[models].auto` must not coexist with it. Resolution uses `X-Model-Purpose`: owner purpose hierarchy, owner default, deployment purpose hierarchy, deployment default. Dash segments fall back from specific to general (`supervisor-gate` → `supervisor`). Stale owner targets are skipped. An absent/empty `[auto]` leaves `auto` as an ordinary alias.

Media validation rejects empty candidate lists, malformed entries, unknown providers, and aliases colliding with completion/embedding maps. Media adapter `api_base` overrides are exported to the adapter's own environment variable once at startup; an existing env value wins. `MEDIA_API_BASE_ENVS` is the mapping authority.

### SIGHUP lifecycle

On Unix, SIGHUP reloads the supplied config path with the same CLI bind override. Invalid config keeps the current configuration. New requests read cloned `Arc` snapshots through `Live<T> = Arc<RwLock<Arc<T>>>`; keep read/write locks brief and never hold them across `.await`.

Config and provider concurrency limiter handles are replaced separately. Do not describe this as one atomic swap of both handles. Listener address, storage connection, logging setup, and metrics listener/recorder initialization remain fixed until restart; media endpoint env exports are not reapplied. Provider semaphores are rebuilt, while owner budgets, provider rate counters, and provider cooldown state survive reload in separate `Arc`s. Those counters are process-local and reset on restart.

## Proxy Implementation Rules

- Keep HTTP parsing/conversion in `api`, provider orchestration in `proxy`, and SQL in `storage`. Both completion formats must continue through `ProxyEngine::process`.
- Database APIs are synchronous: follow the client/proxy pattern of `tokio::task::spawn_blocking`, moving owned arguments and an `Arc<dyn Storage>` into the closure. Some existing admin handlers still call storage directly; do not copy that blocking pattern into new async paths or hold synchronous locks across awaits.
- Acquire permits with the existing limiter helpers and retain the returned RAII guard for the intended lifetime. Provider names in limits are case-insensitive; absent/zero limits mean unlimited.
- Provider request/token limits use fixed 60-second and UTC-day windows. Requests count at admission; completion/embedding input+output tokens count after usage arrives. Tokens are not pre-reserved. Exhausted candidates are skipped; all exhausted returns 429 with `Retry-After`.
- Chained completions prefer the last serving provider for cache affinity. Optional provider-fault failover moves to another candidate for timeout/connect/429/5xx errors; ordinary upstream 4xx errors must remain client errors. Provider queue timeout is a local capacity failure, not an upstream fault.
- With cooldown enabled, three consecutive provider faults deprioritize the provider; cooling candidates remain fallback choices. Success clears the failure streak. Reuse `pick_admitted`, `is_provider_fault`, and the shared trackers rather than duplicating policy.
- Modality mismatch has its own candidate filtering: retain mirrors of the same model on other providers. Preserve `ModalityNotSupportedError` and exact `error.type = "modality_not_supported"`; clients in sibling projects branch on it.
- A `json_schema` request keeps only candidates whose octolib provider reports `enforces_response_schema`; none left → `SchemaNotEnforcedError` (400) before any upstream call. octolib owns that flag — fix a wrong answer there, never with a local override list.
- Keep `ProxyTimeoutError` mapping: provider queue timeout → 503; upstream timeout → 504. Model restrictions → 403; owner/provider budget exhaustion → 429. Reuse classifiers and preserve full contextual error chains where current handlers expose them.
- Call `octolib::llm::chat_completion_enforced` for completions. Preserve Responses-style `text.format` conversion to structured-output requests and the timeout around the full upstream operation, including library retries/parsing. The current classic chat request has no `response_format` field and converts with `text: None`; do not assume feature parity between the two wire formats.
- Forward supported sampling, reasoning, tool, multimodal, and cache-control fields through existing wire conversions. Read `X-Model-Purpose`, `X-Title`, and `HTTP-Referer` before consuming the body; attribution headers continue upstream.

### Completion replay

- Reconstruct stored chains oldest-first, then append live input. Live instructions take precedence over stored instructions. Inherit the resolved chain's session ID and prefer its provider.
- Current chain-read errors fall back to inline input; unresolved previous IDs are not persisted as dangling links. Do not assume chain lookup is scoped by key: the current `walk_chain` trait accepts only an ID, unlike media lookups.
- Preserve reasoning blocks, assistant tool calls, and tool results during conversion/replay. Coalesce replayed function calls and thinking onto the appropriate assistant message using existing helpers.
- Live cache-control markers must reach upstream, including tool-result markers. Stored replay strips ephemeral markers; do not resurrect old cache breakpoints.

### Media lifecycle

- Wire sources accept URLs and base64 data. Do not expose server-local file paths through the client API. Decode/validate using `WireMediaSource` and enforce configured transport caps.
- Reject client `provider_options.<namespace>.cost_estimate` using the existing validator, and filter options to the selected provider's namespace. Pricing belongs to `octolib`; preserve unknown cost as absent with warnings instead of manufacturing zero cost. Unsupported parameters default to `error`; `warn_and_drop` is explicit.
- Persist the submitted operation and resumable `JobHandle` **before inline waiting**. A pending job and its terminal result share one `StoredMedia` row. Return 202 while pending and 200 when terminal; `wait` defaults to true.
- Hold a provider permit for submit only; release it before long polling. Hold the owner permit through the create request. Submit failover stops once a job has been accepted; polling uses the stored provider and handle.
- Terminal reads use stored results without another upstream call. Terminal transitions clear the handle. Cancellation is provider-dependent; reuse the existing cancel path rather than introducing a second job state machine.
- Redact inline binary request payloads before persistence. Preserve result artifacts, warnings, errors, and usage in their existing normalized columns/envelope. There is no background job worker; requests drive polling.

## Storage Patterns

| Backend | DSN | Implementation |
|---|---|---|
| SQLite | `sqlite://path` or bare path | Bundled `rusqlite`, `Mutex<Connection>`, WAL, foreign keys enabled |
| MySQL | `mysql://user:pass@host:port/db` | `mysql::Pool` |
| PostgreSQL | `postgres://...` or `postgresql://...` | `r2d2` + `r2d2_postgres`, current manager uses `NoTls` |

- Storage changes must cover the trait, all three backends, and affected record/filter/usage conversions. Do not implement a feature only in SQLite.
- Schema setup and additive column upgrades run in backend constructors. Extend their existing idempotent initialization paths for both fresh and existing databases; there is no separate migration runner.
- Bind SQL values with the backend's parameter APIs. Keep JSON/null behavior aligned using shared encoding/decoding helpers; PostgreSQL uses JSONB and explicit casts where the existing implementation requires them.
- Preserve the distinction between SQL NULL (unrestricted models) and `[]` (lockout). Current `decode_allowed_models` treats malformed JSON as unrestricted and logs a warning; do not describe it as fail-closed validation.
- Usage aggregates span completions, embeddings, and media, with key/time filters and optional buckets. Unpriced records contribute no priced amount. Keep `usage.cost` and backend aggregate expressions aligned when touching accounting.

## Observability

- Use `tracing` fields and the existing request span, not ad hoc prints. Never log bearer tokens or upstream credentials.
- Logging level precedence is `OCTOHUB_LOG_LEVEL` → `[logging].level` → `RUST_LOG` → `info`. Levels accept `EnvFilter` directives. Auto format uses compact output for a stdout TTY and JSON otherwise; ANSI is disabled without a stdout TTY.
- `route` is a bounded label; `path` is the actual path. Update dispatch and `classify_route` when adding routes, collapsing record IDs to a stable label. Current unmatched classifications use `other`.
- Request IDs accept 1–64 ASCII characters from `[A-Za-z0-9._-]`; otherwise generate a ULID. Preserve `X-Request-Id` on every response through the router.
- Existing span fields include `req_id`, `method`, `route`, `path`, `remote`, `status`, `dur_ms`, `api_key_id`, `model`, `provider`, `queued_ms`, `chain_ms`, `upstream_ms`, `store_ms`, `tok_in`, and `tok_out`. Populate relevant fields without inventing measurements for paths that do not record them.
- `trust_forwarded_for` enables `Forwarded`/`X-Forwarded-For` remote detection. Keep it false unless deployment uses a trusted reverse proxy.
- Prometheus `/metrics` runs on a separate listener. Reuse `record_completion`, `record_embedding`, `record_media`, request in-flight guards, queue/rate/failover helpers, and provider gauges. Preserve metric names/labels in `src/metrics.rs`; keep key labels opt-in via `metrics.per_key` and avoid IDs in other labels.
- Health is derived from actual traffic through the metrics helpers, without probes. `GET /v1/admin/status` reports observed models only; unobserved models are unknown. Classification uses consecutive failures, not total latency. Virtual `auto` is excluded from the health registry. This model status is separate from provider cooldowns.

## Verification and Documentation

For Rust changes, use the existing local checks:

```bash
cargo fmt --all -- --check
cargo check --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

- Use `cargo fmt --all` when formatting changed Rust code is needed. `.pre-commit-config.yaml` runs formatting, Clippy, and checking for Rust files, plus general whitespace/conflict/TOML/YAML checks.
- Tests live alongside implementation in `#[cfg(test)] mod tests`; async cases use `#[tokio::test]`. Extend meaningful nearby regressions for behavior changes, especially conversion/replay, authorization, routing/admission, error mapping, media lifecycle, and storage upgrades.
- `.github/workflows/ci.yml` delegates to the shared Muvon Rust workflow. Its explicit test script uses `cargo test --verbose`, with `--no-default-features` on Windows. Keep Unix-specific signal handling gated with `#[cfg(unix)]`.
- Documentation-only changes need source/link review and `git diff --check`; they do not require Cargo builds/tests. If the user restricts execution, honor that and state what remains unverified. Do not equate source review or SQLite tests with live provider/MySQL/PostgreSQL validation.
- Public API/config changes should update the relevant `API.md`, `README.md`, `doc/` chapter, and example TOML where affected. Use `doc/03-configuration.md`, `doc/05-api-client.md`, `doc/06-api-admin.md`, `doc/07-observability.md`, and `doc/11-media.md` for detailed reference. `doc/SPEC-media.md` is design context; source determines current behavior.
