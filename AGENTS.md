# OctoHub Development Instructions

## Core Principles

### Code Quality
- **Zero Warnings**: All code must pass `cargo clippy` without warnings
- **DRY Principle**: Don't repeat yourself - reuse existing patterns
- **KISS Principle**: Keep it simple, stupid - avoid over-engineering
- **Fail Fast**: Validate inputs early and return clear error messages

## Project Structure

### Core Modules
- `src/main.rs` - Application entry point and HTTP server setup
- `src/config.rs` - Configuration loading and management
- `src/auth.rs` - Authentication handling
- `src/api/handler.rs` - Client endpoint handlers (`/v1/completions`, `/v1/embeddings`)
- `src/api/admin.rs` - Admin endpoint handlers (`/v1/admin/*`)
- `src/storage/mod.rs` - Storage trait, types, and factory (`from_url`)
- `src/storage/sqlite.rs` - SQLite storage implementation
- `src/storage/mysql.rs` - MySQL storage implementation
- `src/storage/postgres.rs` - PostgreSQL storage implementation
- `src/proxy/engine.rs` - Proxy engine (routes requests to providers)

### Dependencies
- `octolib` - Shared library from parent directory (default-features disabled)
- `rusqlite` - SQLite database (bundled)
- `mysql` - MySQL database
- `postgres` - PostgreSQL database
- `hyper` - HTTP server
- `tokio` - Async runtime
- `clap` - CLI argument parsing
- `tracing` - Structured logging

## Authentication Architecture

**There are two completely independent authentication systems. They do not interact.**

### 1. Client Auth — DB keys (`/v1/completions`, `/v1/embeddings`)
- Keys are stored in the `api_keys` database table
- Created and managed via admin endpoints
- Every request must supply a valid active key: `Authorization: Bearer <client-key>`
- Validated by `authenticate_client()` in `src/auth.rs` — looks up the key hash in the DB
- On success: the key's `id` is attached to the stored completion/embedding record
- **Completely independent of the master key** — the master key config has zero effect on this path

### 2. Admin Auth — master key from config (`/v1/admin/*`)

- The master key is set in `octohub.toml` as `server.api_key`
- Used exclusively to protect admin endpoints (create/revoke keys, query usage/logs)
- Validated by `authenticate_admin()` in `src/auth.rs` — compares bearer token to config value
- If `api_key` is not set in config: admin endpoints are **disabled** (always return 401), server still starts with a warning
- **Has no effect on client endpoint auth** — client endpoints always require a DB key regardless

### Summary

| | Client endpoints | Admin endpoints |
|---|---|---|
| Auth source | `api_keys` DB table | `server.api_key` in config |
| No key configured | Always requires DB key | Disabled (401) |
| Controlled by | Admin API | `octohub.toml` |

## Configuration

### Config File Location
Configuration is stored in `octohub.toml` in the current working directory.

### Default Configuration
```toml
# OctoHub Configuration

[server]
host = "127.0.0.1"
port = 8080
db_url = "sqlite://octohub.db"  # Database DSN (see below)
# api_key = "your-master-secret"  # Optional: enables /v1/admin/* endpoints

# Model mappings: model_name -> list of fully qualified "provider:model" strings
# When resolving, randomly pick one from the list (simple load balancing)
# You can also use "provider:model" directly in API calls to bypass mapping

[models]
# Single provider (keys with special chars must be quoted)
"minimax-m2.7" = ["minimax:minimax-m2.7"]

# Multiple providers (random selection)
# "my-model" = ["minimax:minimax-m2.7", "ollama:minimax-m2.7"]
```

### Database Configuration

OctoHub supports three database backends via the `db_url` setting:

| Backend | DSN format | Example |
|---|---|---|
| **SQLite** (default) | `sqlite://path` or bare path | `sqlite://octohub.db` |
| **MySQL** | `mysql://user:pass@host:port/db` | `mysql://root:secret@localhost:3306/octohub` |
| **PostgreSQL** | `postgres://user:pass@host:port/db` | `postgres://postgres:secret@localhost:5432/octohub` |

Schema is created automatically on first connection. No manual migration needed.

Environment variable `OCTOHUB_DB_URL` overrides the config file value.

## Development Workflow

### Build Commands
```bash
cargo build
cargo check
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

### Code Quality Standards
- **Zero clippy warnings** - All code must pass `cargo clippy` without warnings
- **Minimal dependencies** - Reuse existing dependencies before adding new ones
- **Error handling** - Use proper `Result<T>` types and meaningful error messages

## Quick Start Checklist

1. Run clippy before finalizing code
2. Run fmt to ensure consistent formatting
3. Test changes with cargo test

## Observability

### Logging

OctoHub uses `tracing-subscriber` with structured, span-based logging.

**Format auto-detection** (default `Auto`):
- If stdout is a TTY → compact pretty format
- If stdout is not a TTY → JSON format

Override via config `[logging]` section or environment variables:

| Variable | Values | Default |
|---|---|---|
| `OCTOHUB_LOG_FORMAT` | `auto`, `pretty`, `json` | `auto` |
| `OCTOHUB_LOG_LEVEL` | `trace`, `debug`, `info`, `warn`, `error` | `info` (or `RUST_LOG`) |

Precedence for level: `OCTOHUB_LOG_LEVEL` → `RUST_LOG` → `"info"`.

**Sample pretty line:**
```
2024-01-15T10:30:01.234Z INFO octohub starting version="0.1.0" bind="127.0.0.1:8080" db="sqlite" admin_auth=true providers="none" models=1 embed_models=1 metrics=true
```

**Sample JSON line:**
```json
{"timestamp":"2024-01-15T10:30:01.234Z","level":"INFO","message":"request completed","req_id":"01HMQGSB3R","method":"POST","route":"/v1/completions","path":"/v1/completions","remote":"10.0.0.1","status":200,"dur_ms":1523,"api_key_id":1,"model":"minimax-m2.7","provider":"minimax","queued_ms":0,"tok_in":56,"tok_out":120}
```

**Request span fields:**

| Field | Description |
|---|---|
| `req_id` | ULID request ID (or client-provided `X-Request-Id`) |
| `method` | HTTP method |
| `route` | Low-cardinality route label (e.g. `/v1/completions`, `/v1/admin/keys`, `/health`, `other`) |
| `path` | Full request path |
| `remote` | Effective remote address (peer or `X-Forwarded-For` if trusted) |
| `status` | HTTP response status code |
| `dur_ms` | Total request duration in milliseconds |
| `api_key_id` | Client API key ID (after successful auth) |
| `model` | Request model name |
| `provider` | Resolved provider name |
| `queued_ms` | Time queued waiting for provider permit (0 if no wait) |
| `tok_in` | Input tokens from response usage |
| `tok_out` | Output tokens from response usage |

### Metrics

Prometheus metrics exposed via a separate HTTP endpoint.

**Configuration:**

```toml
[metrics]
enabled = true
bind = "127.0.0.1:9090"
per_key = false  # Add api_key_id label to completion/embedding metrics
```

Environment overrides: `OCTOHUB_METRICS_ENABLED` (`true`/`false`/`1`/`0`), `OCTOHUB_METRICS_BIND`.

**Metrics exposed:**

| Name | Type | Labels | Description |
|---|---|---|---|
| `octohub_build_info` | gauge | `version` | Build version (always 1) |
| `octohub_requests_total` | counter | `route`, `method`, `status` | Total HTTP requests |
| `octohub_request_duration_seconds` | histogram | `route`, `method` | Request duration |
| `octohub_requests_in_flight` | gauge | `route` | Current in-flight requests |
| `octohub_completions_total` | counter | `model`, `provider`, `status`, `api_key_id`* | Completion requests |
| `octohub_completion_duration_seconds` | histogram | `model`, `provider` | Upstream call duration |
| `octohub_completion_tokens_total` | counter | `model`, `provider`, `direction` | Token counts (`direction=in\|out`) |
| `octohub_embeddings_total` | counter | `model`, `provider`, `status`, `api_key_id`* | Embedding requests |
| `octohub_embedding_duration_seconds` | histogram | `model`, `provider` | Upstream call duration |
| `octohub_embedding_tokens_total` | counter | `model`, `provider`, `direction` | Token counts |
| `octohub_provider_queue_wait_seconds` | histogram | `provider` | Time queued for permit |
| `octohub_provider_permits_available` | gauge | `provider` | Available concurrency permits |
| `octohub_provider_in_flight` | gauge | `provider` | In-flight requests |

\* `api_key_id` label only present when `per_key = true`.

**Prometheus scrape config:**
```yaml
scrape_configs:
  - job_name: octohub
    static_configs:
      - targets: ['127.0.0.1:9090']
```

**Sample queries:**
```promql
# P95 request latency by route
histogram_quantile(0.95, sum(rate(octohub_request_duration_seconds_bucket[5m])) by (route, le))

# Error rate by route
sum(rate(octohub_requests_total{status=~"5.."}[5m])) by (route) / sum(rate(octohub_requests_total[5m])) by (route)

# Provider queue wait P99
histogram_quantile(0.99, sum(rate(octohub_provider_queue_wait_seconds_bucket[5m])) by (provider, le))

# Completion tokens/sec by model
sum(rate(octohub_completion_tokens_total{direction="out"}[5m])) by (model)
```

### Request ID

Every request is assigned a unique ID and returned in the `X-Request-Id` response header.

- If the incoming request includes a valid `X-Request-Id` header (1–64 chars, `[A-Za-z0-9._-]`), it is passed through unchanged.
- Otherwise, a new ULID is generated.

### X-Forwarded-For

Set `trust_forwarded_for = true` under `[server]` to use `Forwarded` (RFC 7239) or `X-Forwarded-For` headers for the `remote` log field.

**Security warning:** Only enable this when OctoHub runs behind a trusted reverse proxy. With this enabled, clients can spoof their IP address in logs.
