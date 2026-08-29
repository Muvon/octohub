# OctoHub

> One OpenAI-style API in front of all your LLM providers — with multi-tenant keys, full request logging, and usage analytics.

[![CI](https://github.com/Muvon/octohub/actions/workflows/ci.yml/badge.svg)](https://github.com/Muvon/octohub/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Muvon/octohub)](https://github.com/Muvon/octohub/releases)
[![License](https://img.shields.io/github/license/Muvon/octohub)](LICENSE)

OctoHub is a high-performance LLM proxy written in Rust. Clients talk to one
stable endpoint; OctoHub talks to whichever providers you configure — OpenAI,
Anthropic, Google, DeepSeek, Ollama, OpenRouter, Modal-hosted vLLM, and more —
and records every request and response in your own database.

```
            ┌──────────────────┐
 client  →  │  OctoHub proxy   │  →  provider A (openai)
            │  (Rust / hyper)  │  →  provider B (anthropic)
            │                  │  →  provider C (ollama, your GPU)
            └──────────────────┘  →  provider D (Modal-hosted vLLM)
                     │
                     └─→ your DB (SQLite / MySQL / PostgreSQL)
```

## Why OctoHub?

Provider APIs don't give you per-customer usage tracking, audit logs, or a way
to swap models without breaking clients. OctoHub is the missing front door:
one interface, many backends, everything recorded.

## Quick Start

```bash
# Build
cargo build --release

# Drop in a config
cat > octohub.toml <<'EOF'
[server]
host = "127.0.0.1"
port = 8080
api_key = "your-master-secret"
db_url = "sqlite://octohub.db"

[models]
"minimax-m2.7" = ["minimax:minimax-m2.7"]
EOF

# Run
./target/release/octohub
```

Create a client key and make your first call:

```bash
curl -sX POST http://127.0.0.1:8080/v1/admin/keys \
  -H "Authorization: Bearer your-master-secret" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app"}'
# → {"id":1,"key":"nYwT8kQ2v…", ...}

curl -sX POST http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer nYwT8kQ2v…" \
  -H "Content-Type: application/json" \
  -d '{"model":"minimax-m2.7","input":"Say hi"}'
```

## Installation

**Prebuilt binaries** — every [release](https://github.com/Muvon/octohub/releases)
ships static binaries for 6 targets:

| OS | Targets |
|---|---|
| Linux (musl, static) | `x86_64-unknown-linux-musl`, `aarch64-unknown-linux-musl` |
| macOS | `x86_64-apple-darwin`, `aarch64-apple-darwin` |
| Windows | `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc` |

```bash
# Example: Linux x86_64
curl -fsSL https://github.com/Muvon/octohub/releases/download/0.7.8/octohub-0.7.8-x86_64-unknown-linux-musl.tar.gz | tar xz
./octohub
```

**From source**

```bash
cargo build --release
./target/release/octohub
```

## Features

- **One endpoint, many providers** — OpenAI-style `POST /v1/completions`,
  `/v1/chat/completions`, and `/v1/embeddings` routed to 20+ upstream providers
- **Completion chaining** — pass `previous_completion_id` and prior turns are
  replayed automatically, including reasoning/thinking blocks (DeepSeek-compatible)
- **Model mapping & load balancing** — map short names to `provider:model`
  lists; OctoHub picks randomly across them
- **Multi-tenant API keys** — issue/revoke per-client keys with per-key model
  allow-lists and usage attribution
- **Full request/response logging** — every call stored in SQLite (default),
  MySQL, or PostgreSQL for audit, replay, and analytics
- **Usage analytics** — aggregated stats by key and time bucket (hour/day/week/month)
- **Per-provider concurrency limits** — throttle any upstream independently
- **Observability** — structured JSON/pretty logs, Prometheus metrics on a
  separate port, `X-Request-Id` correlation
- **Ops-friendly** — `GET /health`, SIGHUP config hot-reload, `octohub-admin.sh`
  wrapper for daily admin tasks

## Configuration

Create `octohub.toml` in your working directory:

```toml
[server]
host = "127.0.0.1"
port = 8080
db_url = "sqlite://octohub.db"
api_key = "your-master-secret"   # master key for the admin API

[models]
# short name -> list of "provider:model" (random pick = simple load balancing)
"my-model" = ["minimax:minimax-m2.7", "ollama:minimax-m2.7"]

[embedding_models]
"voyage-3.5" = ["voyage:voyage-3.5"]
```

| Backend | DSN example |
|---|---|
| SQLite (default) | `sqlite://octohub.db` |
| MySQL | `mysql://user:pass@host:3306/octohub` |
| PostgreSQL | `postgres://user:pass@host:5432/octohub` |

Schema is created automatically on first connection. `OCTOHUB_DB_URL` overrides
the config file. If `api_key` is omitted, client endpoints run open and admin
endpoints are disabled (a warning is printed at startup).

Full reference: [doc/03 — Configuration](doc/03-configuration.md).

## Authentication

Two independent layers:

| Layer | Endpoints | Key source |
|---|---|---|
| **Client** | `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings` | Keys from the `api_keys` DB table |
| **Admin** | `/v1/admin/*` | Master key from `octohub.toml` |

Details: [doc/04 — Authentication](doc/04-authentication.md).

## Documentation

- **[User docs](doc/README.md)** — installation, configuration, auth, deployment, troubleshooting
- **[API reference](API.md)** — full endpoint reference
- **[Providers](doc/09-providers.md)** — supported `provider:model` prefixes
- **[Changelog](CHANGELOG.md)**

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Credits

Developed by Muvon Un Limited.
