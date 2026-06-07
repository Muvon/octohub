# 01 — Overview

## What OctoHub solves

If you ship a product that calls LLM APIs, you eventually need:

- **One stable interface** even as you swap or A/B test providers
- **Cost and usage tracking** per customer, team, or feature
- **Audit logs** of every prompt and response (for compliance, debugging,
  or post-hoc evaluation)
- **A way to take a model offline** without breaking clients

Provider-native APIs don't give you any of that. OctoHub is the missing
front door: clients talk to it, it talks to whoever you tell it to, and
everything in between is recorded.

## What it is, structurally

```
            ┌──────────────────┐
 client  →  │  OctoHub proxy   │  →  provider A (openai)
            │  (Rust / hyper)  │  →  provider B (anthropic)
            │                  │  →  provider C (ollama, your GPU)
            └──────────────────┘  →  provider D (Modal-hosted vLLM)
                     │
                     └─→ your DB (SQLite / MySQL / PostgreSQL)
                         (api_keys, completions, embeddings, usage)
```

The proxy is stateless across requests. State lives in two places:

- **Config** (`octohub.toml`) — model name → provider mappings, provider
  concurrency limits, master key, log/metrics settings
- **Database** — client API keys, every completion request and response,
  every embedding request, and aggregate usage

The CLI is a single Rust binary (`octohub`). A companion shell script
(`octohub-admin.sh`) wraps the admin API for daily operations.

## What it is not

- **Not a chat UI.** It's a backend proxy. Point your application or
  framework at it.
- **Not a model router with semantic fall-back.** Model selection is the
  caller's job (or a random pick from a configured list). OctoHub does
  not pick a "best" model for a given prompt.
- **Not a vector store.** Embeddings pass through and are logged, but
  OctoHub doesn't index them.

## Feature list

Verified against the current source tree:

| Feature | Where |
|---|---|
| OpenAI-style `POST /v1/completions` (Responses API) and `POST /v1/embeddings` | `src/api/handler.rs:44`, `src/api/handler.rs:146` |
| Classic OpenAI `POST /v1/chat/completions` (Chat Completions API) | `src/api/handler.rs:284` |
| Multi-turn chain (`previous_completion_id`) | `src/proxy/engine.rs:79` |
| Reasoning/thinking replay (DeepSeek-compatible) | `src/proxy/engine.rs:248` |
| Structured output (`text.format` → JSON schema) | `src/proxy/engine.rs:195` |
| Tool calling (function definitions) | `src/proxy/engine.rs:145` |
| Image and video input parts | `src/api/types.rs:130` |
| Per-key model allow-lists | `src/api/admin.rs:80`, `src/storage/mod.rs:34` |
| `GET /health` | `src/api/handler.rs:141` |
| Admin: create/list/get/revoke keys | `src/api/admin.rs:60`–`205` |
| Admin: aggregated usage by time bucket | `src/api/admin.rs:208` |
| Admin: raw completion and embedding history | `src/api/admin.rs:251`, `src/api/admin.rs:296` |
| Per-provider concurrency throttling | `src/proxy/limiter.rs:19` |
| Prometheus metrics on a separate port | `src/metrics.rs:22` |
| Structured JSON / pretty logging | `src/logging.rs:5` |
| `X-Request-Id` passthrough + ULID generation | `src/main.rs:183` |
| `X-Forwarded-For` / `Forwarded` (RFC 7239) | `src/http_util.rs:9` |
| SQLite, MySQL, PostgreSQL backends | `src/storage/{sqlite,mysql,postgres}.rs` |
| 20+ upstream providers (openai, anthropic, ollama, deepseek, …) | octolib `ProviderFactory` |

## When to reach for OctoHub

Reach for it when you have **more than one model behind one application**,
or when **more than one application talks to one model**, or when you
need **persistent logs of every request**. If you're calling a single
provider from a single script, OctoHub is overkill.

The Modal scripts in `llm/modal/` show one common pattern: OctoHub fronts
a fleet of self-hosted vLLM instances (one per model) and the rest of the
stack talks only to OctoHub.

## Architecture in 30 seconds

| Layer | File |
|---|---|
| HTTP server, routing, request ID, span setup | `src/main.rs` |
| Client endpoint handlers | `src/api/handler.rs` |
| Admin endpoint handlers | `src/api/admin.rs` |
| Wire types (request/response shapes) | `src/api/types.rs` |
| Authentication (client + admin) | `src/auth.rs` |
| Proxy engine (chain walk, provider call, persistence) | `src/proxy/engine.rs` |
| Per-provider concurrency gate | `src/proxy/limiter.rs` |
| Storage trait + DSN factory | `src/storage/mod.rs` |
| Config loading + env overrides | `src/config.rs` |
| Metrics init, recording, scrape endpoint | `src/metrics.rs` |
| Logging init (auto/pretty/json) | `src/logging.rs` |
| Remote-IP detection | `src/http_util.rs` |

The proxy is a thin pass-through: every meaningful field returned by the
upstream provider is surfaced to the client verbatim. OctoHub's job is to
**add** things (auth, logging, rate-limiting, multi-provider), not to
mutate what providers return.
