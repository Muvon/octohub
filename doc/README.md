# OctoHub — User Documentation

OctoHub is a high-performance LLM proxy server written in Rust. It sits between
your applications and a fleet of upstream LLM providers, adding four things
that raw provider APIs don't give you on their own:

1. **A single OpenAI-style endpoint** that routes to many providers
   (OpenAI, Anthropic, Google, Ollama, Modal-hosted vLLM, OpenRouter, …) — for
   chat, embeddings, and media generation (images, video, speech, transcription).
2. **Multi-tenant API keys** issued and revoked from an admin API, with
   per-key usage attribution and rate-limiting by model allow-list.
3. **Full request/response logging** in your own database (SQLite, MySQL,
   or PostgreSQL) for observability, audit, and replay.
4. **Multi-turn chain support** so a client can return a `previous_completion_id`
   and have the prior turns of the conversation automatically replayed, including
   reasoning/thinking blocks that providers like DeepSeek require.

Under the hood OctoHub is built on [octolib](https://crates.io/crates/octolib)
(provider abstraction), `hyper` 1 + `tokio` (HTTP server), and `tracing`
(structured logging).

---

## Document index

| # | Topic |
|---|---|
| [01 — Overview](./01-overview.md) | What OctoHub is, how it works, when to use it |
| [02 — Installation](./02-installation.md) | Build from source, first run, CLI flags |
| [03 — Configuration](./03-configuration.md) | Full `octohub.toml` reference, env overrides |
| [04 — Authentication](./04-authentication.md) | Client keys vs. master key, model allow-lists |
| [05 — Client API](./05-api-client.md) | `POST /v1/completions`, `POST /v1/embeddings` |
| [06 — Admin API](./06-api-admin.md) | Key management, usage, raw logs |
| [07 — Observability](./07-observability.md) | Structured logs, Prometheus metrics, request IDs |
| [08 — Deployment](./08-deployment.md) | Reverse proxies, `X-Forwarded-For`, security |
| [09 — Providers](./09-providers.md) | Supported `provider:model` prefixes |
| [10 — Troubleshooting](./10-troubleshooting.md) | Common errors and fixes |
| [11 — Media](./11-media.md) | Image, video, speech and transcription endpoints |

---

## Quick start

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

Then create a client key and call the API:

```bash
# Create a client key (admin endpoint)
curl -sX POST http://127.0.0.1:8080/v1/admin/keys \
  -H "Authorization: Bearer your-master-secret" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app"}'
# → {"id":1,"key":"nYwT8kQ2v…","key_hint":"…xY9z", ...}

# Call a completion (client endpoint)
curl -sX POST http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer nYwT8kQ2v…" \
  -H "Content-Type: application/json" \
  -d '{"model":"minimax-m2.7","input":"Say hi"}'
```

See [02 — Installation](./02-installation.md) and
[04 — Authentication](./04-authentication.md) for the full walkthrough.

---

## License

Apache-2.0. See [LICENSE](../LICENSE).
