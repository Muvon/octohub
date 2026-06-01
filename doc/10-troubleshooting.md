# 10 — Troubleshooting

Symptom-driven. Every entry here is grounded in a specific code path.

## Startup

### Server starts but logs `WARN Master API key is empty`

`src/main.rs:64` fires this when `[server].api_key` is empty. The server
keeps running, but **every admin endpoint returns 401**.

Fix: set `[server].api_key` in `octohub.toml` and restart. If you're
already setting it, the config file you're pointing at is not the one
the server is reading. Check the path passed to `-c` and the
`OCTOHUB_*` env overrides documented in
[03 — Configuration](./03-configuration.md#environment-variables).

### Bind fails / "Address already in use"

Another process is on `host:port`. Common offenders: a second OctoHub
instance, a leftover dev server, or another tool on 8080/9090.

```bash
lsof -iTCP:8080 -sTCP:LISTEN
```

Either kill the other process or change `[server].port` /
`[metrics].bind`.

### SQLite "unable to open database file"

- The directory doesn't exist. SQLite needs a writable directory
  before it can create the file.
- Filesystem permissions. The user running the binary can't write to
  the configured path.
- SELinux / AppArmor. On RHEL/Fedora in enforcing mode, non-default
  paths need a custom policy.

Quick check:

```bash
touch octohub.db && ls -la octohub.db
```

If that fails, fix the directory before starting the server.

### MySQL / Postgres connection refused

- Wrong host/port: `mysql://root:secret@localhost:3306/octohub` —
  check the actual MySQL bind.
- Auth failure: the user `octohub` doesn't exist, or the password is
  wrong.
- The database `octohub` doesn't exist. OctoHub creates tables but
  assumes the database itself is there.
- TLS: the storage layer doesn't pass `sslmode` / `sslrootcert`. If
  your server requires TLS, you may need to use a local-only
  connection or open a tunnel.

The startup line includes `db="mysql"` or `db="postgres"` — if you see
`db="sqlite"` you loaded the wrong config.

## Auth

### `401 Missing API key` on a client call

The `Authorization` header is missing. Most HTTP clients (curl, axios,
requests, the OpenAI Python SDK) accept a callable credential and add
the header for you; check that the call site is actually using the
client key, not the master key, and that the key is being passed to
the SDK.

### `401 Invalid or revoked API key`

Three common causes:

1. The key was revoked via the admin API. List keys to confirm:
   ```bash
   curl -s http://127.0.0.1:8080/v1/admin/keys \
     -H "Authorization: Bearer $MASTER" | jq '.data[] | {id, name, status}'
   ```
2. The `Authorization` header is missing the `Bearer ` prefix. The
   validator at `src/auth.rs:16` requires the literal string `Bearer `
   with a trailing space.
3. The key was typed wrong. Re-issue by revoking and creating again;
   remember the full key is only returned once at creation time
   (`src/api/admin.rs:134`).

### `403 model 'X' is not permitted for this API key`

The key has an `allowed_models` list and the model name you sent isn't
in it. The match is **case-sensitive and exact** against the `model`
field as-sent.

Fix: either remove the allow-list on the key, or add the model name
you're sending (including the alias or the `provider:model` form,
whichever you're using):

```bash
curl -sX POST http://127.0.0.1:8080/v1/admin/keys/$ID/revoke   # if you want to recreate
# OR just send a model name that IS in the list
```

This 403 path is special-cased at `src/api/handler.rs:239` — it's
distinct from the 400/500 paths for other engine errors.

### Admin endpoints all return 401

`[server].api_key` is empty in the loaded config. Confirm with the
startup line — `admin_auth=true` means it's set, `admin_auth=false`
means it's empty.

## Models and resolution

### `400 Failed to resolve model 'X'`

The `model` field in your request is not in `[models]` (for
`/v1/completions`) or `[embedding_models]` (for `/v1/embeddings`),
**and** doesn't contain a colon (so the alias-bypass path doesn't
trigger).

Fix: add an alias in the right table, or change your client to send a
`provider:model` string directly.

### `400 model 'X' is not found in config`

Same as above, distinct error message. This comes from
`src/proxy/engine.rs` — different code path, same fix.

### `400 model 'Y' is not available for provider 'X'`

The provider string in your config doesn't match what octolib
recognizes. Most common cause: typo, or a provider name that requires
a feature flag in octolib. Check that the provider is in the list at
[09 — Providers](./09-providers.md).

## Upstream calls

### `500 Provider 'X' chat_completion failed: …`

The proxy reached the provider, the provider returned an error. Common
shapes:

- **`401 Unauthorized`** — the provider's API key is wrong, missing, or
  revoked. OctoHub doesn't manage provider credentials; they're read by
  octolib from the environment (`OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, etc.). Check `env | grep -i api_key`.
- **`429 Too Many Requests`** — you hit the provider's rate limit.
  Add a `[providers.<name>].concurrency` cap to smooth bursts, or back
  off client-side.
- **`5xx` from the provider** — provider outage. The full anyhow chain
  is returned to the client via `{:#}` formatting at
  `src/api/handler.rs:263`, so the actual provider error body is in
  the response.

The full chain (`Caused by: …` lines for each wrap layer) is the
diagnostic. If the response is just the outer wrap, you're looking at
an old build — re-check the `classify_engine_error` logic at
`src/api/handler.rs:233`.

### Client request hangs forever (no error, no response)

Almost always a per-provider concurrency limit. The HTTP request is
sitting waiting for a permit on the provider's semaphore
(`src/proxy/limiter.rs:49`).

What to check:

1. Is `[providers.<name>]` set with a `concurrency` cap? Check the
   startup line — `providers="ollama:4,openai:32"` if so.
2. Are upstream calls actually completing? Hit the provider directly
   and confirm.
3. Is there a stuck long-running call holding a permit? Check
   `octohub_provider_in_flight` on the metrics endpoint.

Fixes:

- Raise the cap (or remove the section entirely).
- Add timeouts on the client side.
- Restart the server to clear any in-process state.

### Metrics endpoint unreachable (connection refused)

The metrics listener (`src/metrics.rs:112`) is independent. If it
fails to bind, a warning is logged at `src/metrics.rs:117` and the
listener is **silently disabled** — the main API keeps running, so a
scrape just gets connection refused (there is no 503; the port simply
isn't listening).

Check `[metrics].bind` isn't already in use. The startup line
`metrics endpoint listening bind=…` confirms the bind succeeded.

## Storage

### Schema drift after upgrade

OctoHub adds columns additively (`ensure_column` calls in the storage
files). For **minor** upgrades this happens automatically on first
connect. For **major** schema breaks, you would have seen a release
note.

If you suspect drift, restart the server and watch the startup logs
for migration errors. The schema definitions are in
`src/storage/sqlite.rs:27` (and the equivalent lines in
`src/storage/mysql.rs`, `src/storage/postgres.rs`).

### Query is slow

Most likely cause: the `completions` and `embeddings` tables grow
unbounded. There is **no built-in retention** — you set the policy.

Operational options:

- A periodic `DELETE FROM completions WHERE created_at < ?` job
  (whatever cutoff you want).
- Partitioning (PostgreSQL) on `created_at`.
- A separate metrics-only store: copy the counts you need to a
  long-term store and prune the source.

If you need guidance on a specific backend, the storage trait
(`src/storage/mod.rs`) is a stable abstraction — you can plug in
your own backend without modifying the rest of the codebase.

### MySQL `Incorrect datetime value` on insert

The `created_at` column is a Unix timestamp (BIGINT). If you see
datetime errors, someone ran a manual `ALTER TABLE` and changed the
type. Revert it, or drop and re-init the schema (will lose data).

## Observability

### No `req_id` in the log line

Either:

- The log format is `pretty` and the span field collapsed into a
  compact form (still present, just abbreviated).
- The log level is `warn` or above — span fields are attached to the
  span, not the log event. Use `RUST_LOG=info` or
  `[logging].level = "info"`.

### Metrics endpoint returns 404 for everything except `/metrics`

Correct — the metrics listener only serves `/metrics` and 404s on
anything else (`src/metrics.rs:144`). It does not proxy to the main
API. If you want a single port for everything, put both behind the
same reverse proxy.

### `octohub_provider_permits_available` is missing for a provider

The provider isn't in `[providers.<name>]` with a positive
`concurrency`. Unconfigured providers are absent from the gauge loop
(`src/metrics.rs:168`). To get metrics, add the section:

```toml
[providers.openai]
concurrency = 30
```

### Prometheus scrape returns garbage / wrong format

The endpoint returns `text/plain; version=0.0.4` — the Prometheus
exposition format. If you're scraping with something that expects
JSON, it'll fail. Use a real Prometheus client.

## Logs are unreadable in production

`format = "auto"` picks `pretty` only when stdout is a TTY. In a
container, stdout is **not** a TTY, so you'll get JSON. That's the
right call for log aggregation. If you want pretty output in a non-TTY
context (e.g. piped to `tee`, a log file, or journald), set
`format = "pretty"` explicitly. ANSI color codes are auto-suppressed
whenever stdout isn't a terminal (`src/logging.rs:18`), so the text stays
clean regardless of format.

The same applies inside a container: `format = "pretty"` (or
`OCTOHUB_LOG_FORMAT=pretty`) gives you the compact human-readable layout,
and because container stdout isn't a TTY the ANSI codes stay **off** — no
escape sequences for your log shipper to strip.

## Something's broken and it's not in this list

The diagnostic flow:

1. Grab the `X-Request-Id` from the failing response.
2. Grep your server logs for `req_id=<id>`. You'll get the request
   span and any error events.
3. If the request never logged, it didn't reach the server —
   networking, reverse proxy, or DNS.
4. If it logged a 5xx, the error message in the response **is** the
   full `anyhow` chain. Read it.
5. If it logged a 4xx, it's your input. The 4xx message tells you
   which one.

For deeper issues, the route label + status from
`octohub_requests_total` in your metrics tells you the request class.
Histogram quantiles by route tell you whether the issue is latency
(queued too long, slow upstream) or outright failures.
