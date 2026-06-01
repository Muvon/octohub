# 08 — Deployment

This document covers what you need to do *after* the binary builds and
the config loads. Most of it is the same as deploying any
`hyper`/`tokio` Rust binary, with a few OctoHub-specific gotchas.

## Picking a host

There are no platform-specific dependencies in `Cargo.toml` beyond
`rusqlite` (bundled) and the network stack. OctoHub runs on:

- Linux x86_64 / aarch64 (glibc and musl)
- macOS x86_64 / aarch64
- (in principle) Windows, though it's untested

Minimum RAM is in the tens of MB — this is a thin proxy. Disk is
dominated by your database.

## Single-user / single-host (default)

The default `octohub.toml` binds `127.0.0.1:8080` and uses SQLite. That
configuration is suitable for:

- A laptop dev setup
- A single-box install where the proxy and its callers share a host
- Container side-cars that talk to localhost

```bash
./target/release/octohub
# {"level":"INFO","message":"octohub starting","bind":"127.0.0.1:8080",…}
```

Nothing further to do.

## Production / multi-host

You'll want:

1. **A real database** — Postgres or MySQL. SQLite is fine for
   single-process, low-write workloads, but the moment you have
   multiple OctoHub instances hitting the same DB, switch to a network
   DB. See [03 — Configuration](./03-configuration.md#database) for DSN
   formats.
2. **`host = "0.0.0.0"`** to listen on all interfaces, or whatever
   internal IP the reverse proxy talks to.
3. **A reverse proxy** in front (see below).
4. **Process supervision** — systemd unit, Kubernetes Deployment, or
   equivalent. OctoHub doesn't daemonize itself.
5. **Persistent disk for the SQLite file**, if you keep SQLite.

### systemd unit

```ini
# /etc/systemd/system/octohub.service
[Unit]
Description=OctoHub LLM proxy
After=network.target

[Service]
Type=simple
User=octohub
Group=octohub
WorkingDirectory=/var/lib/octohub
ExecStart=/usr/local/bin/octohub -c /etc/octohub/octohub.toml
Restart=always
RestartSec=5
# Hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/octohub
AmbientCapabilities=
CapabilityBoundingSet=
# Open file limits
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

State to back up: the database (`octohub.db` or your Postgres/MySQL
data) and the config file. That's it.

## Running behind a reverse proxy

OctoHub speaks plain HTTP/1.1 and HTTP/2 (auto-negotiated by `hyper`,
`src/main.rs:132`–`138`). Put it behind nginx, Caddy, Envoy, ALB, or
whatever you already use. The proxy just needs to forward bytes.

### Caddy

```caddyfile
api.example.com {
    reverse_proxy 127.0.0.1:8080 {
        header_up X-Forwarded-For {remote_host}
    }
}
```

If you go this route, set `trust_forwarded_for = true` in
`octohub.toml` so the `remote` log field reflects the real client IP
rather than the proxy's loopback address. See
[X-Forwarded-For](#x-forwarded-for) below.

### nginx

```nginx
upstream octohub {
    server 127.0.0.1:8080;
    keepalive 16;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate     /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    # Streaming response — disable buffering so token-by-token output
    # actually streams through to the client.
    proxy_buffering off;
    proxy_request_buffering off;
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;

    location / {
        proxy_pass http://octohub;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Tunables worth knowing:

- `proxy_read_timeout 600s` — LLM completions can run for minutes on
  long outputs. The default 60s will cut you off.
- `proxy_buffering off` — OctoHub's HTTP/1 implementation closes each
  connection after the response (`src/main.rs:138`). nginx buffering is
  what hurts you in the middle: the upstream response is sitting
  generated but not yet sent. Disable both request and response
  buffering to let bytes flow.
- HTTP/2 over nginx → OctoHub works; `hyper` auto-negotiates down to
  HTTP/1 if needed.

## TLS

OctoHub does not do TLS itself. Terminate at the reverse proxy. There
is no plan to add native TLS in the proxy — putting a reverse proxy in
front is the right architectural call regardless.

## Health checks

Two endpoints useful to a load balancer / orchestrator:

- `GET /health` — process is alive. **No auth.** `200 OK` with
  `{"status":"ok"}`. Suitable for `livenessProbe` in Kubernetes.
- A real client request with a known key — for `readinessProbe`. The
  cheapest such request is a single-string embedding call against the
  smallest configured model.

There is no separate `/ready` endpoint. Don't add a check that pokes
the upstream providers — those are slow and external; you don't want
your readiness check to flap because Anthropic had a bad minute.

## `X-Forwarded-For`

Source: `src/http_util.rs:9`. Behavior:

| `trust_forwarded_for` | `Forwarded` header | `X-Forwarded-For` header | Resulting `remote` log field |
|---|---|---|---|
| `false` (default) | any | any | TCP peer address |
| `true` | present, parseable | (ignored) | First `for=…` value |
| `true` | absent | present, first entry parses as IP | First comma-separated entry |
| `true` | malformed / no IP | absent | Falls back to peer |

Both IPv4 and bracketed IPv6 are handled (tests in
`src/http_util.rs:91`–`143`).

### Security warning

**Only enable `trust_forwarded_for` behind a proxy you control.** With
it enabled, an attacker that can reach the proxy port can put any IP
they want in the headers and your logs will record it. This affects
*only* the `remote` log field — it does not affect auth or
rate-limiting — but it does make log-based abuse detection useless.

If your reverse proxy is on the same host (e.g. `127.0.0.1:8080`), the
peer is the proxy and you can leave `trust_forwarded_for = false` and
still know which client connected (via the proxy's logs).

## HTTP/1 keep-alive is disabled

This is a deliberate choice in `src/main.rs:138`. Read the comment:

> HTTP/1 keep-alive stays disabled so each request closes its
> connection — prevents clients from reusing a stale pooled connection
> silently dropped by NAT/firewall during long LLM gaps.

What it means for you in practice: the client will see a fresh TCP
connection per request. If you have a connection-pooled SDK on the
client side, expect each request to pay the TCP+TLS handshake. This is
the right trade for LLM workloads where the response time is
dominated by upstream latency, not connection setup.

## Container deployment

There's no official image. The build is straightforward:

```dockerfile
FROM rust:1.83-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/octohub /usr/local/bin/octohub
COPY octohub.toml /etc/octohub/octohub.toml
WORKDIR /var/lib/octohub
EXPOSE 8080
USER 65534
ENTRYPOINT ["/usr/local/bin/octohub", "-c", "/etc/octohub/octohub.toml"]
```

Drop a `Dockerfile` like the above at the repo root. The default
`octohub.toml` in the repo binds to `127.0.0.1`; for containers, set
`host = "0.0.0.0"` in the in-image config or override at runtime.

## Multiple instances

You can run more than one OctoHub against the same database. The only
shared state is the DB. Be aware:

- **Per-provider concurrency limits are per-process.** If you set
  `concurrency = 4` for `ollama` and run two OctoHub instances, you
  get 8 concurrent requests to Ollama from the cluster, not 4. The
  limit is local to the process. To get a cluster-wide limit, you
  need an external rate-limiter in front.
- **No leader election.** Multiple instances all serve traffic. The DB
  is the only coordination point.
- **Prometheus scraping** needs to hit each instance's metrics
  endpoint, or you need to put a sidecar / pushgateway in front.

## Database maintenance

The schema is created on first connect (see
[03 — Configuration](./03-configuration.md#database)). The proxy never
runs `DROP` or destructive migrations. Additive migrations (new
columns) happen in `ensure_column` calls in the storage files.

Useful operator queries:

```sql
-- Total completions by status, last 7 days
SELECT
  strftime('%Y-%m-%d', created_at, 'unixepoch') AS day,
  COUNT(*) AS total
FROM completions
WHERE created_at > strftime('%s', 'now', '-7 days')
GROUP BY day ORDER BY day DESC;

-- Top 10 keys by output tokens
SELECT api_key_id, SUM(usage_json_extract) ... ;   -- (syntax depends on backend)
```

For a full schema listing, see `src/storage/sqlite.rs` (the SQLite
file is the most readable of the three) — the table shapes are
identical across backends.

## Backup

What's worth backing up:

| Asset | Why |
|---|---|
| `octohub.toml` | All your model mappings, provider concurrency limits, log format settings |
| The database | All API keys, all completions, all embeddings, all usage history |
| TLS certs and reverse-proxy config | Standard |

What's **not** worth backing up: nothing else. Logs go to stdout — if
you want them persisted, that's a job for your log shipper.
