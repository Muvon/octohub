# 06 — Admin API

Every endpoint here requires the **master** key from `[server].api_key`
in `octohub.toml`. The master key is a plain equality match against the
`Authorization: Bearer <master>` header — see
[04 — Authentication](./04-authentication.md#admin-auth-in-detail).

If `[server].api_key` is empty, all of these endpoints return
`401 Unauthorized` and the server logs a warning at startup
(`src/main.rs:63`–`65`).

Source: [`src/api/admin.rs`](../../src/api/admin.rs). All admin handlers
go through the `check_admin` helper at `src/api/admin.rs:34`–`57`.

## Endpoints

| Method | Path | Handler | Purpose |
|---|---|---|---|
| `POST`   | `/v1/admin/keys`                  | `handle_create_key`      | `src/api/admin.rs:60`   | Create a client API key |
| `GET`    | `/v1/admin/keys`                  | `handle_list_keys`       | `src/api/admin.rs:140`  | List all client keys (masked) |
| `GET`    | `/v1/admin/keys/:id`              | `handle_get_key`         | `src/api/admin.rs:163`  | Get one client key (masked) |
| `POST`   | `/v1/admin/keys/:id/revoke`       | `handle_revoke_key`      | `src/api/admin.rs:184`  | Revoke a client key |
| `GET`    | `/v1/admin/usage`                 | `handle_usage`           | `src/api/admin.rs:208`  | Aggregated usage by time bucket |
| `GET`    | `/v1/admin/completions`           | `handle_list_completions`| `src/api/admin.rs:251`  | Raw completion history |
| `GET`    | `/v1/admin/embeddings`            | `handle_list_embeddings` | `src/api/admin.rs:296`  | Raw embedding history |

Routing: see `classify_route` in `src/main.rs:163` for the low-cardinality
labels these produce in metrics.

## Conventions

- All list endpoints return `{"data": [...]}`.
- All timestamps in responses are **Unix seconds (u64)**, not ISO-8601.
- All `id` fields are **i64** database row IDs.
- Errors: `{"error":{"message":"…","type":"invalid_request_error"}}` —
  same shape as the client API. 4xx is your fault (bad input), 5xx is
  the server's.

## `POST /v1/admin/keys` — create

Request:

```json
{
  "name": "frontend-app",
  "allowed_models": ["minimax-m2.7", "voyage"]   // optional
}
```

- `name` (required) — human-readable label. Must be non-empty after trim
  (`src/api/admin.rs:101`).
- `allowed_models` (optional, default `null`):
  - `null` or absent → **unrestricted** (every model allowed).
  - `[]` → **lockout** (every model request returns 403).
  - `["a", "b"]` → only those exact `model` strings allowed.

Response (`201 Created`):

```json
{
  "id": 1,
  "name": "frontend-app",
  "key": "9fK2mQxZ7rT4pL1nB6sH0jW3cY5gAuEoI2bTfQwRdYz",
  "key_hint": "…RdYz",
  "status": "active",
  "allowed_models": ["minimax-m2.7", "voyage"],
  "created_at": 1717171717
}
```

**The `key` field is only returned here, at creation time.** Save it
now. Every later read returns only `key_hint` (the last 4 characters).

The key is 32 random bytes, URL-safe-base64 with no padding and **no
prefix** — a 43-character token (`generate_api_key`,
`src/storage/mod.rs:149`).

## `GET /v1/admin/keys` — list

Returns all keys. The `key` field is **omitted**; only `key_hint` is
present:

```json
{
  "data": [
    {
      "id": 1,
      "name": "frontend-app",
      "key_hint": "…RdYz",
      "status": "active",
      "allowed_models": ["minimax-m2.7", "voyage"],
      "created_at": 1717171717
    },
    {
      "id": 2,
      "name": "old-internal",
      "key_hint": "…xY9z",
      "status": "revoked",
      "allowed_models": null,
      "created_at": 1717000000
    }
  ]
}
```

No pagination — returns everything. If you have millions of keys, you
have a different problem.

## `GET /v1/admin/keys/:id` — get one

Same masked shape. `404 Not Found` if the id doesn't exist.

## `POST /v1/admin/keys/:id/revoke` — revoke

No body. Sets `status = "revoked"`. The key immediately stops
authenticating. **There is no un-revoke endpoint.** To restore access,
create a new key.

Response on success:

```json
{ "status": "revoked" }
```

`404` if the id doesn't exist. Revoking an already-revoked key is a no-op
that returns the same success response.

## `GET /v1/admin/usage` — aggregated usage

Aggregates completions and embeddings over time buckets, optionally
filtered by key. Source: `handle_usage` at `src/api/admin.rs:208`.

Query parameters:

| Param | Type | Default | Description |
|---|---|---|---|
| `key_id` | comma-separated i64 | (all keys) | Filter to these key IDs only |
| `bucket` | `hour`\|`day`\|`week`\|`month` | (none) | Time bucket granularity. Omitted → a single total row per key with `period: null` |
| `since` | u64 (unix seconds) | (unbounded) | Inclusive (`>=`) lower bound on `created_at` |
| `until` | u64 (unix seconds) | (unbounded) | Inclusive (`<=`) upper bound on `created_at` |

Response:

```json
{
  "data": [
    {
      "period": 1717200000,
      "key_id": 1,
      "key_name": "frontend-app",
      "completions_count": 142,
      "embeddings_count": 8,
      "total_input_tokens": 19230,
      "total_output_tokens": 88412
    }
  ]
}
```

`period` is the bucket start in Unix seconds. With `bucket=hour`, an
entry per hour per key. With `bucket=month`, one row per key per month.
With **no** `bucket`, you get one total row per key and `period` is
`null`.

Bucket values are parsed in `parse_bucket` at `src/api/admin.rs:363`. Any
other value (including typo'd `Hour`, `daily`, etc.) silently falls
back to **no bucketing** — a single total aggregate with `period: null`,
exactly as if `bucket` were omitted (`src/api/admin.rs:221`). **There's
no error response for an unknown bucket** — double-check your query
string.

### Use cases

```bash
# Tokens per key over the last 24 hours, hourly
curl -sG "http://127.0.0.1:8080/v1/admin/usage" \
  -H "Authorization: Bearer $MASTER" \
  --data-urlencode "bucket=hour" \
  --data-urlencode "since=$(( $(date +%s) - 86400 ))"

# Cost report for keys 1 and 3 over the last 30 days
curl -sG "http://127.0.0.1:8080/v1/admin/usage" \
  -H "Authorization: Bearer $MASTER" \
  --data-urlencode "key_id=1,3" \
  --data-urlencode "bucket=day" \
  --data-urlencode "since=$(( $(date +%s) - 2592000 ))"
```

## `GET /v1/admin/completions` — raw history

Returns raw completion rows. Source: `handle_list_completions` at
`src/api/admin.rs:251`.

Query parameters:

| Param | Type | Default | Description |
|---|---|---|---|
| `key_id` | comma-separated i64 | (all keys) | Filter to these key IDs only |
| `since` | u64 | (unbounded) | Inclusive (`>=`) lower bound on `created_at` |
| `until` | u64 | (unbounded) | Inclusive (`<=`) upper bound on `created_at` |
| `limit`  | u32 | `100` | Max rows returned (hard-capped at 1000; `limit=0` ⇒ 50) |
| `offset` | u32 | `0`   | Pagination offset |

Filter parsing: `build_filter` at `src/api/admin.rs:373`. Unknown query
keys are ignored.

Response:

```json
{
  "data": [
    {
      "id": "01HMQGSB3R...",
      "api_key_id": 1,
      "session_id": "...",
      "input_model": "minimax-m2.7",
      "resolved_model": "minimax:minimax-m2.7",
      "provider": "minimax",
      "usage": { "input_tokens": 56, "output_tokens": 120 },
      "input": { /* original request input */ },
      "output": { /* provider response output */ },
      "created_at": 1717171717
    }
  ]
}
```

The `id` is a **ULID** (string), not a database integer. It's the same
value you get back in the client's response, and the same value a client
should pass as `previous_completion_id` to continue the chain.

Pagination is `limit`/`offset`. For large ranges, iterate `offset` in
steps of `limit` until you get fewer rows than `limit`.

## `GET /v1/admin/embeddings` — raw history

Same shape as completions, minus `session_id` and `output`. Source:
`handle_list_embeddings` at `src/api/admin.rs:296`.

## The `octohub-admin.sh` wrapper

For daily operations there's a shell script at
[`octohub-admin.sh`](../../octohub-admin.sh) that wraps these endpoints.
It reads the master key and base URL from environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `OCTOHUB_URL`           | `http://127.0.0.1:8080` | Base URL |
| `OCTOHUB_SERVER_HOST`   | `127.0.0.1` | Host |
| `OCTOHUB_SERVER_PORT`   | `8080` | Port |
| `OCTOHUB_MASTER_KEY`    | (none) | Master key — must be set |

The `OCTOHUB_URL` var is used by the script; the others are convenience
defaults. The server itself does not read any of these (the server only
reads `OCTOHUB_MASTER_KEY` in the form of `[server].api_key` or, in
degenerate no-config mode, the env fallback).

Subcommands the script exposes:

```bash
./octohub-admin.sh create-key <name>     # POST /v1/admin/keys
./octohub-admin.sh list-keys             # GET  /v1/admin/keys
./octohub-admin.sh get-key <id>          # GET  /v1/admin/keys/:id
./octohub-admin.sh revoke-key <id>       # POST /v1/admin/keys/:id/revoke
./octohub-admin.sh usage [key_id]        # GET  /v1/admin/usage
./octohub-admin.sh completions [key_id]  # GET  /v1/admin/completions
./octohub-admin.sh embeddings [key_id]   # GET  /v1/admin/embeddings
./octohub-admin.sh help
```

It's a thin wrapper. If it doesn't do what you need, drop down to curl
or your HTTP client of choice — the wire protocol is plain JSON.

## Error summary

| Status | When |
|---|---|
| `200` / `201` | Success |
| `400` | Body parse error, empty `name` field |
| `401` | Missing/invalid `Authorization` header, or empty master key in config |
| `404` | Key id not found (revoke / get-one) |
| `500` | Database error (rare; check logs) |


---

## Usage: media and cost fields

`GET /v1/admin/usage` rows carry two fields beyond the token counts:

| Field | Meaning |
|---|---|
| `media_count` | Media requests in the bucket |
| `total_cost` | Summed USD across completions, embeddings and media |

`total_cost` only counts what could actually be priced. A media request nothing
could price contributes nothing rather than zero — see
[11 — Media](./11-media.md#cost) for what makes a request unpriced, and the
`octohub_media_cost_unknown_total` metric for how often it happens.
