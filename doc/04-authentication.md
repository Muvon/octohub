# 04 — Authentication

OctoHub runs **two completely independent auth systems**. They never interact
and never fall through to each other. Knowing which one applies to which
endpoint is the single most important thing about running this proxy in
production.

Source: [`src/auth.rs`](../../src/auth.rs) (the whole file is 81 lines —
worth reading).

## The two systems at a glance

| | **Client** | **Admin** |
|---|---|---|
| **Endpoints** | `POST /v1/completions`, `POST /v1/embeddings` | `GET/POST /v1/admin/*` |
| **Where the key comes from** | `api_keys` table in the database | `server.api_key` in `octohub.toml` |
| **Validator** | `authenticate_client()` at `src/auth.rs:22` | `authenticate_admin()` at `src/auth.rs:35` |
| **Header expected** | `Authorization: Bearer <token>` | `Authorization: Bearer <token>` |
| **If no key configured** | Always 401 — client endpoints never run unauthenticated | All admin endpoints return 401, server logs a warning and keeps running |
| **Failure response** | `{"error":{"message":"Invalid or revoked API key","type":"invalid_request_error"}}` | `{"error":{"message":"Invalid or missing admin API key","type":"invalid_request_error"}}` |
| **What client gets on success** | The `ApiKey` record — its `id` is attached to every stored completion/embedding | A `bool`; the request proceeds |

## The "no key configured" case

A common source of confusion: **leaving `api_key` empty in `octohub.toml`
does not disable authentication**. It only disables the **admin** layer.

- Client endpoints (`/v1/completions`, `/v1/embeddings`) **always require a
  DB-stored key** that is present in the `api_keys` table. There is no
  bypass. With an empty database table, every client request returns 401.
- Admin endpoints return 401 with no exceptions.

The startup line that confirms which mode you're in
(`src/main.rs:63`–`65`):

```
WARN Master API key is empty (server.api_key). Admin endpoints disabled.
```

…is your cue to set `api_key` if you intended to use the admin API.

## The bearer parsing rule

Both validators go through the same `extract_bearer()` helper
(`src/auth.rs:16`):

- `Authorization: Bearer <token>` → token = `<token>`
- `Authorization: <token>` (no `Bearer ` prefix) → returns `None` →
  authentication fails
- Header missing → returns `None` → authentication fails
- Empty `Bearer ` (token missing) → returns `Some("")` — **an empty bearer
  against an empty master key matches** (see `auth.rs:60`). Don't rely on
  this; it's a degenerate case.

A practical consequence: clients must include the literal `Bearer `
prefix, including the space. The `OpenAI` and `Anthropic` SDKs both send
this by default.

## Client auth in detail

`authenticate_client(header, storage)` does the following
(`src/auth.rs:22`–`31`):

1. Extract the bearer token.
2. Look it up in the `api_keys` table by the stored hash.
3. If the row is found **and** its `status == "active"`, return
   `ClientAuth::Ok(key)`. Otherwise return `ClientAuth::Invalid`.
4. Revoked keys (`status == "revoked"`) behave identically to keys that
   were never created — they all 401.

The matching `ApiKey` is returned to the request handler, which records
its `id` in the request span field `api_key_id` and attaches it to every
completion/embedding row that gets persisted.

### Lifecycle of a client key

1. **Created** via `POST /v1/admin/keys` — see [06 — Admin API](./06-api-admin.md).
   The full key value is returned **once** in the `key` field; subsequent
   reads return only `key_hint` (a masked suffix, e.g. `"…xY9z"`).
2. **Active** by default. Stored with `status = "active"` and
   `allowed_models` either `NULL` (unrestricted) or a list.
3. **Revoked** via `POST /v1/admin/keys/:id/revoke`. Sets
   `status = "revoked"`. Revocation is **immediate and non-reversible**
   in the current schema — there is no "unrevoke" endpoint.

The key is 32 bytes of OS randomness, URL-safe-base64-encoded with no
padding and **no prefix** — a 43-character token (`generate_api_key` at
`src/storage/mod.rs:149`). 256 bits of entropy; the hint is the last 4
characters.

## Per-key model allow-lists

Each client key carries an optional `allowed_models` list
(`src/storage/mod.rs:14`–`39`). Three states:

| `allowed_models` | Behavior |
|---|---|
| `None` (field absent at creation) | Unrestricted. Every model request is permitted. |
| `Some(["a", "b", "c"])` | Only these exact `model` strings are permitted. Match is **case-sensitive** and **exact** against the request's `model` field **as-sent** — i.e. whatever the client put in the JSON, before OctoHub resolves the alias. |
| `Some([])` (empty list) | Lockout. **Every model request returns 403.** This is the right tool for disabling a key without revoking it. |

The check happens in `ensure_model_allowed` at `src/proxy/engine.rs:781`.
When it fails, the error embeds the literal `MODEL_FORBIDDEN_MARKER`
string. The HTTP layer (`classify_engine_error` at
`src/api/handler.rs:239`) detects that marker and returns **`403
Forbidden`** with the model name in the message:

```json
{
  "error": {
    "message": "model 'gpt-4o' is not permitted for this API key",
    "type": "invalid_request_error"
  }
}
```

(That 403 path is the **only** use of `MODEL_FORBIDDEN_MARKER` — it's a
routing hint, not a user-visible error code.)

### Matching against the request's `model` field

A common mistake: a key with `allowed_models = ["minimax-m2.7"]` does
**not** automatically allow `minimax:minimax-m2.7`, and vice versa. The
list must contain the exact string the client sends. If you want to
allow a mapped alias *and* a direct provider call, list both:

```bash
curl -sX POST http://127.0.0.1:8080/v1/admin/keys \
  -H "Authorization: Bearer your-master-secret" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "internal",
    "allowed_models": ["minimax-m2.7", "minimax:minimax-m2.7"]
  }'
```

### Allow-list use cases

- **Isolate teams** — the `frontend` key can only hit the cheap/fast
  model; the `eval-pipeline` key is the only one allowed to call the
  expensive reasoning model.
- **Quota-by-allow-list** — leave a single model assigned per key and
  count via the admin usage endpoint.
- **Kill switch without revocation** — `allowed_models: []` for a
  customer that disputed a charge. They can't be billed for new calls,
  the key record stays intact for audit.

## Admin auth in detail

`authenticate_admin(header, master_key)` is a single equality check
(`src/auth.rs:35`–`40`):

```rust
extract_bearer(header) == Some(master_key)
```

That's it. No hashing, no DB lookup, no per-request rotation. The master
key is read once at startup from `[server].api_key` and held in `Config`.
If you change it, you must restart the server.

**The empty-master-key edge case** — if `api_key = ""` and a client sends
`Authorization: Bearer ` (with a literal space and no token), the empty
strings compare equal and admin access is granted. Don't leave
`api_key` empty in production. The startup warning at `src/main.rs:64`
exists for a reason.

## Setting up auth end-to-end

```bash
# 1. Put a strong master key in octohub.toml
[server]
api_key = "lkAbaTWoBs6HLVG2Kf46"   # openssl rand -base64 24

# 2. Start the server — the startup line should show admin_auth=true
./target/release/octohub
# {"level":"INFO","message":"octohub starting","admin_auth":true,...}

# 3. Create your first client key
curl -sX POST http://127.0.0.1:8080/v1/admin/keys \
  -H "Authorization: Bearer lkAbaTWoBs6HLVG2Kf46" \
  -H "Content-Type: application/json" \
  -d '{"name":"default"}'
# → {"id":1,"key":"nYwT8kQ2v…","key_hint":"…xY9z","status":"active",
#    "allowed_models":null,"created_at":1717171717}

# 4. Use the client key
curl -sX POST http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer nYwT8kQ2v…" \
  -H "Content-Type: application/json" \
  -d '{"model":"minimax-m2.7","input":"hello"}'
```

The `key` value is the **only** time the full token is shown. Store it
in your secret manager immediately. Re-issuing requires revoking the old
key and creating a new one (no rotation endpoint).

## Common auth mistakes

| Symptom | Cause |
|---|---|
| 401 with "Invalid or revoked API key" on a brand-new key | Bearer prefix missing, or the key was typed wrong, or the key was created with an empty `allowed_models` and the model name doesn't match |
| 401 on **all** admin endpoints after editing config | `[server].api_key` is empty in the config the server actually loaded. Check the path passed to `-c`. The startup line shows `admin_auth=true|false`. |
| 403 with "model 'X' is not permitted for this API key" | The key's `allowed_models` is set and doesn't contain the literal `model` string you sent. See the matching section above. |
| 401 on admin endpoints when the master key is right | The header is `Authorization: Token …` instead of `Authorization: Bearer …`. The `extract_bearer` helper requires the `Bearer ` prefix. |
| Client request hangs forever (no error) | Auth succeeded, but the request is queued behind a per-provider concurrency limit. See [03 — Configuration](./03-configuration.md#providersname). |
