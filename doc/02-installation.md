# 02 — Installation

## Requirements

- **Rust** 1.75 or newer (the project uses `edition = "2021"` and `hyper` 1
  / `clap` 4; see [`Cargo.toml`](../../Cargo.toml))
- A C toolchain (only required for the SQLite backend's bundled feature
  — see [Disabling bundled SQLite](#disabling-bundled-sqlite))
- ~50 MB of disk for the release build

## Build from source

```bash
git clone <your-octohub-fork> octohub
cd octohub
cargo build --release
```

The binary lands at `target/release/octohub`.

For development:

```bash
cargo build              # debug build, faster compile, slower runtime
cargo check              # type-check only, fastest
cargo test               # run the test suite
cargo clippy --all-targets -- -D warnings   # lint (the project uses -D warnings)
cargo fmt --all -- --check                 # format check
```

The `cargo clippy -- -D warnings` line is enforced in the project's
release pipeline — see `INSTRUCTIONS.md` at the repo root for the
contributor rules.

## First run

The server needs a config file. The minimum useful config is:

```toml
# octohub.toml
[server]
host = "127.0.0.1"
port = 8080
api_key = "your-master-secret"   # protects /v1/admin/*
db_url = "sqlite://octohub.db"   # SQLite, MySQL, or PostgreSQL DSN

[models]
"minimax-m2.7" = ["minimax:minimax-m2.7"]
```

Start the server:

```bash
./target/release/octohub
# or
./target/release/octohub -c /path/to/octohub.toml
# or
./target/release/octohub --bind 0.0.0.0:8080   # overrides [server] host/port
```

You should see a structured startup line in JSON or pretty format
(depending on whether stdout is a TTY):

```json
{"timestamp":"...","level":"INFO","message":"octohub starting","version":"0.1.0","bind":"127.0.0.1:8080","db":"sqlite","admin_auth":true,"providers":"none","models":1,"embed_models":0,"metrics":true}
```

(See [07 — Observability](./07-observability.md) for the full list of
startup fields.)

Hit the health endpoint to confirm the server is up:

```bash
curl http://127.0.0.1:8080/health
# → {"status":"ok"}
```

## CLI flags

OctoHub uses `clap` derive. The binary accepts:

| Flag | Effect |
|---|---|
| `-c`, `--config <PATH>` | Path to a TOML config file. If omitted, defaults + env vars only. |
| `--bind <HOST:PORT>` | Override `[server].host` and `[server].port` from the config. |

The `--bind` flag is parsed by `Args::parse()` in `src/main.rs:33`.
Invalid input fails fast with a clear error.

## Database setup

The schema is created automatically the first time the binary connects.
**No manual migration step is needed** — see
[`src/storage/sqlite.rs:27`](../../src/storage/sqlite.rs) for the
`CREATE TABLE IF NOT EXISTS` statements. The three supported backends
are documented in [03 — Configuration](./03-configuration.md#database).

For MySQL or PostgreSQL, make sure the database exists and the user has
`CREATE TABLE` and `ALTER TABLE` privileges. SQLite needs no setup
beyond a writable directory.

## Disabling bundled SQLite

The default `Cargo.toml` pulls in SQLite with the `bundled` feature so
the binary is self-contained. If your host already has a SQLite library
you want to link against, edit the `rusqlite` line in `Cargo.toml`:

```toml
rusqlite = { version = "0.35" }   # drop the `bundled` feature
```

## Verifying the install

A quick end-to-end check:

```bash
# 1. Health
curl -s http://127.0.0.1:8080/health

# 2. Create a client key (uses the master key from config)
KEY=$(curl -sX POST http://127.0.0.1:8080/v1/admin/keys \
  -H "Authorization: Bearer your-master-secret" \
  -H "Content-Type: application/json" \
  -d '{"name":"smoke-test"}' | jq -r .key)

# 3. Make a completion
curl -sX POST http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"minimax-m2.7","input":"hello"}'
```

If step 3 returns a 4xx, jump to [10 — Troubleshooting](./10-troubleshooting.md).

## Updating

```bash
git pull
cargo build --release
```

The database schema is migrated in place on connect
(see the `ensure_column` calls in `src/storage/sqlite.rs:77` for
examples of additive migration logic). If you're upgrading from a very
old version, back up the DB first.
