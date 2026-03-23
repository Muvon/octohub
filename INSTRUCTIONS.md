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
- `src/storage/mod.rs` - Storage trait and types
- `src/storage/sqlite.rs` - SQLite storage implementation
- `src/proxy/engine.rs` - Proxy engine (routes requests to providers)

### Dependencies
- `octolib` - Shared library from parent directory (default-features disabled)
- `rusqlite` - SQLite database (bundled)
- `hyper` - HTTP server
- `tokio` - Async runtime
- `clap` - CLI argument parsing
- `tracing` - Structured logging

## Authentication Architecture

**There are two completely independent authentication systems. They do not interact.**

### 1. Client Auth — DB keys (`/v1/completions`, `/v1/embeddings`)

- Keys are stored in the `api_keys` SQLite table
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
db_path = "octohub.db"
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
