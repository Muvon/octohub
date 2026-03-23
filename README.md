# OctoHub - High-Performance LLM Proxy Server

A high-performance LLM proxy server with completion chaining, full request/response logging, and multi-tenant API key management.

## Features

- **LLM Proxy**: Route requests to multiple LLM providers with load balancing
- **Model Mapping**: Map short model names to fully qualified provider:model strings
- **Request/Response Logging**: Full logging of all API requests and responses
- **Multi-tenant API Keys**: Issue and revoke per-client API keys; all usage is tracked per key
- **Usage Analytics**: Aggregated usage stats by key and time bucket (hour/day/week/month)
- **Admin API**: Manage keys and query stored data via a master-key-protected admin API
- **Multi-Database Support**: SQLite (default), MySQL, and PostgreSQL backends

## Installation

### From Source

```bash
cargo build --release
./target/release/octohub
```

## Configuration

Create `octohub.toml` in your working directory:

```toml
# OctoHub Configuration

[server]
host = "127.0.0.1"
port = 8080
db_url = "sqlite://octohub.db"  # Database DSN (see below)
# api_key = "your-master-secret"  # Optional: enables authentication (see below)

# Model mappings: model_name -> list of fully qualified "provider:model" strings
# When resolving, randomly pick one from the list (simple load balancing)
# You can also use "provider:model" directly in API calls to bypass mapping

[models]
# Single provider
"minimax-m2.7" = ["minimax:minimax-m2.7"]

# Multiple providers (random selection)
# "my-model" = ["minimax:minimax-m2.7", "ollama:minimax-m2.7"]

[embedding_models]
"voyage-3.5" = ["voyage:voyage-3.5"]
```

### Database Configuration

OctoHub supports three database backends via the `db_url` setting:

| Backend | DSN format | Example |
|---|---|---|
| **SQLite** (default) | `sqlite://path` or bare path | `sqlite://octohub.db` |
| **MySQL** | `mysql://user:pass@host:port/db` | `mysql://root:secret@localhost:3306/octohub` |
| **PostgreSQL** | `postgres://user:pass@host:port/db` | `postgres://postgres:secret@localhost:5432/octohub` |

Schema is created automatically on first connection. No manual migration needed.

You can also set the database URL via the `OCTOHUB_DB_URL` environment variable (overrides config file).

> **Note**: `api_key` is optional. If omitted (or set to an empty string), the server starts without authentication — all client endpoints are open and admin endpoints are disabled. A warning is printed at startup. Set it to enable full auth.

## Usage

Start the server:

```bash
./octohub
```

The server will start on `http://127.0.0.1:8080` by default.

## Authentication Model

OctoHub uses two separate authentication layers:

| Layer | Endpoints | Key source |
|---|---|---|
| **Client** | `POST /v1/completions`, `POST /v1/embeddings` | API keys from the `api_keys` database table |
| **Admin** | `GET/POST /v1/admin/*` | Master key from `octohub.toml` (`api_key`) |

**Workflow:**
1. Set `api_key` in config (master key).
2. Use the admin API to create client keys: `POST /v1/admin/keys`.
3. Distribute client keys to your users/services.
4. All completions and embeddings are tagged with the issuing key ID for per-key usage tracking.

See [API.md](API.md) for full endpoint reference.

## License

Apache-2.0

## Credits

Developed by Muvon Un Limited.
