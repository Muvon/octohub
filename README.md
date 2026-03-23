# OctoHub - High-Performance LLM Proxy Server

A high-performance LLM proxy server with completion chaining and full request/response logging.

## Features

- **LLM Proxy**: Route requests to multiple LLM providers with load balancing
- **Model Mapping**: Map short model names to fully qualified provider:model strings
- **Request/Response Logging**: Full logging of all API requests and responses
- **Authentication**: Optional API key authentication
- **SQLite Storage**: Persistent storage for logs and data

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
db_path = "octohub.db"
# api_key = "your-secret-key"  # Optional: set to enable authentication

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

## Usage

Start the server:

```bash
./octohub
```

The server will start on `http://127.0.0.1:8080` by default.

## License

Apache-2.0

## Credits

Developed by Muvon Un Limited.
