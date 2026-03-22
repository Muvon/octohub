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

### Dependencies
- `octolib` - Shared library from parent directory (default-features disabled)
- `rusqlite` - SQLite database (bundled)
- `hyper` - HTTP server
- `tokio` - Async runtime
- `clap` - CLI argument parsing
- `tracing` - Structured logging

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
# api_key = "your-secret-key"  # Optional: set to enable authentication

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
