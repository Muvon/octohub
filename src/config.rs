use std::env;

/// Server configuration loaded from environment variables
pub struct Config {
    /// Host to bind to (default: 127.0.0.1)
    pub host: String,
    /// Port to bind to (default: 8080)
    pub port: u16,
    /// API key for authentication (optional, if not set auth is disabled)
    pub api_key: Option<String>,
    /// SQLite database path (default: octohub.db)
    pub db_path: String,
}

impl Config {
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        Self {
            host: env::var("OCTOHUB_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            port: env::var("OCTOHUB_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080),
            api_key: env::var("OCTOHUB_API_KEY").ok(),
            db_path: env::var("OCTOHUB_DB_PATH").unwrap_or_else(|_| "octohub.db".to_string()),
        }
    }
}
