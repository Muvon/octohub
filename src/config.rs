use std::collections::HashMap;
use std::env;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Server configuration loaded from TOML file (with env fallback)
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Server settings
    #[serde(default)]
    pub server: ServerConfig,
    /// Model mappings: model_name -> list of fully qualified "provider:model" strings
    /// Example: "minimax-m2.7" -> ["minimax:minimax-m2.7", "ollama:minimax-m2.7"]
    /// When resolving, randomly pick one from the list
    #[serde(default)]
    pub models: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    /// Host to bind to
    #[serde(default = "default_host")]
    pub host: String,
    /// Port to bind to
    #[serde(default = "default_port")]
    pub port: u16,
    /// API key for authentication (optional)
    pub api_key: Option<String>,
    /// SQLite database path
    #[serde(default = "default_db_path")]
    pub db_path: String,
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_db_path() -> String {
    "octohub.db".to_string()
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            api_key: None,
            db_path: default_db_path(),
        }
    }
}

impl Config {
    /// Load configuration from optional file path, with environment variable fallbacks
    /// If no path provided, use defaults (no config file)
    pub fn load(config_path: Option<String>) -> Result<Self> {
        // Try to load from config file if path provided
        if let Some(path) = config_path {
            let content = std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read config file: {}", path))?;
            let mut config: Config = toml::from_str(&content)
                .with_context(|| format!("Failed to parse config file: {}", path))?;

            // Override with environment variables if set
            if let Ok(host) = env::var("OCTOHUB_HOST") {
                config.server.host = host;
            }
            if let Some(port) = env::var("OCTOHUB_PORT").ok().and_then(|p| p.parse().ok()) {
                config.server.port = port;
            }
            if let Ok(api_key) = env::var("OCTOHUB_API_KEY") {
                config.server.api_key = Some(api_key);
            }
            if let Ok(db_path) = env::var("OCTOHUB_DB_PATH") {
                config.server.db_path = db_path;
            }

            tracing::info!("Loaded config from {}", path);
            Ok(config)
        } else {
            // No config file - use defaults with env overrides
            tracing::info!("No config file specified, using defaults");
            Ok(Self::from_env())
        }
    }

    /// Load from environment variables only (fallback)
    fn from_env() -> Self {
        Self {
            server: ServerConfig {
                host: env::var("OCTOHUB_HOST").unwrap_or_else(|_| default_host()),
                port: env::var("OCTOHUB_PORT")
                    .ok()
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(default_port()),
                api_key: env::var("OCTOHUB_API_KEY").ok(),
                db_path: env::var("OCTOHUB_DB_PATH").unwrap_or_else(|_| default_db_path()),
            },
            models: HashMap::new(),
        }
    }

    /// Resolve a model name to (provider, model_name)
    /// If model is already in "provider:model" format, use directly
    /// Otherwise look up in config and randomly pick one from the list
    pub fn resolve_model(&self, model: &str) -> Result<(String, String)> {
        // Check if model is already in provider:model format
        if let Some(pos) = model.find(':') {
            let provider = model[..pos].to_string();
            let model_name = model[pos + 1..].to_string();
            return Ok((provider, model_name));
        }

        // Look up model in config mapping
        let providers = self.models.get(model).with_context(|| {
            format!(
                "Model '{}' not found in config. Available models: {}",
                model,
                self.models.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        })?;

        // Randomly pick one provider from the list
        let idx = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as usize)
            % providers.len();

        let selected = &providers[idx];

        let pos = selected.find(':').context(format!(
            "Invalid model mapping '{}': expected 'provider:model' format",
            selected
        ))?;

        let provider = selected[..pos].to_string();
        let model_name = selected[pos + 1..].to_string();

        Ok((provider, model_name))
    }
}
