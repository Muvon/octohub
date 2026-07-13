use std::collections::HashMap;
use std::env;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Logging output format.
#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    /// Pretty if stdout is a terminal, JSON otherwise.
    #[default]
    Auto,
    /// Human-readable compact format.
    Pretty,
    /// Machine-readable JSON.
    Json,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct LoggingConfig {
    #[serde(default)]
    pub format: LogFormat,
    #[serde(default)]
    pub level: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MetricsConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_metrics_bind")]
    pub bind: String,
    #[serde(default)]
    pub per_key: bool,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bind: default_metrics_bind(),
            per_key: false,
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_metrics_bind() -> String {
    "127.0.0.1:9090".to_string()
}

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
    /// Embedding model mappings (same format as models)
    #[serde(default)]
    pub embedding_models: HashMap<String, Vec<String>>,
    /// Per-provider tuning (concurrency, etc.). Keyed by lowercase provider
    /// name as returned by octolib (`ollama`, `openai`, `anthropic`, ...).
    /// Providers not listed have no concurrency limit applied.
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,
    /// Metrics configuration
    #[serde(default)]
    pub metrics: MetricsConfig,
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
    pub api_key: String,
    /// Database URL (DSN). Supported schemes: sqlite://, mysql://, postgres://
    /// Examples:
    ///   sqlite://octohub.db       (or just a path for backward compat)
    ///   mysql://user:pass@host:3306/dbname
    ///   postgres://user:pass@host:5432/dbname
    #[serde(default = "default_db_url", alias = "db_path")]
    pub db_url: String,
    /// Trust X-Forwarded-For / Forwarded headers for remote IP detection.
    /// Only enable behind a trusted reverse proxy.
    #[serde(default)]
    pub trust_forwarded_for: bool,
    /// Maximum time to wait for a provider concurrency permit.
    #[serde(default = "default_provider_queue_timeout_secs")]
    pub provider_queue_timeout_secs: u64,
    /// Maximum time for the complete upstream provider operation, including
    /// octolib retries and response parsing.
    #[serde(default = "default_upstream_timeout_secs")]
    pub upstream_timeout_secs: u64,
    /// Retry the request on the next candidate provider when an upstream
    /// call fails with a provider-side error (timeout, connect, 429, 5xx).
    /// Only kicks in when the model alias has another candidate left; the
    /// request's own errors (4xx) always go straight back to the client.
    #[serde(default)]
    pub failover_on_error: bool,
    /// After 3 consecutive provider-side failures, deprioritize the provider
    /// for this many seconds — while cooling it is only used when no healthy
    /// candidate can admit the request. 0 = disabled.
    #[serde(default)]
    pub provider_error_cooldown_secs: u64,
}

/// Per-provider configuration knobs.
///
/// Every limit shares one contract: absent or `0` = unlimited.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ProviderConfig {
    /// Max in-flight requests this process will send to this provider.
    /// When the limit is reached, additional requests queue (the client
    /// connection hangs) until a permit is released. `None` = unlimited.
    #[serde(default)]
    pub concurrency: Option<u32>,
    /// Max requests admitted per fixed 60s window (provider "RPM").
    #[serde(default)]
    pub requests_per_minute: Option<u64>,
    /// Max tokens per fixed 60s window (provider "TPM"). Counted from the
    /// provider-reported usage (input + output) AFTER each response — a
    /// request is rejected once the window's budget is already spent, not
    /// pre-reserved.
    #[serde(default)]
    pub tokens_per_minute: Option<u64>,
    /// Max requests admitted per UTC day (provider "RPD").
    #[serde(default)]
    pub requests_per_day: Option<u64>,
    /// Max tokens per UTC day (provider "TPD").
    #[serde(default)]
    pub tokens_per_day: Option<u64>,
}

impl ProviderConfig {
    /// True when any request/token window is configured with a positive
    /// value (0 = unlimited, same contract as absent).
    pub fn has_rate_limits(&self) -> bool {
        [
            self.requests_per_minute,
            self.tokens_per_minute,
            self.requests_per_day,
            self.tokens_per_day,
        ]
        .iter()
        .any(|l| l.is_some_and(|n| n > 0))
    }
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_db_url() -> String {
    "sqlite://octohub.db".to_string()
}

fn default_provider_queue_timeout_secs() -> u64 {
    60
}

fn default_upstream_timeout_secs() -> u64 {
    6 * 60
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            api_key: String::new(),
            db_url: default_db_url(),
            trust_forwarded_for: false,
            provider_queue_timeout_secs: default_provider_queue_timeout_secs(),
            upstream_timeout_secs: default_upstream_timeout_secs(),
            failover_on_error: false,
            provider_error_cooldown_secs: 0,
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
            if let Ok(master_key) = env::var("OCTOHUB_MASTER_KEY") {
                config.server.api_key = master_key;
            }
            if let Ok(db_url) = env::var("OCTOHUB_DB_URL") {
                config.server.db_url = db_url;
            }
            Self::apply_env_overrides(&mut config);

            tracing::info!("Loaded config from {}", path);
            Ok(config)
        } else {
            // No config file - use defaults with env overrides
            tracing::info!("No config file specified, using defaults");
            Ok(Self::from_env())
        }
    }

    /// Apply environment variable overrides to a loaded config.
    fn apply_env_overrides(config: &mut Config) {
        if let Ok(fmt) = env::var("OCTOHUB_LOG_FORMAT") {
            match fmt.to_lowercase().as_str() {
                "auto" => config.logging.format = LogFormat::Auto,
                "pretty" => config.logging.format = LogFormat::Pretty,
                "json" => config.logging.format = LogFormat::Json,
                _ => {}
            }
        }
        if let Ok(level) = env::var("OCTOHUB_LOG_LEVEL") {
            config.logging.level = Some(level);
        }
        if let Ok(bind) = env::var("OCTOHUB_METRICS_BIND") {
            config.metrics.bind = bind;
        }
        if let Ok(val) = env::var("OCTOHUB_METRICS_ENABLED") {
            config.metrics.enabled = val == "true" || val == "1";
        }
        if let Ok(val) = env::var("OCTOHUB_PROVIDER_QUEUE_TIMEOUT_SECS") {
            if let Ok(seconds) = val.parse() {
                config.server.provider_queue_timeout_secs = seconds;
            }
        }
        if let Ok(val) = env::var("OCTOHUB_UPSTREAM_TIMEOUT_SECS") {
            if let Ok(seconds) = val.parse() {
                config.server.upstream_timeout_secs = seconds;
            }
        }
        if let Ok(val) = env::var("OCTOHUB_FAILOVER_ON_ERROR") {
            config.server.failover_on_error = val == "true" || val == "1";
        }
        if let Ok(val) = env::var("OCTOHUB_PROVIDER_ERROR_COOLDOWN_SECS") {
            if let Ok(seconds) = val.parse() {
                config.server.provider_error_cooldown_secs = seconds;
            }
        }
    }

    /// Load from environment variables only (fallback)
    fn from_env() -> Self {
        let mut config = Self {
            server: ServerConfig {
                host: env::var("OCTOHUB_HOST").unwrap_or_else(|_| default_host()),
                port: env::var("OCTOHUB_PORT")
                    .ok()
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(default_port()),
                api_key: env::var("OCTOHUB_MASTER_KEY").unwrap_or_default(),
                db_url: env::var("OCTOHUB_DB_URL").unwrap_or_else(|_| default_db_url()),
                trust_forwarded_for: false,
                provider_queue_timeout_secs: default_provider_queue_timeout_secs(),
                upstream_timeout_secs: default_upstream_timeout_secs(),
                failover_on_error: false,
                provider_error_cooldown_secs: 0,
            },
            models: HashMap::new(),
            embedding_models: HashMap::new(),
            providers: HashMap::new(),
            logging: LoggingConfig::default(),
            metrics: MetricsConfig::default(),
        };
        Self::apply_env_overrides(&mut config);
        config
    }

    /// All (provider, model_name) candidates for `model`, rotated to a
    /// random starting point so load still spreads across entries while
    /// letting the caller fall through to the next candidate when one is
    /// rate-limited. A direct "provider:model" input yields one candidate.
    pub fn model_candidates(&self, model: &str) -> Result<Vec<(String, String)>> {
        self.candidates_from_map(model, &self.models, "model")
    }

    /// Embedding counterpart of [`model_candidates`](Self::model_candidates).
    pub fn embedding_model_candidates(&self, model: &str) -> Result<Vec<(String, String)>> {
        self.candidates_from_map(model, &self.embedding_models, "embedding model")
    }

    /// Case-insensitive `[providers.<name>]` lookup — the limiter lowercases
    /// its keys the same way (`ProviderLimiter::from_config`).
    pub fn provider_config(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, cfg)| cfg)
    }

    fn candidates_from_map(
        &self,
        model: &str,
        map: &HashMap<String, Vec<String>>,
        kind: &str,
    ) -> Result<Vec<(String, String)>> {
        fn split(entry: &str, kind: &str) -> Result<(String, String)> {
            let pos = entry.find(':').with_context(|| {
                format!(
                    "Invalid {} mapping '{}': expected 'provider:model' format",
                    kind, entry
                )
            })?;
            Ok((entry[..pos].to_string(), entry[pos + 1..].to_string()))
        }

        // Model already in provider:model format — single fixed candidate.
        if model.contains(':') {
            return Ok(vec![split(model, kind)?]);
        }

        let providers = map.get(model).with_context(|| {
            format!(
                "{} '{}' not found in config. Available: {}",
                kind,
                model,
                map.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        })?;
        if providers.is_empty() {
            anyhow::bail!("{} '{}' has an empty provider list in config", kind, model);
        }

        let start = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as usize)
            % providers.len();

        (0..providers.len())
            .map(|i| split(&providers[(start + i) % providers.len()], kind))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_timeout_defaults_are_bounded() {
        let config = ServerConfig::default();
        assert_eq!(config.provider_queue_timeout_secs, 60);
        assert_eq!(config.upstream_timeout_secs, 360);
        assert!(!config.failover_on_error, "failover is opt-in");
        assert_eq!(config.provider_error_cooldown_secs, 0, "cooldown is opt-in");
    }

    #[test]
    fn failover_and_cooldown_deserialize_from_toml() {
        let config: Config = toml::from_str(
            r#"
            [server]
            api_key = ""
            failover_on_error = true
            provider_error_cooldown_secs = 120
            "#,
        )
        .unwrap();

        assert!(config.server.failover_on_error);
        assert_eq!(config.server.provider_error_cooldown_secs, 120);
    }

    #[test]
    fn provider_rate_limits_deserialize_from_toml() {
        let config: Config = toml::from_str(
            r#"
            [server]
            api_key = ""

            [providers.moonshot]
            concurrency = 50
            requests_per_minute = 200
            tokens_per_minute = 128000
            requests_per_day = 5000
            tokens_per_day = 1500000
            "#,
        )
        .unwrap();

        let cfg = config.provider_config("moonshot").unwrap();
        assert_eq!(cfg.concurrency, Some(50));
        assert_eq!(cfg.requests_per_minute, Some(200));
        assert_eq!(cfg.tokens_per_minute, Some(128000));
        assert_eq!(cfg.requests_per_day, Some(5000));
        assert_eq!(cfg.tokens_per_day, Some(1500000));
        assert!(cfg.has_rate_limits());
    }

    #[test]
    fn zero_and_unset_limits_mean_unlimited() {
        let none = ProviderConfig::default();
        assert!(!none.has_rate_limits());

        let zeroed = ProviderConfig {
            requests_per_minute: Some(0),
            tokens_per_minute: Some(0),
            requests_per_day: Some(0),
            tokens_per_day: Some(0),
            ..Default::default()
        };
        assert!(!zeroed.has_rate_limits());
    }

    #[test]
    fn provider_config_lookup_is_case_insensitive() {
        let mut providers = HashMap::new();
        providers.insert("OpenAI".to_string(), ProviderConfig::default());
        let config = Config {
            server: Default::default(),
            models: HashMap::new(),
            embedding_models: HashMap::new(),
            providers,
            logging: Default::default(),
            metrics: Default::default(),
        };
        assert!(config.provider_config("openai").is_some());
        assert!(config.provider_config("OPENAI").is_some());
        assert!(config.provider_config("ollama").is_none());
    }

    #[test]
    fn model_candidates_cover_all_entries_and_direct_format() {
        let mut models = HashMap::new();
        models.insert(
            "fast".to_string(),
            vec!["openai:gpt-5-nano".to_string(), "groq:llama".to_string()],
        );
        let config = Config {
            server: Default::default(),
            models,
            embedding_models: HashMap::new(),
            providers: HashMap::new(),
            logging: Default::default(),
            metrics: Default::default(),
        };

        let mut candidates = config.model_candidates("fast").unwrap();
        candidates.sort();
        assert_eq!(
            candidates,
            vec![
                ("groq".to_string(), "llama".to_string()),
                ("openai".to_string(), "gpt-5-nano".to_string()),
            ]
        );

        // Direct provider:model bypasses the mapping — one fixed candidate,
        // colons after the first stay in the model name (ollama tags).
        let direct = config.model_candidates("ollama:llama3.3:70b").unwrap();
        assert_eq!(
            direct,
            vec![("ollama".to_string(), "llama3.3:70b".to_string())]
        );

        assert!(config.model_candidates("unknown").is_err());
    }

    #[test]
    fn server_timeouts_deserialize_from_toml() {
        let config: Config = toml::from_str(
            r#"
            [server]
            api_key = ""
            provider_queue_timeout_secs = 45
            upstream_timeout_secs = 420
            "#,
        )
        .unwrap();

        assert_eq!(config.server.provider_queue_timeout_secs, 45);
        assert_eq!(config.server.upstream_timeout_secs, 420);
    }
}
