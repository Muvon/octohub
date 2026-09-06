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

/// Server-side media knobs. Every field maps onto an octolib
/// `RequestOptions` field; wait deadlines deliberately reuse
/// `server.upstream_timeout_secs` rather than growing a second timeout.
#[derive(Debug, Clone, Deserialize)]
pub struct MediaConfig {
    #[serde(default = "default_max_source_bytes")]
    pub max_source_bytes: usize,
    #[serde(default = "default_max_response_bytes")]
    pub max_response_bytes: usize,
    #[serde(default = "default_polling_interval_secs")]
    pub polling_interval_secs: u64,
    #[serde(default = "default_submit_timeout_secs")]
    pub submit_timeout_secs: u64,
}

impl Default for MediaConfig {
    fn default() -> Self {
        Self {
            max_source_bytes: default_max_source_bytes(),
            max_response_bytes: default_max_response_bytes(),
            polling_interval_secs: default_polling_interval_secs(),
            submit_timeout_secs: default_submit_timeout_secs(),
        }
    }
}

fn default_max_source_bytes() -> usize {
    20 * 1024 * 1024
}

fn default_max_response_bytes() -> usize {
    100 * 1024 * 1024
}

fn default_polling_interval_secs() -> u64 {
    2
}

fn default_submit_timeout_secs() -> u64 {
    120
}

/// Per-adapter endpoint override, keyed by octolib's provider name. Applied by
/// exporting the adapter's own base-URL environment variable at startup — the
/// adapters read it from the process environment, so there is exactly one
/// custom endpoint per adapter per deployment.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct MediaProviderConfig {
    pub api_base: Option<String>,
}

/// The base-URL environment variable each octolib media adapter reads.
pub const MEDIA_API_BASE_ENVS: &[(&str, &str)] = &[
    ("elevenlabs", "ELEVENLABS_API_URL"),
    ("fal", "FAL_API_URL"),
    ("openrouter", "OPENROUTER_MEDIA_API_URL"),
    ("replicate", "REPLICATE_API_URL"),
    ("runway", "RUNWAY_API_URL"),
];

fn default_metrics_bind() -> String {
    "127.0.0.1:9090".to_string()
}

/// The reserved name that triggers purpose-based routing when `[auto]` is
/// configured. Clients send `model: "auto"` plus an optional
/// `X-Model-Purpose` header; the proxy picks the real alias.
pub const AUTO_MODEL: &str = "auto";
/// The reserved purpose key that terminates the resolution chain.
pub const AUTO_DEFAULT_KEY: &str = "default";

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
    /// Media model mappings — identical shape and resolution to `models`:
    /// alias -> list of `provider:model` mirrors, tried in rotation. Pricing
    /// is octolib's (`media::reference_pricing`), so nothing else lives here.
    #[serde(default)]
    pub media_models: HashMap<String, Vec<String>>,
    /// The virtual `auto` model: purpose → model-alias map, the deployment's
    /// floor for purpose-based routing. When non-empty, a request for model
    /// "auto" is rewritten to the alias this map (or the key owner's stored
    /// override) picks for the request's `X-Model-Purpose` header, and THEN
    /// routed through `[models]` like any other request. The reserved
    /// `default` key is the ultimate fallback and must be present; every
    /// value must be a `[models]` alias. Absent/empty = `auto` is not a
    /// virtual model (a literal `[models.auto]` entry keeps working).
    #[serde(default)]
    pub auto: HashMap<String, String>,
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
    /// Media transport knobs.
    #[serde(default)]
    pub media: MediaConfig,
    /// Per-adapter media endpoint overrides, keyed by octolib provider name.
    #[serde(default)]
    pub media_providers: HashMap<String, MediaProviderConfig>,
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
            config.validate_auto()?;
            config.validate_media()?;
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

    /// True when `model` is the virtual `auto` model on this deployment.
    /// An empty `[auto]` means the feature is off and "auto" is an ordinary
    /// (probably unknown) model name — old behavior, untouched.
    pub fn is_auto(&self, model: &str) -> bool {
        !self.auto.is_empty() && model == AUTO_MODEL
    }

    /// Fail loudly at load on an `[auto]` section that cannot work. Every value
    /// must be a `[models]` alias (ONE indirection — auto resolves to an alias,
    /// the alias resolves to providers), `default` must exist so resolution
    /// always terminates, and a literal `[models.auto]` entry alongside the
    /// virtual model would be ambiguous.
    fn validate_auto(&self) -> Result<()> {
        if self.auto.is_empty() {
            return Ok(());
        }
        if self.models.contains_key(AUTO_MODEL) {
            anyhow::bail!(
                "[auto] is configured but '{}' is also defined in [models] — remove one",
                AUTO_MODEL
            );
        }
        if !self.auto.contains_key(AUTO_DEFAULT_KEY) {
            anyhow::bail!("[auto] must define a '{}' entry", AUTO_DEFAULT_KEY);
        }
        for (purpose, target) in &self.auto {
            if target == AUTO_MODEL {
                anyhow::bail!(
                    "[auto].{} must not point at '{}' itself",
                    purpose,
                    AUTO_MODEL
                );
            }
            if !self.models.contains_key(target) {
                anyhow::bail!(
                    "[auto].{} points at '{}', which is not defined in [models]",
                    purpose,
                    target
                );
            }
        }
        Ok(())
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
            media_models: HashMap::new(),
            auto: HashMap::new(),
            providers: HashMap::new(),
            logging: LoggingConfig::default(),
            metrics: MetricsConfig::default(),
            media: MediaConfig::default(),
            media_providers: HashMap::new(),
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

    /// Media counterpart of [`model_candidates`](Self::model_candidates).
    pub fn media_model_candidates(&self, model: &str) -> Result<Vec<(String, String)>> {
        self.candidates_from_map(model, &self.media_models, "media model")
    }

    /// Fail loudly at load on a `[media_models]` section that cannot work.
    /// An unknown provider or a malformed entry is a typo the operator should
    /// learn about at boot, not on the first paid request. There is no pricing
    /// validation: rates live in octolib, and an unpriced model degrades to a
    /// `cost_unavailable` warning on the response, not a config error.
    fn validate_media(&self) -> Result<()> {
        let supported = octolib::MediaProviderFactory::supported_providers();
        for (alias, entries) in &self.media_models {
            if self.models.contains_key(alias) || self.embedding_models.contains_key(alias) {
                anyhow::bail!(
                    "[media_models].{} collides with an alias already defined in [models] or [embedding_models]",
                    alias
                );
            }
            if entries.is_empty() {
                anyhow::bail!("[media_models].{} has an empty provider list", alias);
            }
            for entry in entries {
                let (provider, model) = entry.split_once(':').with_context(|| {
                    format!(
                        "[media_models].{} entry '{}' is not in 'provider:model' format",
                        alias, entry
                    )
                })?;
                if model.trim().is_empty() {
                    anyhow::bail!(
                        "[media_models].{} entry '{}' has an empty model",
                        alias,
                        entry
                    );
                }
                if !supported.iter().any(|s| s.eq_ignore_ascii_case(provider)) {
                    anyhow::bail!(
                        "[media_models].{} entry '{}' names unknown media provider '{}'. Available: {}",
                        alias,
                        entry,
                        provider,
                        supported.join(", ")
                    );
                }
            }
        }
        for name in self.media_providers.keys() {
            if !MEDIA_API_BASE_ENVS
                .iter()
                .any(|(provider, _)| provider.eq_ignore_ascii_case(name))
            {
                anyhow::bail!(
                    "[media_providers].{} is not an octolib media adapter. Available: {}",
                    name,
                    supported.join(", ")
                );
            }
        }
        Ok(())
    }

    /// Export the configured media endpoint overrides into the process
    /// environment, where the octolib adapters read them. Called once at
    /// startup; an already-set variable wins so an operator can still override
    /// the config from the environment.
    pub fn export_media_provider_endpoints(&self) {
        for (name, cfg) in &self.media_providers {
            let Some(api_base) = cfg
                .api_base
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
            else {
                continue;
            };
            let Some((_, variable)) = MEDIA_API_BASE_ENVS
                .iter()
                .find(|(provider, _)| provider.eq_ignore_ascii_case(name))
            else {
                continue;
            };
            if std::env::var(variable).is_ok() {
                tracing::info!(provider = %name, variable, "media endpoint override skipped; environment already set");
                continue;
            }
            std::env::set_var(variable, api_base);
            tracing::info!(provider = %name, api_base, "media endpoint override applied");
        }
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
            media_models: HashMap::new(),
            auto: HashMap::new(),
            providers,
            logging: Default::default(),
            metrics: Default::default(),
            media: Default::default(),
            media_providers: HashMap::new(),
        };
        assert!(config.provider_config("openai").is_some());
        assert!(config.provider_config("OPENAI").is_some());
        assert!(config.provider_config("ollama").is_none());
    }

    fn config_with_auto(models: &[(&str, &str)], auto: &[(&str, &str)]) -> Config {
        Config {
            server: Default::default(),
            models: models
                .iter()
                .map(|(k, v)| (k.to_string(), vec![v.to_string()]))
                .collect(),
            embedding_models: HashMap::new(),
            media_models: HashMap::new(),
            auto: auto
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            providers: HashMap::new(),
            logging: Default::default(),
            metrics: Default::default(),
            media: Default::default(),
            media_providers: HashMap::new(),
        }
    }

    #[test]
    fn auto_validation_accepts_a_complete_map_and_stays_off_when_empty() {
        let ok = config_with_auto(
            &[("glm", "zai:glm-4.7"), ("cheap", "deepseek:v4")],
            &[("default", "glm"), ("compression", "cheap")],
        );
        assert!(ok.validate_auto().is_ok());
        assert!(ok.is_auto("auto"));
        assert!(!ok.is_auto("glm"));

        let off = config_with_auto(&[("glm", "zai:glm-4.7")], &[]);
        assert!(off.validate_auto().is_ok());
        // Empty [auto] = feature off: "auto" is an ordinary model name.
        assert!(!off.is_auto("auto"));
    }

    #[test]
    fn auto_validation_fails_loudly_on_broken_maps() {
        // No default → resolution could dead-end.
        let c = config_with_auto(&[("glm", "z:g")], &[("compression", "glm")]);
        assert!(c
            .validate_auto()
            .unwrap_err()
            .to_string()
            .contains("default"));

        // Target not in [models] → typo caught at load, not at request time.
        let c = config_with_auto(&[("glm", "z:g")], &[("default", "nope")]);
        assert!(c.validate_auto().is_err());

        // A literal models.auto alongside [auto] is ambiguous.
        let c = config_with_auto(&[("auto", "z:g"), ("glm", "z:g")], &[("default", "glm")]);
        assert!(c.validate_auto().is_err());

        // Self-reference can never resolve.
        let c = config_with_auto(&[("glm", "z:g")], &[("default", "auto")]);
        assert!(c.validate_auto().is_err());
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
            media_models: HashMap::new(),
            auto: HashMap::new(),
            providers: HashMap::new(),
            logging: Default::default(),
            metrics: Default::default(),
            media: Default::default(),
            media_providers: HashMap::new(),
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
    fn media_models_deserialize_and_resolve_like_models() {
        let config: Config = toml::from_str(
            r#"
            [server]
            api_key = ""

            [media_models]
            flux = ["fal:fal-ai/flux/dev", "replicate:black-forest-labs/flux-1.1-pro"]

            [media]
            max_source_bytes = 1024
            polling_interval_secs = 5

            [media_providers.fal]
            api_base = "https://fal.internal.example"
            "#,
        )
        .unwrap();

        assert!(config.validate_media().is_ok());
        assert_eq!(config.media.max_source_bytes, 1024);
        assert_eq!(config.media.polling_interval_secs, 5);
        // Unset media knobs keep their defaults rather than zeroing out.
        assert_eq!(config.media.submit_timeout_secs, 120);

        let mut candidates = config.media_model_candidates("flux").unwrap();
        candidates.sort();
        assert_eq!(
            candidates,
            vec![
                ("fal".to_string(), "fal-ai/flux/dev".to_string()),
                (
                    "replicate".to_string(),
                    "black-forest-labs/flux-1.1-pro".to_string()
                ),
            ]
        );

        // A literal provider:model bypasses the map, same as [models].
        assert_eq!(
            config.media_model_candidates("runway:gen4_turbo").unwrap(),
            vec![("runway".to_string(), "gen4_turbo".to_string())]
        );
        assert!(config.media_model_candidates("unknown").is_err());
    }

    fn media_config(media: &[(&str, &[&str])], models: &[(&str, &str)]) -> Config {
        Config {
            media_models: media
                .iter()
                .map(|(k, v)| {
                    (
                        k.to_string(),
                        v.iter().map(|e| e.to_string()).collect::<Vec<_>>(),
                    )
                })
                .collect(),
            models: models
                .iter()
                .map(|(k, v)| (k.to_string(), vec![v.to_string()]))
                .collect(),
            server: Default::default(),
            embedding_models: HashMap::new(),
            auto: HashMap::new(),
            providers: HashMap::new(),
            logging: Default::default(),
            metrics: Default::default(),
            media: Default::default(),
            media_providers: HashMap::new(),
        }
    }

    /// A typo here costs real money on the first request, so it has to fail at
    /// boot instead.
    #[test]
    fn media_validation_fails_loudly_on_broken_maps() {
        // Unknown provider.
        let c = media_config(&[("flux", &["nope:some-model"])], &[]);
        assert!(c
            .validate_media()
            .unwrap_err()
            .to_string()
            .contains("unknown media provider"));

        // Not provider:model.
        assert!(media_config(&[("flux", &["fal-ai/flux/dev"])], &[])
            .validate_media()
            .is_err());

        // Empty model half.
        assert!(media_config(&[("flux", &["fal:"])], &[])
            .validate_media()
            .is_err());

        // Empty mirror list resolves to nothing.
        assert!(media_config(&[("flux", &[])], &[])
            .validate_media()
            .is_err());

        // An alias shared with [models] makes routing ambiguous.
        let c = media_config(
            &[("flux", &["fal:fal-ai/flux/dev"])],
            &[("flux", "openai:gpt-5")],
        );
        assert!(c
            .validate_media()
            .unwrap_err()
            .to_string()
            .contains("collides"));

        // An override for something that is not an adapter is a typo.
        let mut c = media_config(&[], &[]);
        c.media_providers
            .insert("openai".to_string(), MediaProviderConfig::default());
        assert!(c.validate_media().is_err());
    }

    #[test]
    fn media_validation_accepts_every_supported_adapter() {
        for provider in octolib::MediaProviderFactory::supported_providers() {
            let entry = format!("{provider}:some/model");
            let config = media_config(&[("alias", &[entry.as_str()])], &[]);
            assert!(
                config.validate_media().is_ok(),
                "{provider} should be accepted"
            );
        }
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
