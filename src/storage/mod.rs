pub mod mysql;
pub mod postgres;
pub mod sqlite;

use anyhow::Result;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use rand::RngCore;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Stored API key record
#[derive(Debug, Clone)]
pub struct ApiKey {
    pub id: i64,
    pub name: String,
    /// Full key (only populated on creation)
    pub key: String,
    /// Masked hint for display (e.g., "...xY9z")
    pub key_hint: String,
    /// "active" or "revoked"
    pub status: String,
    /// Models this key may request (matched against the `model` field as
    /// sent — alias or `provider:model`). `None` means unrestricted; the
    /// admin omitted the list at creation time. An empty `Some(vec![])`
    /// is an explicit lockout (admin disabled all models).
    pub allowed_models: Option<Vec<String>>,
    /// Opaque grouping label for shared limits — e.g. the tenant/account that
    /// owns this key in the operator's own system. OctoHub attaches no
    /// semantics beyond "keys with the same owner share limits". `None` =
    /// ungrouped (behaves exactly as before the field existed).
    pub owner: Option<String>,
    /// Max in-flight proxy requests (completions + embeddings together)
    /// shared by ALL keys with the same `owner`. `None` or 0 = unlimited.
    pub owner_concurrency: Option<u32>,
    pub created_at: u64,
}

impl ApiKey {
    /// Check whether this key is permitted to call `model`. Unrestricted
    /// keys (`allowed_models == None`) pass everything through.
    pub fn is_model_allowed(&self, model: &str) -> bool {
        match &self.allowed_models {
            None => true,
            Some(list) => list.iter().any(|m| m == model),
        }
    }
}

/// Stored completion record from the database
#[derive(Debug, Clone)]
pub struct StoredCompletion {
    pub id: String,
    pub api_key_id: i64,
    pub session_id: String,
    pub previous_completion_id: Option<String>,
    /// Model name as sent by user (e.g., "minimax-m2.7")
    pub input_model: String,
    /// Resolved model sent to provider (e.g., "minimax-m2.7" - same as input for now)
    pub resolved_model: String,
    /// Provider name (e.g., "minimax")
    pub provider: String,
    pub input: serde_json::Value,
    pub output: serde_json::Value,
    pub instructions: Option<String>,
    pub exchange: serde_json::Value,
    pub usage: serde_json::Value,
    pub created_at: u64,
}

/// Stored embedding record from the database
#[derive(Debug, Clone)]
pub struct StoredEmbedding {
    pub id: String,
    pub api_key_id: i64,
    /// Model name as sent by user
    pub input_model: String,
    /// Resolved model sent to provider
    pub resolved_model: String,
    /// Provider name (e.g., "voyage")
    pub provider: String,
    /// Input texts as JSON array
    pub input: serde_json::Value,
    /// Usage stats (input_tokens, total_tokens, request_time_ms)
    pub usage: serde_json::Value,
    pub created_at: u64,
}

/// Aggregated usage row for reporting
#[derive(Debug, Clone)]
pub struct UsageRow {
    /// Time bucket start (unix timestamp), None for total aggregation
    pub period: Option<u64>,
    pub key_id: i64,
    pub key_name: String,
    pub completions_count: u64,
    pub embeddings_count: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
}

/// Query filters for listing completions/embeddings
#[derive(Debug, Default)]
pub struct ListFilter {
    pub key_ids: Vec<i64>,
    pub limit: u32,
    pub offset: u32,
    pub since: Option<u64>,
    pub until: Option<u64>,
}

/// Time bucket for usage aggregation
#[derive(Debug, Clone, Copy)]
pub enum TimeBucket {
    Hour,
    Day,
    Week,
    Month,
}

/// Storage trait for completion persistence
pub trait Storage: Send + Sync {
    // API key management
    /// Create a new API key. `allowed_models` is persisted as-is: `None`
    /// means unrestricted, `Some(list)` restricts to that exact set.
    /// `owner`/`owner_concurrency` group the key into a shared concurrency
    /// budget (see [`ApiKey::owner`]); both optional.
    fn create_api_key(
        &self,
        name: &str,
        allowed_models: Option<&[String]>,
        owner: Option<&str>,
        owner_concurrency: Option<u32>,
    ) -> Result<ApiKey>;
    fn list_api_keys(&self) -> Result<Vec<ApiKey>>;
    fn get_api_key(&self, id: i64) -> Result<Option<ApiKey>>;
    fn revoke_api_key(&self, id: i64) -> Result<bool>;
    /// Replace an active key's allowed-models list in place (`None` =
    /// unrestricted). The key value stays the same — this is how a plan
    /// change grants/removes models without breaking deployed credentials.
    fn update_api_key_models(&self, id: i64, allowed_models: Option<&[String]>) -> Result<bool>;
    /// Replace an active key's owner grouping + shared concurrency budget in
    /// place (same in-place contract as `update_api_key_models` — deployed
    /// credentials keep working while their limits follow the plan).
    fn update_api_key_owner(
        &self,
        id: i64,
        owner: Option<&str>,
        owner_concurrency: Option<u32>,
    ) -> Result<bool>;
    /// Look up an active API key by its raw key value (for auth)
    fn get_api_key_by_key(&self, key: &str) -> Result<Option<ApiKey>>;

    // Owner auto-model maps (virtual `auto` model, see proxy::auto)
    /// The stored purpose→alias override map for an owner group, or `None`
    /// when the owner has no override (the `[auto]` config floor applies).
    fn get_owner_auto_map(
        &self,
        owner: &str,
    ) -> Result<Option<std::collections::HashMap<String, String>>>;
    /// Replace an owner's auto map. `None` or an empty map clears the row —
    /// the owner falls back to the `[auto]` config floor.
    fn set_owner_auto_map(
        &self,
        owner: &str,
        map: Option<&std::collections::HashMap<String, String>>,
    ) -> Result<()>;

    // Completions
    fn store_completion(&self, completion: &StoredCompletion) -> Result<()>;
    #[allow(dead_code)]
    fn get_completion(&self, id: &str) -> Result<Option<StoredCompletion>>;
    /// Walk the chain of previous_completion_id links, returning completions oldest-first.
    /// Only populates the fields needed for chain replay: id, session_id,
    /// previous_completion_id, input, output, instructions.
    fn walk_chain(&self, id: &str) -> Result<Vec<StoredCompletion>>;
    fn list_completions(&self, filter: &ListFilter) -> Result<Vec<StoredCompletion>>;

    // Embeddings
    fn store_embedding(&self, embedding: &StoredEmbedding) -> Result<()>;
    fn list_embeddings(&self, filter: &ListFilter) -> Result<Vec<StoredEmbedding>>;

    // Usage
    fn get_usage(
        &self,
        key_ids: &[i64],
        bucket: Option<TimeBucket>,
        since: Option<u64>,
        until: Option<u64>,
    ) -> Result<Vec<UsageRow>>;
}

/// Generate a cryptographically secure API key (32 random bytes, base64url-encoded)
pub(crate) fn generate_api_key() -> String {
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

/// Build a masked hint from the last 4 characters of a key
pub(crate) fn make_key_hint(key: &str) -> String {
    let suffix: String = key
        .chars()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("...{}", suffix)
}

/// Encode an `allowed_models` list into the JSON text we persist. Returns
/// `None` for the unrestricted case so the column stores SQL NULL.
pub(crate) fn encode_allowed_models(models: Option<&[String]>) -> Option<String> {
    models.map(|m| serde_json::to_string(m).unwrap_or_else(|_| "[]".to_string()))
}

/// Encode an owner auto map for persistence (JSON object). Empty maps are
/// never stored — callers clear the row instead.
pub(crate) fn encode_auto_map(map: &std::collections::HashMap<String, String>) -> String {
    serde_json::to_string(map).unwrap_or_else(|_| "{}".to_string())
}

/// Decode a stored owner auto map. Invalid JSON decodes to `None` (config
/// floor applies) — same fail-open stance as `decode_allowed_models`: a
/// hand-edited row must degrade routing, not break it.
pub(crate) fn decode_auto_map(
    raw: Option<String>,
) -> Option<std::collections::HashMap<String, String>> {
    let raw = raw?;
    match serde_json::from_str::<std::collections::HashMap<String, String>>(&raw) {
        Ok(map) if !map.is_empty() => Some(map),
        Ok(_) => None,
        Err(err) => {
            tracing::warn!(error = %err, raw = %raw, "Invalid owner auto map JSON — falling back to [auto] config");
            None
        }
    }
}

/// Decode an `allowed_models` JSON string back into a list. Invalid JSON
/// falls back to unrestricted rather than locking the key out — fail open
/// for a hand-edited DB row beats a 403 for the operator.
pub(crate) fn decode_allowed_models(raw: Option<String>) -> Option<Vec<String>> {
    let raw = raw?;
    match serde_json::from_str::<Vec<String>>(&raw) {
        Ok(list) => Some(list),
        Err(err) => {
            tracing::warn!(error = %err, raw = %raw, "Invalid allowed_models JSON — treating key as unrestricted");
            None
        }
    }
}

/// Current unix timestamp in seconds
pub(crate) fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Create a storage backend from a DSN URL.
///
/// Supported schemes:
/// - `sqlite://path` or bare path (backward compat) → SQLite
/// - `mysql://user:pass@host:port/db` → MySQL
/// - `postgres://user:pass@host:port/db` → PostgreSQL
pub fn from_url(url: &str) -> Result<Arc<dyn Storage>> {
    if let Some(path) = url.strip_prefix("sqlite://") {
        Ok(Arc::new(sqlite::SqliteStorage::new(path)?))
    } else if url.starts_with("mysql://") {
        Ok(Arc::new(mysql::MysqlStorage::new(url)?))
    } else if url.starts_with("postgres://") || url.starts_with("postgresql://") {
        Ok(Arc::new(postgres::PostgresStorage::new(url)?))
    } else {
        // Bare path — treat as SQLite for backward compatibility
        Ok(Arc::new(sqlite::SqliteStorage::new(url)?))
    }
}
