pub mod sqlite;

use anyhow::Result;

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
    pub created_at: u64,
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
    fn create_api_key(&self, name: &str) -> Result<ApiKey>;
    fn list_api_keys(&self) -> Result<Vec<ApiKey>>;
    fn get_api_key(&self, id: i64) -> Result<Option<ApiKey>>;
    fn revoke_api_key(&self, id: i64) -> Result<bool>;
    /// Look up an active API key by its raw key value (for auth)
    fn get_api_key_by_key(&self, key: &str) -> Result<Option<ApiKey>>;

    // Completions
    fn store_completion(&self, completion: &StoredCompletion) -> Result<()>;
    fn get_completion(&self, id: &str) -> Result<Option<StoredCompletion>>;
    /// Get session_id for a given completion id (used to inherit session on chained requests)
    fn get_session_id(&self, id: &str) -> Result<Option<String>>;
    /// Walk the chain of previous_completion_id links, returning completions oldest-first
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
