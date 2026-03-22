pub mod sqlite;

use anyhow::Result;

/// Stored response record from the database
#[derive(Debug, Clone)]
pub struct StoredResponse {
    pub id: String,
    pub session_id: String,
    pub previous_response_id: Option<String>,
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

/// Storage trait for response persistence
pub trait Storage: Send + Sync {
    fn store_response(&self, response: &StoredResponse) -> Result<()>;
    fn get_response(&self, id: &str) -> Result<Option<StoredResponse>>;
    /// Get session_id for a given response id (used to inherit session on chained requests)
    fn get_session_id(&self, id: &str) -> Result<Option<String>>;
    /// Walk the chain of previous_response_id links, returning responses oldest-first
    fn walk_chain(&self, id: &str) -> Result<Vec<StoredResponse>>;
    /// Store an embedding request for observability
    fn store_embedding(&self, embedding: &StoredEmbedding) -> Result<()>;
}
