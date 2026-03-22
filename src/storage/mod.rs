pub mod sqlite;

use anyhow::Result;

/// Stored response record from the database
#[derive(Debug, Clone)]
pub struct StoredResponse {
    pub id: String,
    pub previous_response_id: Option<String>,
    pub model: String,
    pub provider: String,
    pub input: serde_json::Value,
    pub output: serde_json::Value,
    pub instructions: Option<String>,
    pub exchange: serde_json::Value,
    pub usage: serde_json::Value,
    pub created_at: u64,
}

/// Storage trait for response persistence
pub trait Storage: Send + Sync {
    fn store_response(&self, response: &StoredResponse) -> Result<()>;
    fn get_response(&self, id: &str) -> Result<Option<StoredResponse>>;
    /// Walk the chain of previous_response_id links, returning responses oldest-first
    fn walk_chain(&self, id: &str) -> Result<Vec<StoredResponse>>;
}
