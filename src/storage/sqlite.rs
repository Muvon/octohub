use super::{Storage, StoredResponse};
use anyhow::{Context, Result};
use rusqlite::Connection;
use std::sync::Mutex;

/// SQLite-backed storage implementation
pub struct SqliteStorage {
    conn: Mutex<Connection>,
}

impl SqliteStorage {
    /// Create a new SQLite storage, initializing the schema
    pub fn new(path: &str) -> Result<Self> {
        let conn = if path == ":memory:" {
            Connection::open_in_memory()?
        } else {
            Connection::open(path)?
        };

        // Enable WAL mode for better concurrent read performance
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS responses (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                previous_response_id TEXT,
                input_model TEXT NOT NULL,
                resolved_model TEXT NOT NULL,
                provider TEXT NOT NULL,
                input TEXT NOT NULL,
                output TEXT NOT NULL,
                instructions TEXT,
                exchange TEXT NOT NULL,
                usage TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_responses_session ON responses(session_id);
            CREATE INDEX IF NOT EXISTS idx_responses_previous ON responses(previous_response_id);
            CREATE INDEX IF NOT EXISTS idx_responses_created ON responses(created_at);",
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }
}

impl Storage for SqliteStorage {
    fn store_response(&self, response: &StoredResponse) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        conn.execute(
            "INSERT INTO responses (id, session_id, previous_response_id, input_model, resolved_model, provider, input, output, instructions, exchange, usage, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            rusqlite::params![
                response.id,
                response.session_id,
                response.previous_response_id,
                response.input_model,
                response.resolved_model,
                response.provider,
                response.input.to_string(),
                response.output.to_string(),
                response.instructions,
                response.exchange.to_string(),
                response.usage.to_string(),
                response.created_at,
            ],
        )?;
        Ok(())
    }

    fn get_response(&self, id: &str) -> Result<Option<StoredResponse>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        let mut stmt = conn.prepare(
            "SELECT id, session_id, previous_response_id, input_model, resolved_model, provider, input, output, instructions, exchange, usage, created_at
             FROM responses WHERE id = ?1",
        )?;

        let result = stmt.query_row(rusqlite::params![id], |row| {
            Ok(StoredResponse {
                id: row.get(0)?,
                session_id: row.get(1)?,
                previous_response_id: row.get(2)?,
                input_model: row.get(3)?,
                resolved_model: row.get(4)?,
                provider: row.get(5)?,
                input: serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or_default(),
                output: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default(),
                instructions: row.get(8)?,
                exchange: serde_json::from_str(&row.get::<_, String>(9)?).unwrap_or_default(),
                usage: serde_json::from_str(&row.get::<_, String>(10)?).unwrap_or_default(),
                created_at: row.get(11)?,
            })
        });

        match result {
            Ok(r) => Ok(Some(r)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn get_session_id(&self, id: &str) -> Result<Option<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        let mut stmt = conn.prepare("SELECT session_id FROM responses WHERE id = ?1")?;
        let result = stmt.query_row(rusqlite::params![id], |row| row.get(0));
        match result {
            Ok(session_id) => Ok(Some(session_id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn walk_chain(&self, id: &str) -> Result<Vec<StoredResponse>> {
        let mut chain = Vec::new();
        let mut current_id = Some(id.to_string());

        // Walk backwards through the chain
        while let Some(ref cid) = current_id {
            let response = self
                .get_response(cid)?
                .with_context(|| format!("Response '{}' not found in chain", cid))?;
            let prev = response.previous_response_id.clone();
            chain.push(response);
            current_id = prev;
        }

        // Reverse to get oldest-first order
        chain.reverse();
        Ok(chain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stored_response(
        id: &str,
        prev: Option<&str>,
        input: &str,
        output: &str,
    ) -> StoredResponse {
        StoredResponse {
            id: id.to_string(),
            session_id: format!("sess_test_{}", id),
            previous_response_id: prev.map(|s| s.to_string()),
            input_model: "gpt-4o".to_string(),
            resolved_model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            input: serde_json::json!({"content": input}),
            output: serde_json::json!({"content": output}),
            instructions: None,
            exchange: serde_json::json!({}),
            usage: serde_json::json!({"input_tokens": 10, "output_tokens": 5}),
            created_at: 1700000000,
        }
    }

    #[test]
    fn test_store_and_retrieve() {
        let storage = SqliteStorage::new(":memory:").unwrap();
        let resp = make_stored_response("resp_001", None, "Hello", "Hi there!");

        storage.store_response(&resp).unwrap();
        let retrieved = storage.get_response("resp_001").unwrap().unwrap();

        assert_eq!(retrieved.id, "resp_001");
        assert_eq!(retrieved.input_model, "gpt-4o");
        assert!(retrieved.previous_response_id.is_none());
    }

    #[test]
    fn test_get_nonexistent() {
        let storage = SqliteStorage::new(":memory:").unwrap();
        let result = storage.get_response("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_walk_chain_single() {
        let storage = SqliteStorage::new(":memory:").unwrap();
        let resp = make_stored_response("resp_001", None, "Hello", "Hi!");
        storage.store_response(&resp).unwrap();

        let chain = storage.walk_chain("resp_001").unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].id, "resp_001");
    }

    #[test]
    fn test_walk_chain_multi() {
        let storage = SqliteStorage::new(":memory:").unwrap();

        storage
            .store_response(&make_stored_response(
                "resp_001",
                None,
                "What is Rust?",
                "Rust is a systems language",
            ))
            .unwrap();
        storage
            .store_response(&make_stored_response(
                "resp_002",
                Some("resp_001"),
                "And its async story?",
                "Tokio is the main runtime",
            ))
            .unwrap();
        storage
            .store_response(&make_stored_response(
                "resp_003",
                Some("resp_002"),
                "Tell me more",
                "Tokio provides...",
            ))
            .unwrap();

        let chain = storage.walk_chain("resp_003").unwrap();
        assert_eq!(chain.len(), 3);
        // Oldest first
        assert_eq!(chain[0].id, "resp_001");
        assert_eq!(chain[1].id, "resp_002");
        assert_eq!(chain[2].id, "resp_003");
    }

    #[test]
    fn test_walk_chain_missing_link() {
        let storage = SqliteStorage::new(":memory:").unwrap();
        // resp_002 references resp_001 which doesn't exist
        storage
            .store_response(&make_stored_response(
                "resp_002",
                Some("resp_001"),
                "test",
                "test",
            ))
            .unwrap();

        let result = storage.walk_chain("resp_002");
        assert!(result.is_err());
    }

    #[test]
    fn test_store_with_instructions() {
        let storage = SqliteStorage::new(":memory:").unwrap();
        let mut resp = make_stored_response("resp_001", None, "Hello", "Hi!");
        resp.instructions = Some("You are helpful".to_string());
        storage.store_response(&resp).unwrap();

        let retrieved = storage.get_response("resp_001").unwrap().unwrap();
        assert_eq!(retrieved.instructions.as_deref(), Some("You are helpful"));
    }
}
