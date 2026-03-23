use super::{Storage, StoredCompletion, StoredEmbedding};
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
            "CREATE TABLE IF NOT EXISTS completions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                previous_completion_id TEXT,
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
            CREATE INDEX IF NOT EXISTS idx_completions_session ON completions(session_id);
            CREATE INDEX IF NOT EXISTS idx_completions_previous ON completions(previous_completion_id);
            CREATE INDEX IF NOT EXISTS idx_completions_created ON completions(created_at);

            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                input_model TEXT NOT NULL,
                resolved_model TEXT NOT NULL,
                provider TEXT NOT NULL,
                input TEXT NOT NULL,
                usage TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_embeddings_created ON embeddings(created_at);",
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }
}

impl Storage for SqliteStorage {
    fn store_completion(&self, completion: &StoredCompletion) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        conn.execute(
            "INSERT INTO completions (id, session_id, previous_completion_id, input_model, resolved_model, provider, input, output, instructions, exchange, usage, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            rusqlite::params![
                completion.id,
                completion.session_id,
                completion.previous_completion_id,
                completion.input_model,
                completion.resolved_model,
                completion.provider,
                completion.input.to_string(),
                completion.output.to_string(),
                completion.instructions,
                completion.exchange.to_string(),
                completion.usage.to_string(),
                completion.created_at,
            ],
        )?;
        Ok(())
    }

    fn get_completion(&self, id: &str) -> Result<Option<StoredCompletion>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        let mut stmt = conn.prepare(
            "SELECT id, session_id, previous_completion_id, input_model, resolved_model, provider, input, output, instructions, exchange, usage, created_at
             FROM completions WHERE id = ?1",
        )?;

        let result = stmt.query_row(rusqlite::params![id], |row| {
            Ok(StoredCompletion {
                id: row.get(0)?,
                session_id: row.get(1)?,
                previous_completion_id: row.get(2)?,
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
        let mut stmt = conn.prepare("SELECT session_id FROM completions WHERE id = ?1")?;
        let result = stmt.query_row(rusqlite::params![id], |row| row.get(0));
        match result {
            Ok(session_id) => Ok(Some(session_id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn walk_chain(&self, id: &str) -> Result<Vec<StoredCompletion>> {
        let mut chain = Vec::new();
        let mut current_id = Some(id.to_string());

        // Walk backwards through the chain
        while let Some(ref cid) = current_id {
            let completion = self
                .get_completion(cid)?
                .with_context(|| format!("Completion '{}' not found in chain", cid))?;
            let prev = completion.previous_completion_id.clone();
            chain.push(completion);
            current_id = prev;
        }

        // Reverse to get oldest-first order
        chain.reverse();
        Ok(chain)
    }

    fn store_embedding(&self, embedding: &StoredEmbedding) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        conn.execute(
            "INSERT INTO embeddings (id, input_model, resolved_model, provider, input, usage, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                embedding.id,
                embedding.input_model,
                embedding.resolved_model,
                embedding.provider,
                embedding.input.to_string(),
                embedding.usage.to_string(),
                embedding.created_at,
            ],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stored_completion(
        id: &str,
        prev: Option<&str>,
        input: &str,
        output: &str,
    ) -> StoredCompletion {
        StoredCompletion {
            id: id.to_string(),
            session_id: format!("sess_test_{}", id),
            previous_completion_id: prev.map(|s| s.to_string()),
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
        let cmpl = make_stored_completion("cmpl_001", None, "Hello", "Hi there!");

        storage.store_completion(&cmpl).unwrap();
        let retrieved = storage.get_completion("cmpl_001").unwrap().unwrap();

        assert_eq!(retrieved.id, "cmpl_001");
        assert_eq!(retrieved.input_model, "gpt-4o");
        assert!(retrieved.previous_completion_id.is_none());
    }

    #[test]
    fn test_get_nonexistent() {
        let storage = SqliteStorage::new(":memory:").unwrap();
        let result = storage.get_completion("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_walk_chain_single() {
        let storage = SqliteStorage::new(":memory:").unwrap();
        let cmpl = make_stored_completion("cmpl_001", None, "Hello", "Hi!");
        storage.store_completion(&cmpl).unwrap();

        let chain = storage.walk_chain("cmpl_001").unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].id, "cmpl_001");
    }

    #[test]
    fn test_walk_chain_multi() {
        let storage = SqliteStorage::new(":memory:").unwrap();

        storage
            .store_completion(&make_stored_completion(
                "cmpl_001",
                None,
                "What is Rust?",
                "Rust is a systems language",
            ))
            .unwrap();
        storage
            .store_completion(&make_stored_completion(
                "cmpl_002",
                Some("cmpl_001"),
                "And its async story?",
                "Tokio is the main runtime",
            ))
            .unwrap();
        storage
            .store_completion(&make_stored_completion(
                "cmpl_003",
                Some("cmpl_002"),
                "Tell me more",
                "Tokio provides...",
            ))
            .unwrap();

        let chain = storage.walk_chain("cmpl_003").unwrap();
        assert_eq!(chain.len(), 3);
        // Oldest first
        assert_eq!(chain[0].id, "cmpl_001");
        assert_eq!(chain[1].id, "cmpl_002");
        assert_eq!(chain[2].id, "cmpl_003");
    }

    #[test]
    fn test_walk_chain_missing_link() {
        let storage = SqliteStorage::new(":memory:").unwrap();
        // cmpl_002 references cmpl_001 which doesn't exist
        storage
            .store_completion(&make_stored_completion(
                "cmpl_002",
                Some("cmpl_001"),
                "test",
                "test",
            ))
            .unwrap();

        let result = storage.walk_chain("cmpl_002");
        assert!(result.is_err());
    }

    #[test]
    fn test_store_with_instructions() {
        let storage = SqliteStorage::new(":memory:").unwrap();
        let mut cmpl = make_stored_completion("cmpl_001", None, "Hello", "Hi!");
        cmpl.instructions = Some("You are helpful".to_string());
        storage.store_completion(&cmpl).unwrap();

        let retrieved = storage.get_completion("cmpl_001").unwrap().unwrap();
        assert_eq!(retrieved.instructions.as_deref(), Some("You are helpful"));
    }
}
