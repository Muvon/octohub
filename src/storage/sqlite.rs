use super::{
    generate_api_key, make_key_hint, now_unix, ApiKey, ListFilter, Storage, StoredCompletion,
    StoredEmbedding, TimeBucket, UsageRow,
};
use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension};
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
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;",
        )?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                key TEXT NOT NULL UNIQUE,
                key_hint TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_api_keys_key ON api_keys(key);
            CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status);

            CREATE TABLE IF NOT EXISTS completions (
                id TEXT PRIMARY KEY,
                api_key_id INTEGER NOT NULL REFERENCES api_keys(id),
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
            CREATE INDEX IF NOT EXISTS idx_completions_api_key ON completions(api_key_id);
            CREATE INDEX IF NOT EXISTS idx_completions_session ON completions(session_id);
            CREATE INDEX IF NOT EXISTS idx_completions_previous ON completions(previous_completion_id);
            CREATE INDEX IF NOT EXISTS idx_completions_created ON completions(created_at);

            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                api_key_id INTEGER NOT NULL REFERENCES api_keys(id),
                input_model TEXT NOT NULL,
                resolved_model TEXT NOT NULL,
                provider TEXT NOT NULL,
                input TEXT NOT NULL,
                usage TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_embeddings_api_key ON embeddings(api_key_id);
            CREATE INDEX IF NOT EXISTS idx_embeddings_created ON embeddings(created_at);",
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }
}

fn lock_conn(conn: &Mutex<Connection>) -> Result<std::sync::MutexGuard<'_, Connection>> {
    conn.lock()
        .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))
}

fn read_completion(row: &rusqlite::Row) -> rusqlite::Result<StoredCompletion> {
    Ok(StoredCompletion {
        id: row.get(0)?,
        api_key_id: row.get(1)?,
        session_id: row.get(2)?,
        previous_completion_id: row.get(3)?,
        input_model: row.get(4)?,
        resolved_model: row.get(5)?,
        provider: row.get(6)?,
        input: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default(),
        output: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
        instructions: row.get(9)?,
        exchange: serde_json::from_str(&row.get::<_, String>(10)?).unwrap_or_default(),
        usage: serde_json::from_str(&row.get::<_, String>(11)?).unwrap_or_default(),
        created_at: row.get(12)?,
    })
}

const COMPLETION_COLUMNS: &str =
    "id, api_key_id, session_id, previous_completion_id, input_model, resolved_model, provider, input, output, instructions, exchange, usage, created_at";

/// Build WHERE clause and params for key_ids + since/until filters.
/// Returns (clause_string, param_values) where clause starts with " WHERE " or is empty.
fn build_time_and_key_filter(
    key_ids: &[i64],
    since: Option<u64>,
    until: Option<u64>,
) -> (String, Vec<Box<dyn rusqlite::types::ToSql>>) {
    let mut conditions: Vec<String> = Vec::new();
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if !key_ids.is_empty() {
        let placeholders: Vec<String> = key_ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect();
        conditions.push(format!("api_key_id IN ({})", placeholders.join(", ")));
        for &kid in key_ids {
            params.push(Box::new(kid));
        }
    }

    if let Some(s) = since {
        let idx = params.len() + 1;
        conditions.push(format!("created_at >= ?{}", idx));
        params.push(Box::new(s as i64));
    }

    if let Some(u) = until {
        let idx = params.len() + 1;
        conditions.push(format!("created_at <= ?{}", idx));
        params.push(Box::new(u as i64));
    }

    let clause = if conditions.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", conditions.join(" AND "))
    };

    (clause, params)
}

impl Storage for SqliteStorage {
    fn create_api_key(&self, name: &str) -> Result<ApiKey> {
        let conn = lock_conn(&self.conn)?;
        let key = generate_api_key();
        let key_hint = make_key_hint(&key);
        let now = now_unix();

        conn.execute(
            "INSERT INTO api_keys (name, key, key_hint, status, created_at) VALUES (?1, ?2, ?3, 'active', ?4)",
            rusqlite::params![name, key, key_hint, now],
        )?;

        let id = conn.last_insert_rowid();

        Ok(ApiKey {
            id,
            name: name.to_string(),
            key,
            key_hint,
            status: "active".to_string(),
            created_at: now,
        })
    }

    fn list_api_keys(&self) -> Result<Vec<ApiKey>> {
        let conn = lock_conn(&self.conn)?;
        let mut stmt = conn
            .prepare("SELECT id, name, key_hint, status, created_at FROM api_keys ORDER BY id")?;
        let rows = stmt.query_map([], |row| {
            Ok(ApiKey {
                id: row.get(0)?,
                name: row.get(1)?,
                key: String::new(), // Never expose full key in list
                key_hint: row.get(2)?,
                status: row.get(3)?,
                created_at: row.get(4)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    fn get_api_key(&self, id: i64) -> Result<Option<ApiKey>> {
        let conn = lock_conn(&self.conn)?;
        conn.query_row(
            "SELECT id, name, key_hint, status, created_at FROM api_keys WHERE id = ?1",
            rusqlite::params![id],
            |row| {
                Ok(ApiKey {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    key: String::new(),
                    key_hint: row.get(2)?,
                    status: row.get(3)?,
                    created_at: row.get(4)?,
                })
            },
        )
        .optional()
        .map_err(Into::into)
    }

    fn revoke_api_key(&self, id: i64) -> Result<bool> {
        let conn = lock_conn(&self.conn)?;
        let affected = conn.execute(
            "UPDATE api_keys SET status = 'revoked' WHERE id = ?1 AND status = 'active'",
            rusqlite::params![id],
        )?;
        Ok(affected > 0)
    }

    fn get_api_key_by_key(&self, key: &str) -> Result<Option<ApiKey>> {
        let conn = lock_conn(&self.conn)?;
        conn.query_row(
            "SELECT id, name, key, key_hint, status, created_at FROM api_keys WHERE key = ?1",
            rusqlite::params![key],
            |row| {
                Ok(ApiKey {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    key: row.get(2)?,
                    key_hint: row.get(3)?,
                    status: row.get(4)?,
                    created_at: row.get(5)?,
                })
            },
        )
        .optional()
        .map_err(Into::into)
    }

    fn store_completion(&self, completion: &StoredCompletion) -> Result<()> {
        let conn = lock_conn(&self.conn)?;
        conn.execute(
            "INSERT INTO completions (id, api_key_id, session_id, previous_completion_id, input_model, resolved_model, provider, input, output, instructions, exchange, usage, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            rusqlite::params![
                completion.id,
                completion.api_key_id,
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
        let conn = lock_conn(&self.conn)?;
        let sql = format!(
            "SELECT {} FROM completions WHERE id = ?1",
            COMPLETION_COLUMNS
        );
        conn.query_row(&sql, rusqlite::params![id], read_completion)
            .optional()
            .map_err(Into::into)
    }

    fn get_session_id(&self, id: &str) -> Result<Option<String>> {
        let conn = lock_conn(&self.conn)?;
        conn.query_row(
            "SELECT session_id FROM completions WHERE id = ?1",
            rusqlite::params![id],
            |row| row.get(0),
        )
        .optional()
        .map_err(Into::into)
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

    fn list_completions(&self, filter: &ListFilter) -> Result<Vec<StoredCompletion>> {
        let conn = lock_conn(&self.conn)?;
        let (where_clause, filter_params) =
            build_time_and_key_filter(&filter.key_ids, filter.since, filter.until);

        let limit_idx = filter_params.len() + 1;
        let offset_idx = filter_params.len() + 2;
        let sql = format!(
            "SELECT {} FROM completions{} ORDER BY created_at DESC LIMIT ?{} OFFSET ?{}",
            COMPLETION_COLUMNS, where_clause, limit_idx, offset_idx
        );

        let mut all_params: Vec<Box<dyn rusqlite::types::ToSql>> = filter_params;
        let limit = if filter.limit == 0 {
            50
        } else {
            filter.limit.min(1000)
        };
        all_params.push(Box::new(limit));
        all_params.push(Box::new(filter.offset));

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            all_params.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(param_refs.as_slice(), read_completion)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    fn store_embedding(&self, embedding: &StoredEmbedding) -> Result<()> {
        let conn = lock_conn(&self.conn)?;
        conn.execute(
            "INSERT INTO embeddings (id, api_key_id, input_model, resolved_model, provider, input, usage, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                embedding.id,
                embedding.api_key_id,
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

    fn list_embeddings(&self, filter: &ListFilter) -> Result<Vec<StoredEmbedding>> {
        let conn = lock_conn(&self.conn)?;
        let (where_clause, filter_params) =
            build_time_and_key_filter(&filter.key_ids, filter.since, filter.until);

        let limit_idx = filter_params.len() + 1;
        let offset_idx = filter_params.len() + 2;
        let sql = format!(
            "SELECT id, api_key_id, input_model, resolved_model, provider, input, usage, created_at \
             FROM embeddings{} ORDER BY created_at DESC LIMIT ?{} OFFSET ?{}",
            where_clause, limit_idx, offset_idx
        );

        let mut all_params: Vec<Box<dyn rusqlite::types::ToSql>> = filter_params;
        let limit = if filter.limit == 0 {
            50
        } else {
            filter.limit.min(1000)
        };
        all_params.push(Box::new(limit));
        all_params.push(Box::new(filter.offset));

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            all_params.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            Ok(StoredEmbedding {
                id: row.get(0)?,
                api_key_id: row.get(1)?,
                input_model: row.get(2)?,
                resolved_model: row.get(3)?,
                provider: row.get(4)?,
                input: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
                usage: serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or_default(),
                created_at: row.get(7)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    fn get_usage(
        &self,
        key_ids: &[i64],
        bucket: Option<TimeBucket>,
        since: Option<u64>,
        until: Option<u64>,
    ) -> Result<Vec<UsageRow>> {
        let conn = lock_conn(&self.conn)?;

        // Build the time-bucketing expression for SQLite
        let bucket_expr = match bucket {
            Some(TimeBucket::Hour) => "(created_at / 3600) * 3600",
            Some(TimeBucket::Day) => "(created_at / 86400) * 86400",
            Some(TimeBucket::Week) => "(created_at / 604800) * 604800",
            Some(TimeBucket::Month) => "(created_at / 2592000) * 2592000",
            None => "0", // No bucketing — single aggregate
        };

        let (where_clause, filter_params) = build_time_and_key_filter(key_ids, since, until);

        // Query completions usage
        let compl_sql = format!(
            "SELECT api_key_id, {bucket} AS period, \
             COUNT(*) AS cnt, \
             COALESCE(SUM(json_extract(usage, '$.input_tokens')), 0) AS inp, \
             COALESCE(SUM(json_extract(usage, '$.output_tokens')), 0) AS outp \
             FROM completions{where_clause} \
             GROUP BY api_key_id, period",
            bucket = bucket_expr,
            where_clause = where_clause,
        );

        let emb_sql = format!(
            "SELECT api_key_id, {bucket} AS period, \
             COUNT(*) AS cnt, \
             COALESCE(SUM(json_extract(usage, '$.input_tokens')), 0) AS inp \
             FROM embeddings{where_clause} \
             GROUP BY api_key_id, period",
            bucket = bucket_expr,
            where_clause = where_clause,
        );

        // Collect completion stats
        use std::collections::HashMap;
        type BucketKey = (i64, u64); // (key_id, period)
        let mut usage_map: HashMap<BucketKey, UsageRow> = HashMap::new();

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            filter_params.iter().map(|p| p.as_ref()).collect();

        {
            let mut stmt = conn.prepare(&compl_sql)?;
            let rows = stmt.query_map(param_refs.as_slice(), |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, u64>(1)?,
                    row.get::<_, u64>(2)?,
                    row.get::<_, u64>(3)?,
                    row.get::<_, u64>(4)?,
                ))
            })?;
            for row in rows {
                let (key_id, period, cnt, inp, outp) = row?;
                let entry = usage_map
                    .entry((key_id, period))
                    .or_insert_with(|| UsageRow {
                        period: if bucket.is_some() { Some(period) } else { None },
                        key_id,
                        key_name: String::new(),
                        completions_count: 0,
                        embeddings_count: 0,
                        total_input_tokens: 0,
                        total_output_tokens: 0,
                    });
                entry.completions_count = cnt;
                entry.total_input_tokens += inp;
                entry.total_output_tokens += outp;
            }
        }

        // Collect embedding stats
        {
            let mut stmt = conn.prepare(&emb_sql)?;
            let rows = stmt.query_map(param_refs.as_slice(), |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, u64>(1)?,
                    row.get::<_, u64>(2)?,
                    row.get::<_, u64>(3)?,
                ))
            })?;
            for row in rows {
                let (key_id, period, cnt, inp) = row?;
                let entry = usage_map
                    .entry((key_id, period))
                    .or_insert_with(|| UsageRow {
                        period: if bucket.is_some() { Some(period) } else { None },
                        key_id,
                        key_name: String::new(),
                        completions_count: 0,
                        embeddings_count: 0,
                        total_input_tokens: 0,
                        total_output_tokens: 0,
                    });
                entry.embeddings_count = cnt;
                entry.total_input_tokens += inp;
            }
        }

        // Resolve key names
        let key_name_map: HashMap<i64, String> = {
            let mut stmt = conn.prepare("SELECT id, name FROM api_keys")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })?;
            rows.filter_map(|r| r.ok()).collect()
        };

        let mut results: Vec<UsageRow> = usage_map
            .into_values()
            .map(|mut row| {
                row.key_name = key_name_map.get(&row.key_id).cloned().unwrap_or_default();
                row
            })
            .collect();

        // Sort by period then key_id
        results.sort_by(|a, b| a.period.cmp(&b.period).then(a.key_id.cmp(&b.key_id)));

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_storage() -> SqliteStorage {
        SqliteStorage::new(":memory:").unwrap()
    }

    fn create_test_key(storage: &SqliteStorage) -> ApiKey {
        storage.create_api_key("test-key").unwrap()
    }

    fn make_stored_completion(
        id: &str,
        api_key_id: i64,
        prev: Option<&str>,
        input: &str,
        output: &str,
    ) -> StoredCompletion {
        StoredCompletion {
            id: id.to_string(),
            api_key_id,
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

    // ── API key tests ──

    #[test]
    fn test_create_api_key() {
        let storage = create_test_storage();
        let key = storage.create_api_key("my-app").unwrap();
        assert_eq!(key.name, "my-app");
        assert_eq!(key.status, "active");
        assert!(!key.key.is_empty());
        assert!(key.key_hint.starts_with("..."));
        assert_eq!(key.key_hint.len(), 7); // "..." + 4 chars
    }

    #[test]
    fn test_list_api_keys_hides_full_key() {
        let storage = create_test_storage();
        storage.create_api_key("key-1").unwrap();
        storage.create_api_key("key-2").unwrap();
        let keys = storage.list_api_keys().unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys[0].key.is_empty()); // Full key not exposed
        assert!(keys[1].key.is_empty());
    }

    #[test]
    fn test_get_api_key() {
        let storage = create_test_storage();
        let created = storage.create_api_key("my-app").unwrap();
        let fetched = storage.get_api_key(created.id).unwrap().unwrap();
        assert_eq!(fetched.name, "my-app");
        assert!(fetched.key.is_empty()); // Full key not exposed in get
    }

    #[test]
    fn test_revoke_api_key() {
        let storage = create_test_storage();
        let key = storage.create_api_key("my-app").unwrap();
        assert!(storage.revoke_api_key(key.id).unwrap());
        // Second revoke returns false (already revoked)
        assert!(!storage.revoke_api_key(key.id).unwrap());
        let fetched = storage.get_api_key(key.id).unwrap().unwrap();
        assert_eq!(fetched.status, "revoked");
    }

    #[test]
    fn test_get_api_key_by_key() {
        let storage = create_test_storage();
        let created = storage.create_api_key("my-app").unwrap();
        let found = storage.get_api_key_by_key(&created.key).unwrap().unwrap();
        assert_eq!(found.id, created.id);
        assert_eq!(found.name, "my-app");
    }

    #[test]
    fn test_get_api_key_by_key_not_found() {
        let storage = create_test_storage();
        let result = storage.get_api_key_by_key("nonexistent").unwrap();
        assert!(result.is_none());
    }

    // ── Completion tests ──

    #[test]
    fn test_store_and_retrieve() {
        let storage = create_test_storage();
        let key = create_test_key(&storage);
        let cmpl = make_stored_completion("cmpl_001", key.id, None, "Hello", "Hi there!");

        storage.store_completion(&cmpl).unwrap();
        let retrieved = storage.get_completion("cmpl_001").unwrap().unwrap();

        assert_eq!(retrieved.id, "cmpl_001");
        assert_eq!(retrieved.api_key_id, key.id);
        assert_eq!(retrieved.input_model, "gpt-4o");
        assert!(retrieved.previous_completion_id.is_none());
    }

    #[test]
    fn test_get_nonexistent() {
        let storage = create_test_storage();
        let result = storage.get_completion("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_walk_chain_single() {
        let storage = create_test_storage();
        let key = create_test_key(&storage);
        let cmpl = make_stored_completion("cmpl_001", key.id, None, "Hello", "Hi!");
        storage.store_completion(&cmpl).unwrap();

        let chain = storage.walk_chain("cmpl_001").unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].id, "cmpl_001");
    }

    #[test]
    fn test_walk_chain_multi() {
        let storage = create_test_storage();
        let key = create_test_key(&storage);

        storage
            .store_completion(&make_stored_completion(
                "cmpl_001",
                key.id,
                None,
                "What is Rust?",
                "Rust is a systems language",
            ))
            .unwrap();
        storage
            .store_completion(&make_stored_completion(
                "cmpl_002",
                key.id,
                Some("cmpl_001"),
                "And its async story?",
                "Tokio is the main runtime",
            ))
            .unwrap();
        storage
            .store_completion(&make_stored_completion(
                "cmpl_003",
                key.id,
                Some("cmpl_002"),
                "Tell me more",
                "Tokio provides...",
            ))
            .unwrap();

        let chain = storage.walk_chain("cmpl_003").unwrap();
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0].id, "cmpl_001");
        assert_eq!(chain[1].id, "cmpl_002");
        assert_eq!(chain[2].id, "cmpl_003");
    }

    #[test]
    fn test_walk_chain_missing_link() {
        let storage = create_test_storage();
        let key = create_test_key(&storage);
        storage
            .store_completion(&make_stored_completion(
                "cmpl_002",
                key.id,
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
        let storage = create_test_storage();
        let key = create_test_key(&storage);
        let mut cmpl = make_stored_completion("cmpl_001", key.id, None, "Hello", "Hi!");
        cmpl.instructions = Some("You are helpful".to_string());
        storage.store_completion(&cmpl).unwrap();

        let retrieved = storage.get_completion("cmpl_001").unwrap().unwrap();
        assert_eq!(retrieved.instructions.as_deref(), Some("You are helpful"));
    }

    #[test]
    fn test_list_completions_filtered() {
        let storage = create_test_storage();
        let key1 = storage.create_api_key("app-1").unwrap();
        let key2 = storage.create_api_key("app-2").unwrap();

        storage
            .store_completion(&make_stored_completion("c1", key1.id, None, "a", "b"))
            .unwrap();
        storage
            .store_completion(&make_stored_completion("c2", key2.id, None, "c", "d"))
            .unwrap();
        storage
            .store_completion(&make_stored_completion("c3", key1.id, None, "e", "f"))
            .unwrap();

        // Filter by key1 only
        let filter = ListFilter {
            key_ids: vec![key1.id],
            limit: 50,
            ..Default::default()
        };
        let results = storage.list_completions(&filter).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|c| c.api_key_id == key1.id));

        // No filter — all
        let filter = ListFilter {
            limit: 50,
            ..Default::default()
        };
        let results = storage.list_completions(&filter).unwrap();
        assert_eq!(results.len(), 3);
    }

    // ── Usage tests ──

    #[test]
    fn test_get_usage_total() {
        let storage = create_test_storage();
        let key = create_test_key(&storage);

        storage
            .store_completion(&make_stored_completion("c1", key.id, None, "a", "b"))
            .unwrap();
        storage
            .store_completion(&make_stored_completion("c2", key.id, None, "c", "d"))
            .unwrap();

        let usage = storage.get_usage(&[], None, None, None).unwrap();
        assert_eq!(usage.len(), 1);
        assert_eq!(usage[0].completions_count, 2);
        assert_eq!(usage[0].total_input_tokens, 20); // 10 * 2
        assert_eq!(usage[0].total_output_tokens, 10); // 5 * 2
    }
}
