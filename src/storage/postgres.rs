use super::{
    generate_api_key, make_key_hint, now_unix, ApiKey, ListFilter, Storage, StoredCompletion,
    StoredEmbedding, TimeBucket, UsageRow,
};
use anyhow::{Context, Result};
use postgres::{Client, NoTls};
use std::collections::HashMap;
use std::sync::Mutex;

/// PostgreSQL-backed storage implementation
pub struct PostgresStorage {
    client: Mutex<Client>,
}

impl PostgresStorage {
    /// Create a new PostgreSQL storage, initializing the schema
    pub fn new(url: &str) -> Result<Self> {
        let client = Client::connect(url, NoTls).context("Failed to connect to PostgreSQL")?;
        let storage = Self {
            client: Mutex::new(client),
        };
        storage.init_schema()?;
        Ok(storage)
    }

    fn init_schema(&self) -> Result<()> {
        let mut client = self.lock_client()?;
        client.batch_execute(
            "CREATE TABLE IF NOT EXISTS api_keys (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                key TEXT NOT NULL UNIQUE,
                key_hint TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at BIGINT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_api_keys_key ON api_keys(key);
            CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status);

            CREATE TABLE IF NOT EXISTS completions (
                id TEXT PRIMARY KEY,
                api_key_id BIGINT NOT NULL REFERENCES api_keys(id),
                session_id TEXT NOT NULL,
                previous_completion_id TEXT,
                input_model TEXT NOT NULL,
                resolved_model TEXT NOT NULL,
                provider TEXT NOT NULL,
                input JSONB NOT NULL,
                output JSONB NOT NULL,
                instructions TEXT,
                exchange JSONB NOT NULL,
                usage JSONB NOT NULL,
                created_at BIGINT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_completions_api_key ON completions(api_key_id);
            CREATE INDEX IF NOT EXISTS idx_completions_session ON completions(session_id);
            CREATE INDEX IF NOT EXISTS idx_completions_previous ON completions(previous_completion_id);
            CREATE INDEX IF NOT EXISTS idx_completions_created ON completions(created_at);

            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                api_key_id BIGINT NOT NULL REFERENCES api_keys(id),
                input_model TEXT NOT NULL,
                resolved_model TEXT NOT NULL,
                provider TEXT NOT NULL,
                input JSONB NOT NULL,
                usage JSONB NOT NULL,
                created_at BIGINT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_embeddings_api_key ON embeddings(api_key_id);
            CREATE INDEX IF NOT EXISTS idx_embeddings_created ON embeddings(created_at);",
        )?;
        Ok(())
    }

    fn lock_client(&self) -> Result<std::sync::MutexGuard<'_, Client>> {
        self.client
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))
    }
}

/// Build WHERE clause with numbered placeholders ($1, $2, ...) for PostgreSQL.
/// Returns (clause, params, next_param_index).
fn build_filter(
    key_ids: &[i64],
    since: Option<u64>,
    until: Option<u64>,
) -> (String, Vec<Box<dyn postgres::types::ToSql + Sync>>, usize) {
    let mut conditions: Vec<String> = Vec::new();
    let mut params: Vec<Box<dyn postgres::types::ToSql + Sync>> = Vec::new();
    let mut idx = 1usize;

    if !key_ids.is_empty() {
        let placeholders: Vec<String> = key_ids
            .iter()
            .map(|_| {
                let p = format!("${}", idx);
                idx += 1;
                p
            })
            .collect();
        conditions.push(format!("api_key_id IN ({})", placeholders.join(", ")));
        for &kid in key_ids {
            params.push(Box::new(kid));
        }
    }

    if let Some(s) = since {
        conditions.push(format!("created_at >= ${}", idx));
        params.push(Box::new(s as i64));
        idx += 1;
    }

    if let Some(u) = until {
        conditions.push(format!("created_at <= ${}", idx));
        params.push(Box::new(u as i64));
        idx += 1;
    }

    let clause = if conditions.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", conditions.join(" AND "))
    };

    (clause, params, idx)
}

fn read_completion(row: &postgres::Row) -> StoredCompletion {
    StoredCompletion {
        id: row.get("id"),
        api_key_id: row.get("api_key_id"),
        session_id: row.get("session_id"),
        previous_completion_id: row.get("previous_completion_id"),
        input_model: row.get("input_model"),
        resolved_model: row.get("resolved_model"),
        provider: row.get("provider"),
        input: row.get("input"),
        output: row.get("output"),
        instructions: row.get("instructions"),
        exchange: row.get("exchange"),
        usage: row.get("usage"),
        created_at: row.get::<_, i64>("created_at") as u64,
    }
}

fn read_embedding(row: &postgres::Row) -> StoredEmbedding {
    StoredEmbedding {
        id: row.get("id"),
        api_key_id: row.get("api_key_id"),
        input_model: row.get("input_model"),
        resolved_model: row.get("resolved_model"),
        provider: row.get("provider"),
        input: row.get("input"),
        usage: row.get("usage"),
        created_at: row.get::<_, i64>("created_at") as u64,
    }
}

fn effective_limit(limit: u32) -> i64 {
    if limit == 0 {
        50
    } else {
        limit.min(1000) as i64
    }
}

impl Storage for PostgresStorage {
    fn create_api_key(&self, name: &str) -> Result<ApiKey> {
        let mut client = self.lock_client()?;
        let key = generate_api_key();
        let key_hint = make_key_hint(&key);
        let now = now_unix() as i64;

        let row = client.query_one(
            "INSERT INTO api_keys (name, key, key_hint, status, created_at) VALUES ($1, $2, $3, 'active', $4) RETURNING id",
            &[&name, &key, &key_hint, &now],
        )?;
        let id: i64 = row.get(0);

        Ok(ApiKey {
            id,
            name: name.to_string(),
            key,
            key_hint,
            status: "active".to_string(),
            created_at: now as u64,
        })
    }

    fn list_api_keys(&self) -> Result<Vec<ApiKey>> {
        let mut client = self.lock_client()?;
        let rows = client.query(
            "SELECT id, name, key_hint, status, created_at FROM api_keys ORDER BY id",
            &[],
        )?;
        Ok(rows
            .iter()
            .map(|row| ApiKey {
                id: row.get("id"),
                name: row.get("name"),
                key: String::new(),
                key_hint: row.get("key_hint"),
                status: row.get("status"),
                created_at: row.get::<_, i64>("created_at") as u64,
            })
            .collect())
    }

    fn get_api_key(&self, id: i64) -> Result<Option<ApiKey>> {
        let mut client = self.lock_client()?;
        let rows = client.query(
            "SELECT id, name, key_hint, status, created_at FROM api_keys WHERE id = $1",
            &[&id],
        )?;
        Ok(rows.first().map(|row| ApiKey {
            id: row.get("id"),
            name: row.get("name"),
            key: String::new(),
            key_hint: row.get("key_hint"),
            status: row.get("status"),
            created_at: row.get::<_, i64>("created_at") as u64,
        }))
    }

    fn revoke_api_key(&self, id: i64) -> Result<bool> {
        let mut client = self.lock_client()?;
        let affected = client.execute(
            "UPDATE api_keys SET status = 'revoked' WHERE id = $1 AND status = 'active'",
            &[&id],
        )?;
        Ok(affected > 0)
    }

    fn get_api_key_by_key(&self, key: &str) -> Result<Option<ApiKey>> {
        let mut client = self.lock_client()?;
        let rows = client.query(
            "SELECT id, name, key, key_hint, status, created_at FROM api_keys WHERE key = $1",
            &[&key],
        )?;
        Ok(rows.first().map(|row| ApiKey {
            id: row.get("id"),
            name: row.get("name"),
            key: row.get("key"),
            key_hint: row.get("key_hint"),
            status: row.get("status"),
            created_at: row.get::<_, i64>("created_at") as u64,
        }))
    }

    fn store_completion(&self, completion: &StoredCompletion) -> Result<()> {
        let mut client = self.lock_client()?;
        let created_at = completion.created_at as i64;
        client.execute(
            "INSERT INTO completions (id, api_key_id, session_id, previous_completion_id, input_model, resolved_model, provider, input, output, instructions, exchange, usage, created_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
            &[
                &completion.id,
                &completion.api_key_id,
                &completion.session_id,
                &completion.previous_completion_id,
                &completion.input_model,
                &completion.resolved_model,
                &completion.provider,
                &completion.input,
                &completion.output,
                &completion.instructions,
                &completion.exchange,
                &completion.usage,
                &created_at,
            ],
        )?;
        Ok(())
    }

    fn get_completion(&self, id: &str) -> Result<Option<StoredCompletion>> {
        let mut client = self.lock_client()?;
        let rows = client.query("SELECT * FROM completions WHERE id = $1", &[&id])?;
        Ok(rows.first().map(read_completion))
    }

    fn get_session_id(&self, id: &str) -> Result<Option<String>> {
        let mut client = self.lock_client()?;
        let rows = client.query("SELECT session_id FROM completions WHERE id = $1", &[&id])?;
        Ok(rows.first().map(|row| row.get("session_id")))
    }

    fn walk_chain(&self, id: &str) -> Result<Vec<StoredCompletion>> {
        let mut chain = Vec::new();
        let mut current_id = Some(id.to_string());

        while let Some(ref cid) = current_id {
            let completion = self
                .get_completion(cid)?
                .with_context(|| format!("Completion '{}' not found in chain", cid))?;
            let prev = completion.previous_completion_id.clone();
            chain.push(completion);
            current_id = prev;
        }

        chain.reverse();
        Ok(chain)
    }

    fn list_completions(&self, filter: &ListFilter) -> Result<Vec<StoredCompletion>> {
        let mut client = self.lock_client()?;
        let (where_clause, mut params, mut idx) =
            build_filter(&filter.key_ids, filter.since, filter.until);

        let limit = effective_limit(filter.limit);
        let offset = filter.offset as i64;

        let sql = format!(
            "SELECT * FROM completions{} ORDER BY created_at DESC LIMIT ${} OFFSET ${}",
            where_clause,
            idx,
            idx + 1
        );
        idx += 2;
        let _ = idx; // suppress unused warning

        params.push(Box::new(limit));
        params.push(Box::new(offset));

        let param_refs: Vec<&(dyn postgres::types::ToSql + Sync)> =
            params.iter().map(|p| p.as_ref()).collect();
        let rows = client.query(&sql, &param_refs)?;
        Ok(rows.iter().map(read_completion).collect())
    }

    fn store_embedding(&self, embedding: &StoredEmbedding) -> Result<()> {
        let mut client = self.lock_client()?;
        let created_at = embedding.created_at as i64;
        client.execute(
            "INSERT INTO embeddings (id, api_key_id, input_model, resolved_model, provider, input, usage, created_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            &[
                &embedding.id,
                &embedding.api_key_id,
                &embedding.input_model,
                &embedding.resolved_model,
                &embedding.provider,
                &embedding.input,
                &embedding.usage,
                &created_at,
            ],
        )?;
        Ok(())
    }

    fn list_embeddings(&self, filter: &ListFilter) -> Result<Vec<StoredEmbedding>> {
        let mut client = self.lock_client()?;
        let (where_clause, mut params, mut idx) =
            build_filter(&filter.key_ids, filter.since, filter.until);

        let limit = effective_limit(filter.limit);
        let offset = filter.offset as i64;

        let sql = format!(
            "SELECT * FROM embeddings{} ORDER BY created_at DESC LIMIT ${} OFFSET ${}",
            where_clause,
            idx,
            idx + 1
        );
        idx += 2;
        let _ = idx;

        params.push(Box::new(limit));
        params.push(Box::new(offset));

        let param_refs: Vec<&(dyn postgres::types::ToSql + Sync)> =
            params.iter().map(|p| p.as_ref()).collect();
        let rows = client.query(&sql, &param_refs)?;
        Ok(rows.iter().map(read_embedding).collect())
    }

    fn get_usage(
        &self,
        key_ids: &[i64],
        bucket: Option<TimeBucket>,
        since: Option<u64>,
        until: Option<u64>,
    ) -> Result<Vec<UsageRow>> {
        let mut client = self.lock_client()?;

        let bucket_expr = match bucket {
            Some(TimeBucket::Hour) => "(created_at / 3600) * 3600",
            Some(TimeBucket::Day) => "(created_at / 86400) * 86400",
            Some(TimeBucket::Week) => "(created_at / 604800) * 604800",
            Some(TimeBucket::Month) => "(created_at / 2592000) * 2592000",
            None => "0",
        };

        let (where_clause, filter_params, _) = build_filter(key_ids, since, until);

        let compl_sql = format!(
            "SELECT api_key_id, {bucket} AS period, \
             COUNT(*) AS cnt, \
             COALESCE(SUM((usage->>'input_tokens')::bigint), 0) AS inp, \
             COALESCE(SUM((usage->>'output_tokens')::bigint), 0) AS outp \
             FROM completions{where_clause} \
             GROUP BY api_key_id, period",
            bucket = bucket_expr,
            where_clause = where_clause,
        );

        let emb_sql = format!(
            "SELECT api_key_id, {bucket} AS period, \
             COUNT(*) AS cnt, \
             COALESCE(SUM((usage->>'input_tokens')::bigint), 0) AS inp \
             FROM embeddings{where_clause} \
             GROUP BY api_key_id, period",
            bucket = bucket_expr,
            where_clause = where_clause,
        );

        let param_refs: Vec<&(dyn postgres::types::ToSql + Sync)> =
            filter_params.iter().map(|p| p.as_ref()).collect();

        type BucketKey = (i64, u64);
        let mut usage_map: HashMap<BucketKey, UsageRow> = HashMap::new();

        // Completion stats
        let compl_rows = client.query(&compl_sql, &param_refs)?;
        for row in &compl_rows {
            let key_id: i64 = row.get("api_key_id");
            let period: i64 = row.get("period");
            let period = period as u64;
            let cnt: i64 = row.get("cnt");
            let inp: i64 = row.get("inp");
            let outp: i64 = row.get("outp");

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
            entry.completions_count = cnt as u64;
            entry.total_input_tokens += inp as u64;
            entry.total_output_tokens += outp as u64;
        }

        // Embedding stats
        let emb_rows = client.query(&emb_sql, &param_refs)?;
        for row in &emb_rows {
            let key_id: i64 = row.get("api_key_id");
            let period: i64 = row.get("period");
            let period = period as u64;
            let cnt: i64 = row.get("cnt");
            let inp: i64 = row.get("inp");

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
            entry.embeddings_count = cnt as u64;
            entry.total_input_tokens += inp as u64;
        }

        // Resolve key names
        let key_name_map: HashMap<i64, String> = {
            let rows = client.query("SELECT id, name FROM api_keys", &[])?;
            rows.iter()
                .map(|r| (r.get::<_, i64>("id"), r.get::<_, String>("name")))
                .collect()
        };

        let mut results: Vec<UsageRow> = usage_map
            .into_values()
            .map(|mut row| {
                row.key_name = key_name_map.get(&row.key_id).cloned().unwrap_or_default();
                row
            })
            .collect();

        results.sort_by(|a, b| a.period.cmp(&b.period).then(a.key_id.cmp(&b.key_id)));
        Ok(results)
    }
}
