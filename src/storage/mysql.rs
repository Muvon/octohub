use super::{
    decode_allowed_models, decode_auto_map, decode_json_column, encode_allowed_models,
    encode_auto_map, encode_json_column, generate_api_key, make_key_hint, now_unix, ApiKey,
    ListFilter, Storage, StoredCompletion, StoredEmbedding, StoredMedia, TimeBucket, UsageRow,
};
use anyhow::{Context, Result};
use mysql::prelude::*;
use mysql::{Opts, Pool};
use std::collections::HashMap;

/// MySQL-backed storage implementation
pub struct MysqlStorage {
    pool: Pool,
}

impl MysqlStorage {
    /// Create a new MySQL storage, initializing the schema
    pub fn new(url: &str) -> Result<Self> {
        let opts = Opts::from_url(url).context("Invalid MySQL DSN")?;
        let pool = Pool::new(opts).context("Failed to create MySQL connection pool")?;
        let storage = Self { pool };
        storage.init_schema()?;
        Ok(storage)
    }

    fn init_schema(&self) -> Result<()> {
        let mut conn = self.pool.get_conn().context("MySQL connection failed")?;
        conn.query_drop(
            "CREATE TABLE IF NOT EXISTS api_keys (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                `key` VARCHAR(255) NOT NULL UNIQUE,
                key_hint VARCHAR(32) NOT NULL,
                status VARCHAR(16) NOT NULL DEFAULT 'active',
                allowed_models JSON,
                owner VARCHAR(64) NULL,
                owner_concurrency INT UNSIGNED NULL,
                created_at BIGINT UNSIGNED NOT NULL,
                INDEX idx_api_keys_key (`key`),
                INDEX idx_api_keys_status (status)
            )",
        )?;
        // Additive migration for older deployments. Not all MySQL versions
        // support `ADD COLUMN IF NOT EXISTS`, so we probe information_schema.
        ensure_column(
            &mut conn,
            "api_keys",
            "allowed_models",
            "JSON NULL AFTER status",
        )?;
        ensure_column(
            &mut conn,
            "api_keys",
            "owner",
            "VARCHAR(64) NULL AFTER allowed_models",
        )?;
        ensure_column(
            &mut conn,
            "api_keys",
            "owner_concurrency",
            "INT UNSIGNED NULL AFTER owner",
        )?;
        conn.query_drop(
            "CREATE TABLE IF NOT EXISTS completions (
                id VARCHAR(255) PRIMARY KEY,
                api_key_id BIGINT NOT NULL,
                session_id VARCHAR(255) NOT NULL,
                previous_completion_id VARCHAR(255),
                input_model VARCHAR(255) NOT NULL,
                resolved_model VARCHAR(255) NOT NULL,
                provider VARCHAR(255) NOT NULL,
                input JSON NOT NULL,
                output JSON NOT NULL,
                instructions TEXT,
                exchange JSON NOT NULL,
                `usage` JSON NOT NULL,
                created_at BIGINT UNSIGNED NOT NULL,
                INDEX idx_completions_api_key (api_key_id),
                INDEX idx_completions_session (session_id),
                INDEX idx_completions_previous (previous_completion_id),
                INDEX idx_completions_created (created_at)
            )",
        )?;
        conn.query_drop(
            "CREATE TABLE IF NOT EXISTS embeddings (
                id VARCHAR(255) PRIMARY KEY,
                api_key_id BIGINT NOT NULL,
                input_model VARCHAR(255) NOT NULL,
                resolved_model VARCHAR(255) NOT NULL,
                provider VARCHAR(255) NOT NULL,
                input JSON NOT NULL,
                `usage` JSON NOT NULL,
                created_at BIGINT UNSIGNED NOT NULL,
                INDEX idx_embeddings_api_key (api_key_id),
                INDEX idx_embeddings_created (created_at)
            )",
        )?;
        conn.query_drop(
            "CREATE TABLE IF NOT EXISTS media (
                id VARCHAR(255) PRIMARY KEY,
                api_key_id BIGINT NOT NULL,
                task VARCHAR(32) NOT NULL,
                status VARCHAR(32) NOT NULL,
                input_model VARCHAR(255) NOT NULL,
                resolved_model VARCHAR(255) NOT NULL,
                provider VARCHAR(255) NOT NULL,
                request JSON NOT NULL,
                handle JSON,
                result JSON,
                `usage` JSON,
                warnings JSON,
                error JSON,
                created_at BIGINT UNSIGNED NOT NULL,
                completed_at BIGINT UNSIGNED NULL,
                INDEX idx_media_api_key (api_key_id),
                INDEX idx_media_created (created_at),
                INDEX idx_media_status (status)
            )",
        )?;
        conn.query_drop(
            "CREATE TABLE IF NOT EXISTS owner_auto_models (
                owner VARCHAR(64) PRIMARY KEY,
                map JSON NOT NULL,
                updated_at BIGINT UNSIGNED NOT NULL
            )",
        )?;
        Ok(())
    }
}

/// Add `column` of `decl` to `table` if it doesn't exist yet. Uses
/// `information_schema.columns` to stay compatible across MySQL versions
/// that don't support `ADD COLUMN IF NOT EXISTS`.
fn ensure_column(
    conn: &mut mysql::PooledConn,
    table: &str,
    column: &str,
    decl: &str,
) -> Result<()> {
    let exists: Option<i64> = conn.exec_first(
        "SELECT 1 FROM information_schema.columns \
         WHERE table_schema = DATABASE() AND table_name = ? AND column_name = ?",
        (table, column),
    )?;
    if exists.is_none() {
        conn.query_drop(format!(
            "ALTER TABLE {} ADD COLUMN {} {}",
            table, column, decl
        ))?;
    }
    Ok(())
}

/// Build WHERE clause and positional params for key_ids + since/until filters (MySQL uses `?`).
fn build_filter(
    key_ids: &[i64],
    since: Option<u64>,
    until: Option<u64>,
) -> (String, Vec<mysql::Value>) {
    let mut conditions: Vec<String> = Vec::new();
    let mut params: Vec<mysql::Value> = Vec::new();

    if !key_ids.is_empty() {
        let placeholders: Vec<&str> = key_ids.iter().map(|_| "?").collect();
        conditions.push(format!("api_key_id IN ({})", placeholders.join(", ")));
        for &kid in key_ids {
            params.push(kid.into());
        }
    }

    if let Some(s) = since {
        conditions.push("created_at >= ?".to_string());
        params.push((s as i64).into());
    }

    if let Some(u) = until {
        conditions.push("created_at <= ?".to_string());
        params.push((u as i64).into());
    }

    let clause = if conditions.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", conditions.join(" AND "))
    };

    (clause, params)
}

/// Read an `ApiKey` from a row without exposing the `key` column (used by
/// list/get endpoints where the full key must never leave the database).
fn read_api_key_masked(row: mysql::Row) -> ApiKey {
    ApiKey {
        id: row.get("id").unwrap_or_default(),
        name: row.get("name").unwrap_or_default(),
        key: String::new(),
        key_hint: row.get("key_hint").unwrap_or_default(),
        status: row.get("status").unwrap_or_default(),
        allowed_models: decode_allowed_models(row.get("allowed_models")),
        owner: row.get("owner").unwrap_or_default(),
        owner_concurrency: row.get("owner_concurrency").unwrap_or_default(),
        created_at: row.get("created_at").unwrap_or_default(),
    }
}

fn read_completion(row: mysql::Row) -> Result<StoredCompletion> {
    Ok(StoredCompletion {
        id: row.get("id").unwrap_or_default(),
        api_key_id: row.get("api_key_id").unwrap_or_default(),
        session_id: row.get("session_id").unwrap_or_default(),
        previous_completion_id: row.get("previous_completion_id").unwrap_or_default(),
        input_model: row.get("input_model").unwrap_or_default(),
        resolved_model: row.get("resolved_model").unwrap_or_default(),
        provider: row.get("provider").unwrap_or_default(),
        input: serde_json::from_str(&row.get::<String, _>("input").unwrap_or_default())
            .unwrap_or_default(),
        output: serde_json::from_str(&row.get::<String, _>("output").unwrap_or_default())
            .unwrap_or_default(),
        instructions: row.get("instructions").unwrap_or_default(),
        exchange: serde_json::from_str(&row.get::<String, _>("exchange").unwrap_or_default())
            .unwrap_or_default(),
        usage: serde_json::from_str(&row.get::<String, _>("usage").unwrap_or_default())
            .unwrap_or_default(),
        created_at: row.get("created_at").unwrap_or_default(),
    })
}

fn read_embedding(row: mysql::Row) -> Result<StoredEmbedding> {
    Ok(StoredEmbedding {
        id: row.get("id").unwrap_or_default(),
        api_key_id: row.get("api_key_id").unwrap_or_default(),
        input_model: row.get("input_model").unwrap_or_default(),
        resolved_model: row.get("resolved_model").unwrap_or_default(),
        provider: row.get("provider").unwrap_or_default(),
        input: serde_json::from_str(&row.get::<String, _>("input").unwrap_or_default())
            .unwrap_or_default(),
        usage: serde_json::from_str(&row.get::<String, _>("usage").unwrap_or_default())
            .unwrap_or_default(),
        created_at: row.get("created_at").unwrap_or_default(),
    })
}

/// Read a nullable JSON text column. `Row::get` panics when SQL NULL is
/// converted straight to `String`, so the value has to be fetched as
/// `Option<String>` first.
fn json_column(row: &mysql::Row, name: &str) -> Option<serde_json::Value> {
    decode_json_column(row.get::<Option<String>, _>(name).unwrap_or_default())
}

fn read_media(row: mysql::Row) -> Result<StoredMedia> {
    Ok(StoredMedia {
        id: row.get("id").unwrap_or_default(),
        api_key_id: row.get("api_key_id").unwrap_or_default(),
        task: row.get("task").unwrap_or_default(),
        status: row.get("status").unwrap_or_default(),
        input_model: row.get("input_model").unwrap_or_default(),
        resolved_model: row.get("resolved_model").unwrap_or_default(),
        provider: row.get("provider").unwrap_or_default(),
        request: serde_json::from_str(&row.get::<String, _>("request").unwrap_or_default())
            .unwrap_or_default(),
        handle: json_column(&row, "handle"),
        result: json_column(&row, "result"),
        usage: json_column(&row, "usage"),
        warnings: json_column(&row, "warnings"),
        error: json_column(&row, "error"),
        created_at: row.get("created_at").unwrap_or_default(),
        completed_at: row.get("completed_at").unwrap_or_default(),
    })
}

fn effective_limit(limit: u32) -> u32 {
    if limit == 0 {
        50
    } else {
        limit.min(1000)
    }
}

impl Storage for MysqlStorage {
    fn create_api_key(
        &self,
        name: &str,
        allowed_models: Option<&[String]>,
        owner: Option<&str>,
        owner_concurrency: Option<u32>,
    ) -> Result<ApiKey> {
        let mut conn = self.pool.get_conn()?;
        let key = generate_api_key();
        let key_hint = make_key_hint(&key);
        let now = now_unix();
        let allowed_models_json = encode_allowed_models(allowed_models);

        conn.exec_drop(
            "INSERT INTO api_keys (name, `key`, key_hint, status, allowed_models, owner, owner_concurrency, created_at)
             VALUES (?, ?, ?, 'active', ?, ?, ?, ?)",
            (name, &key, &key_hint, &allowed_models_json, owner, owner_concurrency, now),
        )?;

        let id: i64 = conn.query_first("SELECT LAST_INSERT_ID()")?.unwrap_or(0);

        Ok(ApiKey {
            id,
            name: name.to_string(),
            key,
            key_hint,
            status: "active".to_string(),
            allowed_models: allowed_models.map(|m| m.to_vec()),
            owner: owner.map(str::to_string),
            owner_concurrency,
            created_at: now,
        })
    }

    fn list_api_keys(&self) -> Result<Vec<ApiKey>> {
        let mut conn = self.pool.get_conn()?;
        let rows: Vec<mysql::Row> = conn.query(
            "SELECT id, name, key_hint, status, allowed_models, owner, owner_concurrency, created_at \
             FROM api_keys ORDER BY id",
        )?;
        Ok(rows.into_iter().map(read_api_key_masked).collect())
    }

    fn get_api_key(&self, id: i64) -> Result<Option<ApiKey>> {
        let mut conn = self.pool.get_conn()?;
        let row: Option<mysql::Row> = conn.exec_first(
            "SELECT id, name, key_hint, status, allowed_models, owner, owner_concurrency, created_at \
             FROM api_keys WHERE id = ?",
            (id,),
        )?;
        Ok(row.map(read_api_key_masked))
    }

    fn revoke_api_key(&self, id: i64) -> Result<bool> {
        let mut conn = self.pool.get_conn()?;
        conn.exec_drop(
            "UPDATE api_keys SET status = 'revoked' WHERE id = ? AND status = 'active'",
            (id,),
        )?;
        Ok(conn.affected_rows() > 0)
    }

    fn update_api_key_models(&self, id: i64, allowed_models: Option<&[String]>) -> Result<bool> {
        let mut conn = self.pool.get_conn()?;
        conn.exec_drop(
            "UPDATE api_keys SET allowed_models = ? WHERE id = ? AND status = 'active'",
            (encode_allowed_models(allowed_models), id),
        )?;
        Ok(conn.affected_rows() > 0)
    }

    fn update_api_key_owner(
        &self,
        id: i64,
        owner: Option<&str>,
        owner_concurrency: Option<u32>,
    ) -> Result<bool> {
        let mut conn = self.pool.get_conn()?;
        conn.exec_drop(
            "UPDATE api_keys SET owner = ?, owner_concurrency = ? WHERE id = ? AND status = 'active'",
            (owner, owner_concurrency, id),
        )?;
        Ok(conn.affected_rows() > 0)
    }

    fn get_owner_auto_map(
        &self,
        owner: &str,
    ) -> Result<Option<std::collections::HashMap<String, String>>> {
        let mut conn = self.pool.get_conn()?;
        let raw: Option<String> = conn.exec_first(
            "SELECT map FROM owner_auto_models WHERE owner = ?",
            (owner,),
        )?;
        Ok(decode_auto_map(raw))
    }

    fn set_owner_auto_map(
        &self,
        owner: &str,
        map: Option<&std::collections::HashMap<String, String>>,
    ) -> Result<()> {
        let mut conn = self.pool.get_conn()?;
        match map.filter(|m| !m.is_empty()) {
            Some(m) => {
                conn.exec_drop(
                    "INSERT INTO owner_auto_models (owner, map, updated_at) VALUES (?, ?, ?) \
                     ON DUPLICATE KEY UPDATE map = VALUES(map), updated_at = VALUES(updated_at)",
                    (owner, encode_auto_map(m), now_unix()),
                )?;
            }
            None => {
                conn.exec_drop("DELETE FROM owner_auto_models WHERE owner = ?", (owner,))?;
            }
        }
        Ok(())
    }

    fn get_api_key_by_key(&self, key: &str) -> Result<Option<ApiKey>> {
        let mut conn = self.pool.get_conn()?;
        let row: Option<mysql::Row> = conn.exec_first(
            "SELECT id, name, `key`, key_hint, status, allowed_models, owner, owner_concurrency, created_at \
             FROM api_keys WHERE `key` = ?",
            (key,),
        )?;
        Ok(row.map(|r| ApiKey {
            id: r.get("id").unwrap_or_default(),
            name: r.get("name").unwrap_or_default(),
            key: r.get("key").unwrap_or_default(),
            key_hint: r.get("key_hint").unwrap_or_default(),
            status: r.get("status").unwrap_or_default(),
            allowed_models: decode_allowed_models(r.get("allowed_models")),
            owner: r.get("owner").unwrap_or_default(),
            owner_concurrency: r.get("owner_concurrency").unwrap_or_default(),
            created_at: r.get("created_at").unwrap_or_default(),
        }))
    }

    fn store_completion(&self, completion: &StoredCompletion) -> Result<()> {
        let mut conn = self.pool.get_conn()?;
        conn.exec_drop(
            "INSERT INTO completions (id, api_key_id, session_id, previous_completion_id, input_model, resolved_model, provider, input, output, instructions, exchange, `usage`, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            mysql::Params::Positional(vec![
                completion.id.clone().into(),
                completion.api_key_id.into(),
                completion.session_id.clone().into(),
                completion.previous_completion_id.clone().into(),
                completion.input_model.clone().into(),
                completion.resolved_model.clone().into(),
                completion.provider.clone().into(),
                completion.input.to_string().into(),
                completion.output.to_string().into(),
                completion.instructions.clone().into(),
                completion.exchange.to_string().into(),
                completion.usage.to_string().into(),
                completion.created_at.into(),
            ]),
        )?;
        Ok(())
    }

    fn get_completion(&self, id: &str) -> Result<Option<StoredCompletion>> {
        let mut conn = self.pool.get_conn()?;
        let row: Option<mysql::Row> =
            conn.exec_first("SELECT * FROM completions WHERE id = ?", (id,))?;
        row.map(read_completion).transpose()
    }

    fn walk_chain(&self, id: &str) -> Result<Vec<StoredCompletion>> {
        let mut conn = self.pool.get_conn()?;
        // Single recursive CTE — one round-trip regardless of chain depth.
        let rows: Vec<mysql::Row> = conn.exec(
            "WITH RECURSIVE chain AS (\
               SELECT * FROM completions WHERE id = ? \
               UNION ALL \
               SELECT c.* FROM completions c \
               JOIN chain ON c.id = chain.previous_completion_id \
             ) \
             SELECT * FROM chain ORDER BY created_at ASC",
            (id,),
        )?;
        rows.into_iter().map(read_completion).collect()
    }

    fn list_completions(&self, filter: &ListFilter) -> Result<Vec<StoredCompletion>> {
        let mut conn = self.pool.get_conn()?;
        let (where_clause, mut params) = build_filter(&filter.key_ids, filter.since, filter.until);

        let limit = effective_limit(filter.limit);
        params.push(limit.into());
        params.push(filter.offset.into());

        let sql = format!(
            "SELECT * FROM completions{} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            where_clause
        );

        let rows: Vec<mysql::Row> = conn.exec(sql, mysql::Params::Positional(params))?;
        rows.into_iter().map(read_completion).collect()
    }

    fn store_embedding(&self, embedding: &StoredEmbedding) -> Result<()> {
        let mut conn = self.pool.get_conn()?;
        conn.exec_drop(
            "INSERT INTO embeddings (id, api_key_id, input_model, resolved_model, provider, input, `usage`, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                &embedding.id,
                embedding.api_key_id,
                &embedding.input_model,
                &embedding.resolved_model,
                &embedding.provider,
                embedding.input.to_string(),
                embedding.usage.to_string(),
                embedding.created_at,
            ),
        )?;
        Ok(())
    }

    fn list_embeddings(&self, filter: &ListFilter) -> Result<Vec<StoredEmbedding>> {
        let mut conn = self.pool.get_conn()?;
        let (where_clause, mut params) = build_filter(&filter.key_ids, filter.since, filter.until);

        let limit = effective_limit(filter.limit);
        params.push(limit.into());
        params.push(filter.offset.into());

        let sql = format!(
            "SELECT * FROM embeddings{} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            where_clause
        );

        let rows: Vec<mysql::Row> = conn.exec(sql, mysql::Params::Positional(params))?;
        rows.into_iter().map(read_embedding).collect()
    }

    fn store_media(&self, record: &StoredMedia) -> Result<()> {
        let mut conn = self.pool.get_conn()?;
        conn.exec_drop(
            "INSERT INTO media (id, api_key_id, task, status, input_model, resolved_model, provider, request, handle, result, `usage`, warnings, error, created_at, completed_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            mysql::Params::Positional(vec![
                record.id.clone().into(),
                record.api_key_id.into(),
                record.task.clone().into(),
                record.status.clone().into(),
                record.input_model.clone().into(),
                record.resolved_model.clone().into(),
                record.provider.clone().into(),
                serde_json::to_string(&record.request)?.into(),
                encode_json_column(record.handle.as_ref()).into(),
                encode_json_column(record.result.as_ref()).into(),
                encode_json_column(record.usage.as_ref()).into(),
                encode_json_column(record.warnings.as_ref()).into(),
                encode_json_column(record.error.as_ref()).into(),
                record.created_at.into(),
                record.completed_at.into(),
            ]),
        )?;
        Ok(())
    }

    fn update_media(&self, record: &StoredMedia) -> Result<()> {
        let mut conn = self.pool.get_conn()?;
        // Only the mutable half is written: identity, ownership and the
        // original request are immutable once the job has been submitted.
        conn.exec_drop(
            "UPDATE media SET status = ?, handle = ?, result = ?, `usage` = ?, \
             warnings = ?, error = ?, completed_at = ? WHERE id = ?",
            (
                &record.status,
                encode_json_column(record.handle.as_ref()),
                encode_json_column(record.result.as_ref()),
                encode_json_column(record.usage.as_ref()),
                encode_json_column(record.warnings.as_ref()),
                encode_json_column(record.error.as_ref()),
                record.completed_at,
                &record.id,
            ),
        )?;
        Ok(())
    }

    fn get_media(&self, id: &str, api_key_id: i64) -> Result<Option<StoredMedia>> {
        let mut conn = self.pool.get_conn()?;
        let row: Option<mysql::Row> = conn.exec_first(
            "SELECT * FROM media WHERE id = ? AND api_key_id = ?",
            (id, api_key_id),
        )?;
        row.map(read_media).transpose()
    }

    fn list_media(&self, filter: &ListFilter) -> Result<Vec<StoredMedia>> {
        let mut conn = self.pool.get_conn()?;
        let (where_clause, mut params) = build_filter(&filter.key_ids, filter.since, filter.until);

        let limit = effective_limit(filter.limit);
        params.push(limit.into());
        params.push(filter.offset.into());

        let sql = format!(
            "SELECT * FROM media{} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            where_clause
        );

        let rows: Vec<mysql::Row> = conn.exec(sql, mysql::Params::Positional(params))?;
        rows.into_iter().map(read_media).collect()
    }

    fn get_usage(
        &self,
        key_ids: &[i64],
        bucket: Option<TimeBucket>,
        since: Option<u64>,
        until: Option<u64>,
    ) -> Result<Vec<UsageRow>> {
        let mut conn = self.pool.get_conn()?;

        let bucket_expr = match bucket {
            Some(TimeBucket::Hour) => "(created_at DIV 3600) * 3600",
            Some(TimeBucket::Day) => "(created_at DIV 86400) * 86400",
            Some(TimeBucket::Week) => "(created_at DIV 604800) * 604800",
            Some(TimeBucket::Month) => "(created_at DIV 2592000) * 2592000",
            None => "0",
        };

        let (where_clause, filter_params) = build_filter(key_ids, since, until);

        let compl_sql = format!(
            "SELECT api_key_id, {bucket} AS period, \
             COUNT(*) AS cnt, \
             COALESCE(SUM(JSON_EXTRACT(`usage`, '$.input_tokens')), 0) AS inp, \
             COALESCE(SUM(JSON_EXTRACT(`usage`, '$.output_tokens')), 0) AS outp, \
             COALESCE(SUM(JSON_EXTRACT(`usage`, '$.cost')), 0.0) AS cost \
             FROM completions{where_clause} \
             GROUP BY api_key_id, period",
            bucket = bucket_expr,
            where_clause = where_clause,
        );

        let emb_sql = format!(
            "SELECT api_key_id, {bucket} AS period, \
             COUNT(*) AS cnt, \
             COALESCE(SUM(JSON_EXTRACT(`usage`, '$.input_tokens')), 0) AS inp, \
             COALESCE(SUM(JSON_EXTRACT(`usage`, '$.cost')), 0.0) AS cost \
             FROM embeddings{where_clause} \
             GROUP BY api_key_id, period",
            bucket = bucket_expr,
            where_clause = where_clause,
        );

        // Media rows carry cost but no tokens; an unpriced row contributes
        // nothing rather than zero-filling the total.
        let media_sql = format!(
            "SELECT api_key_id, {bucket} AS period, \
             COUNT(*) AS cnt, \
             COALESCE(SUM(JSON_EXTRACT(`usage`, '$.cost')), 0.0) AS cost \
             FROM media{where_clause} \
             GROUP BY api_key_id, period",
            bucket = bucket_expr,
            where_clause = where_clause,
        );

        type BucketKey = (i64, u64);
        let mut usage_map: HashMap<BucketKey, UsageRow> = HashMap::new();

        // Completion stats
        let compl_rows: Vec<mysql::Row> =
            conn.exec(&compl_sql, mysql::Params::Positional(filter_params.clone()))?;
        for row in compl_rows {
            let key_id: i64 = row.get("api_key_id").unwrap_or_default();
            let period: u64 = row.get("period").unwrap_or_default();
            let cnt: u64 = row.get("cnt").unwrap_or_default();
            let inp: u64 = row.get("inp").unwrap_or_default();
            let outp: u64 = row.get("outp").unwrap_or_default();
            let cost: f64 = row.get("cost").unwrap_or_default();

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
                    media_count: 0,
                    total_cost: 0.0,
                });
            entry.completions_count = cnt;
            entry.total_input_tokens += inp;
            entry.total_output_tokens += outp;
            entry.total_cost += cost;
        }

        // Embedding stats
        let emb_rows: Vec<mysql::Row> =
            conn.exec(&emb_sql, mysql::Params::Positional(filter_params.clone()))?;
        for row in emb_rows {
            let key_id: i64 = row.get("api_key_id").unwrap_or_default();
            let period: u64 = row.get("period").unwrap_or_default();
            let cnt: u64 = row.get("cnt").unwrap_or_default();
            let inp: u64 = row.get("inp").unwrap_or_default();
            let cost: f64 = row.get("cost").unwrap_or_default();

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
                    media_count: 0,
                    total_cost: 0.0,
                });
            entry.embeddings_count = cnt;
            entry.total_input_tokens += inp;
            entry.total_cost += cost;
        }

        // Media stats
        let media_rows: Vec<mysql::Row> =
            conn.exec(&media_sql, mysql::Params::Positional(filter_params))?;
        for row in media_rows {
            let key_id: i64 = row.get("api_key_id").unwrap_or_default();
            let period: u64 = row.get("period").unwrap_or_default();
            let cnt: u64 = row.get("cnt").unwrap_or_default();
            let cost: f64 = row.get("cost").unwrap_or_default();

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
                    media_count: 0,
                    total_cost: 0.0,
                });
            entry.media_count = cnt;
            entry.total_cost += cost;
        }

        // Resolve key names
        let key_name_map: HashMap<i64, String> = {
            let rows: Vec<mysql::Row> = conn.query("SELECT id, name FROM api_keys")?;
            rows.into_iter()
                .filter_map(|r| {
                    let id: i64 = r.get("id")?;
                    let name: String = r.get("name")?;
                    Some((id, name))
                })
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
