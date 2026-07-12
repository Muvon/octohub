use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{Request, Response, StatusCode};

use crate::auth::authenticate_admin;
use crate::storage::{ListFilter, Storage, TimeBucket};

type BoxBody = Full<Bytes>;

fn json_response(status: StatusCode, body: serde_json::Value) -> Response<BoxBody> {
    let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(body_bytes)))
        .unwrap()
}

fn error_response(status: StatusCode, message: &str) -> Response<BoxBody> {
    json_response(
        status,
        serde_json::json!({
            "error": {
                "message": message,
                "type": "invalid_request_error"
            }
        }),
    )
}

/// Verify admin auth from request headers against master key
fn check_admin(
    req: &Request<hyper::body::Incoming>,
    master_key: &str,
) -> Result<(), Box<Response<BoxBody>>> {
    let header = req
        .headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok());

    if authenticate_admin(header, master_key) {
        Ok(())
    } else {
        let reason = if header.and_then(|h| h.strip_prefix("Bearer ")).is_some() {
            "invalid_token"
        } else {
            "missing_token"
        };
        tracing::warn!(kind = "admin", reason = reason, "auth failed");
        Err(Box::new(error_response(
            StatusCode::UNAUTHORIZED,
            "Invalid or missing admin API key",
        )))
    }
}

/// POST /v1/admin/keys - Create a new API key
pub async fn handle_create_key(
    req: Request<hyper::body::Incoming>,
    storage: Arc<dyn Storage>,
    master_key: &str,
) -> Response<BoxBody> {
    if let Err(resp) = check_admin(&req, master_key) {
        return *resp;
    }

    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Failed to read request body: {}", e),
            );
        }
    };

    #[derive(serde::Deserialize)]
    struct CreateKeyRequest {
        name: String,
        /// `None` (field absent) → unrestricted, all models allowed.
        /// `Some(list)` → only these model names accepted on /v1/completions
        /// and /v1/embeddings. Match is exact against the `model` request
        /// field as-sent (alias from [models]/[embedding_models] or raw
        /// `provider:model`).
        #[serde(default)]
        allowed_models: Option<Vec<String>>,
        /// Opaque grouping label: keys sharing an owner share one in-flight
        /// budget (`owner_concurrency`). Both optional — absent keeps the key
        /// ungrouped/unlimited, exactly as before the fields existed.
        #[serde(default)]
        owner: Option<String>,
        #[serde(default)]
        owner_concurrency: Option<u32>,
    }

    let create_req: CreateKeyRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Invalid request JSON: {}", e),
            );
        }
    };

    if create_req.name.trim().is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "Key name must not be empty");
    }

    let allowed_models = create_req.allowed_models.as_deref();
    match storage.create_api_key(
        create_req.name.trim(),
        allowed_models,
        create_req.owner.as_deref(),
        create_req.owner_concurrency,
    ) {
        Ok(key) => {
            // On creation, return the full key (only time it's visible)
            json_response(StatusCode::CREATED, api_key_response(&key, true))
        }
        Err(e) => {
            tracing::error!(error = %e, "create key failed");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Failed to create API key: {}", e),
            )
        }
    }
}

/// POST /v1/admin/keys/:id/owner - Replace a key's owner grouping + shared
/// concurrency budget in place (same contract as the /models endpoint: the
/// key value keeps working while its limits follow the operator's plan).
pub async fn handle_update_key_owner(
    req: Request<hyper::body::Incoming>,
    storage: Arc<dyn Storage>,
    master_key: &str,
    key_id: i64,
) -> Response<BoxBody> {
    if let Err(resp) = check_admin(&req, master_key) {
        return *resp;
    }

    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Failed to read request body: {}", e),
            );
        }
    };

    #[derive(serde::Deserialize)]
    struct UpdateOwnerRequest {
        /// Absent/`null` owner ungroups the key; absent/`null`/0 concurrency
        /// means unlimited. Same semantics as key creation.
        #[serde(default)]
        owner: Option<String>,
        #[serde(default)]
        owner_concurrency: Option<u32>,
    }

    let update_req: UpdateOwnerRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Invalid request JSON: {}", e),
            );
        }
    };

    match storage.update_api_key_owner(
        key_id,
        update_req.owner.as_deref(),
        update_req.owner_concurrency,
    ) {
        Ok(true) => json_response(StatusCode::OK, serde_json::json!({"status": "updated"})),
        Ok(false) => error_response(StatusCode::NOT_FOUND, "API key not found"),
        Err(e) => {
            tracing::error!(error = %e, "update key owner failed");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to update API key owner",
            )
        }
    }
}

/// Build the JSON view of an `ApiKey`. The full `key` field is included only
/// at creation time (`include_full_key=true`); list/get responses return only
/// the masked `key_hint`.
fn api_key_response(key: &crate::storage::ApiKey, include_full_key: bool) -> serde_json::Value {
    let mut body = serde_json::json!({
        "id": key.id,
        "name": key.name,
        "key_hint": key.key_hint,
        "status": key.status,
        "allowed_models": key.allowed_models,
        "owner": key.owner,
        "owner_concurrency": key.owner_concurrency,
        "created_at": key.created_at,
    });
    if include_full_key {
        body["key"] = serde_json::Value::String(key.key.clone());
    }
    body
}

/// GET /v1/admin/keys - List all API keys (key field masked)
pub async fn handle_list_keys(
    req: Request<hyper::body::Incoming>,
    storage: Arc<dyn Storage>,
    master_key: &str,
) -> Response<BoxBody> {
    if let Err(resp) = check_admin(&req, master_key) {
        return *resp;
    }

    match storage.list_api_keys() {
        Ok(keys) => {
            let items: Vec<serde_json::Value> =
                keys.iter().map(|k| api_key_response(k, false)).collect();
            json_response(StatusCode::OK, serde_json::json!({ "data": items }))
        }
        Err(e) => {
            tracing::error!(error = %e, "list keys failed");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, "Failed to list API keys")
        }
    }
}

/// GET /v1/admin/keys/:id - Get a single API key (masked)
pub async fn handle_get_key(
    req: Request<hyper::body::Incoming>,
    storage: Arc<dyn Storage>,
    master_key: &str,
    key_id: i64,
) -> Response<BoxBody> {
    if let Err(resp) = check_admin(&req, master_key) {
        return *resp;
    }

    match storage.get_api_key(key_id) {
        Ok(Some(k)) => json_response(StatusCode::OK, api_key_response(&k, false)),
        Ok(None) => error_response(StatusCode::NOT_FOUND, "API key not found"),
        Err(e) => {
            tracing::error!(error = %e, "get key failed");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, "Failed to get API key")
        }
    }
}

/// POST /v1/admin/keys/:id/revoke - Revoke an API key
pub async fn handle_revoke_key(
    req: Request<hyper::body::Incoming>,
    storage: Arc<dyn Storage>,
    master_key: &str,
    key_id: i64,
) -> Response<BoxBody> {
    if let Err(resp) = check_admin(&req, master_key) {
        return *resp;
    }

    match storage.revoke_api_key(key_id) {
        Ok(true) => json_response(StatusCode::OK, serde_json::json!({"status": "revoked"})),
        Ok(false) => error_response(StatusCode::NOT_FOUND, "API key not found"),
        Err(e) => {
            tracing::error!(error = %e, "revoke key failed");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to revoke API key",
            )
        }
    }
}

/// POST /v1/admin/keys/:id/models - Replace a key's allowed-models list in
/// place. The key value is untouched, so deployed credentials keep working —
/// this is the plan-change path (upgrade grants models, downgrade removes).
pub async fn handle_update_key_models(
    req: Request<hyper::body::Incoming>,
    storage: Arc<dyn Storage>,
    master_key: &str,
    key_id: i64,
) -> Response<BoxBody> {
    if let Err(resp) = check_admin(&req, master_key) {
        return *resp;
    }

    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Failed to read request body: {}", e),
            );
        }
    };

    #[derive(serde::Deserialize)]
    struct UpdateModelsRequest {
        /// Same semantics as key creation: absent/`null` → unrestricted,
        /// a list → exact-match allow-list.
        #[serde(default)]
        allowed_models: Option<Vec<String>>,
    }

    let update_req: UpdateModelsRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Invalid request JSON: {}", e),
            );
        }
    };

    match storage.update_api_key_models(key_id, update_req.allowed_models.as_deref()) {
        Ok(true) => json_response(StatusCode::OK, serde_json::json!({"status": "updated"})),
        Ok(false) => error_response(StatusCode::NOT_FOUND, "API key not found"),
        Err(e) => {
            tracing::error!(error = %e, "update key models failed");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to update API key models",
            )
        }
    }
}

/// GET /v1/admin/usage?key_id=1,3&bucket=hour&since=...&until=...
pub async fn handle_usage(
    req: Request<hyper::body::Incoming>,
    storage: Arc<dyn Storage>,
    master_key: &str,
) -> Response<BoxBody> {
    if let Err(resp) = check_admin(&req, master_key) {
        return *resp;
    }

    let query = req.uri().query().unwrap_or("");
    let params = parse_query(query);

    let key_ids = parse_key_ids(params.get("key_id"));
    let bucket = params.get("bucket").and_then(|b| parse_bucket(b));
    let since = params.get("since").and_then(|s| s.parse::<u64>().ok());
    let until = params.get("until").and_then(|s| s.parse::<u64>().ok());

    match storage.get_usage(&key_ids, bucket, since, until) {
        Ok(rows) => {
            let items: Vec<serde_json::Value> = rows
                .into_iter()
                .map(|r| {
                    serde_json::json!({
                        "period": r.period,
                        "key_id": r.key_id,
                        "key_name": r.key_name,
                        "completions_count": r.completions_count,
                        "embeddings_count": r.embeddings_count,
                        "total_input_tokens": r.total_input_tokens,
                        "total_output_tokens": r.total_output_tokens,
                    })
                })
                .collect();
            json_response(StatusCode::OK, serde_json::json!({ "data": items }))
        }
        Err(e) => {
            tracing::error!(error = %e, "get usage failed");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, "Failed to get usage")
        }
    }
}

/// GET /v1/admin/completions?key_id=1,3&since=...&until=...
pub async fn handle_list_completions(
    req: Request<hyper::body::Incoming>,
    storage: Arc<dyn Storage>,
    master_key: &str,
) -> Response<BoxBody> {
    if let Err(resp) = check_admin(&req, master_key) {
        return *resp;
    }

    let query = req.uri().query().unwrap_or("");
    let params = parse_query(query);
    let filter = build_filter(&params);

    match storage.list_completions(&filter) {
        Ok(completions) => {
            let items: Vec<serde_json::Value> = completions
                .into_iter()
                .map(|c| {
                    serde_json::json!({
                        "id": c.id,
                        "api_key_id": c.api_key_id,
                        "session_id": c.session_id,
                        "input_model": c.input_model,
                        "resolved_model": c.resolved_model,
                        "provider": c.provider,
                        "usage": c.usage,
                        "input": c.input,
                        "output": c.output,
                        "created_at": c.created_at,
                    })
                })
                .collect();
            json_response(StatusCode::OK, serde_json::json!({ "data": items }))
        }
        Err(e) => {
            tracing::error!(error = %e, "list completions failed");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to list completions",
            )
        }
    }
}

/// GET /v1/admin/embeddings?key_id=1,3&since=...&until=...
pub async fn handle_list_embeddings(
    req: Request<hyper::body::Incoming>,
    storage: Arc<dyn Storage>,
    master_key: &str,
) -> Response<BoxBody> {
    if let Err(resp) = check_admin(&req, master_key) {
        return *resp;
    }

    let query = req.uri().query().unwrap_or("");
    let params = parse_query(query);
    let filter = build_filter(&params);

    match storage.list_embeddings(&filter) {
        Ok(embeddings) => {
            let items: Vec<serde_json::Value> = embeddings
                .into_iter()
                .map(|e| {
                    serde_json::json!({
                        "id": e.id,
                        "api_key_id": e.api_key_id,
                        "input_model": e.input_model,
                        "resolved_model": e.resolved_model,
                        "provider": e.provider,
                        "usage": e.usage,
                        "input": e.input,
                        "created_at": e.created_at,
                    })
                })
                .collect();
            json_response(StatusCode::OK, serde_json::json!({ "data": items }))
        }
        Err(e) => {
            tracing::error!(error = %e, "list embeddings failed");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to list embeddings",
            )
        }
    }
}

// --- Query parsing helpers ---

fn parse_query(query: &str) -> std::collections::HashMap<String, String> {
    query
        .split('&')
        .filter(|s| !s.is_empty())
        .filter_map(|pair| {
            let mut parts = pair.splitn(2, '=');
            let key = parts.next()?;
            let value = parts.next().unwrap_or("");
            Some((key.to_string(), value.to_string()))
        })
        .collect()
}

fn parse_key_ids(value: Option<&String>) -> Vec<i64> {
    value
        .map(|s| {
            s.split(',')
                .filter_map(|v| v.trim().parse::<i64>().ok())
                .collect()
        })
        .unwrap_or_default()
}

fn parse_bucket(value: &str) -> Option<TimeBucket> {
    match value {
        "hour" => Some(TimeBucket::Hour),
        "day" => Some(TimeBucket::Day),
        "week" => Some(TimeBucket::Week),
        "month" => Some(TimeBucket::Month),
        _ => None,
    }
}

fn build_filter(params: &std::collections::HashMap<String, String>) -> ListFilter {
    ListFilter {
        key_ids: parse_key_ids(params.get("key_id")),
        limit: params
            .get("limit")
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(100),
        offset: params
            .get("offset")
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0),
        since: params.get("since").and_then(|s| s.parse::<u64>().ok()),
        until: params.get("until").and_then(|s| s.parse::<u64>().ok()),
    }
}
