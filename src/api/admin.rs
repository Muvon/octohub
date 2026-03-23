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

    match storage.create_api_key(create_req.name.trim()) {
        Ok(key) => {
            // On creation, return the full key (only time it's visible)
            json_response(
                StatusCode::CREATED,
                serde_json::json!({
                    "id": key.id,
                    "name": key.name,
                    "key": key.key,
                    "key_hint": key.key_hint,
                    "status": key.status,
                    "created_at": key.created_at,
                }),
            )
        }
        Err(e) => {
            tracing::error!("Failed to create API key: {:?}", e);
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Failed to create API key: {}", e),
            )
        }
    }
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
            let items: Vec<serde_json::Value> = keys
                .into_iter()
                .map(|k| {
                    serde_json::json!({
                        "id": k.id,
                        "name": k.name,
                        "key_hint": k.key_hint,
                        "status": k.status,
                        "created_at": k.created_at,
                    })
                })
                .collect();
            json_response(StatusCode::OK, serde_json::json!({ "data": items }))
        }
        Err(e) => {
            tracing::error!("Failed to list API keys: {:?}", e);
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
        Ok(Some(k)) => json_response(
            StatusCode::OK,
            serde_json::json!({
                "id": k.id,
                "name": k.name,
                "key_hint": k.key_hint,
                "status": k.status,
                "created_at": k.created_at,
            }),
        ),
        Ok(None) => error_response(StatusCode::NOT_FOUND, "API key not found"),
        Err(e) => {
            tracing::error!("Failed to get API key: {:?}", e);
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
            tracing::error!("Failed to revoke API key: {:?}", e);
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to revoke API key",
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
            tracing::error!("Failed to get usage: {:?}", e);
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
            tracing::error!("Failed to list completions: {:?}", e);
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
            tracing::error!("Failed to list embeddings: {:?}", e);
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
