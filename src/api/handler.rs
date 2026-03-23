use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{Request, Response, StatusCode};

use crate::api::types::{CreateCompletionRequest, CreateEmbeddingRequest};
use crate::auth::{authenticate_client, ClientAuth};
use crate::proxy::engine::ProxyEngine;
use crate::storage::Storage;

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

/// Extract Authorization header value from request
fn auth_header(req: &Request<hyper::body::Incoming>) -> Option<String> {
    req.headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

/// Handle POST /v1/completions
pub async fn handle_create_completion(
    req: Request<hyper::body::Incoming>,
    engine: Arc<ProxyEngine>,
    storage: Arc<dyn Storage>,
) -> Response<BoxBody> {
    let header = auth_header(&req);
    let api_key_id = match authenticate_client(header.as_deref(), &storage) {
        ClientAuth::Ok(key) => key.id,
        ClientAuth::Missing => return error_response(StatusCode::UNAUTHORIZED, "Missing API key"),
        ClientAuth::Invalid => {
            return error_response(StatusCode::UNAUTHORIZED, "Invalid or revoked API key")
        }
    };

    // Read body
    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Failed to read request body: {}", e),
            );
        }
    };

    // Parse request
    let create_req: CreateCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Invalid request JSON: {}", e),
            );
        }
    };

    // Process
    match engine.process(create_req, api_key_id).await {
        Ok(response) => {
            let body = serde_json::to_value(&response).unwrap_or_default();
            json_response(StatusCode::OK, body)
        }
        Err(e) => {
            let (status, msg) = classify_engine_error(&e);
            if status.is_server_error() {
                tracing::error!("Request processing failed: {:?}", e);
            } else {
                tracing::warn!("Request rejected: {}", msg);
            }
            error_response(status, &msg)
        }
    }
}

/// Handle GET /health
pub fn handle_health() -> Response<BoxBody> {
    json_response(StatusCode::OK, serde_json::json!({"status": "ok"}))
}

/// Handle POST /v1/embeddings
pub async fn handle_create_embedding(
    req: Request<hyper::body::Incoming>,
    engine: Arc<ProxyEngine>,
    storage: Arc<dyn Storage>,
) -> Response<BoxBody> {
    let header = auth_header(&req);
    let api_key_id = match authenticate_client(header.as_deref(), &storage) {
        ClientAuth::Ok(key) => key.id,
        ClientAuth::Missing => return error_response(StatusCode::UNAUTHORIZED, "Missing API key"),
        ClientAuth::Invalid => {
            return error_response(StatusCode::UNAUTHORIZED, "Invalid or revoked API key")
        }
    };

    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Failed to read request body: {}", e),
            );
        }
    };

    let create_req: CreateEmbeddingRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Invalid request JSON: {}", e),
            );
        }
    };

    match engine.process_embedding(create_req, api_key_id).await {
        Ok(response) => {
            let body = serde_json::to_value(&response).unwrap_or_default();
            json_response(StatusCode::OK, body)
        }
        Err(e) => {
            let (status, msg) = classify_engine_error(&e);
            if status.is_server_error() {
                tracing::error!("Embedding request failed: {:?}", e);
            } else {
                tracing::warn!("Embedding request rejected: {}", msg);
            }
            error_response(status, &msg)
        }
    }
}

/// Classify engine errors into HTTP status codes.
/// Client errors (bad model, bad input) → 400, server errors → 500.
fn classify_engine_error(error: &anyhow::Error) -> (StatusCode, String) {
    let msg = format!("{}", error);

    // Model/provider resolution failures are client errors
    let is_client_error = msg.contains("not found in config")
        || msg.contains("Failed to resolve model")
        || msg.contains("Failed to resolve embedding model")
        || msg.contains("not available")
        || msg.contains("Invalid request")
        || msg.contains("Failed to walk chain");

    let status = if is_client_error {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };

    (status, msg)
}
