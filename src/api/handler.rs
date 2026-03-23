use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{Request, Response, StatusCode};

use crate::api::types::{CreateCompletionRequest, CreateEmbeddingRequest};
use crate::proxy::engine::ProxyEngine;

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

/// Handle POST /v1/completions
pub async fn handle_create_completion(
    req: Request<hyper::body::Incoming>,
    engine: Arc<ProxyEngine>,
    api_key: Option<String>,
) -> Response<BoxBody> {
    // Auth check
    let auth_header = req
        .headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok());

    if !crate::auth::check_auth(auth_header, api_key.as_deref()) {
        return error_response(StatusCode::UNAUTHORIZED, "Invalid or missing API key");
    }

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
    match engine.process(create_req).await {
        Ok(response) => {
            let body = serde_json::to_value(&response).unwrap_or_default();
            json_response(StatusCode::OK, body)
        }
        Err(e) => {
            tracing::error!("Request processing failed: {:?}", e);
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Processing failed: {}", e),
            )
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
    api_key: Option<String>,
) -> Response<BoxBody> {
    let auth_header = req
        .headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok());

    if !crate::auth::check_auth(auth_header, api_key.as_deref()) {
        return error_response(StatusCode::UNAUTHORIZED, "Invalid or missing API key");
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

    let create_req: CreateEmbeddingRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Invalid request JSON: {}", e),
            );
        }
    };

    match engine.process_embedding(create_req).await {
        Ok(response) => {
            let body = serde_json::to_value(&response).unwrap_or_default();
            json_response(StatusCode::OK, body)
        }
        Err(e) => {
            tracing::error!("Embedding request failed: {:?}", e);
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Embedding processing failed: {}", e),
            )
        }
    }
}
