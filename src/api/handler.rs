use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{Request, Response, StatusCode};

use crate::api::types::{CreateCompletionRequest, CreateEmbeddingRequest};
use crate::auth::{authenticate_client, ClientAuth};
use crate::proxy::engine::{ProxyEngine, MODEL_FORBIDDEN_MARKER};
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
    let api_key = match authenticate_client(header.as_deref(), &storage) {
        ClientAuth::Ok(key) => key,
        ClientAuth::Missing => {
            tracing::warn!(kind = "client", reason = "missing_token", "auth failed");
            return error_response(StatusCode::UNAUTHORIZED, "Missing API key");
        }
        ClientAuth::Invalid => {
            tracing::warn!(kind = "client", reason = "invalid_token", "auth failed");
            return error_response(StatusCode::UNAUTHORIZED, "Invalid or revoked API key");
        }
    };

    tracing::Span::current().record("api_key_id", api_key.id);

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

    tracing::Span::current().record("model", create_req.model.as_str());

    let per_key = engine.config().metrics.per_key;

    // Process
    let start = Instant::now();
    let model_label = create_req.model.clone();
    match engine.process(create_req, &api_key).await {
        Ok(response) => {
            let elapsed = start.elapsed();
            tracing::Span::current().record("tok_in", response.usage.input_tokens);
            tracing::Span::current().record("tok_out", response.usage.output_tokens);

            crate::metrics::record_completion(
                &response.model,
                &response.provider,
                "ok",
                elapsed,
                response.usage.input_tokens,
                response.usage.output_tokens,
                Some(api_key.id),
                per_key,
            );

            let body = serde_json::to_value(&response).unwrap_or_default();
            json_response(StatusCode::OK, body)
        }
        Err(e) => {
            let elapsed = start.elapsed();
            let (status, msg) = classify_engine_error(&e);
            if status.is_server_error() {
                tracing::error!(error = %e, "completion failed");
            } else {
                tracing::warn!(reason = %msg, "request rejected");
            }

            crate::metrics::record_completion(
                &model_label,
                "unknown",
                "error",
                elapsed,
                0,
                0,
                Some(api_key.id),
                per_key,
            );

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
    let api_key = match authenticate_client(header.as_deref(), &storage) {
        ClientAuth::Ok(key) => key,
        ClientAuth::Missing => {
            tracing::warn!(kind = "client", reason = "missing_token", "auth failed");
            return error_response(StatusCode::UNAUTHORIZED, "Missing API key");
        }
        ClientAuth::Invalid => {
            tracing::warn!(kind = "client", reason = "invalid_token", "auth failed");
            return error_response(StatusCode::UNAUTHORIZED, "Invalid or revoked API key");
        }
    };

    tracing::Span::current().record("api_key_id", api_key.id);

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

    tracing::Span::current().record("model", create_req.model.as_str());

    let per_key = engine.config().metrics.per_key;

    let model_label = create_req.model.clone();
    let start = Instant::now();
    match engine.process_embedding(create_req, &api_key).await {
        Ok(response) => {
            let elapsed = start.elapsed();

            crate::metrics::record_embedding(
                &model_label,
                "unknown",
                "ok",
                elapsed,
                0,
                Some(api_key.id),
                per_key,
            );

            let body = serde_json::to_value(&response).unwrap_or_default();
            json_response(StatusCode::OK, body)
        }
        Err(e) => {
            let elapsed = start.elapsed();
            let (status, msg) = classify_engine_error(&e);
            if status.is_server_error() {
                tracing::error!(error = %e, "embedding failed");
            } else {
                tracing::warn!(reason = %msg, "embedding request rejected");
            }

            crate::metrics::record_embedding(
                &model_label,
                "unknown",
                "error",
                elapsed,
                0,
                Some(api_key.id),
                per_key,
            );

            error_response(status, &msg)
        }
    }
}

/// Classify engine errors into HTTP status codes.
/// Per-key model restriction → 403, bad model/input → 400, otherwise 500.
fn classify_engine_error(error: &anyhow::Error) -> (StatusCode, String) {
    let msg = format!("{}", error);

    if msg.contains(MODEL_FORBIDDEN_MARKER) {
        // Strip the internal marker before returning the message to clients —
        // it's a routing hint, not part of the user-facing error.
        let cleaned = msg
            .replace(&format!("{}: ", MODEL_FORBIDDEN_MARKER), "")
            .replace(MODEL_FORBIDDEN_MARKER, "");
        return (StatusCode::FORBIDDEN, cleaned);
    }

    let is_client_error = msg.contains("not found in config")
        || msg.contains("Failed to resolve model")
        || msg.contains("Failed to resolve embedding model")
        || msg.contains("not available")
        || msg.contains("Invalid request");

    let status = if is_client_error {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };

    (status, msg)
}
