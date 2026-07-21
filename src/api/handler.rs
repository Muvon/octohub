use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{Request, Response, StatusCode};

use crate::api::types::{
    ChatCompletionRequest, ChatCompletionResponse, CreateCompletionRequest, CreateEmbeddingRequest,
};
use crate::auth::{authenticate_client, ClientAuth};
use crate::proxy::engine::{
    upstream_status_code, ProxyEngine, ProxyTimeoutError, RateLimitedError, MODEL_FORBIDDEN_MARKER,
    OWNER_LIMIT_MARKER,
};
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

/// `error_response` plus a `Retry-After` header when the failure is a
/// provider rate-window rejection, so well-behaved clients back off for
/// exactly as long as the window needs.
fn engine_error_response(
    error: &anyhow::Error,
    status: StatusCode,
    message: &str,
) -> Response<BoxBody> {
    let mut response = error_response(status, message);
    if let Some(rate) = error.downcast_ref::<RateLimitedError>() {
        let secs = rate.retry_after.as_secs().max(1);
        if let Ok(value) = hyper::header::HeaderValue::from_str(&secs.to_string()) {
            response
                .headers_mut()
                .insert(hyper::header::RETRY_AFTER, value);
        }
    }
    response
}

/// Extract Authorization header value from request
fn auth_header(req: &Request<hyper::body::Incoming>) -> Option<String> {
    req.headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

/// The client's routing purpose for the virtual `auto` model. Read BEFORE the
/// body is consumed; absent/blank → `None` (the `[auto]` default applies).
fn model_purpose(req: &Request<hyper::body::Incoming>) -> Option<String> {
    req.headers()
        .get("X-Model-Purpose")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Handle POST /v1/completions
pub async fn handle_create_completion(
    req: Request<hyper::body::Incoming>,
    engine: Arc<ProxyEngine>,
    storage: Arc<dyn Storage>,
) -> Response<BoxBody> {
    let header = auth_header(&req);
    let storage_clone = storage.clone();
    let auth_result =
        tokio::task::spawn_blocking(move || authenticate_client(header.as_deref(), &storage_clone))
            .await
            .unwrap_or(ClientAuth::Invalid);
    let api_key = match auth_result {
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
    let purpose = model_purpose(&req);

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
    let model_label = create_req.model.clone();
    match engine.process(create_req, &api_key, purpose).await {
        Ok((response, upstream_duration)) => {
            tracing::Span::current().record("tok_in", response.usage.input_tokens);
            tracing::Span::current().record("tok_out", response.usage.output_tokens);

            crate::metrics::record_completion(
                &response.model,
                &response.provider,
                "ok",
                upstream_duration,
                response.usage.input_tokens,
                response.usage.output_tokens,
                Some(api_key.id),
                per_key,
            );

            let body = serde_json::to_value(&response).unwrap_or_default();
            json_response(StatusCode::OK, body)
        }
        Err(e) => {
            let (status, msg) = classify_engine_error(&e);
            if status.is_server_error() {
                // `?e` (Debug) prints the full anyhow chain — "Caused by: …" lines
                // for each layer. `%e` (Display) only shows the outermost context,
                // which hid the actual upstream error body and HTTP status.
                tracing::error!(error = ?e, "completion failed");
            } else {
                tracing::warn!(reason = %msg, "request rejected");
            }

            // Provider/duration unknown on the error path — the call either
            // never reached an upstream or failed before producing usage.
            crate::metrics::record_completion(
                &model_label,
                "unknown",
                "error",
                std::time::Duration::ZERO,
                0,
                0,
                Some(api_key.id),
                per_key,
            );

            engine_error_response(&e, status, &msg)
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
    let storage_clone = storage.clone();
    let auth_result =
        tokio::task::spawn_blocking(move || authenticate_client(header.as_deref(), &storage_clone))
            .await
            .unwrap_or(ClientAuth::Invalid);
    let api_key = match auth_result {
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
    match engine.process_embedding(create_req, &api_key).await {
        Ok(outcome) => {
            tracing::Span::current().record("tok_in", outcome.input_tokens);

            crate::metrics::record_embedding(
                &model_label,
                &outcome.provider,
                "ok",
                outcome.upstream_duration,
                outcome.input_tokens,
                Some(api_key.id),
                per_key,
            );

            let body = serde_json::to_value(&outcome.response).unwrap_or_default();
            json_response(StatusCode::OK, body)
        }
        Err(e) => {
            let (status, msg) = classify_engine_error(&e);
            if status.is_server_error() {
                tracing::error!(error = ?e, "embedding failed");
            } else {
                tracing::warn!(reason = %msg, "embedding request rejected");
            }

            crate::metrics::record_embedding(
                &model_label,
                "unknown",
                "error",
                std::time::Duration::ZERO,
                0,
                Some(api_key.id),
                per_key,
            );

            engine_error_response(&e, status, &msg)
        }
    }
}

/// Classify engine errors into HTTP status codes.
/// Per-key model restriction → 403, bad model/input → 400, otherwise 500.
fn classify_engine_error(error: &anyhow::Error) -> (StatusCode, String) {
    // Top-level message — used only for classification heuristics and for
    // client-error responses (where our own marker is the correct user-facing
    // text). For 500s we surface the full chain below.
    let top = format!("{}", error);
    let full = format!("{:#}", error);

    if let Some(timeout) = error.downcast_ref::<ProxyTimeoutError>() {
        let status = match timeout {
            ProxyTimeoutError::ProviderQueue { .. } => StatusCode::SERVICE_UNAVAILABLE,
            ProxyTimeoutError::Upstream { .. } => StatusCode::GATEWAY_TIMEOUT,
        };
        return (status, timeout.to_string());
    }

    if top.contains(MODEL_FORBIDDEN_MARKER) {
        // Strip the internal marker before returning the message to clients —
        // it's a routing hint, not part of the user-facing error.
        let cleaned = top
            .replace(&format!("{}: ", MODEL_FORBIDDEN_MARKER), "")
            .replace(MODEL_FORBIDDEN_MARKER, "");
        return (StatusCode::FORBIDDEN, cleaned);
    }

    if error.downcast_ref::<RateLimitedError>().is_some() {
        // Every candidate provider's rate window is exhausted — retryable,
        // just not yet. `engine_error_response` adds the Retry-After header.
        return (StatusCode::TOO_MANY_REQUESTS, top);
    }

    if top.contains(OWNER_LIMIT_MARKER) {
        // The tenant's shared in-flight budget stayed saturated for the whole
        // queue wait — tell the caller THEY are the bottleneck (429), not us.
        let cleaned = top
            .replace(&format!("{}: ", OWNER_LIMIT_MARKER), "")
            .replace(OWNER_LIMIT_MARKER, "");
        return (StatusCode::TOO_MANY_REQUESTS, cleaned);
    }

    let is_client_error = top.contains("not found in config")
        || top.contains("Failed to resolve model")
        || top.contains("Failed to resolve embedding model")
        || top.contains("not available")
        || top.contains("Invalid request");

    if is_client_error {
        return (StatusCode::BAD_REQUEST, top);
    }

    // Upstream provider client-errors (4xx) are PERMANENT for this request — most
    // importantly a 400 "prompt is too long" (context overflow). octolib formats
    // them as "... API error <code> ...". Returning 500 here makes octolib's retry
    // classifier (`is_retryable_status`: 429 || >=500) re-send the IDENTICAL request
    // in a backoff loop forever. Map an upstream 4xx to the same status so the caller
    // fails fast; a 429 stays legitimately retryable, and 5xx still flows to 500 below.
    if let Some(code) = upstream_status_code(&full) {
        if (400..500).contains(&code) {
            let status = StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_REQUEST);
            return (status, full);
        }
    }

    // Server-side failure (most commonly an upstream provider 5xx). Surface the
    // full anyhow chain via `{:#}` so callers see "Provider 'anthropic'
    // chat_completion failed: Anthropic API error 503: { ... }" instead of just
    // the outer wrap. Without this the client gets a useless top message and has
    // to read server logs to diagnose anything.
    (StatusCode::INTERNAL_SERVER_ERROR, full)
}

/// Handle POST /v1/chat/completions (classic OpenAI-compatible)
pub async fn handle_chat_completion(
    req: Request<hyper::body::Incoming>,
    engine: Arc<ProxyEngine>,
    storage: Arc<dyn Storage>,
) -> Response<BoxBody> {
    let header = auth_header(&req);
    let storage_clone = storage.clone();
    let auth_result =
        tokio::task::spawn_blocking(move || authenticate_client(header.as_deref(), &storage_clone))
            .await
            .unwrap_or(ClientAuth::Invalid);
    let api_key = match auth_result {
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
    let purpose = model_purpose(&req);

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

    // Parse classic chat request
    let chat_req: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Invalid request JSON: {}", e),
            );
        }
    };

    if chat_req.stream {
        return error_response(
            StatusCode::NOT_IMPLEMENTED,
            "Streaming is not supported on this endpoint",
        );
    }

    // Convert to internal representation — same engine path as /v1/completions
    let model_label = chat_req.model.clone();
    let create_req: CreateCompletionRequest = chat_req.into();
    tracing::Span::current().record("model", create_req.model.as_str());

    let per_key = engine.config().metrics.per_key;

    match engine.process(create_req, &api_key, purpose).await {
        Ok((response, upstream_duration)) => {
            tracing::Span::current().record("tok_in", response.usage.input_tokens);
            tracing::Span::current().record("tok_out", response.usage.output_tokens);

            crate::metrics::record_completion(
                &response.model,
                &response.provider,
                "ok",
                upstream_duration,
                response.usage.input_tokens,
                response.usage.output_tokens,
                Some(api_key.id),
                per_key,
            );

            let chat_resp: ChatCompletionResponse = response.into();
            let body = serde_json::to_value(&chat_resp).unwrap_or_default();
            json_response(StatusCode::OK, body)
        }
        Err(e) => {
            let (status, msg) = classify_engine_error(&e);
            if status.is_server_error() {
                tracing::error!(error = ?e, "completion failed");
            } else {
                tracing::warn!(reason = %msg, "request rejected");
            }

            crate::metrics::record_completion(
                &model_label,
                "unknown",
                "error",
                std::time::Duration::ZERO,
                0,
                0,
                Some(api_key.id),
                per_key,
            );

            engine_error_response(&e, status, &msg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    #[test]
    fn queue_timeout_maps_to_service_unavailable() {
        let error = anyhow::Error::new(ProxyTimeoutError::ProviderQueue {
            provider: "ollama".to_string(),
            timeout: std::time::Duration::from_secs(60),
        });
        let (status, message) = classify_engine_error(&error);

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(message.contains("waiting for provider 'ollama' capacity"));
    }

    #[test]
    fn wrapped_upstream_timeout_maps_to_gateway_timeout() {
        let error = Err::<(), _>(anyhow::Error::new(ProxyTimeoutError::Upstream {
            provider: "anthropic".to_string(),
            timeout: std::time::Duration::from_secs(360),
        }))
        .context("Provider chat_completion failed")
        .unwrap_err();
        let (status, message) = classify_engine_error(&error);

        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(
            message,
            "provider 'anthropic' exceeded the 360s operation deadline"
        );
    }

    #[test]
    fn owner_limit_marker_maps_to_429() {
        let error = anyhow::anyhow!(
            "{}: owner concurrency limit (10) exhausted — retry shortly",
            OWNER_LIMIT_MARKER
        );
        let (status, msg) = classify_engine_error(&error);
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert!(
            !msg.contains(OWNER_LIMIT_MARKER),
            "internal marker must be stripped from the client message"
        );
    }

    #[test]
    fn rate_limited_maps_to_429_with_retry_after_header() {
        let error = anyhow::anyhow!(RateLimitedError {
            model: "kimi-k2".to_string(),
            retry_after: std::time::Duration::from_secs(37),
        });
        let (status, msg) = classify_engine_error(&error);
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert!(msg.contains("kimi-k2"));

        let response = engine_error_response(&error, status, &msg);
        assert_eq!(
            response.headers().get(hyper::header::RETRY_AFTER).unwrap(),
            "37"
        );
    }

    #[test]
    fn retry_after_is_at_least_one_second() {
        // A sub-second window remainder must not round down to "0" — that
        // tells clients to retry immediately, defeating the backoff.
        let error = anyhow::anyhow!(RateLimitedError {
            model: "m".to_string(),
            retry_after: std::time::Duration::from_millis(200),
        });
        let (status, msg) = classify_engine_error(&error);
        let response = engine_error_response(&error, status, &msg);
        assert_eq!(
            response.headers().get(hyper::header::RETRY_AFTER).unwrap(),
            "1"
        );
    }

    #[test]
    fn non_rate_limit_errors_get_no_retry_after() {
        let error = anyhow::anyhow!("database connection lost");
        let (status, msg) = classify_engine_error(&error);
        let response = engine_error_response(&error, status, &msg);
        assert!(response.headers().get(hyper::header::RETRY_AFTER).is_none());
    }

    #[test]
    fn upstream_400_maps_to_client_400_not_retryable_500() {
        // ollama's 400 "prompt too long" must NOT become a retryable 500, or
        // octolib re-sends the identical oversized request in a loop.
        let error = Err::<(), _>(anyhow::anyhow!(
            "ollama API error 400 Bad Request: {{\"error\":\"The prompt is too long: 380813\"}}"
        ))
        .context("Provider 'ollama' chat_completion failed")
        .unwrap_err();
        let (status, _) = classify_engine_error(&error);
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn upstream_429_stays_retryable() {
        let error = Err::<(), _>(anyhow::anyhow!("anthropic API error 429 Too Many Requests"))
            .context("Provider 'anthropic' chat_completion failed")
            .unwrap_err();
        let (status, _) = classify_engine_error(&error);
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn upstream_503_stays_500_retryable() {
        let error = Err::<(), _>(anyhow::anyhow!("openai API error 503 Service Unavailable"))
            .context("Provider 'openai' chat_completion failed")
            .unwrap_err();
        let (status, _) = classify_engine_error(&error);
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn non_provider_error_stays_500() {
        let error = anyhow::anyhow!("database connection lost");
        let (status, _) = classify_engine_error(&error);
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    }
}
