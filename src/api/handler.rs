use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{Request, Response, StatusCode};

use crate::api::types::{
    ChatCompletionRequest, ChatCompletionResponse, CreateCompletionRequest, CreateEmbeddingRequest,
};
use crate::auth::{authenticate_client, ClientAuth};
use crate::proxy::engine::{
    upstream_status_code, ModalityNotSupportedError, ProxyEngine, ProxyTimeoutError,
    RateLimitedError, MODEL_FORBIDDEN_MARKER, OWNER_LIMIT_MARKER,
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

/// CROSS-REPO CONTRACT: the `error.type` value clients match on to detect that
/// the selected model cannot accept an attached modality. `octomind` pre-checks
/// capability locally and relies on this string as the backstop for models its
/// own capability table does not know, so it must never be reworded — callers
/// must not have to regex the prose message to identify the condition.
pub const MODALITY_ERROR_TYPE: &str = "modality_not_supported";

/// The default OpenAI-compatible error discriminator.
const DEFAULT_ERROR_TYPE: &str = "invalid_request_error";

fn error_response(status: StatusCode, message: &str) -> Response<BoxBody> {
    error_response_typed(status, message, DEFAULT_ERROR_TYPE)
}

fn error_response_typed(status: StatusCode, message: &str, error_type: &str) -> Response<BoxBody> {
    json_response(
        status,
        serde_json::json!({
            "error": {
                "message": message,
                "type": error_type
            }
        }),
    )
}

/// The `error.type` discriminator for an engine failure. Conditions a caller
/// must BRANCH on get their own string; everything else stays the generic
/// OpenAI-compatible default.
fn engine_error_type(error: &anyhow::Error) -> &'static str {
    if error.downcast_ref::<ModalityNotSupportedError>().is_some() {
        return MODALITY_ERROR_TYPE;
    }
    DEFAULT_ERROR_TYPE
}

/// `error_response` plus a `Retry-After` header when the failure is a
/// provider rate-window rejection, so well-behaved clients back off for
/// exactly as long as the window needs, and a specific `error.type` for
/// conditions a caller has to branch on rather than merely display.
fn engine_error_response(
    error: &anyhow::Error,
    status: StatusCode,
    message: &str,
) -> Response<BoxBody> {
    let mut response = error_response_typed(status, message, engine_error_type(error));
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

/// OpenRouter-style attribution headers from the client (X-Title, HTTP-Referer),
/// forwarded upstream so the originating app — not this proxy — gets credited.
/// Read BEFORE the body is consumed.
fn attribution_headers(
    req: &Request<hyper::body::Incoming>,
) -> Option<std::collections::HashMap<String, String>> {
    let mut map = std::collections::HashMap::new();
    for name in ["X-Title", "HTTP-Referer"] {
        if let Some(v) = req.headers().get(name).and_then(|v| v.to_str().ok()) {
            map.insert(name.to_string(), v.to_string());
        }
    }
    (!map.is_empty()).then_some(map)
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
    let attribution = attribution_headers(&req);

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
    match engine
        .process(create_req, &api_key, purpose, attribution)
        .await
    {
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

    if error.downcast_ref::<ModalityNotSupportedError>().is_some() {
        return (StatusCode::BAD_REQUEST, top);
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
    let attribution = attribution_headers(&req);

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

    // Streaming is emulated: the upstream call stays buffered (same engine
    // path as non-streaming), and the finished response is re-framed as
    // OpenAI `chat.completion.chunk` SSE on the way out. Clients that require
    // `stream: true` get a fully OpenAI-compatible `text/event-stream`;
    // time-to-first-token is the same as non-streaming (see
    // `chat_completion_stream_body`).
    let stream = chat_req.stream;

    // Convert to internal representation — same engine path as /v1/completions
    let model_label = chat_req.model.clone();
    let create_req: CreateCompletionRequest = chat_req.into();
    tracing::Span::current().record("model", create_req.model.as_str());

    let per_key = engine.config().metrics.per_key;

    match engine
        .process(create_req, &api_key, purpose, attribution)
        .await
    {
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
            if stream {
                let body = Bytes::from(chat_completion_stream_body(&chat_resp));
                Response::builder()
                    .status(StatusCode::OK)
                    .header("Content-Type", "text/event-stream")
                    .header("Cache-Control", "no-cache")
                    .body(Full::new(body))
                    .unwrap()
            } else {
                let body = serde_json::to_value(&chat_resp).unwrap_or_default();
                json_response(StatusCode::OK, body)
            }
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

/// Render a finished classic chat response as OpenAI `chat.completion.chunk`
/// SSE. The upstream is called buffered (see `handle_chat_completion`), so this
/// is format-compatibility streaming: content is re-framed into a sequence of
/// deltas the way a real stream would arrive, then a `[DONE]` sentinel. The
/// concatenated `delta.content` of every chunk reassembles the exact original
/// text.
fn chat_completion_stream_body(resp: &ChatCompletionResponse) -> String {
    let choice = resp
        .choices
        .first()
        .expect("chat completion has one choice");
    let id = &resp.id;
    let created = resp.created;
    let model = &resp.model;
    let finish_reason = &choice.finish_reason;

    let base = |delta: serde_json::Value| {
        serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": serde_json::Value::Null,
            }],
        })
    };

    let mut out = String::new();

    // First delta carries the assistant role, matching a real stream's opening
    // chunk so clients that await `delta.role` before reading content see it.
    out.push_str(&sse_chunk(base(serde_json::json!({ "role": "assistant" }))));

    // Split content on whitespace boundaries so token-counting clients see a
    // sequence of deltas rather than one blob. Each chunk is the *increment*
    // since the previous boundary, so concatenating all `delta.content` values
    // reassembles the exact original text (SSE deltas are not running prefixes).
    if let Some(content) = &choice.message.content {
        let mut prev = 0;
        for word_end in split_words(content) {
            out.push_str(&sse_chunk(base(serde_json::json!({
                "content": &content[prev..word_end],
            }))));
            prev = word_end;
        }
    }

    // Terminal chunk: empty delta carrying the finish reason.
    out.push_str(&sse_chunk(serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": serde_json::json!({}),
            "finish_reason": finish_reason,
        }],
    })));

    out.push_str("data: [DONE]\n\n");
    out
}

/// Offsets of whitespace-delimited word boundaries in `s`, so the full text can
/// be sliced into per-word SSE deltas. Returns the end index of each word.
fn split_words(s: &str) -> Vec<usize> {
    let mut ends = Vec::new();
    let mut prev = 0;
    for (i, c) in s.char_indices() {
        if c.is_whitespace() {
            if i > prev {
                ends.push(i);
            }
            prev = i + c.len_utf8();
        }
    }
    if prev < s.len() {
        ends.push(s.len());
    }
    ends
}

fn sse_chunk(value: serde_json::Value) -> String {
    format!(
        "data: {}\n\n",
        serde_json::to_string(&value).unwrap_or_default()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::{ChatChoice, ChatResponseMessage, ChatUsage};
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
    fn modality_exhaustion_maps_to_distinct_client_error() {
        let error = anyhow::anyhow!(ModalityNotSupportedError {
            provider: "ollama".to_string(),
            model: "llama3.2".to_string(),
            modality: "image".to_string(),
        });

        let (status, message) = classify_engine_error(&error);

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(message.contains("llama3.2"));
        assert!(message.contains("image"));
        assert!(!message.contains("No provider candidate"));
    }

    #[test]
    fn modality_error_carries_the_cross_repo_discriminator() {
        // octomind branches on `error.type`, never on the prose message.
        let error = anyhow::anyhow!(ModalityNotSupportedError {
            provider: "ollama".to_string(),
            model: "llama3.2".to_string(),
            modality: "image".to_string(),
        });

        assert_eq!(engine_error_type(&error), "modality_not_supported");
    }

    #[test]
    fn ordinary_engine_errors_keep_the_generic_discriminator() {
        let error = anyhow::anyhow!("database connection lost");

        assert_eq!(engine_error_type(&error), "invalid_request_error");
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

    #[test]
    fn stream_body_concatenated_deltas_reassemble_original_text() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion",
            created: 1_700_000_000,
            model: "gpt-test".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatResponseMessage {
                    role: "assistant",
                    content: Some("Hello, world!  This is a test.".to_string()),
                    tool_calls: None,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: ChatUsage {
                prompt_tokens: 3,
                completion_tokens: 7,
                total_tokens: 10,
            },
        };

        let body = chat_completion_stream_body(&resp);

        // Terminates with the [DONE] sentinel.
        assert!(body.ends_with("data: [DONE]\n\n"), "body: {body}");

        // Concatenating every chunk's delta.content must reassemble the exact
        // original text (SSE deltas are increments, not running prefixes).
        let mut reassembled = String::new();
        for data_line in body.lines().filter(|l| l.starts_with("data: ")) {
            if data_line == "data: [DONE]" {
                continue;
            }
            let chunk: serde_json::Value = serde_json::from_str(&data_line[6..]).unwrap();
            let delta = &chunk["choices"][0]["delta"];
            // Skip the role-only opening chunk and the empty terminal delta.
            if let Some(content) = delta["content"].as_str() {
                reassembled.push_str(content);
            }
        }
        assert_eq!(reassembled, "Hello, world!  This is a test.");
    }

    #[test]
    fn split_words_returns_increment_boundaries() {
        assert_eq!(split_words("Hello world"), vec![5, 11]);
        assert_eq!(split_words("  leading and  trailing  "), vec![9, 13, 23]);
        assert_eq!(split_words("single"), vec![6]);
        assert_eq!(split_words("abc  def"), vec![3, 8]);
        assert_eq!(split_words(""), Vec::<usize>::new());
    }
}

// ── Media ──

/// Authenticate a media request, returning the key or the ready-made 401.
async fn media_auth(
    req: &Request<hyper::body::Incoming>,
    storage: Arc<dyn Storage>,
) -> Result<crate::storage::ApiKey, Response<BoxBody>> {
    let header = auth_header(req);
    let auth_result =
        tokio::task::spawn_blocking(move || authenticate_client(header.as_deref(), &storage))
            .await
            .unwrap_or(ClientAuth::Invalid);
    match auth_result {
        ClientAuth::Ok(key) => {
            tracing::Span::current().record("api_key_id", key.id);
            Ok(key)
        }
        ClientAuth::Missing => {
            tracing::warn!(kind = "client", reason = "missing_token", "auth failed");
            Err(error_response(StatusCode::UNAUTHORIZED, "Missing API key"))
        }
        ClientAuth::Invalid => {
            tracing::warn!(kind = "client", reason = "invalid_token", "auth failed");
            Err(error_response(
                StatusCode::UNAUTHORIZED,
                "Invalid or revoked API key",
            ))
        }
    }
}

/// Map an octolib `MediaError` onto its HTTP status and error discriminator.
///
/// `WaitTimeout` and `LocalWaitCancelled` are absent on purpose: they are not
/// failures. The engine converts them into a 202 with a resumable job id so a
/// paid job is never lost to a slow client.
fn classify_media_error(error: &anyhow::Error) -> (StatusCode, String, &'static str) {
    use octolib::media::{FailureCategory, MediaError};

    if let Some(unsupported) = error.downcast_ref::<crate::proxy::media::MediaTaskUnsupported>() {
        return (
            StatusCode::BAD_REQUEST,
            unsupported.to_string(),
            "media_task_not_supported",
        );
    }

    let Some(media) = error.downcast_ref::<MediaError>() else {
        // Not an octolib media failure — reuse the shared engine classifier so
        // model-restriction, owner-budget and queue-timeout paths behave
        // identically across every endpoint.
        let (status, message) = classify_engine_error(error);
        return (status, message, engine_error_type(error));
    };

    let message = media.to_string();
    match media {
        MediaError::InvalidModelFormat(_)
        | MediaError::InvalidRequest(_)
        | MediaError::SourceTooLarge { .. } => {
            (StatusCode::BAD_REQUEST, message, DEFAULT_ERROR_TYPE)
        }
        MediaError::UnsupportedParameter { .. } => {
            (StatusCode::BAD_REQUEST, message, "unsupported_parameter")
        }
        MediaError::UnsupportedProvider(_) | MediaError::UnsupportedTask { .. } => {
            (StatusCode::BAD_REQUEST, message, "media_task_not_supported")
        }
        MediaError::MissingApiKey(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            message,
            "configuration_error",
        ),
        MediaError::Authentication { .. } | MediaError::Permission { .. } => {
            (StatusCode::BAD_GATEWAY, message, "upstream_auth_error")
        }
        MediaError::InsufficientCredits { .. } => (
            StatusCode::BAD_GATEWAY,
            message,
            "upstream_insufficient_credits",
        ),
        MediaError::RateLimit { .. } => {
            (StatusCode::TOO_MANY_REQUESTS, message, "rate_limit_error")
        }
        MediaError::Api { status, .. } => {
            // An upstream 4xx is permanent for this request; passing the code
            // through stops a client retrying something that can never succeed.
            let status = if (400..500).contains(status) {
                StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_REQUEST)
            } else {
                StatusCode::BAD_GATEWAY
            };
            (status, message, "upstream_error")
        }
        MediaError::RemoteFailure(failure) => {
            // A content-policy rejection is the request's fault; retrying it
            // upstream is pointless, so it reads as 400 rather than 502.
            let status = match failure.category {
                FailureCategory::ContentPolicy | FailureCategory::InvalidInput => {
                    StatusCode::BAD_REQUEST
                }
                FailureCategory::RateLimit => StatusCode::TOO_MANY_REQUESTS,
                _ => StatusCode::BAD_GATEWAY,
            };
            (status, message, "media_generation_failed")
        }
        MediaError::WrongJobHandle { .. } => {
            (StatusCode::INTERNAL_SERVER_ERROR, message, "internal_error")
        }
        _ => (StatusCode::BAD_GATEWAY, message, "upstream_error"),
    }
}

fn media_error_response(error: &anyhow::Error) -> Response<BoxBody> {
    let (status, message, error_type) = classify_media_error(error);
    if status.is_server_error() {
        tracing::error!(error = ?error, "media request failed");
    } else {
        tracing::warn!(reason = %message, "media request rejected");
    }
    let mut response = error_response_typed(status, &message, error_type);
    if let Some(octolib::media::MediaError::RateLimit {
        retry_after_secs: Some(secs),
        ..
    }) = error.downcast_ref::<octolib::media::MediaError>()
    {
        if let Ok(value) = hyper::header::HeaderValue::from_str(&secs.to_string()) {
            response
                .headers_mut()
                .insert(hyper::header::RETRY_AFTER, value);
        }
    }
    response
}

/// The metric label for a finished call. `terminal` is not the same as
/// `succeeded` — a job that came back Failed or Expired must not land in the
/// success bucket and drag its cost along with it.
fn media_status_label(status: octolib::media::OperationStatus) -> &'static str {
    use octolib::media::OperationStatus;
    match status {
        OperationStatus::Succeeded => "ok",
        OperationStatus::Failed => "failed",
        OperationStatus::Expired => "expired",
        OperationStatus::Cancelled => "cancelled",
        // Queued, Running, CancellationRequested — the caller got a 202.
        _ => "accepted",
    }
}

/// Render a media outcome: 200 when terminal, 202 while the job is still live.
fn media_outcome_response(outcome: crate::proxy::media::MediaOutcome) -> Response<BoxBody> {
    let status = if outcome.accepted {
        StatusCode::ACCEPTED
    } else {
        StatusCode::OK
    };
    let body = serde_json::to_value(&outcome.response).unwrap_or_default();
    json_response(status, body)
}

/// Shared body for the four create endpoints: parse, dispatch, record metrics.
async fn handle_media_create(
    req: Request<hyper::body::Incoming>,
    engine: Arc<ProxyEngine>,
    storage: Arc<dyn Storage>,
    build: fn(&[u8]) -> Result<crate::proxy::media::MediaRequest, String>,
) -> Response<BoxBody> {
    let api_key = match media_auth(&req, storage).await {
        Ok(key) => key,
        Err(response) => return response,
    };

    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Failed to read request body: {}", e),
            )
        }
    };

    let media_request = match build(&body_bytes) {
        Ok(request) => request,
        Err(message) => return error_response(StatusCode::BAD_REQUEST, &message),
    };

    let model_label = media_request.model_label().to_string();
    let task_label = media_request.task_label();
    let per_key = engine.config().metrics.per_key;

    tracing::Span::current().record("model", model_label.as_str());

    match engine.process_media(media_request, &api_key).await {
        Ok(outcome) => {
            crate::metrics::record_media(crate::metrics::MediaCall {
                task: task_label,
                model: &model_label,
                provider: &outcome.response.provider,
                status: media_status_label(outcome.response.status),
                duration: outcome.upstream_duration,
                usage: outcome.response.usage.as_ref(),
                api_key_id: Some(api_key.id),
                per_key,
            });
            media_outcome_response(outcome)
        }
        Err(e) => {
            crate::metrics::record_media(crate::metrics::MediaCall {
                task: task_label,
                model: &model_label,
                // The call never reached an upstream, so there is no provider
                // to attribute the failure to.
                provider: "unknown",
                status: "error",
                duration: std::time::Duration::ZERO,
                usage: None,
                api_key_id: Some(api_key.id),
                per_key,
            });
            media_error_response(&e)
        }
    }
}

pub async fn handle_image_generation(
    req: Request<hyper::body::Incoming>,
    engine: Arc<ProxyEngine>,
    storage: Arc<dyn Storage>,
) -> Response<BoxBody> {
    handle_media_create(req, engine, storage, |bytes| {
        serde_json::from_slice(bytes)
            .map(|r| crate::proxy::media::MediaRequest::Image(Box::new(r)))
            .map_err(|e| format!("Invalid request JSON: {e}"))
    })
    .await
}

pub async fn handle_video_generation(
    req: Request<hyper::body::Incoming>,
    engine: Arc<ProxyEngine>,
    storage: Arc<dyn Storage>,
) -> Response<BoxBody> {
    handle_media_create(req, engine, storage, |bytes| {
        serde_json::from_slice(bytes)
            .map(|r| crate::proxy::media::MediaRequest::Video(Box::new(r)))
            .map_err(|e| format!("Invalid request JSON: {e}"))
    })
    .await
}

pub async fn handle_speech(
    req: Request<hyper::body::Incoming>,
    engine: Arc<ProxyEngine>,
    storage: Arc<dyn Storage>,
) -> Response<BoxBody> {
    handle_media_create(req, engine, storage, |bytes| {
        serde_json::from_slice(bytes)
            .map(|r| crate::proxy::media::MediaRequest::Speech(Box::new(r)))
            .map_err(|e| format!("Invalid request JSON: {e}"))
    })
    .await
}

pub async fn handle_transcription(
    req: Request<hyper::body::Incoming>,
    engine: Arc<ProxyEngine>,
    storage: Arc<dyn Storage>,
) -> Response<BoxBody> {
    handle_media_create(req, engine, storage, |bytes| {
        serde_json::from_slice(bytes)
            .map(|r| crate::proxy::media::MediaRequest::Transcription(Box::new(r)))
            .map_err(|e| format!("Invalid request JSON: {e}"))
    })
    .await
}

/// GET /v1/media/{id} — advance and return a job.
pub async fn handle_media_get(
    req: Request<hyper::body::Incoming>,
    engine: Arc<ProxyEngine>,
    storage: Arc<dyn Storage>,
    id: &str,
) -> Response<BoxBody> {
    let api_key = match media_auth(&req, storage).await {
        Ok(key) => key,
        Err(response) => return response,
    };
    match engine.poll_media(id, &api_key).await {
        // Absent OR owned by another key: both read as 404 so record ids
        // cannot be probed across tenants.
        Ok(None) => {
            error_response_typed(StatusCode::NOT_FOUND, "Media record not found", "not_found")
        }
        Ok(Some(outcome)) => media_outcome_response(outcome),
        Err(e) => media_error_response(&e),
    }
}

/// POST /v1/media/{id}/cancel — best-effort remote cancellation.
pub async fn handle_media_cancel(
    req: Request<hyper::body::Incoming>,
    engine: Arc<ProxyEngine>,
    storage: Arc<dyn Storage>,
    id: &str,
) -> Response<BoxBody> {
    let api_key = match media_auth(&req, storage).await {
        Ok(key) => key,
        Err(response) => return response,
    };
    match engine.cancel_media(id, &api_key).await {
        Ok(None) => {
            error_response_typed(StatusCode::NOT_FOUND, "Media record not found", "not_found")
        }
        Ok(Some(outcome)) => json_response(
            StatusCode::OK,
            serde_json::to_value(&outcome.response).unwrap_or_default(),
        ),
        Err(e) => media_error_response(&e),
    }
}

/// GET /v1/media/models — capability and pricing discovery.
pub async fn handle_media_models(
    req: Request<hyper::body::Incoming>,
    engine: Arc<ProxyEngine>,
    storage: Arc<dyn Storage>,
) -> Response<BoxBody> {
    if let Err(response) = media_auth(&req, storage).await {
        return response;
    }
    json_response(StatusCode::OK, engine.media_models())
}
