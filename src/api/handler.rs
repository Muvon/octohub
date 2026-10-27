use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full, LengthLimitError, Limited};
use hyper::{Request, Response, StatusCode};

use crate::api::types::{
	ChatCompletionRequest, ChatCompletionResponse, CreateCompletionRequest, CreateEmbeddingRequest,
	CreateEvaluationRequest,
};
use crate::auth::{authenticate_admin, authenticate_client, ClientAuth};
use crate::proxy::engine::{
	enforces_response_schema, evaluation_error, upstream_status_code, ModalityNotSupportedError, ProxyEngine,
	ProxyTimeoutError, RateLimitedError, SchemaNotEnforcedError, MODEL_FORBIDDEN_MARKER, OWNER_LIMIT_MARKER,
};
use crate::storage::Storage;
use octolib::evaluation::EvaluationError;

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

/// Largest request body a client endpoint accepts. A stored completion carries
/// the request twice (`input` and `exchange.request`) plus the answer in one
/// INSERT, and MySQL's default `max_allowed_packet` is 64 MiB: a bigger body
/// would pay for the upstream call and then fail to store. Rejected up front
/// with 413 instead.
const MAX_REQUEST_BODY_BYTES: usize = 16 * 1024 * 1024;

async fn read_body<B>(req: Request<B>) -> Result<Bytes, Box<Response<BoxBody>>>
where
	B: hyper::body::Body,
	B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
	match Limited::new(req.into_body(), MAX_REQUEST_BODY_BYTES).collect().await {
		Ok(collected) => Ok(collected.to_bytes()),
		Err(e) if e.is::<LengthLimitError>() => Err(Box::new(error_response(
			StatusCode::PAYLOAD_TOO_LARGE,
			&format!("Request body exceeds {} MiB", MAX_REQUEST_BODY_BYTES / (1024 * 1024)),
		))),
		Err(e) => Err(Box::new(error_response(
			StatusCode::BAD_REQUEST,
			&format!("Failed to read request body: {}", e),
		))),
	}
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
fn engine_error_response(error: &anyhow::Error, status: StatusCode, message: &str) -> Response<BoxBody> {
	let mut response = error_response_typed(status, message, engine_error_type(error));
	let retry_after_secs = match error.downcast_ref::<RateLimitedError>() {
		Some(rate) => Some(rate.retry_after.as_secs().max(1)),
		None => match evaluation_error(error) {
			Some(EvaluationError::RateLimit {
				retry_after_secs: Some(secs),
				..
			}) => Some((*secs).max(1)),
			_ => None,
		},
	};
	if let Some(secs) = retry_after_secs {
		if let Ok(value) = hyper::header::HeaderValue::from_str(&secs.to_string()) {
			response.headers_mut().insert(hyper::header::RETRY_AFTER, value);
		}
	}
	response
}

/// Extract Authorization header value from request
fn auth_header<B>(req: &Request<B>) -> Option<String> {
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

/// OpenRouter-style attribution headers from the client (X-Title, HTTP-Referer,
/// X-OpenRouter-Categories), forwarded upstream so the originating app — not
/// this proxy — gets credited.
/// Read BEFORE the body is consumed.
fn attribution_headers(req: &Request<hyper::body::Incoming>) -> Option<std::collections::HashMap<String, String>> {
	let mut map = std::collections::HashMap::new();
	for name in ["X-Title", "HTTP-Referer", "X-OpenRouter-Categories"] {
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
	let auth_result = tokio::task::spawn_blocking(move || authenticate_client(header.as_deref(), &storage_clone))
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

	let body_bytes = match read_body(req).await {
		Ok(bytes) => bytes,
		Err(response) => return *response,
	};

	// Parse request
	let create_req: CreateCompletionRequest = match serde_json::from_slice(&body_bytes) {
		Ok(r) => r,
		Err(e) => {
			return error_response(StatusCode::BAD_REQUEST, &format!("Invalid request JSON: {}", e));
		}
	};

	tracing::Span::current().record("model", create_req.model.as_str());

	let per_key = engine.config().metrics.per_key;

	// Process
	let model_label = create_req.model.clone();
	match engine.process(create_req, &api_key, purpose, attribution).await {
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
	let auth_result = tokio::task::spawn_blocking(move || authenticate_client(header.as_deref(), &storage_clone))
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

	let body_bytes = match read_body(req).await {
		Ok(bytes) => bytes,
		Err(response) => return *response,
	};

	let create_req: CreateEmbeddingRequest = match serde_json::from_slice(&body_bytes) {
		Ok(r) => r,
		Err(e) => {
			return error_response(StatusCode::BAD_REQUEST, &format!("Invalid request JSON: {}", e));
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

/// Handle POST /v1/evaluations
pub async fn handle_create_evaluation(
	req: Request<hyper::body::Incoming>,
	engine: Arc<ProxyEngine>,
	storage: Arc<dyn Storage>,
) -> Response<BoxBody> {
	let header = auth_header(&req);
	let storage_clone = storage.clone();
	let auth_result = tokio::task::spawn_blocking(move || authenticate_client(header.as_deref(), &storage_clone))
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

	let body_bytes = match read_body(req).await {
		Ok(bytes) => bytes,
		Err(response) => return *response,
	};

	let create_req: CreateEvaluationRequest = match serde_json::from_slice(&body_bytes) {
		Ok(r) => r,
		Err(e) => {
			return error_response(StatusCode::BAD_REQUEST, &format!("Invalid request JSON: {}", e));
		}
	};

	tracing::Span::current().record("model", create_req.model.as_str());

	let per_key = engine.config().metrics.per_key;
	let model_label = create_req.model.clone();
	match engine.process_evaluation(create_req, &api_key).await {
		Ok(outcome) => {
			tracing::Span::current().record("tok_in", outcome.input_tokens);
			tracing::Span::current().record("tok_out", outcome.output_tokens);

			crate::metrics::record_evaluation(
				&model_label,
				&outcome.provider,
				"ok",
				outcome.upstream_duration,
				outcome.input_tokens,
				outcome.output_tokens,
				Some(api_key.id),
				per_key,
			);

			let body = serde_json::to_value(&outcome.response).unwrap_or_default();
			json_response(StatusCode::OK, body)
		}
		Err(e) => {
			let (status, msg) = classify_engine_error(&e);
			if status.is_server_error() {
				tracing::error!(error = ?e, "evaluation failed");
			} else {
				tracing::warn!(reason = %msg, "evaluation request rejected");
			}

			crate::metrics::record_evaluation(
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

/// Classify engine errors into HTTP status codes.
/// Per-key model restriction → 403, bad model/input → 400, otherwise 500.
fn classify_engine_error(error: &anyhow::Error) -> (StatusCode, String) {
	// Top-level message — used only for classification heuristics and for
	// client-error responses (where our own marker is the correct user-facing
	// text). For 500s we surface the full chain below.
	let top = format!("{}", error);
	let full = format!("{:#}", error);

	// octolib's evaluation adapters keep their errors typed: map the variant
	// instead of scanning for an embedded status code.
	if let Some(evaluation) = evaluation_error(error) {
		return match evaluation {
			EvaluationError::InvalidRequest(_)
			| EvaluationError::InvalidModelFormat(_)
			| EvaluationError::UnsupportedProvider(_)
			| EvaluationError::UnsupportedModel { .. } => (StatusCode::BAD_REQUEST, full),
			EvaluationError::RateLimit { .. } => (StatusCode::TOO_MANY_REQUESTS, full),
			EvaluationError::Authentication { .. } => (StatusCode::UNAUTHORIZED, full),
			EvaluationError::Permission { .. } => (StatusCode::FORBIDDEN, full),
			EvaluationError::Api { status, .. } if (400..500).contains(status) => {
				(StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_REQUEST), full)
			}
			_ => (StatusCode::INTERNAL_SERVER_ERROR, full),
		};
	}

	if let Some(byok) = error.downcast_ref::<crate::proxy::engine::ByokFailure>() {
		return (byok.status, byok.to_string());
	}

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

	if error.downcast_ref::<ModalityNotSupportedError>().is_some()
		|| error.downcast_ref::<SchemaNotEnforcedError>().is_some()
	{
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
		|| top.contains("Failed to resolve evaluation model")
		|| top.contains("Failed to resolve media model")
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
	let auth_result = tokio::task::spawn_blocking(move || authenticate_client(header.as_deref(), &storage_clone))
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

	let body_bytes = match read_body(req).await {
		Ok(bytes) => bytes,
		Err(response) => return *response,
	};

	// Parse classic chat request
	let chat_req: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
		Ok(r) => r,
		Err(e) => {
			return error_response(StatusCode::BAD_REQUEST, &format!("Invalid request JSON: {}", e));
		}
	};

	// Streaming is emulated: the upstream call stays buffered (same engine
	// path as non-streaming), and the finished response is re-framed as
	// OpenAI `chat.completion.chunk` SSE on the way out. Clients that require
	// `stream: true` get a fully OpenAI-compatible `text/event-stream`;
	// time-to-first-token is the same as non-streaming (see
	// `chat_completion_stream_body`).
	let stream = chat_req.stream;
	let include_usage = chat_req
		.stream_options
		.as_ref()
		.is_some_and(|options| options.include_usage);

	// Convert to internal representation — same engine path as /v1/completions
	let model_label = chat_req.model.clone();
	let create_req: CreateCompletionRequest = chat_req.into();
	tracing::Span::current().record("model", create_req.model.as_str());

	let per_key = engine.config().metrics.per_key;

	match engine.process(create_req, &api_key, purpose, attribution).await {
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
				let body = Bytes::from(chat_completion_stream_body(&chat_resp, include_usage));
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
fn chat_completion_stream_body(resp: &ChatCompletionResponse, include_usage: bool) -> String {
	let choice = resp.choices.first().expect("chat completion has one choice");
	let id = &resp.id;
	let created = resp.created;
	let model = &resp.model;
	let finish_reason = &choice.finish_reason;

	let base = |delta: serde_json::Value| {
		let mut chunk = serde_json::json!({
			"id": id,
			"object": "chat.completion.chunk",
			"created": created,
			"model": model,
			"choices": [{
				"index": 0,
				"delta": delta,
				"finish_reason": serde_json::Value::Null,
			}],
		});
		if include_usage {
			chunk["usage"] = serde_json::Value::Null;
		}
		chunk
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
	if let Some(tool_calls) = &choice.message.tool_calls {
		for (index, call) in tool_calls.iter().enumerate() {
			out.push_str(&sse_chunk(base(serde_json::json!({
				"tool_calls": [{
					"index": index,
					"id": call.id,
					"type": call.call_type,
					"function": call.function,
				}],
			}))));
		}
	}

	// Terminal chunk: empty delta carrying the finish reason.
	let mut terminal = base(serde_json::json!({}));
	terminal["choices"][0]["finish_reason"] = serde_json::json!(finish_reason);
	out.push_str(&sse_chunk(terminal));

	if include_usage {
		out.push_str(&sse_chunk(serde_json::json!({
			"id": id,
			"object": "chat.completion.chunk",
			"created": created,
			"model": model,
			"choices": [],
			"usage": resp.usage,
		})));
	}

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
	format!("data: {}\n\n", serde_json::to_string(&value).unwrap_or_default())
}

// ── Media ──

/// Authenticate a media request, returning the key or the ready-made 401.
async fn media_auth<B>(
	req: &Request<B>,
	storage: Arc<dyn Storage>,
) -> Result<crate::storage::ApiKey, Box<Response<BoxBody>>> {
	let header = auth_header(req);
	let auth_result = tokio::task::spawn_blocking(move || authenticate_client(header.as_deref(), &storage))
		.await
		.unwrap_or(ClientAuth::Invalid);
	match auth_result {
		ClientAuth::Ok(key) => {
			tracing::Span::current().record("api_key_id", key.id);
			Ok(key)
		}
		ClientAuth::Missing => {
			tracing::warn!(kind = "client", reason = "missing_token", "auth failed");
			Err(Box::new(error_response(StatusCode::UNAUTHORIZED, "Missing API key")))
		}
		ClientAuth::Invalid => {
			tracing::warn!(kind = "client", reason = "invalid_token", "auth failed");
			Err(Box::new(error_response(
				StatusCode::UNAUTHORIZED,
				"Invalid or revoked API key",
			)))
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
		MediaError::InvalidModelFormat(_) | MediaError::InvalidRequest(_) | MediaError::SourceTooLarge { .. } => {
			(StatusCode::BAD_REQUEST, message, DEFAULT_ERROR_TYPE)
		}
		MediaError::UnsupportedParameter { .. } => (StatusCode::BAD_REQUEST, message, "unsupported_parameter"),
		MediaError::UnsupportedProvider(_) | MediaError::UnsupportedTask { .. } => {
			(StatusCode::BAD_REQUEST, message, "media_task_not_supported")
		}
		MediaError::MissingApiKey(_) => (StatusCode::INTERNAL_SERVER_ERROR, message, "configuration_error"),
		MediaError::Authentication { .. } | MediaError::Permission { .. } => {
			(StatusCode::BAD_GATEWAY, message, "upstream_auth_error")
		}
		MediaError::InsufficientCredits { .. } => (StatusCode::BAD_GATEWAY, message, "upstream_insufficient_credits"),
		MediaError::RateLimit { .. } => (StatusCode::TOO_MANY_REQUESTS, message, "rate_limit_error"),
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
				FailureCategory::ContentPolicy | FailureCategory::InvalidInput => StatusCode::BAD_REQUEST,
				FailureCategory::RateLimit => StatusCode::TOO_MANY_REQUESTS,
				_ => StatusCode::BAD_GATEWAY,
			};
			(status, message, "media_generation_failed")
		}
		MediaError::WrongJobHandle { .. } => (StatusCode::INTERNAL_SERVER_ERROR, message, "internal_error"),
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
			response.headers_mut().insert(hyper::header::RETRY_AFTER, value);
		}
	}
	response
}

/// The metric label for a finished call. `terminal` is not the same as
/// `succeeded` — a job that came back Failed or Expired must not land in the
/// success bucket and drag its cost along with it. A job the provider refused
/// for its content is `rejected`: the prompt decided that, not the model.
fn media_outcome_label(response: &crate::api::media_types::MediaResponse) -> &'static str {
	use octolib::media::{GenerationFailure, OperationStatus};
	match response.status {
		OperationStatus::Succeeded => "ok",
		OperationStatus::Failed
			if response
				.error
				.clone()
				.and_then(|error| serde_json::from_value::<GenerationFailure>(error).ok())
				.is_some_and(|failure| refused_for_content(&failure.category)) =>
		{
			"rejected"
		}
		OperationStatus::Failed => "failed",
		OperationStatus::Expired => "expired",
		OperationStatus::Cancelled => "cancelled",
		// Queued, Running, CancellationRequested — the caller got a 202.
		_ => "accepted",
	}
}

/// The metric label for a call that failed outright. Only a failure on the
/// provider's side is evidence about the model, an upstream 4xx included: a
/// request refused for its own shape or content, or refused by octohub before
/// any provider saw it, is `rejected`.
fn media_failure_label(error: &anyhow::Error) -> &'static str {
	use octolib::media::MediaError;
	match error.downcast_ref::<MediaError>() {
		Some(
			MediaError::InvalidModelFormat(_)
			| MediaError::InvalidRequest(_)
			| MediaError::SourceTooLarge { .. }
			| MediaError::UnsupportedParameter { .. }
			| MediaError::UnsupportedTask { .. },
		) => "rejected",
		Some(MediaError::RemoteFailure(failure)) if refused_for_content(&failure.category) => "rejected",
		Some(_) => "error",
		None if classify_media_error(error).0.is_client_error() => "rejected",
		None => "error",
	}
}

fn refused_for_content(category: &octolib::media::FailureCategory) -> bool {
	use octolib::media::FailureCategory;
	matches!(category, FailureCategory::ContentPolicy | FailureCategory::InvalidInput)
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
		Err(response) => return *response,
	};

	let body_bytes = match read_body(req).await {
		Ok(bytes) => bytes,
		Err(response) => return *response,
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
				status: media_outcome_label(&outcome.response),
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
				status: media_failure_label(&e),
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
		Err(response) => return *response,
	};
	match engine.poll_media(id, &api_key).await {
		// Absent OR owned by another key: both read as 404 so record ids
		// cannot be probed across tenants.
		Ok(None) => error_response_typed(StatusCode::NOT_FOUND, "Media record not found", "not_found"),
		Ok(Some(outcome)) => {
			// The poll that finishes an async job is that job's real outcome, so
			// it goes through the same funnel as a submit; a model whose jobs fail
			// then reads as failing on `/v1/admin/status`.
			if outcome.finished_now {
				crate::metrics::record_media(crate::metrics::MediaCall {
					task: crate::proxy::media::task_family(outcome.response.task),
					model: &outcome.alias,
					provider: &outcome.response.provider,
					status: media_outcome_label(&outcome.response),
					// The job's own run time, not this one poll's round trip.
					duration: std::time::Duration::from_secs(
						outcome
							.response
							.completed_at
							.map_or(0, |done| done.saturating_sub(outcome.response.created_at)),
					),
					usage: outcome.response.usage.as_ref(),
					api_key_id: Some(api_key.id),
					per_key: engine.config().metrics.per_key,
				});
			}
			media_outcome_response(outcome)
		}
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
		Err(response) => return *response,
	};
	match engine.cancel_media(id, &api_key).await {
		Ok(None) => error_response_typed(StatusCode::NOT_FOUND, "Media record not found", "not_found"),
		Ok(Some(outcome)) => json_response(
			StatusCode::OK,
			serde_json::to_value(&outcome.response).unwrap_or_default(),
		),
		Err(e) => media_error_response(&e),
	}
}

/// GET /v1/models — OpenAI model list; `supported_parameters` carries
/// "structured_outputs" when a lane enforces JSON schemas (OpenRouter's convention).
pub async fn handle_models<B>(
	req: Request<B>,
	engine: Arc<ProxyEngine>,
	storage: Arc<dyn Storage>,
) -> Response<BoxBody> {
	let config = engine.config();
	let api_key = if authenticate_admin(auth_header(&req).as_deref(), &config.server.api_key) {
		None
	} else {
		match media_auth(&req, storage).await {
			Ok(key) => Some(key),
			Err(response) => return *response,
		}
	};
	let mut ids: Vec<_> = config
		.models
		.keys()
		.filter(|id| api_key.as_ref().is_none_or(|key| key.is_model_allowed(id)))
		.collect();
	ids.sort();

	let mut data = Vec::with_capacity(ids.len());
	for id in ids {
		let candidates = match config.model_candidates(id) {
			Ok(candidates) => candidates,
			Err(error) => {
				tracing::error!(model = id, error = %error, "model discovery failed");
				return error_response(StatusCode::INTERNAL_SERVER_ERROR, "Invalid model configuration");
			}
		};
		let supported_parameters = if candidates
			.iter()
			.any(|(provider, model)| enforces_response_schema(provider, model))
		{
			vec!["structured_outputs"]
		} else {
			vec![]
		};
		data.push(serde_json::json!({
			"id": id,
			"object": "model",
			"created": engine.config_loaded_at,
			"owned_by": "octohub",
			"supported_parameters": supported_parameters,
		}));
	}
	json_response(StatusCode::OK, serde_json::json!({"object": "list", "data": data}))
}

/// GET /v1/media/models — capability and pricing discovery.
pub async fn handle_media_models(
	req: Request<hyper::body::Incoming>,
	engine: Arc<ProxyEngine>,
	storage: Arc<dyn Storage>,
) -> Response<BoxBody> {
	if let Err(response) = media_auth(&req, storage).await {
		return *response;
	}
	json_response(StatusCode::OK, engine.media_models())
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::api::types::{ChatChoice, ChatResponseMessage, ChatUsage};
	use crate::config::Config;
	use crate::proxy::limiter::ProviderLimiter;
	use crate::storage::sqlite::SqliteStorage;
	use anyhow::Context;
	use std::sync::RwLock;

	/// Only a provider-side failure is evidence about the model: a route the
	/// model does not serve or a prompt the provider's policy refuses is not,
	/// while an upstream 4xx is — a model answering 404 to every submit is down.
	#[test]
	fn media_failures_count_against_the_model_only_when_the_provider_failed() {
		use octolib::media::{FailureCategory, GenerationFailure, MediaError, MediaTask};
		let refused = anyhow::anyhow!(MediaError::UnsupportedTask {
			provider: "fal".to_string(),
			model: "fal-ai/elevenlabs/music".to_string(),
			task: MediaTask::TextToImage,
		});
		assert_eq!(media_failure_label(&refused), "rejected");

		let policy = anyhow::anyhow!(MediaError::RemoteFailure(GenerationFailure {
			category: FailureCategory::ContentPolicy,
			message: "refused".to_string(),
			provider_code: None,
			http_status: None,
			request_id: None,
			provider_metadata: serde_json::Value::Null,
		}));
		assert_eq!(media_failure_label(&policy), "rejected");

		let gone = anyhow::anyhow!(MediaError::Api {
			provider: "fal".to_string(),
			status: 404,
			message: "application not found".to_string(),
		});
		assert_eq!(media_failure_label(&gone), "error");
	}

	fn models_engine() -> (Arc<ProxyEngine>, crate::Live<Config>) {
		let config: Config = toml::from_str(
			r#"
            [server]
            api_key = "master-secret"
            [models]
            qwen = ["zai:glm-5.3-flash", "alibaba:qwen3.8-flash"]
            glm = ["zai:glm-5.3-flash", "ollama:glm-5.3-flash"]
            [embedding_models]
            embedding = ["openai:text-embedding-3-small"]
            [media_models]
            image = ["openai:gpt-image-1"]
            [evaluation_models]
            judge = ["typesafe:jev"]
            [auto]
            default = "qwen"
            "#,
		)
		.unwrap();
		let limiter = Arc::new(RwLock::new(Arc::new(ProviderLimiter::from_config(&config))));
		let config = Arc::new(RwLock::new(Arc::new(config)));
		let storage = Arc::new(SqliteStorage::new(":memory:").unwrap());
		(
			Arc::new(ProxyEngine::new(
				storage.clone(),
				config.clone(),
				limiter,
				1_720_000_000,
				Arc::new(crate::auth::Credentials::new(storage, None).unwrap()),
			)),
			config,
		)
	}

	fn models_request(auth: Option<&str>) -> Request<()> {
		let mut request = Request::builder().uri("/v1/models");
		if let Some(auth) = auth {
			request = request.header("Authorization", auth);
		}
		request.body(()).unwrap()
	}

	async fn response_json(response: Response<BoxBody>) -> serde_json::Value {
		serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
	}

	#[tokio::test]
	async fn request_body_over_the_cap_is_413_before_any_work() {
		let body = |len: usize| Request::new(Full::new(Bytes::from(vec![b' '; len])));
		assert_eq!(
			read_body(body(MAX_REQUEST_BODY_BYTES)).await.unwrap().len(),
			MAX_REQUEST_BODY_BYTES
		);
		let response = read_body(body(MAX_REQUEST_BODY_BYTES + 1)).await.unwrap_err();
		assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
		assert_eq!(
			response_json(response).await,
			serde_json::json!({"error": {"message": "Request body exceeds 16 MiB", "type": "invalid_request_error"}})
		);
	}

	#[tokio::test]
	async fn models_list_has_openai_shape_and_shared_schema_capabilities() {
		let (engine, _) = models_engine();
		let key = engine.storage.create_api_key("unrestricted", None, None, None).unwrap();
		let response = handle_models(
			models_request(Some(&format!("Bearer {}", key.key))),
			engine.clone(),
			engine.storage.clone(),
		)
		.await;
		assert_eq!(response.status(), StatusCode::OK);
		assert_eq!(response.headers()["Content-Type"], "application/json");
		assert_eq!(
			response_json(response).await,
			serde_json::json!({
				"object": "list",
				"data": [
					{"id": "glm", "object": "model", "created": 1_720_000_000,
					 "owned_by": "octohub", "supported_parameters": []},
					{"id": "qwen", "object": "model", "created": 1_720_000_000,
					 "owned_by": "octohub", "supported_parameters": ["structured_outputs"]}
				]
			})
		);
	}

	#[tokio::test]
	async fn models_list_filters_by_exact_requested_alias() {
		let (engine, _) = models_engine();
		for (allowed, expected) in [
			(None, vec!["glm", "qwen"]),
			(Some(vec![]), vec![]),
			(Some(vec!["qwen".to_string()]), vec!["qwen"]),
			(
				Some(vec![
					"QWEN".to_string(),
					"alibaba:qwen3.8-flash".to_string(),
					"auto".to_string(),
				]),
				vec![],
			),
		] {
			let key = engine
				.storage
				.create_api_key("scoped", allowed.as_deref(), None, None)
				.unwrap();
			let response = handle_models(
				models_request(Some(&format!("Bearer {}", key.key))),
				engine.clone(),
				engine.storage.clone(),
			)
			.await;
			assert_eq!(response.status(), StatusCode::OK);
			let body = response_json(response).await;
			let ids: Vec<_> = body["data"]
				.as_array()
				.unwrap()
				.iter()
				.map(|model| model["id"].as_str().unwrap())
				.collect();
			assert_eq!(ids, expected);
		}
	}

	#[tokio::test]
	async fn models_list_accepts_master_key_without_a_client_key() {
		let (engine, _) = models_engine();
		let response = handle_models(
			models_request(Some("Bearer master-secret")),
			engine.clone(),
			engine.storage.clone(),
		)
		.await;
		assert_eq!(response.status(), StatusCode::OK);
		let body = response_json(response).await;
		assert_eq!(body["data"].as_array().unwrap().len(), 2);
		assert_eq!(body["data"][0]["id"], "glm");
		assert_eq!(body["data"][1]["id"], "qwen");
	}

	#[tokio::test]
	async fn models_list_preserves_media_auth_failures() {
		let (engine, _) = models_engine();
		let key = engine.storage.create_api_key("revoked", None, None, None).unwrap();
		engine.storage.revoke_api_key(key.id).unwrap();
		let revoked = format!("Bearer {}", key.key);
		for auth in [None, Some("Basic token"), Some("Bearer wrong"), Some(&revoked)] {
			let expected = media_auth(&models_request(auth), engine.storage.clone())
				.await
				.err()
				.unwrap();
			let response = handle_models(models_request(auth), engine.clone(), engine.storage.clone()).await;
			assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
			assert_eq!(response_json(response).await, response_json(*expected).await);
		}
	}

	#[tokio::test]
	async fn models_list_uses_reloaded_faces_with_the_startup_timestamp() {
		let (engine, config) = models_engine();
		let mut reloaded = (*engine.config()).clone();
		reloaded.models.remove("glm");
		*config.write().unwrap() = Arc::new(reloaded);
		for _ in 0..2 {
			let response = handle_models(
				models_request(Some("Bearer master-secret")),
				engine.clone(),
				engine.storage.clone(),
			)
			.await;
			let body = response_json(response).await;
			assert_eq!(body["data"].as_array().unwrap().len(), 1);
			assert_eq!(body["data"][0]["id"], "qwen");
			assert_eq!(body["data"][0]["created"], 1_720_000_000);
		}
	}

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
		assert_eq!(message, "provider 'anthropic' exceeded the 360s operation deadline");
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
		assert_eq!(response.headers().get(hyper::header::RETRY_AFTER).unwrap(), "37");
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
	fn unenforceable_schema_is_a_client_error() {
		// A 5xx would make octolib's retry classifier re-send it forever.
		let error = anyhow::anyhow!(SchemaNotEnforcedError {
			model: "glm-5.3-flash".to_string(),
		});

		let (status, message) = classify_engine_error(&error);

		assert_eq!(status, StatusCode::BAD_REQUEST);
		assert!(message.contains("glm-5.3-flash"));
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
		assert_eq!(response.headers().get(hyper::header::RETRY_AFTER).unwrap(), "1");
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
				byok: None,
				prompt_tokens: 3,
				completion_tokens: 7,
				total_tokens: 10,
			},
		};

		for include_usage in [false, true] {
			let body = chat_completion_stream_body(&resp, include_usage);
			assert!(body.ends_with("data: [DONE]\n\n"));
			let chunks: Vec<serde_json::Value> = body
				.lines()
				.filter_map(|line| line.strip_prefix("data: "))
				.filter(|data| *data != "[DONE]")
				.map(|data| serde_json::from_str(data).unwrap())
				.collect();
			if include_usage {
				let usage_chunk = chunks.last().unwrap();
				assert_eq!(usage_chunk["choices"], serde_json::json!([]));
				assert_eq!(usage_chunk["usage"], serde_json::to_value(&resp.usage).unwrap());
				assert_eq!(usage_chunk["id"], resp.id);
				assert_eq!(usage_chunk["model"], resp.model);
				assert_eq!(usage_chunk["created"], resp.created);
				assert_eq!(usage_chunk["object"], "chat.completion.chunk");
				for chunk in &chunks[..chunks.len() - 1] {
					assert_eq!(chunk.get("usage"), Some(&serde_json::Value::Null));
				}
				assert_eq!(chunks[chunks.len() - 2]["choices"][0]["finish_reason"], "stop");
			} else {
				assert!(chunks.iter().all(|chunk| chunk.get("usage").is_none()));
				assert_eq!(chunks.last().unwrap()["choices"][0]["finish_reason"], "stop");
			}
		}
		let body = chat_completion_stream_body(&resp, false);

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
	fn stream_body_preserves_single_and_multiple_tool_calls() {
		for count in [1, 2] {
			for content in [None, Some("Calling tools now. 世界".to_string())] {
				let calls = (0..count)
					.map(|index| crate::api::types::ChatToolCall {
						id: format!("call_{index}"),
						call_type: "function".to_string(),
						function: crate::api::types::ChatToolCallFunction {
							name: format!("weather_{index}"),
							arguments: format!(r#"{{"city":"Oslo 世界","day":{index}}}"#),
						},
					})
					.collect();
				let resp = ChatCompletionResponse {
					id: "chatcmpl-tools".to_string(),
					object: "chat.completion",
					created: 1_700_000_000,
					model: "gpt-test".to_string(),
					choices: vec![ChatChoice {
						index: 0,
						message: ChatResponseMessage {
							role: "assistant",
							content: content.clone(),
							tool_calls: Some(calls),
						},
						finish_reason: "tool_calls".to_string(),
					}],
					usage: ChatUsage {
						byok: None,
						prompt_tokens: 3,
						completion_tokens: 7,
						total_tokens: 10,
					},
				};
				let body = chat_completion_stream_body(&resp, false);
				assert!(body.ends_with("data: [DONE]\n\n"));
				let chunks: Vec<serde_json::Value> = body
					.lines()
					.filter_map(|line| line.strip_prefix("data: "))
					.filter(|data| *data != "[DONE]")
					.map(|data| serde_json::from_str(data).unwrap())
					.collect();
				assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
				let mut reassembled = vec![serde_json::json!({}); count];
				let mut text = String::new();
				for chunk in &chunks {
					assert_eq!(chunk["id"], resp.id);
					assert_eq!(chunk["model"], resp.model);
					assert_eq!(chunk["object"], "chat.completion.chunk");
					let delta = &chunk["choices"][0]["delta"];
					if let Some(part) = delta["content"].as_str() {
						text.push_str(part);
					}
					if let Some(calls) = delta["tool_calls"].as_array() {
						for call in calls {
							let index = call["index"].as_u64().unwrap() as usize;
							let target = &mut reassembled[index];
							for key in ["id", "type"] {
								if let Some(value) = call.get(key) {
									target[key] = value.clone();
								}
							}
							if let Some(name) = call["function"].get("name") {
								target["function"]["name"] = name.clone();
							}
							let mut arguments =
								target["function"]["arguments"].as_str().unwrap_or_default().to_string();
							arguments.push_str(call["function"]["arguments"].as_str().unwrap());
							target["function"]["arguments"] = arguments.into();
						}
					}
				}
				assert_eq!(text, content.unwrap_or_default());
				assert_eq!(
					serde_json::to_value(reassembled).unwrap(),
					serde_json::to_value(resp.choices[0].message.tool_calls.as_ref().unwrap()).unwrap()
				);
				assert_eq!(chunks.last().unwrap()["choices"][0]["finish_reason"], "tool_calls");
			}
		}
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
