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
fn check_admin<B>(req: &Request<B>, master_key: &str) -> Result<(), Box<Response<BoxBody>>> {
	let header = req.headers().get("Authorization").and_then(|v| v.to_str().ok());

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
			return error_response(StatusCode::BAD_REQUEST, &format!("Failed to read request body: {}", e));
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
			return error_response(StatusCode::BAD_REQUEST, &format!("Invalid request JSON: {}", e));
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
			return error_response(StatusCode::BAD_REQUEST, &format!("Failed to read request body: {}", e));
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
			return error_response(StatusCode::BAD_REQUEST, &format!("Invalid request JSON: {}", e));
		}
	};

	match storage.update_api_key_owner(key_id, update_req.owner.as_deref(), update_req.owner_concurrency) {
		Ok(true) => json_response(StatusCode::OK, serde_json::json!({"status": "updated"})),
		Ok(false) => error_response(StatusCode::NOT_FOUND, "API key not found"),
		Err(e) => {
			tracing::error!(error = %e, "update key owner failed");
			error_response(StatusCode::INTERNAL_SERVER_ERROR, "Failed to update API key owner")
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

/// GET /v1/admin/status — per-model health, observed from real traffic.
///
/// The gateway sees every call and every failure, so status is measured rather
/// than probed: no synthetic requests, no probe credential, no spend on models
/// nobody asked for. Models with no traffic are simply absent from the list, and
/// the caller reports them as unknown instead of inventing a verdict.
pub async fn handle_status(req: Request<hyper::body::Incoming>, master_key: &str) -> Response<BoxBody> {
	if let Err(resp) = check_admin(&req, master_key) {
		return *resp;
	}
	json_response(StatusCode::OK, crate::health::snapshot())
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
			let items: Vec<serde_json::Value> = keys.iter().map(|k| api_key_response(k, false)).collect();
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
			error_response(StatusCode::INTERNAL_SERVER_ERROR, "Failed to revoke API key")
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
			return error_response(StatusCode::BAD_REQUEST, &format!("Failed to read request body: {}", e));
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
			return error_response(StatusCode::BAD_REQUEST, &format!("Invalid request JSON: {}", e));
		}
	};

	match storage.update_api_key_models(key_id, update_req.allowed_models.as_deref()) {
		Ok(true) => json_response(StatusCode::OK, serde_json::json!({"status": "updated"})),
		Ok(false) => error_response(StatusCode::NOT_FOUND, "API key not found"),
		Err(e) => {
			tracing::error!(error = %e, "update key models failed");
			error_response(StatusCode::INTERNAL_SERVER_ERROR, "Failed to update API key models")
		}
	}
}

/// GET /v1/admin/owners/:owner/auto — the owner's stored purpose→alias map
/// for the virtual `auto` model. `map: null` = no override, the `[auto]`
/// config floor applies.
pub async fn handle_get_owner_auto(
	req: Request<hyper::body::Incoming>,
	storage: Arc<dyn Storage>,
	master_key: &str,
	owner: &str,
) -> Response<BoxBody> {
	if let Err(resp) = check_admin(&req, master_key) {
		return *resp;
	}

	match storage.get_owner_auto_map(owner) {
		Ok(map) => json_response(StatusCode::OK, serde_json::json!({ "owner": owner, "map": map })),
		Err(e) => {
			tracing::error!(error = %e, "get owner auto map failed");
			error_response(StatusCode::INTERNAL_SERVER_ERROR, "Failed to get owner auto map")
		}
	}
}

/// PUT /v1/admin/owners/:owner/auto — replace the owner's auto map.
/// Body `{"map": {"default": "...", "<purpose>": "..."}}`; `map: null` or an
/// empty object clears the override. Values are NOT validated against the live
/// `[models]` table here — the operator's control plane owns that roster and
/// config may change after a write; the resolver skips stale entries at use.
pub async fn handle_set_owner_auto(
	req: Request<hyper::body::Incoming>,
	storage: Arc<dyn Storage>,
	master_key: &str,
	owner: &str,
) -> Response<BoxBody> {
	if let Err(resp) = check_admin(&req, master_key) {
		return *resp;
	}

	let body_bytes = match req.collect().await {
		Ok(collected) => collected.to_bytes(),
		Err(e) => {
			return error_response(StatusCode::BAD_REQUEST, &format!("Failed to read request body: {}", e));
		}
	};

	#[derive(serde::Deserialize)]
	struct SetAutoRequest {
		#[serde(default)]
		map: Option<std::collections::HashMap<String, String>>,
	}

	let set_req: SetAutoRequest = match serde_json::from_slice(&body_bytes) {
		Ok(r) => r,
		Err(e) => {
			return error_response(StatusCode::BAD_REQUEST, &format!("Invalid request JSON: {}", e));
		}
	};

	// Shape checks only: a purpose or target that is empty, or a target of
	// "auto" itself, can never resolve and is certainly a caller bug.
	if let Some(ref map) = set_req.map {
		for (purpose, target) in map {
			if purpose.trim().is_empty() || target.trim().is_empty() {
				return error_response(StatusCode::BAD_REQUEST, "Auto map keys and values must be non-empty");
			}
			if target == crate::config::AUTO_MODEL {
				return error_response(
					StatusCode::BAD_REQUEST,
					"An auto map entry must not point at 'auto' itself",
				);
			}
		}
	}

	match storage.set_owner_auto_map(owner, set_req.map.as_ref()) {
		Ok(()) => json_response(
			StatusCode::OK,
			serde_json::json!({ "status": "updated", "owner": owner }),
		),
		Err(e) => {
			tracing::error!(error = %e, "set owner auto map failed");
			error_response(StatusCode::INTERNAL_SERVER_ERROR, "Failed to set owner auto map")
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
						"evaluations_count": r.evaluations_count,
						"total_input_tokens": r.total_input_tokens,
						"total_output_tokens": r.total_output_tokens,
						"media_count": r.media_count,
						"total_cost": r.total_cost,
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
			error_response(StatusCode::INTERNAL_SERVER_ERROR, "Failed to list completions")
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
			error_response(StatusCode::INTERNAL_SERVER_ERROR, "Failed to list embeddings")
		}
	}
}

/// GET /v1/admin/evaluations?key_id=1,3&since=...&until=...
pub async fn handle_list_evaluations(
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

	match storage.list_evaluations(&filter) {
		Ok(evaluations) => {
			let items: Vec<serde_json::Value> = evaluations
				.into_iter()
				.map(|e| {
					serde_json::json!({
						"id": e.id,
						"api_key_id": e.api_key_id,
						"input_model": e.input_model,
						"resolved_model": e.resolved_model,
						"provider": e.provider,
						"usage": e.usage,
						"request": e.request,
						"answers": e.answers,
						"created_at": e.created_at,
					})
				})
				.collect();
			json_response(StatusCode::OK, serde_json::json!({ "data": items }))
		}
		Err(e) => {
			tracing::error!(error = %e, "list evaluations failed");
			error_response(StatusCode::INTERNAL_SERVER_ERROR, "Failed to list evaluations")
		}
	}
}

/// GET /v1/admin/media?key_id=1,3&since=...&until=...
pub async fn handle_list_media(
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

	match storage.list_media(&filter) {
		Ok(records) => {
			let items: Vec<serde_json::Value> = records
				.into_iter()
				.map(|m| {
					serde_json::json!({
						"id": m.id,
						"api_key_id": m.api_key_id,
						"task": m.task,
						"status": m.status,
						"input_model": m.input_model,
						"resolved_model": m.resolved_model,
						"provider": m.provider,
						"usage": m.usage,
						// The request is already stored with binary payloads
						// redacted, so it is safe to return verbatim.
						"request": m.request,
						"result": m.result,
						"warnings": m.warnings,
						"error": m.error,
						"created_at": m.created_at,
						"completed_at": m.completed_at,
					})
				})
				.collect();
			json_response(StatusCode::OK, serde_json::json!({ "data": items }))
		}
		Err(e) => {
			tracing::error!(error = %e, "list media failed");
			error_response(StatusCode::INTERNAL_SERVER_ERROR, "Failed to list media")
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
		.map(|s| s.split(',').filter_map(|v| v.trim().parse::<i64>().ok()).collect())
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
		limit: params.get("limit").and_then(|s| s.parse::<u32>().ok()).unwrap_or(100),
		offset: params.get("offset").and_then(|s| s.parse::<u32>().ok()).unwrap_or(0),
		since: params.get("since").and_then(|s| s.parse::<u64>().ok()),
		until: params.get("until").and_then(|s| s.parse::<u64>().ok()),
	}
}

pub async fn handle_credentials<B>(
	req: Request<B>,
	storage: Arc<dyn Storage>,
	credentials: Arc<crate::auth::Credentials>,
	master_key: &str,
	owner: &str,
	provider: Option<&str>,
) -> Response<BoxBody>
where
	B: hyper::body::Body<Data = Bytes> + Send,
{
	use crate::storage::credentials::{CredentialChange, CredentialInputError};
	use hyper::Method;
	if let Err(response) = check_admin(&req, master_key) {
		return *response;
	}
	if !credentials.enabled() {
		return json_response(
			StatusCode::SERVICE_UNAVAILABLE,
			serde_json::json!({"error": {
				"type": "credentials_disabled", "message": "Owner credentials are disabled: configure OCTOHUB_CREDENTIALS_KEY"
			}}),
		);
	}
	if req.method() == Method::PUT && provider.is_some_and(|p| crate::auth::byok_provider_label(p).is_none()) {
		return json_response(
			StatusCode::BAD_REQUEST,
			serde_json::json!({"error": {
				"type": "unsupported_provider", "message": "Unsupported BYOK provider"
			}}),
		);
	}
	let method = req.method().clone();
	let change = match (&method, provider) {
		(&Method::GET, None) => None,
		(&Method::DELETE, None) => Some(CredentialChange::Clear),
		(&Method::DELETE, Some(provider)) => Some(CredentialChange::Delete(provider.to_owned())),
		(&Method::PUT, provider) => {
			let bytes = match req.collect().await {
				Ok(body) => body.to_bytes(),
				Err(_) => return error_response(StatusCode::BAD_REQUEST, "Failed to read credential request body"),
			};
			match provider {
				Some(provider) => {
					#[derive(serde::Deserialize)]
					#[serde(deny_unknown_fields)]
					struct Put {
						key: String,
					}
					let input: Put = match serde_json::from_slice(&bytes) {
						Ok(input) => input,
						Err(_) => {
							return error_response(StatusCode::BAD_REQUEST, "Expected JSON object with a key string")
						}
					};
					if input.key.trim().is_empty() {
						return error_response(StatusCode::BAD_REQUEST, "Key must not be empty");
					}
					match credentials.put(owner, provider, &input.key) {
						Ok(change) => Some(change),
						Err(_) => {
							return error_response(
								StatusCode::INTERNAL_SERVER_ERROR,
								"Failed to encrypt owner credential",
							)
						}
					}
				}
				None => {
					#[derive(serde::Deserialize)]
					#[serde(deny_unknown_fields)]
					struct Order {
						order: Vec<String>,
					}
					match serde_json::from_slice::<Order>(&bytes) {
						Ok(input) => Some(CredentialChange::Order(input.order)),
						Err(_) => {
							return error_response(StatusCode::BAD_REQUEST, "Expected JSON object with an order array")
						}
					}
				}
			}
		}
		_ => return error_response(StatusCode::NOT_FOUND, "Not found"),
	};
	let owner = owner.to_owned();
	let owner_db = owner.clone();
	let result = tokio::task::spawn_blocking(move || match change {
		Some(change) => credentials.update(&owner_db, change),
		None => storage.owner_credentials(&owner_db),
	})
	.await;
	match result {
		Ok(Ok(rows)) => {
			if method == Method::PUT {
				if let Some(provider) = provider {
					let row = rows
						.iter()
						.find(|row| row.provider == provider)
						.expect("upsert returns inserted provider");
					return json_response(StatusCode::OK, serde_json::json!(row));
				}
			}
			if method == Method::GET {
				json_response(StatusCode::OK, serde_json::json!({"owner": owner, "credentials": rows}))
			} else {
				json_response(StatusCode::OK, serde_json::json!({"owner": owner, "status": "updated"}))
			}
		}
		Ok(Err(error)) => {
			if let Some(input) = error.downcast_ref::<CredentialInputError>() {
				let status = match input {
					CredentialInputError::NotFound => StatusCode::NOT_FOUND,
					CredentialInputError::InvalidOrder => StatusCode::BAD_REQUEST,
				};
				return error_response(status, &input.to_string());
			}
			// Database driver errors may contain bound values; never log credential operations verbatim.
			tracing::error!(
				class = %crate::storage::error_class(&error),
				"Owner credential storage operation failed"
			);
			error_response(
				StatusCode::INTERNAL_SERVER_ERROR,
				"Owner credential storage operation failed",
			)
		}
		Err(_) => error_response(
			StatusCode::INTERNAL_SERVER_ERROR,
			"Owner credential storage task failed",
		),
	}
}

#[cfg(test)]
mod credential_tests {
	use super::*;
	use crate::auth::Credentials;
	use crate::storage::sqlite::SqliteStorage;
	use base64::Engine;

	async fn call(
		storage: Arc<dyn Storage>,
		credentials: Arc<Credentials>,
		method: &str,
		provider: Option<&str>,
		body: &str,
		auth: &str,
	) -> (StatusCode, serde_json::Value) {
		let req = Request::builder()
			.method(method)
			.header("Authorization", auth)
			.body(Full::new(Bytes::copy_from_slice(body.as_bytes())))
			.unwrap();
		let response = handle_credentials(req, storage, credentials, "master", "account", provider).await;
		let status = response.status();
		let body = response.into_body().collect().await.unwrap().to_bytes();
		(status, serde_json::from_slice(&body).unwrap())
	}

	#[tokio::test]
	async fn credential_admin_contract() {
		let storage: Arc<dyn Storage> = Arc::new(SqliteStorage::new(":memory:").unwrap());
		let disabled = Arc::new(Credentials::new(storage.clone(), None).unwrap());
		let (status, _) = call(storage.clone(), disabled.clone(), "GET", None, "", "Bearer wrong").await;
		assert_eq!(status, StatusCode::UNAUTHORIZED);
		for (method, provider) in [
			("GET", None),
			("PUT", None),
			("DELETE", None),
			("PUT", Some("zai")),
			("DELETE", Some("zai")),
		] {
			let (status, body) = call(
				storage.clone(),
				disabled.clone(),
				method,
				provider,
				"{}",
				"Bearer master",
			)
			.await;
			assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
			assert_eq!(body["error"]["type"], "credentials_disabled");
		}
		let key = base64::engine::general_purpose::STANDARD.encode([3; 32]);
		let credentials = Arc::new(Credentials::new(storage.clone(), Some(&key)).unwrap());
		let (status, body) = call(storage.clone(), credentials.clone(), "GET", None, "", "Bearer master").await;
		assert_eq!(status, StatusCode::OK);
		assert_eq!(body, serde_json::json!({"owner": "account", "credentials": []}));
		let (status, body) = call(
			storage.clone(),
			credentials.clone(),
			"PUT",
			Some("cloudflare"),
			r#"{"key":"private"}"#,
			"Bearer master",
		)
		.await;
		assert_eq!(status, StatusCode::BAD_REQUEST);
		assert_eq!(body["error"]["type"], "unsupported_provider");
		let (status, _) = call(
			storage.clone(),
			credentials.clone(),
			"DELETE",
			Some("cloudflare"),
			"",
			"Bearer master",
		)
		.await;
		assert_eq!(status, StatusCode::NOT_FOUND);
		for body in [
			r#"{"key":""}"#,
			r#"{"key":"  "}"#,
			r#"{"key":42}"#,
			r#"{"key":"private","private":"extra"}"#,
			"not-json-private",
		] {
			let (status, body) = call(
				storage.clone(),
				credentials.clone(),
				"PUT",
				Some("zai"),
				body,
				"Bearer master",
			)
			.await;
			assert_eq!(status, StatusCode::BAD_REQUEST);
			assert!(!body.to_string().contains("private"));
		}
		for (position, provider) in ["zai", "ollama"].into_iter().enumerate() {
			let (status, body) = call(
				storage.clone(),
				credentials.clone(),
				"PUT",
				Some(provider),
				r#"{"key":"owner-private-secret-1234"}"#,
				"Bearer master",
			)
			.await;
			assert_eq!(status, StatusCode::OK);
			assert_eq!(body["provider"], provider);
			assert_eq!(body["position"], position);
			assert_eq!(body["key_hint"], "...1234");
			assert!(body["last_error"].is_null());
			assert!(body["last_used_at"].is_null());
			assert!(body["created_at"].is_i64());
			assert!(body.get("secret").is_none());
			assert!(!body.to_string().contains("owner-private"));
		}
		for body in [
			r#"{"order":["zai"]}"#,
			r#"{"order":["zai","zai"]}"#,
			r#"{"order":["zai","openai"]}"#,
		] {
			let (status, _) = call(storage.clone(), credentials.clone(), "PUT", None, body, "Bearer master").await;
			assert_eq!(status, StatusCode::BAD_REQUEST);
		}
		let (status, _) = call(
			storage.clone(),
			credentials.clone(),
			"PUT",
			None,
			r#"{"order":["ollama","zai"]}"#,
			"Bearer master",
		)
		.await;
		assert_eq!(status, StatusCode::OK);
		let (status, body) = call(
			storage.clone(),
			credentials.clone(),
			"PUT",
			Some("zai"),
			r#"{"key":"replacement-private-5678"}"#,
			"Bearer master",
		)
		.await;
		assert_eq!(status, StatusCode::OK);
		assert_eq!(body["position"], 1);
		assert_eq!(body["key_hint"], "...5678");
		let (_, body) = call(storage.clone(), credentials.clone(), "GET", None, "", "Bearer master").await;
		assert_eq!(body["credentials"][0]["provider"], "ollama");
		let (status, _) = call(
			storage.clone(),
			credentials.clone(),
			"DELETE",
			Some("ollama"),
			"",
			"Bearer master",
		)
		.await;
		assert_eq!(status, StatusCode::OK);
		let (status, _) = call(
			storage.clone(),
			credentials.clone(),
			"DELETE",
			Some("ollama"),
			"",
			"Bearer master",
		)
		.await;
		assert_eq!(status, StatusCode::NOT_FOUND);
		let (_, body) = call(storage.clone(), credentials.clone(), "GET", None, "", "Bearer master").await;
		assert_eq!(body["credentials"][0]["position"], 0);
		for _ in 0..2 {
			let (status, body) = call(
				storage.clone(),
				credentials.clone(),
				"DELETE",
				None,
				"",
				"Bearer master",
			)
			.await;
			assert_eq!(status, StatusCode::OK);
			assert_eq!(body, serde_json::json!({"owner": "account", "status": "updated"}));
		}
		assert!(!storage.has_credentials().unwrap());
	}
}
