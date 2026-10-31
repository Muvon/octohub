use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use octolib::embedding::{create_embedding_provider_from_parts, InputType};
use octolib::evaluation::{EvaluationError, EvaluationProviderFactory, EvaluationRequest};

use crate::api::types::{CreateEvaluationRequest, CreateEvaluationResponse};
use crate::storage::StoredEvaluation;
use octolib::llm::reference_models::reference_model_id;
use octolib::llm::{
	chat_completion_enforced, ChatCompletionParams, FunctionDefinition, ImageAttachment, ImageData, Message,
	OutputFormat, ProviderFactory, ReasoningEffort, ResponseMode, SourceType, StructuredOutputRequest, ThinkingBlock,
	VideoAttachment, VideoData,
};
use uuid::Uuid;

use crate::api::types::*;
use crate::config::{Config, LaneStrategy};
use crate::proxy::limiter::{OwnerLimiter, ProviderHealth, ProviderLimiter, ProviderRateTracker};
use crate::storage::{ApiKey, Storage, StoredCompletion, StoredEmbedding};

/// Marker added to model-restriction errors so the HTTP layer can map them
/// to `403 Forbidden` instead of the generic 400/500.
pub const MODEL_FORBIDDEN_MARKER: &str = "model_not_allowed_for_key";
/// Marker for owner-budget exhaustion — the handler maps it to HTTP 429.
pub const OWNER_LIMIT_MARKER: &str = "owner_concurrency_exhausted";
const CREDENTIAL_ERROR_MAX_CHARS: usize = 300;

/// Stored-only output item carrying a provider's raw output items (e.g.
/// Anthropic thinking blocks with their signatures). History rebuilt from the
/// chain hands it back to octolib for exact replay; clients never see it.
const PROVIDER_ITEMS_TYPE: &str = "provider_items";

#[derive(Debug)]
pub enum ProxyTimeoutError {
	ProviderQueue {
		provider: String,
		timeout: std::time::Duration,
	},
	Upstream {
		provider: String,
		timeout: std::time::Duration,
	},
}

impl std::fmt::Display for ProxyTimeoutError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::ProviderQueue { provider, timeout } => write!(
				f,
				"timed out after {}s waiting for provider '{}' capacity",
				timeout.as_secs(),
				provider
			),
			Self::Upstream { provider, timeout } => write!(
				f,
				"provider '{}' exceeded the {}s operation deadline",
				provider,
				timeout.as_secs()
			),
		}
	}
}

impl std::error::Error for ProxyTimeoutError {}

/// Every candidate provider for the requested model had an exhausted rate
/// window. The handler maps this to HTTP 429 with a `Retry-After` of
/// `retry_after` — the soonest any candidate's window frees up.
#[derive(Debug)]
pub struct RateLimitedError {
	pub model: String,
	pub retry_after: std::time::Duration,
}

impl std::fmt::Display for RateLimitedError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"rate limit reached for model '{}' on all providers — retry in {}s",
			self.model,
			self.retry_after.as_secs().max(1)
		)
	}
}

impl std::error::Error for RateLimitedError {}

/// A candidate provider does not support a modality required by the request
/// (e.g. video or image attachments). The caller should skip this candidate
/// and try the next one — this is NOT a provider fault and must not trigger
/// health penalties or cooldowns.
#[derive(Debug)]
pub struct ModalityNotSupportedError {
	pub provider: String,
	pub model: String,
	pub modality: String,
}

impl std::fmt::Display for ModalityNotSupportedError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"Provider '{}' model '{}' does not support {} attachments",
			self.provider, self.model, self.modality
		)
	}
}

impl std::error::Error for ModalityNotSupportedError {}

/// The attachment kinds a conversation carries — what any lane serving it
/// must be able to read. Checked per candidate before the upstream call, and
/// per alias when `auto` picks one.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Modalities {
	pub images: bool,
	pub videos: bool,
}

impl Modalities {
	pub(crate) fn of(messages: &[Message]) -> Self {
		Self {
			images: messages
				.iter()
				.any(|m| m.images.as_ref().is_some_and(|i| !i.is_empty())),
			videos: messages
				.iter()
				.any(|m| m.videos.as_ref().is_some_and(|v| !v.is_empty())),
		}
	}

	/// Whether one `provider:model` lane can read everything this carries.
	pub(crate) fn served_by(self, provider: &str, model: &str) -> bool {
		if !self.images && !self.videos {
			return true;
		}
		ProviderFactory::create_provider(provider)
			.is_ok_and(|p| (!self.images || p.supports_vision(model)) && (!self.videos || p.supports_video(model)))
	}

	/// Whether at least one lane of a `[models]` alias can serve this — the
	/// request then routes to it, exactly as an explicit model would.
	pub(crate) fn served_by_alias(self, config: &Config, alias: &str) -> bool {
		config
			.ordered_model_candidates(alias)
			.is_ok_and(|lanes| lanes.iter().any(|(provider, model)| self.served_by(provider, model)))
	}

	/// What of this NO lane of `alias` can read — the attachments that must be
	/// left out for that alias to serve the request at all.
	pub(crate) fn unread_by_alias(self, config: &Config, alias: &str) -> Self {
		let images = Self {
			images: true,
			videos: false,
		};
		let videos = Self {
			images: false,
			videos: true,
		};
		Self {
			images: self.images && !images.served_by_alias(config, alias),
			videos: self.videos && !videos.served_by_alias(config, alias),
		}
	}

	/// Suffix for an unresolvable-`auto` error, naming what ruled entries out.
	fn describe(self) -> &'static str {
		match (self.images, self.videos) {
			(true, true) => " that reads images and video",
			(true, false) => " that reads images",
			(false, true) => " that reads video",
			(false, false) => "",
		}
	}
}

/// Where the turn the model is answering now begins: just after the last
/// assistant message. Everything from there on (the user's message, or the tool
/// results of the model's own last calls) is what it is being asked about; what
/// came before is history it has already answered.
pub(crate) fn current_turn_start(messages: &[Message]) -> usize {
	messages
		.iter()
		.rposition(|m| m.role == "assistant")
		.map_or(0, |i| i + 1)
}

/// Leave out the attachment kinds in `omit`, saying so in the message text so
/// the model knows something was there. The same thing octomind does before it
/// sends history to a model it knows cannot read images.
pub(crate) fn omit_attachments(messages: &mut [Message], omit: Modalities) {
	for m in messages {
		let mut left_out = Vec::new();
		if omit.images {
			if let Some(images) = m.images.take().filter(|i| !i.is_empty()) {
				left_out.push(format!("{} image(s)", images.len()));
			}
		}
		if omit.videos {
			if let Some(videos) = m.videos.take().filter(|v| !v.is_empty()) {
				left_out.push(format!("{} video(s)", videos.len()));
			}
		}
		if !left_out.is_empty() {
			let note = format!(
				"[{} omitted: the model answering now cannot read them]",
				left_out.join(" and ")
			);
			m.content = if m.content.is_empty() {
				note
			} else {
				format!("{}\n\n{}", m.content, note)
			};
		}
	}
}

/// A JSON-schema request for a model none of whose lanes guarantees the
/// schema's shape. A request fault, not a provider fault: every lane was
/// filtered out before any upstream call.
#[derive(Debug)]
pub struct SchemaNotEnforcedError {
	pub model: String,
}

impl std::fmt::Display for SchemaNotEnforcedError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "no provider for model '{}' enforces a JSON schema", self.model)
	}
}

impl std::error::Error for SchemaNotEnforcedError {}

/// Extract the upstream HTTP status embedded in an octolib provider error.
/// octolib formats provider failures as "... API error <code> <message>", e.g.
/// "ollama API error 400 Bad Request: ...". Returns the first such code, if any.
pub(crate) fn upstream_status_code(msg: &str) -> Option<u16> {
	const MARKER: &str = "API error ";
	let start = msg.find(MARKER)? + MARKER.len();
	let digits: String = msg[start..].chars().take_while(|c| c.is_ascii_digit()).collect();
	digits.parse().ok()
}

/// Whether a dispatch failure is the PROVIDER's fault (worth failing over
/// and counting toward its cooldown) rather than the request's. Only a 400 /
/// invalid-request rejects the request itself — every lane would refuse it the
/// same way. Every other upstream status says THIS lane cannot serve right now:
/// 401/403 (key revoked, billing past due), 404 (model gone from the host),
/// 429, 5xx, plus upstream deadlines and errors with no embedded status
/// (connect/transport/unavailable provider). Our own queue timeout is a local
/// capacity failure, never the provider's.
pub(crate) fn is_provider_fault(error: &anyhow::Error) -> bool {
	match error.downcast_ref::<ProxyTimeoutError>() {
		Some(ProxyTimeoutError::Upstream { .. }) => true,
		Some(ProxyTimeoutError::ProviderQueue { .. }) => false,
		None => {
			if let Some(evaluation) = evaluation_error(error) {
				return !matches!(
					evaluation,
					EvaluationError::InvalidRequest(_) | EvaluationError::Api { status: 400, .. }
				);
			}
			upstream_status_code(&format!("{:#}", error)) != Some(400)
		}
	}
}

/// The typed octolib evaluation error inside an anyhow chain, if any. The
/// evaluation adapters keep their errors typed instead of embedding "API
/// error <code>" text, so classification reads the variant.
pub(crate) fn evaluation_error(error: &anyhow::Error) -> Option<&EvaluationError> {
	error.chain().find_map(|cause| cause.downcast_ref())
}

/// Result of `process_embedding`. Carries telemetry alongside the response so
/// the HTTP handler can label `octohub_embedding_*` metrics with the resolved
/// provider name and the approximate input token count — neither of which is
/// part of the user-facing response struct.
pub struct EmbeddingOutcome {
	pub response: CreateEmbeddingResponse,
	pub provider: String,
	pub upstream_duration: std::time::Duration,
	pub input_tokens: u64,
}

/// Result of `process_evaluation`, mirroring [`EmbeddingOutcome`].
pub struct EvaluationOutcome {
	pub response: CreateEvaluationResponse,
	pub provider: String,
	pub upstream_duration: std::time::Duration,
	pub input_tokens: u64,
	pub output_tokens: u64,
}

/// Core proxy engine that processes requests through octolib providers.
///
/// `config` and `limiter` are live handles swapped on SIGHUP reload (see
/// `main`), so each request reads a fresh snapshot instead of a value frozen
/// at startup.
pub struct ProxyEngine {
	pub(crate) credentials: Arc<crate::auth::Credentials>,
	pub(crate) storage: Arc<dyn Storage>,
	/// Startup config load time stays fixed even when SIGHUP replaces the model map.
	pub(crate) config_loaded_at: u64,
	config: crate::Live<Config>,
	limiter: crate::Live<ProviderLimiter>,
	/// Per-owner in-flight budgets (keys sharing `ApiKey::owner`). Plain Arc,
	/// not `Live`: state comes from the key rows, not config, and must survive
	/// SIGHUP so in-flight accounting is never reset by a reload.
	owner_limiter: Arc<OwnerLimiter>,
	/// Per-provider request/token windows. Plain Arc for the same reason as
	/// `owner_limiter`: a SIGHUP reload must update the LIMITS (read from the
	/// live config at admission) without zeroing the day's counters.
	pub(crate) rate_tracker: Arc<ProviderRateTracker>,
	/// Per-provider failure streaks + cooldowns (SIGHUP-safe, like the
	/// trackers above). Only active when `provider_error_cooldown_secs` > 0.
	pub(crate) provider_health: Arc<ProviderHealth>,
}

impl ProxyEngine {
	pub fn new(
		storage: Arc<dyn Storage>,
		config: crate::Live<Config>,
		limiter: crate::Live<ProviderLimiter>,
		config_loaded_at: u64,
		credentials: Arc<crate::auth::Credentials>,
	) -> Self {
		Self {
			credentials,
			storage,
			config_loaded_at,
			config,
			limiter,
			owner_limiter: Arc::new(OwnerLimiter::new()),
			rate_tracker: Arc::new(ProviderRateTracker::new()),
			provider_health: Arc::new(ProviderHealth::new()),
		}
	}

	/// Take a slot in the key's shared owner budget, or fail with the 429
	/// marker after `[server].owner_queue_timeout_secs`. `None` = key is ungrouped/unlimited —
	/// hold the returned permit (if any) for the whole upstream call.
	pub(crate) async fn acquire_owner_slot(
		&self,
		api_key: &ApiKey,
	) -> Result<Option<tokio::sync::OwnedSemaphorePermit>> {
		let (Some(owner), Some(capacity)) = (&api_key.owner, api_key.owner_concurrency) else {
			return Ok(None);
		};
		if capacity == 0 {
			return Ok(None); // 0 = unlimited, same contract as absent
		}
		match self
			.owner_limiter
			.acquire(
				owner,
				capacity,
				std::time::Duration::from_secs(self.config().server.owner_queue_timeout_secs),
			)
			.await
		{
			Ok(permit) => Ok(Some(permit)),
			Err(()) => Err(anyhow!(
				"{}: owner concurrency limit ({}) exhausted — retry shortly",
				OWNER_LIMIT_MARKER,
				capacity
			)),
		}
	}

	/// Current config snapshot. The reload writer only swaps an `Arc` and never
	/// panics while holding the lock, so it can't be poisoned — unwrap is safe.
	pub fn config(&self) -> Arc<Config> {
		self.config.read().unwrap().clone()
	}

	/// Current limiter snapshot. Same poison-free reasoning as `config`.
	pub(crate) fn limiter(&self) -> Arc<ProviderLimiter> {
		self.limiter.read().unwrap().clone()
	}

	/// Process a create completion request, attributing usage to the given API key.
	/// Returns the response plus the upstream provider call duration — the latter
	/// drives `octohub_completion_duration_seconds` so the histogram reflects
	/// provider latency only, excluding our own auth/parse/storage overhead.
	///
	/// `purpose` is the request's `X-Model-Purpose` header, meaningful only when
	/// the model is the virtual `auto` — it steers which alias `auto` resolves to.
	///
	/// `extra_headers` are client attribution headers (X-Title / HTTP-Referer)
	/// forwarded verbatim to the upstream call — octohub is a proxy, so the
	/// originating app's identity goes upstream, not ours.
	pub async fn process(
		&self,
		req: CreateCompletionRequest,
		api_key: &ApiKey,
		purpose: Option<String>,
		extra_headers: Option<std::collections::HashMap<String, String>>,
	) -> Result<(CreateCompletionResponse, std::time::Duration)> {
		// Snapshot config + limiter once so this request keeps a consistent view
		// even if SIGHUP swaps them mid-flight.
		let config = self.config();

		// The as-sent model must be allowed — for `auto` that means the key was
		// explicitly granted "auto". The RESOLVED alias is checked again below,
		// so `auto` can never smuggle in a model the key's roster bans.
		ensure_model_allowed(api_key, &req.model)?;

		// Owner budget slot — held for the WHOLE request (queue + upstream +
		// storage), completions and embeddings drain the same budget.

		let _owner_slot = self.acquire_owner_slot(api_key).await?;

		let limiter = self.limiter();

		// 1. Build conversation history from chain
		let mut messages: Vec<Message> = Vec::new();

		// Live system message from req.instructions has priority over chain-stored
		// instructions because the request carries the current cache_control marker.
		let mut system_msg: Option<Message> = req.instructions.as_ref().map(content_to_system);

		// Tracks whether the supplied previous_completion_id actually resolved.
		// Unknown IDs are accepted (stateless-provider migration path) but must
		// not be persisted — that would create dangling chains forever.
		let mut resolved_prev_id: Option<String> = None;
		// session_id inherited from the tail of the resolved chain.
		let mut inherited_session_id: Option<String> = None;
		// Provider that served the previous turn — preferred at resolution so
		// multi-turn sessions keep hitting the same provider-side prompt cache.
		let mut sticky_provider: Option<String> = None;

		if let Some(ref prev_cmpl_id) = req.previous_completion_id {
			// Unknown IDs are tolerated — the client may be migrating from a stateless
			// provider (Anthropic, etc.) where they pass the full history inline in
			// `input`. Hard-failing would break that workflow. Mirrors OpenAI's
			// guidance: "retry with full input context and previous_response_id null."
			// An unknown id walks to an empty chain. A chain that fails to LOAD is not
			// unknown: its history is stored, and answering from the new input alone
			// would cut the model's memory without a word (prod 2026-10-28, MySQL's
			// recursion cap) — that fails the request.
			let chain_start = std::time::Instant::now();
			let storage = self.storage.clone();
			let prev_id = prev_cmpl_id.clone();
			let chain = tokio::task::spawn_blocking(move || storage.walk_chain(&prev_id))
				.await?
				.with_context(|| format!("loading the chain of previous_completion_id {prev_cmpl_id}"))?;
			let chain_ms = chain_start.elapsed().as_millis() as u64;
			tracing::Span::current().record("chain_ms", chain_ms);
			if chain.is_empty() {
				tracing::warn!(
					previous_completion_id = %prev_cmpl_id,
					"Unknown previous_completion_id — using inline input only",
				);
			} else {
				resolved_prev_id = Some(prev_cmpl_id.clone());
			}

			// Inherit session_id and the serving provider from the most
			// recent (last) chain entry.
			inherited_session_id = chain.last().map(|c| c.session_id.clone());
			sticky_provider = chain.last().map(|c| c.provider.clone());

			for stored in &chain {
				// Fall back to chain-stored instructions only if the request didn't
				// supply any (cache markers are not preserved on this path).
				if system_msg.is_none() {
					if let Some(ref instr) = stored.instructions {
						system_msg = Some(Message::system(instr));
					}
				}

				// Reconstruct input messages
				self.reconstruct_input(&stored.input, &mut messages);

				// Reconstruct output as assistant message
				self.reconstruct_output(&stored.output, &mut messages);
			}
		}

		// 2. Prepend resolved system message (if any) at the head of history
		if let Some(sys) = system_msg {
			messages.insert(0, sys);
		}

		// 3. Append new input
		match &req.input {
			Input::Text(text) => {
				messages.push(Message::user(text));
			}
			Input::Items(items) => {
				push_items(items, &mut messages);
			}
		}

		// Virtual `auto`: pick the real alias from the owner's stored map, then
		// the [auto] config floor (see proxy::auto for the chain). `req.model`
		// deliberately stays "auto" — it is what the client sent, and it lands
		// in `input_model` for observability; `routed_model` feeds routing.
		//
		// Resolved only now that the conversation is assembled, because `auto` is
		// OUR choice of model and must be able to serve the request it is given
		// (prod 2026-10-22: one screenshot made every octomind call on `auto` a
		// modality 400, which failed the turn). The current turn decides: an entry
		// that cannot read the images or video the model is being asked about
		// right now is skipped, as a stale one is. Images further back in history
		// do not move the pick — one old screenshot must not downgrade a whole
		// conversation — they are omitted for a model that cannot read them,
		// exactly what octomind does itself for a known text-only model.
		let turn_start = current_turn_start(&messages);
		let routed_model = if config.is_auto(&req.model) {
			let owner_map = match api_key.owner.clone() {
				Some(owner) => {
					let storage = self.storage.clone();
					tokio::task::spawn_blocking(move || storage.get_owner_auto_map(&owner)).await??
				}
				None => None,
			};
			let turn = Modalities::of(&messages[turn_start..]);
			let resolved = super::auto::resolve(purpose.as_deref(), owner_map.as_ref(), &config.auto, |m| {
				config.models.contains_key(m) && turn.served_by_alias(&config, m)
			})
			.with_context(|| {
				format!(
					"Failed to resolve model 'auto' (purpose '{}'): no usable [auto] mapping{}",
					purpose.as_deref().unwrap_or(""),
					turn.describe()
				)
			})?;
			ensure_model_allowed(api_key, &resolved)?;
			let earlier = Modalities::of(&messages[..turn_start]);
			let omitted = earlier.unread_by_alias(&config, &resolved);
			omit_attachments(&mut messages[..turn_start], omitted);
			tracing::Span::current().record("model", resolved.as_str());
			tracing::info!(
				purpose = purpose.as_deref().unwrap_or(""),
				resolved = %resolved,
				turn_images = turn.images,
				turn_videos = turn.videos,
				omitted_images = omitted.images,
				omitted_videos = omitted.videos,
				"auto model resolved"
			);
			resolved
		} else {
			req.model.clone()
		};
		let needs = Modalities::of(&messages);

		// 4. Resolve provider and model via config. The chain's previous
		//    provider is preferred (provider-side prompt caches are per
		//    provider); candidates whose rate windows are exhausted are
		//    skipped; all exhausted → 429 with Retry-After (`pick_admitted`).
		//    With `failover_on_error` — or on an `ordered` alias, whose list
		//    names its fallbacks — a provider-side failure moves to the next
		//    candidate instead of surfacing to the client.
		let owner_credentials = match api_key.owner.clone() {
			Some(owner) => {
				let credentials = self.credentials.clone();
				tokio::task::spawn_blocking(move || credentials.load(&owner)).await??
			}
			None => Arc::default(),
		};
		let byok_candidates = if !owner_credentials.is_empty() {
			byok_candidates(config.ordered_model_candidates(&routed_model)?, &owner_credentials)
		} else {
			Vec::new()
		};
		let byok = !byok_candidates.is_empty();
		let strategy = config.lane_strategy(&routed_model);
		let mut candidates = if byok {
			byok_candidates
		} else {
			let mut candidates = match strategy {
				LaneStrategy::Rotate => config.model_candidates(&routed_model),
				LaneStrategy::Ordered => config.ordered_model_candidates(&routed_model),
			}
			.with_context(|| format!("Failed to resolve model '{}'", routed_model))?;
			let continuing = sticky_provider
				.as_deref()
				.filter(|sticky| candidates.iter().any(|(p, _)| p.eq_ignore_ascii_case(sticky)));
			match continuing {
				Some(sticky) => prefer_provider(&mut candidates, sticky),
				// A new conversation on an ordered alias takes the first lane with
				// a free slot instead of queueing behind a busy primary. A
				// continuing one keeps its lane and waits for it, since its prompt
				// cache lives there.
				None if strategy == LaneStrategy::Ordered => deprioritize_saturated(&mut candidates, &limiter),
				None => {}
			}
			candidates
		};
		// A JSON-schema request only goes to lanes that guarantee the shape;
		// the rest would approximate it with a forced tool call that can miss
		// required fields. No such lane is an error, never an unenforced answer.
		if req
			.text
			.as_ref()
			.is_some_and(|t| matches!(t.format, TextFormat::JsonSchema { .. }))
		{
			retain_schema_enforcing_candidates(&mut candidates);
			if candidates.is_empty() {
				return Err(anyhow!(SchemaNotEnforcedError {
					model: routed_model.clone(),
				}));
			}
		}

		let cooldown = std::time::Duration::from_secs(config.server.provider_error_cooldown_secs);
		let (provider_name, resolved_model, provider, mut provider_response, upstream_duration) = loop {
			let (provider_name, resolved_model) = if byok {
				candidates.first().context("BYOK candidates exhausted")?.clone()
			} else {
				pick_admitted(
					&self.rate_tracker,
					&self.provider_health,
					&config,
					candidates.clone(),
					&routed_model,
				)?
			};
			let credential = if byok {
				Some(
					owner_credentials
						.iter()
						.find(|c| c.record.provider.eq_ignore_ascii_case(&provider_name))
						.context("Missing credential for BYOK lane")?,
				)
			} else {
				None
			};

			// Record provider in request span
			tracing::Span::current().record("provider", provider_name.as_str());

			let attempt = async {
				// 5. Get provider instance
				let provider = ProviderFactory::create_provider(&provider_name)
					.with_context(|| format!("Provider '{}' not available", provider_name))?;

				// Modality compatibility: fail fast so the outer loop can skip
				// this candidate without counting it as a provider fault.
				if needs.videos && !provider.supports_video(&resolved_model) {
					return Err(anyhow!(ModalityNotSupportedError {
						provider: provider_name.clone(),
						model: resolved_model.clone(),
						modality: "video".to_string(),
					}));
				}

				if needs.images && !provider.supports_vision(&resolved_model) {
					return Err(anyhow!(ModalityNotSupportedError {
						provider: provider_name.clone(),
						model: resolved_model.clone(),
						modality: "image".to_string(),
					}));
				}

				// 6. Build ChatCompletionParams
				let tools = req.tools.as_ref().map(|tools| {
					tools
						.iter()
						.map(|t| FunctionDefinition {
							name: t.name.clone(),
							description: t.description.clone().unwrap_or_default(),
							parameters: t.parameters.clone().unwrap_or(serde_json::json!({})),
							// Pass-through: octohub is a proxy. Whatever cache marker
							// the client attached to this tool goes upstream unchanged.
							cache_control: t.cache_control.clone(),
						})
						.collect::<Vec<_>>()
				});
				// top_k is not part of the Responses-API wire shape (octolib client
				// never sends it), so we leave it at a neutral default; upstream
				// providers that don't honor it ignore it harmlessly.
				let mut params = ChatCompletionParams::new(
					&messages,
					&resolved_model,
					req.temperature,
					req.top_p,
					50,
					req.max_output_tokens,
				);

				if let Some(credential) = credential {
					params = params.with_api_key(credential.key.clone());
					// Each credential gets one attempt; the owner lane loop owns failover.
					params.max_retries = 0;
				}
				if let Some(tools) = tools {
					params.tools = Some(tools);
				}

				// Forwarded client attribution; octolib applies these last with
				// override semantics, beating its own env-based defaults.
				params.extra_headers = extra_headers.clone();

				// Silence limit for streaming lanes: a hung lane fails over on its
				// own clock instead of burning the whole upstream deadline.
				if config.server.upstream_stream_timeout_secs > 0 {
					params = params.with_stream_timeout(Some(std::time::Duration::from_secs(
						config.server.upstream_stream_timeout_secs,
					)));
				}

				// Reasoning effort: parse client string into octolib enum and pass
				// through. Unknown values silently fall back to provider default —
				// we never want a malformed effort hint to fail the whole request.
				if let Some(ref eff) = req.reasoning_effort {
					params.reasoning_effort = match eff.to_lowercase().as_str() {
						"low" => Some(ReasoningEffort::Low),
						"medium" => Some(ReasoningEffort::Medium),
						"high" => Some(ReasoningEffort::High),
						"xhigh" => Some(ReasoningEffort::XHigh),
						"max" => Some(ReasoningEffort::Max),
						_ => None,
					};
				}

				// Structured output: map the Responses-API `text.format` shape onto
				// octolib's StructuredOutputRequest so the upstream provider receives
				// the schema (or json_object request) the client asked for. Without
				// this, JSON-mode and JSON-Schema requests through the proxy silently
				// fall back to free-form text.
				if let Some(ref text) = req.text {
					params.response_format = Some(match &text.format {
						TextFormat::JsonObject => StructuredOutputRequest {
							format: OutputFormat::Json,
							mode: ResponseMode::Auto,
							schema: None,
						},
						TextFormat::JsonSchema { schema, strict } => StructuredOutputRequest {
							format: OutputFormat::JsonSchema,
							mode: if *strict {
								ResponseMode::Strict
							} else {
								ResponseMode::Auto
							},
							schema: Some(schema.clone()),
						},
					});
				}

				// 7. Call provider — hold a concurrency permit (if configured) for the
				//    full duration of the upstream call. Dropping `_permit` after the
				//    `.await` resolves wakes the next queued request.
				let queue_start = std::time::Instant::now();
				let queue_timeout = std::time::Duration::from_secs(config.server.provider_queue_timeout_secs);
				let _permit = tokio::time::timeout(queue_timeout, limiter.acquire(&provider_name))
					.await
					.map_err(|_| {
						anyhow!(ProxyTimeoutError::ProviderQueue {
							provider: provider_name.clone(),
							timeout: queue_timeout,
						})
					})?;
				let queue_wait = queue_start.elapsed();
				if queue_wait.as_millis() > 0 {
					tracing::Span::current().record("queued_ms", queue_wait.as_millis() as u64);
				}
				if queue_wait.as_millis() > 100 {
					tracing::info!(provider = %provider_name, waited_ms = queue_wait.as_millis() as u64, "queued for provider permit");
				}
				crate::metrics::record_queue_wait(&provider_name, queue_wait);

				let upstream_start = std::time::Instant::now();
				let upstream_timeout = std::time::Duration::from_secs(config.server.upstream_timeout_secs);
				// Route through the schema-enforcement fallback: a transparent
				// passthrough unless the client requested a JSON schema that the
				// resolved provider can't natively guarantee (see
				// `octolib::llm::chat_completion_enforced`).
				let provider_response =
					tokio::time::timeout(upstream_timeout, chat_completion_enforced(provider.as_ref(), params))
						.await
						.map_err(|_| {
							anyhow!(ProxyTimeoutError::Upstream {
								provider: provider.name().to_string(),
								timeout: upstream_timeout,
							})
						})?
						.with_context(|| format!("Provider '{}' chat_completion failed", provider.name()))?;
				let upstream_duration = upstream_start.elapsed();
				tracing::Span::current().record("upstream_ms", upstream_duration.as_millis() as u64);
				anyhow::Ok((provider, provider_response, upstream_duration))
			}
			.await;

			match attempt {
				Ok((provider, response, duration)) => {
					if let Some(credential) = credential {
						self.record_credential(api_key, credential, None).await?;
					} else {
						self.provider_health.record_success(&provider_name);
					}
					break (provider_name, resolved_model, provider, response, duration);
				}
				Err(err) => {
					// Modality mismatch is a candidate filter, not a provider fault.
					if let Some(modality_err) = err.downcast_ref::<ModalityNotSupportedError>() {
						retain_same_model_failover_candidates(&mut candidates, modality_err);
						if candidates.is_empty() {
							return Err(err);
						}
						tracing::info!(
							provider = %modality_err.provider,
							model = %modality_err.model,
							modality = %modality_err.modality,
							"provider does not support modality — skipping to next candidate"
						);
						continue;
					}

					if let Some(credential) = credential {
						if matches!(
							err.downcast_ref::<ProxyTimeoutError>(),
							Some(ProxyTimeoutError::ProviderQueue { .. })
						) {
							return Err(err);
						}
						let failure = ByokFailure::from_upstream(&err, &provider_name, &owner_credentials)?;
						self.record_credential(
							api_key,
							credential,
							Some(crate::storage::credentials::CredentialError {
								at: crate::storage::now_unix() as i64,
								status: failure.status.as_u16(),
								message: failure.message.chars().take(CREDENTIAL_ERROR_MAX_CHARS).collect(),
							}),
						)
						.await?;
						candidates.remove(0);
						if !failure.retryable() || candidates.is_empty() {
							return Err(failure.into());
						}
						crate::metrics::record_failover(&provider_name);
						tracing::warn!(provider = %provider_name, status = failure.status.as_u16(), "Owner credential failed; trying next BYOK lane");
						continue;
					}

					let provider_fault = is_provider_fault(&err);
					if provider_fault {
						self.provider_health.record_failure(&provider_name, cooldown);
					}
					// The failed provider is out for THIS request either way —
					// never re-picked within one failover loop.
					candidates.retain(|(p, _)| !p.eq_ignore_ascii_case(&provider_name));
					// An ordered alias's list names its fallbacks, so its lanes fail
					// over even where the deployment leaves failover off.
					let failover = config.server.failover_on_error || strategy == LaneStrategy::Ordered;
					if !(failover && provider_fault) || candidates.is_empty() {
						return Err(err);
					}
					crate::metrics::record_failover(&provider_name);
					tracing::warn!(
						provider = %provider_name,
						error = %err,
						"provider failed — failing over to next candidate"
					);
				}
			}
		};

		if byok {
			provider_response.content = redact_credential_text(&provider_response.content, &owner_credentials);
			if let Some(thinking) = &mut provider_response.thinking {
				thinking.content = redact_credential_text(&thinking.content, &owner_credentials);
			}
			if let Some(tool_calls) = &mut provider_response.tool_calls {
				for call in tool_calls {
					call.id = redact_credential_text(&call.id, &owner_credentials);
					call.name = redact_credential_text(&call.name, &owner_credentials);
					redact_credential_json(&mut call.arguments, &owner_credentials);
				}
			}
			redact_credential_json(&mut provider_response.exchange.request, &owner_credentials);
			redact_credential_json(&mut provider_response.exchange.response, &owner_credentials);
			if let Some(value) = &mut provider_response.structured_output {
				redact_credential_json(value, &owner_credentials);
			}
			if let Some(reason) = &mut provider_response.finish_reason {
				*reason = redact_credential_text(reason, &owner_credentials);
			}
		}

		// 8. Build our response
		let completion_id = format!("cmpl_{}", Uuid::new_v4().simple());
		let now = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.unwrap_or_default()
			.as_secs();

		let mut output = Vec::new();

		// Emit reasoning FIRST so it's replayed before assistant message/tool_calls.
		// DeepSeek requires `reasoning_content` to accompany assistant turns that
		// produced tool_calls; without it the API returns 400.
		if let Some(ref thinking) = provider_response.thinking {
			if !thinking.content.is_empty() {
				output.push(OutputItem::Reasoning {
					id: format!("rsn_{}", Uuid::new_v4().simple()),
					content: vec![ContentPart::OutputText {
						text: thinking.content.clone(),
					}],
				});
			}
		}

		// Add function calls if present
		if let Some(ref tool_calls) = provider_response.tool_calls {
			for tc in tool_calls {
				output.push(OutputItem::FunctionCall {
					id: format!("fc_{}", Uuid::new_v4().simple()),
					call_id: tc.id.clone(),
					name: tc.name.clone(),
					arguments: tc.arguments.to_string(),
				});
			}
		}

		// Add message content if present
		if !provider_response.content.is_empty() {
			output.push(OutputItem::Message {
				id: format!("msg_{}", Uuid::new_v4().simple()),
				role: "assistant".to_string(),
				content: vec![ContentPart::OutputText {
					text: provider_response.content.clone(),
				}],
			});
		}

		let exchange_usage = &provider_response.exchange.usage;
		let usage = Usage {
			byok: byok.then_some(true),
			input_tokens: exchange_usage.as_ref().map(|u| u.input_tokens).unwrap_or(0),
			output_tokens: exchange_usage.as_ref().map(|u| u.output_tokens).unwrap_or(0),
			cache_read_tokens: exchange_usage.as_ref().and_then(|u| {
				if u.cache_read_tokens > 0 {
					Some(u.cache_read_tokens)
				} else {
					None
				}
			}),
			cache_write_tokens: exchange_usage.as_ref().and_then(|u| {
				if u.cache_write_tokens > 0 {
					Some(u.cache_write_tokens)
				} else {
					None
				}
			}),
			total_tokens: exchange_usage.as_ref().map(|u| u.total_tokens).unwrap_or(0),
			reasoning_tokens: exchange_usage.as_ref().and_then(|u| {
				if u.reasoning_tokens > 0 {
					Some(u.reasoning_tokens)
				} else {
					None
				}
			}),
			cost: exchange_usage.as_ref().and_then(|u| u.cost),
			request_time_ms: exchange_usage.as_ref().and_then(|u| u.request_time_ms),
		};

		// Feed actual usage back into the provider's rate windows — the
		// request itself was counted at admission, tokens only now that the
		// provider reported them. `max` covers providers whose total_tokens
		// is absent (0) but that still report the split counts.
		if !byok {
			self.rate_tracker.record_tokens(
				&provider_name,
				usage.total_tokens.max(usage.input_tokens + usage.output_tokens),
			);
		}

		let response = CreateCompletionResponse {
			id: completion_id.clone(),
			object: "completion",
			model: resolved_model.clone(),
			provider: provider.name().to_string(),
			output: output.clone(),
			usage: usage.clone(),
			// Mirror everything the upstream provider returned: clients must
			// not have to re-parse text to recover schema-validated JSON or
			// know why generation stopped.
			structured_output: provider_response.structured_output.clone(),
			finish_reason: provider_response.finish_reason.clone(),
			created_at: now,
		};

		// 9. Store for observability. Use resolved_prev_id (None if chain didn't
		// resolve) so we never persist a link to an unknown ID.
		// session_id comes from the chain tail — no extra DB query needed.
		let session_id = inherited_session_id.unwrap_or_else(|| format!("sess_{}", Uuid::new_v4().simple()));

		let mut stored_output = serde_json::to_value(&output).unwrap_or(serde_json::Value::Null);
		if let Some(provider_items) = stored_provider_items(&provider_response.exchange.response) {
			stored_output
				.as_array_mut()
				.expect("output serializes as an array")
				.push(provider_items);
		}

		let mut stored = StoredCompletion {
			id: completion_id,
			api_key_id: api_key.id,
			session_id,
			previous_completion_id: resolved_prev_id,
			input_model: req.model.clone(),
			resolved_model,
			provider: provider.name().to_string(),
			input: serde_json::to_value(&req.input).unwrap_or(serde_json::Value::Null),
			output: stored_output,
			instructions: req.instructions.as_ref().map(|i| i.text()),
			exchange: serde_json::json!({
				"request": provider_response.exchange.request,
				"response": provider_response.exchange.response,
			}),
			usage: serde_json::to_value(&usage).unwrap_or(serde_json::Value::Null),
			created_at: now,
		};
		if byok {
			redact_credential_json(&mut stored.input, &owner_credentials);
			if let Some(instructions) = &mut stored.instructions {
				*instructions = redact_credential_text(instructions, &owner_credentials);
			}
		}

		let store_start = std::time::Instant::now();
		let storage = self.storage.clone();
		tokio::task::spawn_blocking(move || storage.store_completion(&stored)).await??;
		tracing::Span::current().record("store_ms", store_start.elapsed().as_millis() as u64);

		Ok((response, upstream_duration))
	}

	async fn record_credential(
		&self,
		api_key: &ApiKey,
		credential: &crate::auth::OwnerCredential,
		error: Option<crate::storage::credentials::CredentialError>,
	) -> Result<()> {
		let owner = api_key.owner.clone().context("BYOK request has no owner")?;
		let credentials = self.credentials.clone();
		let change = crate::storage::credentials::CredentialChange::Record {
			provider: credential.record.provider.clone(),
			secret: credential.record.secret.clone(),
			error,
		};
		tokio::task::spawn_blocking(move || credentials.update(&owner, change))
			.await
			.context("Credential telemetry task failed")?
			.map_err(|_| anyhow!("Credential telemetry storage failed"))?;
		Ok(())
	}

	/// Process an embedding request, attributing usage to the given API key.
	/// Returns the response, the upstream call duration (for the histogram),
	/// the resolved provider name (so callers can label metrics correctly),
	/// and the approximate input token count.
	pub async fn process_embedding(&self, req: CreateEmbeddingRequest, api_key: &ApiKey) -> Result<EmbeddingOutcome> {
		ensure_model_allowed(api_key, &req.model)?;
		// Same shared owner budget as completions (see `process`).
		let _owner_slot = self.acquire_owner_slot(api_key).await?;

		// Snapshot config + limiter once (see `process`).
		let config = self.config();
		let limiter = self.limiter();

		let start = std::time::Instant::now();

		// 1. Resolve provider and model — same rate-window admission and
		//    failover policy as completions (both drain the same provider
		//    windows and share the health tracker).
		let mut candidates = config
			.embedding_model_candidates(&req.model)
			.with_context(|| format!("Failed to resolve embedding model '{}'", req.model))?;

		let texts: Vec<String> = match &req.input {
			EmbeddingInput::Single(s) => vec![s.clone()],
			EmbeddingInput::Batch(v) => v.clone(),
		};

		let cooldown = std::time::Duration::from_secs(config.server.provider_error_cooldown_secs);
		let (provider_name, resolved_model, embeddings, usage_calc, upstream_duration) = loop {
			let (provider_name, resolved_model) = pick_admitted(
				&self.rate_tracker,
				&self.provider_health,
				&config,
				candidates.clone(),
				&req.model,
			)?;

			// Record provider in request span
			tracing::Span::current().record("provider", provider_name.as_str());

			let attempt = async {
				// 2. Parse provider type and create provider
				let provider_model = format!("{}:{}", provider_name, resolved_model);
				let (provider_type, model_name) = octolib::embedding::parse_provider_model(&provider_model)?;
				let provider = create_embedding_provider_from_parts(&provider_type, &model_name).await?;

				// 3. Generate embeddings — gated by the same per-provider concurrency
				// limit as completions. Permit released as soon as the call returns.
				let queue_start = std::time::Instant::now();
				let queue_timeout = std::time::Duration::from_secs(config.server.provider_queue_timeout_secs);
				let _permit = tokio::time::timeout(queue_timeout, limiter.acquire(&provider_name))
					.await
					.map_err(|_| {
						anyhow!(ProxyTimeoutError::ProviderQueue {
							provider: provider_name.clone(),
							timeout: queue_timeout,
						})
					})?;
				let queue_wait = queue_start.elapsed();
				if queue_wait.as_millis() > 0 {
					tracing::Span::current().record("queued_ms", queue_wait.as_millis() as u64);
				}
				if queue_wait.as_millis() > 100 {
					tracing::info!(provider = %provider_name, waited_ms = queue_wait.as_millis() as u64, "queued for provider permit");
				}
				crate::metrics::record_queue_wait(&provider_name, queue_wait);

				let upstream_start = std::time::Instant::now();
				let upstream_timeout = std::time::Duration::from_secs(config.server.upstream_timeout_secs);
				let (embeddings, usage_calc) = tokio::time::timeout(
					upstream_timeout,
					provider.generate_embeddings_batch(texts.clone(), InputType::None),
				)
				.await
				.map_err(|_| {
					anyhow!(ProxyTimeoutError::Upstream {
						provider: provider_name.clone(),
						timeout: upstream_timeout,
					})
				})?
				.with_context(|| {
					format!(
						"Embedding provider '{}' failed for model '{}'",
						provider_name, resolved_model
					)
				})?;
				let upstream_duration = upstream_start.elapsed();
				tracing::Span::current().record("upstream_ms", upstream_duration.as_millis() as u64);
				anyhow::Ok((embeddings, usage_calc, upstream_duration))
			}
			.await;

			match attempt {
				Ok((embeddings, usage_calc, duration)) => {
					self.provider_health.record_success(&provider_name);
					break (provider_name, resolved_model, embeddings, usage_calc, duration);
				}
				Err(err) => {
					let provider_fault = is_provider_fault(&err);
					if provider_fault {
						self.provider_health.record_failure(&provider_name, cooldown);
					}
					candidates.retain(|(p, _)| !p.eq_ignore_ascii_case(&provider_name));
					if !(config.server.failover_on_error && provider_fault) || candidates.is_empty() {
						return Err(err);
					}
					crate::metrics::record_failover(&provider_name);
					tracing::warn!(
						provider = %provider_name,
						error = %err,
						"embedding provider failed — failing over to next candidate"
					);
				}
			}
		};

		// Embeddings drain the same provider token windows as completions.
		self.rate_tracker.record_tokens(&provider_name, usage_calc.input_tokens);

		let elapsed_ms = start.elapsed().as_millis() as u64;

		// 4. Build response (simple: just the embedding(s))
		let response = match &req.input {
			EmbeddingInput::Single(_) => {
				CreateEmbeddingResponse::Single(embeddings.into_iter().next().unwrap_or_default())
			}
			EmbeddingInput::Batch(_) => CreateEmbeddingResponse::Batch(embeddings),
		};

		// Usage comes straight from octolib's embedding provider: the real
		// provider-reported token count (or a tiktoken estimate for providers that
		// report none) plus the reference cost (`None` for local/unpriced models).
		// Cost rides in the `usage` JSON, exactly as completions carry it
		// (`usage.cost`), so the billing layer reads it the same way for both.
		let usage = serde_json::json!({
			"input_tokens": usage_calc.input_tokens,
			"total_tokens": usage_calc.input_tokens,
			"cost": usage_calc.cost,
			"request_time_ms": elapsed_ms,
		});

		// 5. Store for observability
		let embedding_id = format!("embd_{}", Uuid::new_v4().simple());
		let now = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.unwrap_or_default()
			.as_secs();

		let stored = StoredEmbedding {
			id: embedding_id,
			api_key_id: api_key.id,
			input_model: req.model.clone(),
			resolved_model,
			provider: provider_name.clone(),
			input: serde_json::to_value(&req.input).unwrap_or(serde_json::Value::Null),
			usage: usage.clone(),
			created_at: now,
		};
		let store_start = std::time::Instant::now();
		let storage = self.storage.clone();
		tokio::task::spawn_blocking(move || storage.store_embedding(&stored)).await??;
		tracing::Span::current().record("store_ms", store_start.elapsed().as_millis() as u64);

		Ok(EmbeddingOutcome {
			response,
			provider: provider_name,
			upstream_duration,
			input_tokens: usage_calc.input_tokens,
		})
	}

	/// Process an evaluation request: same admission, failover and
	/// accounting as embeddings, against octolib's evaluation adapters.
	pub async fn process_evaluation(
		&self,
		req: CreateEvaluationRequest,
		api_key: &ApiKey,
	) -> Result<EvaluationOutcome> {
		ensure_model_allowed(api_key, &req.model)?;
		if req.questions.is_empty() {
			anyhow::bail!("Invalid request: at least one question is required");
		}
		let _owner_slot = self.acquire_owner_slot(api_key).await?;

		let config = self.config();
		let limiter = self.limiter();
		let start = std::time::Instant::now();

		// Virtual `auto`: the owner's `evaluation` entry, then the [auto]
		// floor's. As for completions, `req.model` stays "auto" and lands in
		// `input_model`; the resolved alias must pass the key's roster too.
		let routed_model = if config.is_auto(&req.model) {
			let owner_map = match api_key.owner.clone() {
				Some(owner) => {
					let storage = self.storage.clone();
					tokio::task::spawn_blocking(move || storage.get_owner_auto_map(&owner)).await??
				}
				None => None,
			};
			let resolved = super::auto::resolve_evaluation(owner_map.as_ref(), &config.auto, |m| {
				config.evaluation_models.contains_key(m)
			})
			.context("Failed to resolve model 'auto' for an evaluation: no usable [auto].evaluation mapping")?;
			ensure_model_allowed(api_key, &resolved)?;
			tracing::info!(resolved = %resolved, "auto evaluation model resolved");
			resolved
		} else {
			req.model.clone()
		};

		let mut candidates = config
			.evaluation_model_candidates(&routed_model)
			.with_context(|| format!("Failed to resolve evaluation model '{}'", routed_model))?;

		let cooldown = std::time::Duration::from_secs(config.server.provider_error_cooldown_secs);
		let (provider_name, resolved_model, result, upstream_duration) = loop {
			let (provider_name, resolved_model) = pick_admitted(
				&self.rate_tracker,
				&self.provider_health,
				&config,
				candidates.clone(),
				&routed_model,
			)?;
			tracing::Span::current().record("provider", provider_name.as_str());

			let attempt = async {
				let provider = EvaluationProviderFactory::create_provider(&provider_name)?;
				if !provider.supports_model(&resolved_model) {
					return Err(anyhow!(EvaluationError::UnsupportedModel {
						provider: provider_name.clone(),
						model: resolved_model.clone(),
					}));
				}
				let mut request = EvaluationRequest::new(req.state.clone()).with_model(resolved_model.clone());
				request.questions = req.questions.clone();

				let queue_start = std::time::Instant::now();
				let queue_timeout = std::time::Duration::from_secs(config.server.provider_queue_timeout_secs);
				let _permit = tokio::time::timeout(queue_timeout, limiter.acquire(&provider_name))
					.await
					.map_err(|_| {
						anyhow!(ProxyTimeoutError::ProviderQueue {
							provider: provider_name.clone(),
							timeout: queue_timeout,
						})
					})?;
				let queue_wait = queue_start.elapsed();
				if queue_wait.as_millis() > 0 {
					tracing::Span::current().record("queued_ms", queue_wait.as_millis() as u64);
				}
				crate::metrics::record_queue_wait(&provider_name, queue_wait);

				let upstream_start = std::time::Instant::now();
				let upstream_timeout = std::time::Duration::from_secs(config.server.upstream_timeout_secs);
				let result = tokio::time::timeout(upstream_timeout, provider.evaluate(request))
					.await
					.map_err(|_| {
						anyhow!(ProxyTimeoutError::Upstream {
							provider: provider_name.clone(),
							timeout: upstream_timeout,
						})
					})?
					.with_context(|| {
						format!(
							"Evaluation provider '{}' failed for model '{}'",
							provider_name, resolved_model
						)
					})?;
				let upstream_duration = upstream_start.elapsed();
				tracing::Span::current().record("upstream_ms", upstream_duration.as_millis() as u64);
				anyhow::Ok((result, upstream_duration))
			}
			.await;

			match attempt {
				Ok((result, duration)) => {
					self.provider_health.record_success(&provider_name);
					break (provider_name, resolved_model, result, duration);
				}
				Err(err) => {
					let provider_fault = is_provider_fault(&err);
					if provider_fault {
						self.provider_health.record_failure(&provider_name, cooldown);
					}
					candidates.retain(|(p, _)| !p.eq_ignore_ascii_case(&provider_name));
					if !(config.server.failover_on_error && provider_fault) || candidates.is_empty() {
						return Err(err);
					}
					crate::metrics::record_failover(&provider_name);
					tracing::warn!(
						provider = %provider_name,
						error = %err,
						"evaluation provider failed — failing over to next candidate"
					);
				}
			}
		};

		// Evaluations drain the same provider token windows as completions.
		self.rate_tracker
			.record_tokens(&provider_name, result.usage.input_tokens + result.usage.output_tokens);

		let elapsed_ms = start.elapsed().as_millis() as u64;
		let usage = serde_json::json!({
			"input_tokens": result.usage.input_tokens,
			"output_tokens": result.usage.output_tokens,
			"cost": result.usage.cost,
			"request_time_ms": elapsed_ms,
		});

		let evaluation_id = format!("eval_{}", Uuid::new_v4().simple());
		let now = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.unwrap_or_default()
			.as_secs();
		let stored = StoredEvaluation {
			id: evaluation_id.clone(),
			api_key_id: api_key.id,
			input_model: req.model.clone(),
			resolved_model,
			provider: provider_name.clone(),
			request: serde_json::json!({"state": req.state, "questions": req.questions}),
			answers: serde_json::to_value(&result.answers).unwrap_or(serde_json::Value::Null),
			usage,
			created_at: now,
		};
		let store_start = std::time::Instant::now();
		let storage = self.storage.clone();
		tokio::task::spawn_blocking(move || storage.store_evaluation(&stored)).await??;
		tracing::Span::current().record("store_ms", store_start.elapsed().as_millis() as u64);

		Ok(EvaluationOutcome {
			response: CreateEvaluationResponse {
				id: evaluation_id,
				model: result.model,
				answers: result.answers,
				usage: result.usage.clone(),
			},
			provider: provider_name,
			upstream_duration,
			input_tokens: result.usage.input_tokens,
			output_tokens: result.usage.output_tokens,
		})
	}

	/// Reconstruct input messages from stored JSON
	fn reconstruct_input(&self, input: &serde_json::Value, messages: &mut Vec<Message>) {
		// Input can be a string or array of items
		if let Some(text) = input.as_str() {
			messages.push(Message::user(text));
		} else if let Some(items) = input.as_array() {
			push_stored_items(items, messages);
		}
		// Handle {"Text": "..."} from serde serialization of Input::Text
		else if let Some(text) = input.get("Text").and_then(|v| v.as_str()) {
			messages.push(Message::user(text));
		}
		// Handle {"Items": [...]} from serde serialization of Input::Items
		else if let Some(items) = input.get("Items").and_then(|v| v.as_array()) {
			push_stored_items(items, messages);
		}
	}

	/// Reconstruct output as assistant message(s) from stored JSON
	///
	/// Collects all function calls into a single assistant message with
	/// tool_calls in the unified GenericToolCall format that octolib expects.
	fn reconstruct_output(&self, output: &serde_json::Value, messages: &mut Vec<Message>) {
		if let Some(items) = output.as_array() {
			let mut text_parts: Vec<String> = Vec::new();
			let mut tool_calls: Vec<serde_json::Value> = Vec::new();
			let mut reasoning_text: Option<String> = None;
			let mut provider_items: Option<serde_json::Value> = None;

			for item in items {
				if item["type"] == PROVIDER_ITEMS_TYPE {
					provider_items = Some(serde_json::json!({"provider": item["provider"], "items": item["items"]}));
					continue;
				}
				if let Ok(output_item) = serde_json::from_value::<OutputItem>(item.clone()) {
					match output_item {
						OutputItem::Message { content, .. } => {
							let text: String = content
								.iter()
								.map(|c| match c {
									ContentPart::OutputText { text } => text.as_str(),
								})
								.collect::<Vec<_>>()
								.join("\n");
							if !text.is_empty() {
								text_parts.push(text);
							}
						}
						OutputItem::FunctionCall {
							call_id,
							name,
							arguments,
							..
						} => {
							// Parse arguments string back to JSON value
							let args_value: serde_json::Value =
								serde_json::from_str(&arguments).unwrap_or(serde_json::json!({}));
							tool_calls.push(serde_json::json!({
								"id": call_id,
								"name": name,
								"arguments": args_value,
							}));
						}
						OutputItem::Reasoning { content, .. } => {
							let text: String = content
								.iter()
								.map(|c| match c {
									ContentPart::OutputText { text } => text.as_str(),
								})
								.collect::<Vec<_>>()
								.join("\n");
							if !text.is_empty() {
								reasoning_text = Some(text);
							}
						}
					}
				}
			}

			// Emit a single assistant message with text + tool_calls + thinking.
			// Thinking must accompany tool_calls for providers like DeepSeek that
			// require `reasoning_content` continuity in subsequent turns.
			if !tool_calls.is_empty() || !text_parts.is_empty() || reasoning_text.is_some() {
				let content = text_parts.join("\n");
				let mut msg = Message::assistant(&content);
				if !tool_calls.is_empty() {
					msg.tool_calls = Some(serde_json::Value::Array(tool_calls));
				}
				if let Some(rt) = reasoning_text {
					msg.thinking = Some(ThinkingBlock::new(&rt));
				}
				msg.provider_items = provider_items;
				messages.push(msg);
			}
		}
	}
}

/// The stored-only output item for a provider's captured output items, if the
/// provider captured any. BYOK redaction has already run on `exchange_response`.
fn stored_provider_items(exchange_response: &serde_json::Value) -> Option<serde_json::Value> {
	let captured = exchange_response
		.get("output_items")
		.filter(|captured| !captured.is_null())?;
	Some(serde_json::json!({
		"type": PROVIDER_ITEMS_TYPE,
		"provider": captured["provider"],
		"items": captured["items"],
	}))
}

/// Build an octolib `Message` from an input message, extracting plain text
/// from either `ContentValue::Text` or `ContentValue::Parts`, attaching any
/// `input_image` / `input_video` parts as image/video attachments, and
/// propagating any `cache_control` markers as `cached`/`cache_ttl` flags.
fn content_to_message(role: &str, content: &ContentValue) -> Message {
	let text = content.text();
	let mut msg = match role {
		"user" => Message::user(&text),
		"system" => Message::system(&text),
		"assistant" => Message::assistant(&text),
		_ => Message::user(&text),
	};
	if content.is_cached() {
		msg.cached = true;
		// Proxy semantics: forward the full marker, including TTL. Without
		// this the upstream provider receives `cached` but falls back to its
		// default TTL (5m for Anthropic), so client-requested longer TTLs
		// (e.g. "1h") were silently downgraded.
		msg.cache_ttl = content.cache_ttl();
	}

	attach_media(&mut msg, content);
	msg
}

/// Attach the `input_image` / `input_video` parts of `content` to `msg`.
fn attach_media(msg: &mut Message, content: &ContentValue) {
	let images: Vec<ImageAttachment> = content
		.image_urls()
		.into_iter()
		.filter_map(|u| parse_image_url(&u))
		.collect();
	if !images.is_empty() {
		msg.images = Some(images);
	}

	let videos: Vec<VideoAttachment> = content
		.video_urls()
		.into_iter()
		.filter_map(|u| parse_video_url(&u))
		.collect();
	if !videos.is_empty() {
		msg.videos = Some(videos);
	}
}

/// Parse a Responses-API image URL into an `ImageAttachment`. Supports
/// `data:<media-type>;base64,<data>` data URIs (decoded to `ImageData::Base64`)
/// and plain `http(s)://` URLs (passed through as `ImageData::Url`). Returns
/// `None` for malformed inputs so individual bad parts don't drop the request.
fn parse_image_url(url: &str) -> Option<ImageAttachment> {
	if let Some((media_type, data)) = parse_data_uri(url) {
		Some(ImageAttachment {
			data: ImageData::Base64(data),
			media_type,
			source_type: SourceType::Url,
			dimensions: None,
			size_bytes: None,
		})
	} else if url.starts_with("http://") || url.starts_with("https://") {
		Some(ImageAttachment {
			data: ImageData::Url(url.to_string()),
			media_type: "image/*".to_string(),
			source_type: SourceType::Url,
			dimensions: None,
			size_bytes: None,
		})
	} else {
		tracing::warn!(url = %url, kind = "image", "skipping unrecognized media url");
		None
	}
}

/// Parse a Responses-API video URL into a `VideoAttachment`. Same data-URI /
/// http(s) handling as `parse_image_url`.
fn parse_video_url(url: &str) -> Option<VideoAttachment> {
	if let Some((media_type, data)) = parse_data_uri(url) {
		Some(VideoAttachment {
			data: VideoData::Base64(data),
			media_type,
			source_type: SourceType::Url,
			dimensions: None,
			size_bytes: None,
			duration_secs: None,
		})
	} else if url.starts_with("http://") || url.starts_with("https://") {
		Some(VideoAttachment {
			data: VideoData::Url(url.to_string()),
			media_type: "video/*".to_string(),
			source_type: SourceType::Url,
			dimensions: None,
			size_bytes: None,
			duration_secs: None,
		})
	} else {
		tracing::warn!(url = %url, kind = "video", "skipping unrecognized media url");
		None
	}
}

/// Split a `data:<media-type>;base64,<data>` URI into `(media_type, data)`.
/// Only the `;base64` form is recognized — that's what octolib emits and what
/// upstream providers accept after re-encoding.
fn parse_data_uri(url: &str) -> Option<(String, String)> {
	let rest = url.strip_prefix("data:")?;
	let (header, data) = rest.split_once(',')?;
	let media_type = header.strip_suffix(";base64")?;
	if media_type.is_empty() {
		return None;
	}
	Some((media_type.to_string(), data.to_string()))
}

/// Build the system message from request `instructions`, preserving cache markers.
fn content_to_system(content: &ContentValue) -> Message {
	let mut msg = Message::system(&content.text());
	if content.is_cached() {
		msg.cached = true;
		msg.cache_ttl = content.cache_ttl();
	}
	msg
}

/// Convert input items into octolib `Message`s and append to the list.
///
/// FunctionCall and Reasoning items are coalesced onto the preceding assistant
/// Message (matching how `reconstruct_output` builds them on the way out), so
/// providers receive a single assistant turn carrying text + tool_calls +
/// thinking together — required by DeepSeek's `reasoning_content` rule and by
/// octolib's tool_calls-attached-to-message convention.
fn push_items(items: &[InputItem], messages: &mut Vec<Message>) {
	for item in items {
		match item {
			InputItem::Message { role, content } => {
				messages.push(content_to_message(role, content));
			}
			InputItem::FunctionCallOutput {
				call_id,
				output,
				cache_control,
			} => {
				let mut msg = Message::tool(&output.text(), call_id, "function");
				// A tool result can carry images (browser screenshot); octolib places
				// them per provider, so they must ride on the tool message.
				attach_media(&mut msg, output);
				// Forward the client's cache marker so the upstream provider caches
				// the prefix ending at this tool result — the common agent-loop tail
				// breakpoint. Mirrors `content_to_message` for Message items.
				if let Some(cc) = cache_control {
					msg.cached = true;
					msg.cache_ttl = cc.get("ttl").and_then(|v| v.as_str()).map(String::from);
				}
				messages.push(msg);
			}
			InputItem::FunctionCall {
				call_id,
				name,
				arguments,
			} => {
				let args_value: serde_json::Value = serde_json::from_str(arguments).unwrap_or(serde_json::json!({}));
				let tool_call = serde_json::json!({
					"id": call_id,
					"name": name,
					"arguments": args_value,
				});
				attach_tool_call_to_assistant(messages, tool_call);
			}
			InputItem::Reasoning { content } => {
				let text: String = content
					.iter()
					.map(|c| match c {
						ContentPart::OutputText { text } => text.as_str(),
					})
					.collect::<Vec<_>>()
					.join("\n");
				if !text.is_empty() {
					attach_thinking_to_assistant(messages, text);
				}
			}
		}
	}
}

/// Deserialize stored JSON input items, skipping any that fail to parse
/// (forward-compat with new types), then forward to `push_items`.
///
/// Strips `cache_control` markers from reconstructed messages. `cache_control`
/// is a per-request hint to the upstream provider — "check/extend prompt cache
/// here on THIS request". It is NOT a persistent property of the conversation.
/// Replaying historical markers from every prior completion accumulates them
/// across the chain and blows past Anthropic's hard limit of 4 cache_control
/// blocks per request. Only the live request's `instructions` + `input`
/// (handled in `process` directly, not via this function) should carry markers.
fn push_stored_items(items: &[serde_json::Value], messages: &mut Vec<Message>) {
	let typed: Vec<InputItem> = items
		.iter()
		.filter_map(|v| serde_json::from_value(v.clone()).ok())
		.collect();
	let before_len = messages.len();
	push_items(&typed, messages);
	for msg in &mut messages[before_len..] {
		msg.cached = false;
		msg.cache_ttl = None;
	}
}

/// Append `tool_call` to the trailing assistant Message's `tool_calls` array,
/// or create an empty assistant Message carrying it if none precedes.
fn attach_tool_call_to_assistant(messages: &mut Vec<Message>, tool_call: serde_json::Value) {
	if let Some(last) = messages.last_mut() {
		if last.role == "assistant" {
			match last.tool_calls.as_mut() {
				Some(serde_json::Value::Array(arr)) => arr.push(tool_call),
				_ => last.tool_calls = Some(serde_json::Value::Array(vec![tool_call])),
			}
			return;
		}
	}
	let mut msg = Message::assistant("");
	msg.tool_calls = Some(serde_json::Value::Array(vec![tool_call]));
	messages.push(msg);
}

/// Attach `text` as a ThinkingBlock on the trailing assistant Message, or create
/// an empty assistant Message carrying it if none precedes.
fn attach_thinking_to_assistant(messages: &mut Vec<Message>, text: String) {
	if let Some(last) = messages.last_mut() {
		if last.role == "assistant" {
			last.thinking = Some(ThinkingBlock::new(&text));
			return;
		}
	}
	let mut msg = Message::assistant("");
	msg.thinking = Some(ThinkingBlock::new(&text));
	messages.push(msg);
}

/// Reject the request if `api_key.allowed_models` is set and `model` isn't
/// in it. The error message embeds `MODEL_FORBIDDEN_MARKER` so the HTTP
/// layer can return `403 Forbidden` rather than the default 400/500.
/// Move `provider`'s candidates to the front (stable — order among the rest
/// is preserved) so a session sticks to the provider that served its chain
/// and keeps the provider-side prompt cache warm. `pick_admitted` still
/// falls through to the others when the preferred windows are exhausted.
fn prefer_provider(candidates: &mut [(String, String)], provider: &str) {
	candidates.sort_by_key(|(p, _)| !p.eq_ignore_ascii_case(provider));
}

/// Move lanes whose concurrency is full behind the rest (stable, so list order
/// holds within each group). Deprioritized, not removed: when every lane is
/// busy the first one still takes the request and it queues there.
pub(crate) fn deprioritize_saturated(candidates: &mut [(String, String)], limiter: &ProviderLimiter) {
	candidates.sort_by_key(|(provider, _)| limiter.is_saturated(provider));
}

/// After a modality mismatch, only another host for the same resolved model is
/// a valid failover. A different model under the alias must never receive the
/// request implicitly.
fn retain_same_model_failover_candidates(candidates: &mut Vec<(String, String)>, mismatch: &ModalityNotSupportedError) {
	candidates.retain(|(provider, model)| {
		!provider.eq_ignore_ascii_case(&mismatch.provider) && same_model(model, &mismatch.model)
	});
}

/// Hosts name one model differently (`kimi-2.7-code` on Ollama,
/// `@cf/moonshotai/kimi-k2.7-code` on Workers AI); octolib's reference id is
/// what says two lane models are the same model.
fn same_model(a: &str, b: &str) -> bool {
	a == b || reference_model_id(a).is_some_and(|id| reference_model_id(b) == Some(id))
}

/// Keep the candidates whose provider guarantees a JSON schema's shape —
/// octolib's `enforces_response_schema` is the single source of that fact.
fn retain_schema_enforcing_candidates(candidates: &mut Vec<(String, String)>) {
	candidates.retain(|(provider, model)| enforces_response_schema(provider, model));
}

pub(crate) fn enforces_response_schema(provider: &str, model: &str) -> bool {
	ProviderFactory::create_provider(provider).is_ok_and(|p| p.enforces_response_schema(model))
}

/// Pick the first candidate whose provider rate windows admit a request
/// (counting it against the winner's windows). Cooling providers (see
/// [`ProviderHealth`]) are sorted behind healthy ones — deprioritized, not
/// blocked. Exhausted candidates are skipped; when ALL are exhausted the
/// caller gets [`RateLimitedError`] carrying the soonest retry among them.
pub(crate) fn pick_admitted(
	tracker: &ProviderRateTracker,
	health: &ProviderHealth,
	config: &Config,
	mut candidates: Vec<(String, String)>,
	model: &str,
) -> Result<(String, String)> {
	// Stable sort: healthy candidates keep their (sticky + rotation) order
	// in front, cooling ones follow as a last resort.
	candidates.sort_by_key(|(provider, _)| health.is_cooling(provider));

	let mut soonest_retry: Option<std::time::Duration> = None;
	for (provider, resolved) in candidates {
		// No [providers.<name>] section (or no rate limits in it) — admit.
		let Some(provider_cfg) = config.provider_config(&provider) else {
			return Ok((provider, resolved));
		};
		match tracker.try_admit(&provider, provider_cfg) {
			Ok(()) => return Ok((provider, resolved)),
			Err(retry) => {
				tracing::info!(
					provider = %provider,
					retry_in_s = retry.as_secs(),
					"provider rate window exhausted — skipping candidate"
				);
				crate::metrics::record_rate_limited(&provider);
				soonest_retry = Some(soonest_retry.map_or(retry, |s| s.min(retry)));
			}
		}
	}
	Err(anyhow!(RateLimitedError {
		model: model.to_string(),
		retry_after: soonest_retry.unwrap_or_default(),
	}))
}

pub(crate) fn ensure_model_allowed(api_key: &ApiKey, model: &str) -> Result<()> {
	if api_key.is_model_allowed(model) {
		return Ok(());
	}
	Err(anyhow!(
		"{}: model '{}' is not permitted for this API key",
		MODEL_FORBIDDEN_MARKER,
		model
	))
}

// Implement Serialize for Input so we can store it
impl serde::Serialize for Input {
	fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		match self {
			Input::Text(s) => serializer.serialize_str(s),
			Input::Items(items) => items.serialize(serializer),
		}
	}
}

fn byok_candidates(
	candidates: Vec<(String, String)>,
	credentials: &[crate::auth::OwnerCredential],
) -> Vec<(String, String)> {
	let mut lanes: Vec<_> = candidates
		.into_iter()
		.filter_map(|lane| {
			credentials
				.iter()
				.find(|c| c.record.provider.eq_ignore_ascii_case(&lane.0))
				.map(|c| (c.record.position, lane))
		})
		.collect();
	lanes.sort_by_key(|(position, _)| *position);
	lanes.into_iter().map(|(_, lane)| lane).collect()
}

fn redact_credential_text(text: &str, credentials: &[crate::auth::OwnerCredential]) -> String {
	credentials.iter().fold(text.to_owned(), |text, credential| {
		text.replace(&credential.key, "[REDACTED]")
	})
}

fn redact_credential_json(value: &mut serde_json::Value, credentials: &[crate::auth::OwnerCredential]) {
	match value {
		serde_json::Value::String(text) => *text = redact_credential_text(text, credentials),
		serde_json::Value::Array(values) => {
			for value in values {
				redact_credential_json(value, credentials);
			}
		}
		serde_json::Value::Object(values) => {
			*values = std::mem::take(values)
				.into_iter()
				.map(|(key, mut value)| {
					redact_credential_json(&mut value, credentials);
					(redact_credential_text(&key, credentials), value)
				})
				.collect();
		}
		_ => {}
	}
}

#[derive(Debug)]
pub(crate) struct ByokFailure {
	pub status: hyper::StatusCode,
	message: String,
	label: &'static str,
}

impl ByokFailure {
	fn from_upstream(
		error: &anyhow::Error,
		provider: &str,
		credentials: &[crate::auth::OwnerCredential],
	) -> Result<Self> {
		let upstream_message = format!("{error:#}");
		let status = match error.downcast_ref::<ProxyTimeoutError>() {
			Some(ProxyTimeoutError::Upstream { .. }) => hyper::StatusCode::GATEWAY_TIMEOUT,
			_ => match upstream_status_code(&upstream_message) {
				Some(code) => hyper::StatusCode::from_u16(code).context("Invalid upstream HTTP status")?,
				None => hyper::StatusCode::BAD_GATEWAY,
			},
		};
		Ok(Self {
			status,
			message: redact_credential_text(&upstream_message, credentials),
			label: crate::auth::byok_provider_label(&provider.to_ascii_lowercase()).context("Unsupported BYOK lane")?,
		})
	}
	fn retryable(&self) -> bool {
		matches!(self.status.as_u16(), 401 | 403 | 429 | 500..=599)
	}
}
impl std::fmt::Display for ByokFailure {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "Your {} key failed: {}", self.label, self.message)
	}
}
impl std::error::Error for ByokFailure {}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::config::ProviderConfig;
	use std::collections::HashMap;

	fn config_with_provider(name: &str, cfg: ProviderConfig) -> Config {
		let mut providers = HashMap::new();
		providers.insert(name.to_string(), cfg);
		Config {
			server: Default::default(),
			models: HashMap::new(),
			embedding_models: HashMap::new(),
			media_models: HashMap::new(),
			evaluation_models: HashMap::new(),
			auto: HashMap::new(),
			providers,
			logging: Default::default(),
			metrics: Default::default(),
			media: Default::default(),
			media_providers: HashMap::new(),
			routing: Default::default(),
		}
	}

	fn candidate(provider: &str, model: &str) -> (String, String) {
		(provider.to_string(), model.to_string())
	}

	fn with_image(text: &str) -> Message {
		let mut m = Message::user(text);
		m.images = parse_image_url("https://example.test/shot.png").map(|i| vec![i]);
		m
	}

	#[test]
	fn modalities_come_from_the_whole_conversation() {
		assert_eq!(Modalities::of(&[Message::user("hi")]), Modalities::default());
		// An image anywhere in history counts: providers get the whole history.
		let got = Modalities::of(&[with_image("look"), Message::user("and now?")]);
		assert!(got.images && !got.videos);
	}

	#[test]
	fn auto_skips_entries_whose_lanes_cannot_read_the_images() {
		// Prod 2026-10-22: owner map main = glm-5.3 (text-only), floor default =
		// glm-5.3-flash (multimodal). A turn carrying a screenshot must land on the
		// floor, not 400 on the owner's text model.
		let mut config = config_with_provider("zai", ProviderConfig::default());
		config.models.insert("glm-5.3".into(), vec!["zai:glm-5.3".into()]);
		config
			.models
			.insert("glm-5.3-flash".into(), vec!["zai:glm-5.3-flash".into()]);
		config.auto.insert("default".into(), "glm-5.3-flash".into());
		let owner: HashMap<String, String> = [("main".to_string(), "glm-5.3".to_string())].into();
		let pick = |config: &Config, needs: Modalities| {
			super::super::auto::resolve(Some("main"), Some(&owner), &config.auto, |m| {
				config.models.contains_key(m) && needs.served_by_alias(config, m)
			})
		};
		assert_eq!(
			pick(&config, Modalities::default()).as_deref(),
			Some("glm-5.3"),
			"text keeps the owner's pick"
		);
		let image = Modalities {
			images: true,
			videos: false,
		};
		assert_eq!(
			pick(&config, image).as_deref(),
			Some("glm-5.3-flash"),
			"an image skips the text-only model"
		);

		// Nothing in the chain reads images: unresolvable, never a silent text model.
		config.auto.insert("default".into(), "glm-5.3".into());
		assert_eq!(pick(&config, image), None);
		assert_eq!(image.describe(), " that reads images");
	}

	#[test]
	fn the_current_turn_starts_after_the_last_assistant_message() {
		let mut answer = Message::user("done");
		answer.role = "assistant".into();
		assert_eq!(
			current_turn_start(&[with_image("look")]),
			0,
			"no answer yet: all of it is the turn"
		);
		let convo = [with_image("look"), answer.clone(), Message::user("and now?")];
		assert_eq!(current_turn_start(&convo), 2);
		// The screenshot is history now: it must not move the pick for this turn.
		assert_eq!(Modalities::of(&convo[2..]), Modalities::default());
		assert!(Modalities::of(&convo[..2]).images);
		assert_eq!(
			current_turn_start(&[with_image("look"), answer]),
			2,
			"nothing asked yet"
		);
	}

	#[test]
	fn history_images_are_omitted_for_a_model_that_cannot_read_them() {
		let mut config = config_with_provider("zai", ProviderConfig::default());
		config.models.insert("glm-5.3".into(), vec!["zai:glm-5.3".into()]);
		config
			.models
			.insert("glm-5.3-flash".into(), vec!["zai:glm-5.3-flash".into()]);
		let earlier = Modalities {
			images: true,
			videos: false,
		};
		assert_eq!(
			earlier.unread_by_alias(&config, "glm-5.3"),
			earlier,
			"a text-only alias reads none of it"
		);
		assert_eq!(
			earlier.unread_by_alias(&config, "glm-5.3-flash"),
			Modalities::default(),
			"a vision alias keeps it"
		);

		let mut history = vec![with_image("see this"), Message::user("plain")];
		omit_attachments(&mut history, earlier.unread_by_alias(&config, "glm-5.3"));
		assert!(history[0].images.is_none(), "the image left the request");
		assert!(
			history[0].content.starts_with("see this\n\n[1 image(s) omitted"),
			"and the text says so"
		);
		assert_eq!(
			history[1].content, "plain",
			"messages without attachments are untouched"
		);
		assert_eq!(
			Modalities::of(&history),
			Modalities::default(),
			"nothing left to refuse"
		);

		// Nothing to omit: nothing changes.
		let mut kept = vec![with_image("see this")];
		omit_attachments(&mut kept, Modalities::default());
		assert!(kept[0].images.is_some() && kept[0].content == "see this");
	}

	#[test]
	fn an_alias_serves_a_modality_when_any_lane_does() {
		let mut config = config_with_provider("zai", ProviderConfig::default());
		config
			.models
			.insert("mixed".into(), vec!["zai:glm-5.3".into(), "zai:glm-5.3-flash".into()]);
		config.models.insert("text".into(), vec!["zai:glm-5.3".into()]);
		let image = Modalities {
			images: true,
			videos: false,
		};
		assert!(image.served_by_alias(&config, "mixed"));
		assert!(!image.served_by_alias(&config, "text"));
		assert!(Modalities::default().served_by_alias(&config, "text"));
		assert!(!image.served_by_alias(&config, "missing"));
	}

	/// pick_admitted with a fresh (all-healthy) health tracker.
	fn pick(
		tracker: &ProviderRateTracker,
		config: &Config,
		candidates: Vec<(String, String)>,
		model: &str,
	) -> Result<(String, String)> {
		pick_admitted(tracker, &ProviderHealth::new(), config, candidates, model)
	}

	#[test]
	fn pick_admitted_skips_exhausted_provider() {
		let tracker = ProviderRateTracker::new();
		let config = config_with_provider(
			"moonshot",
			ProviderConfig {
				requests_per_minute: Some(1),
				..Default::default()
			},
		);

		// Consume moonshot's single request slot.
		let cfg = config.provider_config("moonshot").unwrap();
		tracker.try_admit("moonshot", cfg).unwrap();

		let picked = pick(
			&tracker,
			&config,
			vec![candidate("moonshot", "kimi-k2"), candidate("ollama", "kimi-k2")],
			"kimi-k2",
		)
		.unwrap();
		assert_eq!(picked.0, "ollama", "exhausted first candidate is skipped");
	}

	#[test]
	fn pick_admitted_fails_429_when_all_exhausted() {
		let tracker = ProviderRateTracker::new();
		let config = config_with_provider(
			"moonshot",
			ProviderConfig {
				requests_per_minute: Some(1),
				..Default::default()
			},
		);
		let cfg = config.provider_config("moonshot").unwrap();
		tracker.try_admit("moonshot", cfg).unwrap();

		let err = pick(&tracker, &config, vec![candidate("moonshot", "kimi-k2")], "kimi-k2")
			.expect_err("all candidates exhausted");
		let rate = err
			.downcast_ref::<RateLimitedError>()
			.expect("typed error for the 429 mapping");
		assert_eq!(rate.model, "kimi-k2");
		assert!(rate.retry_after > std::time::Duration::ZERO);
	}

	#[test]
	fn pick_admitted_counts_the_winning_request() {
		let tracker = ProviderRateTracker::new();
		let config = config_with_provider(
			"moonshot",
			ProviderConfig {
				requests_per_minute: Some(1),
				..Default::default()
			},
		);

		let candidates = vec![candidate("moonshot", "kimi-k2")];
		assert!(pick(&tracker, &config, candidates.clone(), "kimi-k2").is_ok());
		assert!(
			pick(&tracker, &config, candidates, "kimi-k2").is_err(),
			"admission must have consumed the rpm=1 budget"
		);
	}

	#[test]
	fn prefer_provider_moves_sticky_to_front_preserving_rest() {
		let mut candidates = vec![
			candidate("openai", "m"),
			candidate("groq", "m"),
			candidate("Anthropic", "m"),
		];
		prefer_provider(&mut candidates, "anthropic");
		assert_eq!(
			candidates,
			vec![
				candidate("Anthropic", "m"),
				candidate("openai", "m"),
				candidate("groq", "m"),
			],
			"sticky provider first (case-insensitive), others keep order"
		);
	}

	#[test]
	fn sticky_provider_wins_until_exhausted() {
		let tracker = ProviderRateTracker::new();
		let config = config_with_provider(
			"anthropic",
			ProviderConfig {
				requests_per_minute: Some(1),
				..Default::default()
			},
		);

		let mut candidates = vec![candidate("openai", "m"), candidate("anthropic", "m")];
		prefer_provider(&mut candidates, "anthropic");

		// Turn 1: session sticks to its previous provider.
		let picked = pick(&tracker, &config, candidates.clone(), "m").unwrap();
		assert_eq!(picked.0, "anthropic");

		// Turn 2: sticky provider's window exhausted — fall through, don't fail.
		let picked = pick(&tracker, &config, candidates, "m").unwrap();
		assert_eq!(picked.0, "openai");
	}

	#[tokio::test]
	async fn saturated_lanes_move_behind_free_ones_in_list_order() {
		let config = config_with_provider(
			"xiaomi",
			ProviderConfig {
				concurrency: Some(1),
				..Default::default()
			},
		);
		let limiter = ProviderLimiter::from_config(&config);
		let ordered = vec![
			candidate("xiaomi", "m"),
			candidate("openrouter", "m"),
			candidate("groq", "m"),
		];

		let mut free = ordered.clone();
		deprioritize_saturated(&mut free, &limiter);
		assert_eq!(free, ordered, "a lane with a free slot keeps its priority");

		let _busy = limiter.acquire("xiaomi").await;
		let mut busy = ordered.clone();
		deprioritize_saturated(&mut busy, &limiter);
		assert_eq!(
			busy,
			vec![
				candidate("openrouter", "m"),
				candidate("groq", "m"),
				candidate("xiaomi", "m")
			],
			"a full primary goes last, the rest keep list order"
		);
	}

	#[test]
	fn schema_requests_keep_only_enforcing_lanes_in_order() {
		// Z.ai answers schemas with json_object only; Alibaba's Qwen3.8 family
		// decodes against the schema; an unknown provider can't be trusted.
		let mut candidates = vec![
			candidate("zai", "glm-5.3-flash"),
			candidate("alibaba", "qwen3.8-flash"),
			candidate("no-such-provider", "m"),
			candidate("openai", "gpt-5.5"),
		];
		retain_schema_enforcing_candidates(&mut candidates);
		assert_eq!(
			candidates,
			vec![candidate("alibaba", "qwen3.8-flash"), candidate("openai", "gpt-5.5")]
		);

		let mut candidates = vec![candidate("zai", "glm-5.3-flash")];
		retain_schema_enforcing_candidates(&mut candidates);
		assert!(candidates.is_empty());
	}

	#[test]
	fn modality_failover_keeps_only_same_model_on_another_host() {
		let mut candidates = vec![
			candidate("openai", "vision-model"),
			candidate("ollama", "vision-model"),
			candidate("anthropic", "different-model"),
		];
		let mismatch = ModalityNotSupportedError {
			provider: "openai".to_string(),
			model: "vision-model".to_string(),
			modality: "image".to_string(),
		};

		retain_same_model_failover_candidates(&mut candidates, &mismatch);

		assert_eq!(candidates, vec![candidate("ollama", "vision-model")]);
	}

	#[test]
	fn modality_failover_keeps_other_hosts_that_name_the_model_differently() {
		let mut candidates = vec![
			candidate("ollama", "kimi-2.7-code"),
			candidate("cloudflare", "@cf/moonshotai/kimi-k2.7-code"),
			candidate("together", "moonshotai/Kimi-K2.7-Code"),
			candidate("together", "moonshotai/Kimi-K2.6"),
		];
		let mismatch = ModalityNotSupportedError {
			provider: "ollama".to_string(),
			model: "kimi-2.7-code".to_string(),
			modality: "video".to_string(),
		};

		retain_same_model_failover_candidates(&mut candidates, &mismatch);

		assert_eq!(
			candidates,
			vec![
				candidate("cloudflare", "@cf/moonshotai/kimi-k2.7-code"),
				candidate("together", "moonshotai/Kimi-K2.7-Code"),
			]
		);
	}

	#[test]
	fn anthropic_image_reaches_provider_message_as_base64_attachment() {
		let chat: ChatCompletionRequest = serde_json::from_str(
			r#"{
                "model": "vision-model",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aW1hZ2U="
                        }
                    }]
                }]
            }"#,
		)
		.unwrap();
		let request: CreateCompletionRequest = chat.into();
		let Input::Items(items) = request.input else {
			panic!("chat request must convert to input items");
		};
		let mut messages = Vec::new();

		push_items(&items, &mut messages);

		let images = messages[0]
			.images
			.as_ref()
			.expect("provider message must carry the image");
		assert_eq!(images[0].media_type, "image/png");
		assert!(matches!(
			&images[0].data,
			ImageData::Base64(data) if data == "aW1hZ2U="
		));
	}

	#[test]
	fn cooling_provider_is_deprioritized_not_blocked() {
		let tracker = ProviderRateTracker::new();
		let health = ProviderHealth::new();
		let config = config_with_provider("moonshot", ProviderConfig::default());
		let cooldown = std::time::Duration::from_secs(60);
		for _ in 0..3 {
			health.record_failure("openai", cooldown);
		}

		// Healthy candidate wins even though the cooling one is listed first.
		let candidates = vec![candidate("openai", "m"), candidate("groq", "m")];
		let picked = pick_admitted(&tracker, &health, &config, candidates, "m").unwrap();
		assert_eq!(picked.0, "groq", "cooling provider sorted behind healthy");

		// A cooling provider as the ONLY candidate still dispatches.
		let picked = pick_admitted(&tracker, &health, &config, vec![candidate("openai", "m")], "m").unwrap();
		assert_eq!(picked.0, "openai", "cooldown deprioritizes, never blocks");
	}

	#[test]
	fn evaluation_errors_are_classified_by_variant() {
		let rate = anyhow!(EvaluationError::RateLimit {
			provider: "typesafe".into(),
			message: "slow down".into(),
			retry_after_secs: Some(2),
		})
		.context("Evaluation provider 'typesafe' failed for model 'jev-latest'");
		assert!(is_provider_fault(&rate));
		assert!(matches!(
			evaluation_error(&rate),
			Some(EvaluationError::RateLimit { .. })
		));

		let overloaded = anyhow!(EvaluationError::Api {
			provider: "typesafe".into(),
			status: 529,
			message: "overloaded".into(),
		});
		assert!(is_provider_fault(&overloaded));

		let rejected = anyhow!(EvaluationError::InvalidRequest("bad question".into()));
		assert!(!is_provider_fault(&rejected));

		let bad_request = anyhow!(EvaluationError::Api {
			provider: "typesafe".into(),
			status: 400,
			message: "bad question".into(),
		});
		assert!(!is_provider_fault(&bad_request));

		// A revoked key is this lane's problem; the next lane may be keyed.
		let unauthenticated = anyhow!(EvaluationError::Authentication {
			provider: "typesafe".into(),
			message: "bad key".into(),
		});
		assert!(is_provider_fault(&unauthenticated));

		// An unkeyed lane must not block the keyed one behind it.
		let unkeyed = anyhow!(EvaluationError::MissingApiKey("TYPESAFE_API_KEY".into()));
		assert!(is_provider_fault(&unkeyed));
	}

	#[test]
	fn provider_fault_classification() {
		let upstream_timeout = anyhow::Error::new(ProxyTimeoutError::Upstream {
			provider: "openai".to_string(),
			timeout: std::time::Duration::from_secs(360),
		});
		assert!(is_provider_fault(&upstream_timeout));

		let queue_timeout = anyhow::Error::new(ProxyTimeoutError::ProviderQueue {
			provider: "openai".to_string(),
			timeout: std::time::Duration::from_secs(60),
		});
		assert!(
			!is_provider_fault(&queue_timeout),
			"our own queue capacity is not the provider's fault"
		);

		let upstream_500 = anyhow::anyhow!("openai API error 503 Service Unavailable");
		assert!(is_provider_fault(&upstream_500));

		let upstream_429 = anyhow::anyhow!("anthropic API error 429 Too Many Requests");
		assert!(is_provider_fault(&upstream_429));

		let client_400 = anyhow::anyhow!("ollama API error 400 Bad Request: prompt too long");
		assert!(
			!is_provider_fault(&client_400),
			"400 is the request's fault — every provider rejects it"
		);

		let past_due = anyhow::anyhow!(
			"Provider 'ollama' chat_completion failed: ollama API error 403 Forbidden: \
             your subscription payment is past due"
		);
		assert!(
			is_provider_fault(&past_due),
			"a lane's billing/auth rejection must fail over and cool the lane"
		);

		let model_gone = anyhow::anyhow!("together API error 404 Not Found: model not found");
		assert!(is_provider_fault(&model_gone));

		let connect_error = anyhow::anyhow!("connection refused");
		assert!(
			is_provider_fault(&connect_error),
			"transport errors without a status are provider-side"
		);

		let stream_silence = anyhow::anyhow!("Stream timed out after 30s without response data");
		assert!(
			is_provider_fault(&stream_silence),
			"octolib's StreamTimeout carries no status: a silent lane is provider-side"
		);
	}

	#[test]
	fn pick_admitted_ignores_providers_without_config() {
		let tracker = ProviderRateTracker::new();
		let config = config_with_provider("moonshot", ProviderConfig::default());

		// Unlisted provider and listed-but-unlimited provider both admit.
		for provider in ["ollama", "moonshot"] {
			let picked = pick(&tracker, &config, vec![candidate(provider, "m")], "m").unwrap();
			assert_eq!(picked.0, provider);
		}
	}

	fn item(json: &str) -> InputItem {
		serde_json::from_str(json).unwrap()
	}

	#[test]
	fn function_call_coalesces_onto_preceding_assistant() {
		// Client replays: user → assistant text → function_call → function_call_output → user
		// Expected octolib message shape: user / assistant(text + tool_calls) / tool / user
		let items = vec![
			item(r#"{"type":"message","role":"user","content":"hi"}"#),
			item(r#"{"type":"message","role":"assistant","content":"calling tool"}"#),
			item(r#"{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"}"#),
			item(r#"{"type":"function_call_output","call_id":"c1","output":"ok"}"#),
			item(r#"{"type":"message","role":"user","content":"thanks"}"#),
		];
		let mut messages = Vec::new();
		push_items(&items, &mut messages);

		assert_eq!(messages.len(), 4, "tool_call merges into assistant turn");
		assert_eq!(messages[0].role, "user");
		assert_eq!(messages[1].role, "assistant");
		assert!(messages[1].tool_calls.is_some(), "assistant must carry tool_calls");
		assert_eq!(messages[2].role, "tool");
		assert_eq!(messages[3].role, "user");
	}

	#[test]
	fn function_call_without_preceding_assistant_creates_one() {
		// Edge case: client sends function_call as the first item with no prior assistant.
		let items = vec![
			item(r#"{"type":"message","role":"user","content":"go"}"#),
			item(r#"{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"}"#),
		];
		let mut messages = Vec::new();
		push_items(&items, &mut messages);

		assert_eq!(messages.len(), 2);
		assert_eq!(messages[1].role, "assistant");
		assert!(messages[1].tool_calls.is_some());
	}

	#[test]
	fn function_call_output_forwards_cache_control() {
		// A tool result carrying a cache marker must propagate to the upstream
		// Message (cached + ttl) — it's the agent-loop tail breakpoint. Regression
		// for the proxy silently dropping markers on tool results.
		let cached = item(
			r#"{"type":"function_call_output","call_id":"c1","output":"ok","cache_control":{"type":"ephemeral","ttl":"1h"}}"#,
		);
		let plain = item(r#"{"type":"function_call_output","call_id":"c2","output":"ok"}"#);
		let mut messages = Vec::new();
		push_items(&[cached, plain], &mut messages);

		assert_eq!(messages.len(), 2);
		assert!(messages[0].cached, "marked tool result must stay cached");
		assert_eq!(messages[0].cache_ttl.as_deref(), Some("1h"));
		assert!(!messages[1].cached, "unmarked tool result must not be cached");
	}

	#[test]
	fn stored_items_strip_tool_cache_control() {
		// Replayed-from-chain history must NOT carry markers (anti-accumulation vs
		// Anthropic's 4-breakpoint limit); only live input anchors the breakpoint.
		let stored = vec![serde_json::json!({
			"type": "function_call_output",
			"call_id": "c1",
			"output": "ok",
			"cache_control": {"type": "ephemeral", "ttl": "1h"}
		})];
		let mut messages = Vec::new();
		push_stored_items(&stored, &mut messages);

		assert_eq!(messages.len(), 1);
		assert!(!messages[0].cached, "replayed history markers are stripped");
	}

	#[test]
	fn function_call_output_with_image_parts_carries_text_and_image() {
		// Prod 2026-10-31: a browser screenshot came back as an array of parts and
		// the whole request was rejected. The image must reach the tool Message.
		let shot = item(
			r#"{"type":"function_call_output","call_id":"c1","output":[
				{"type":"input_text","text":"screenshot taken"},
				{"type":"input_image","image_url":"data:image/png;base64,iVBORw0KGgo="}
			]}"#,
		);
		let mut messages = Vec::new();
		push_items(&[shot], &mut messages);

		assert_eq!(messages.len(), 1);
		assert_eq!(messages[0].role, "tool");
		assert_eq!(messages[0].content, "screenshot taken");
		let images = messages[0].images.as_ref().expect("tool image must be forwarded");
		assert_eq!(images.len(), 1);
		assert_eq!(images[0].media_type, "image/png");
	}

	#[test]
	fn stored_items_replay_tool_image_parts() {
		let stored = vec![serde_json::json!({
			"type": "function_call_output",
			"call_id": "c1",
			"output": [
				{"type": "input_text", "text": "screenshot taken"},
				{"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}
			]
		})];
		let mut messages = Vec::new();
		push_stored_items(&stored, &mut messages);

		assert_eq!(
			messages.len(),
			1,
			"array-shaped output must not be skipped as unparseable"
		);
		assert_eq!(messages[0].content, "screenshot taken");
		assert_eq!(messages[0].images.as_ref().map(Vec::len), Some(1));
	}

	#[test]
	fn reasoning_attaches_thinking_to_preceding_assistant() {
		// DeepSeek migration path: reasoning + tool_call replayed together must
		// produce one assistant Message carrying both thinking and tool_calls.
		let items = vec![
			item(r#"{"type":"message","role":"user","content":"q"}"#),
			item(r#"{"type":"message","role":"assistant","content":""}"#),
			item(r#"{"type":"reasoning","content":[{"type":"output_text","text":"think..."}]}"#),
			item(r#"{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"}"#),
		];
		let mut messages = Vec::new();
		push_items(&items, &mut messages);

		assert_eq!(messages.len(), 2);
		assert_eq!(messages[1].role, "assistant");
		assert!(messages[1].thinking.is_some());
		assert!(messages[1].tool_calls.is_some());
	}

	#[test]
	fn multiple_tool_calls_accumulate_on_same_assistant() {
		let items = vec![
			item(r#"{"type":"message","role":"assistant","content":""}"#),
			item(r#"{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"}"#),
			item(r#"{"type":"function_call","call_id":"c2","name":"g","arguments":"{}"}"#),
		];
		let mut messages = Vec::new();
		push_items(&items, &mut messages);

		assert_eq!(messages.len(), 1);
		let arr = messages[0].tool_calls.as_ref().unwrap().as_array().unwrap();
		assert_eq!(arr.len(), 2);
	}
}

#[cfg(test)]
mod byok_tests {
	use super::*;
	use crate::auth::{Credentials, OwnerCredential};
	use crate::storage::credentials::{Credential, CredentialChange};
	use crate::storage::sqlite::SqliteStorage;
	use base64::Engine;
	use std::sync::{Mutex, RwLock};
	use tracing::instrument::WithSubscriber;

	fn credential(provider: &str, position: i64) -> OwnerCredential {
		OwnerCredential {
			key: format!("private-{provider}-key"),
			record: Credential {
				provider: provider.into(),
				secret: "ciphertext".into(),
				key_hint: "...hint".into(),
				position,
				created_at: 1,
				last_used_at: None,
				last_error: None,
			},
		}
	}

	#[test]
	fn lane_order_is_stable_and_never_includes_platform_candidates() {
		let lanes = vec![
			("openai".into(), "platform".into()),
			("ollama".into(), "first".into()),
			("zai".into(), "second".into()),
			("ollama".into(), "third".into()),
		];
		assert!(byok_candidates(lanes.clone(), &[]).is_empty());
		assert!(byok_candidates(lanes.clone(), &[credential("anthropic", 0)]).is_empty());
		assert_eq!(
			byok_candidates(lanes, &[credential("zai", 1), credential("ollama", 0)]),
			[
				("ollama".into(), "first".into()),
				("ollama".into(), "third".into()),
				("zai".into(), "second".into())
			]
		);
	}

	#[test]
	fn failures_preserve_status_and_redact_secrets_and_params() {
		let credentials = [credential("zai", 0)];
		for status in [400, 401, 403, 404, 429, 500, 502, 503] {
			let error = anyhow!("Z.ai API error {status}: rejected private-zai-key");
			let failure = ByokFailure::from_upstream(&error, "zai", &credentials).unwrap();
			assert_eq!(failure.status.as_u16(), status);
			assert_eq!(failure.retryable(), matches!(status, 401 | 403 | 429 | 500..=599));
			assert!(failure.to_string().starts_with("Your Z.ai key failed: "));
			assert!(!format!("{failure:?}").contains("private-zai-key"));
		}
		let failure = ByokFailure::from_upstream(&anyhow!("connection closed"), "zai", &credentials).unwrap();
		assert_eq!(failure.status, hyper::StatusCode::BAD_GATEWAY);
		assert!(failure.retryable());
		let params = ChatCompletionParams::new(&[Message::user("hello")], "glm-5", 1.0, 1.0, 50, 10)
			.with_api_key(credentials[0].key.clone());
		assert!(!format!("{params:?}").contains("private-zai-key"));
		let mut exchange = serde_json::json!({"request": {"private-zai-key": "Bearer private-zai-key"}, "response": ["private-zai-key"]});
		redact_credential_json(&mut exchange, &credentials);
		assert!(!exchange.to_string().contains("private-zai-key"));
	}

	#[derive(Clone, Default)]
	struct CapturedLogs(Arc<Mutex<Vec<u8>>>);

	impl std::io::Write for CapturedLogs {
		fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
			self.0.lock().unwrap().extend_from_slice(bytes);
			Ok(bytes.len())
		}

		fn flush(&mut self) -> std::io::Result<()> {
			Ok(())
		}
	}

	struct EnvGuard(Vec<(&'static str, Option<std::ffi::OsString>)>);
	impl EnvGuard {
		fn set(values: &[(&'static str, String)]) -> Self {
			let previous = values
				.iter()
				.map(|(name, value)| {
					let previous = std::env::var_os(name);
					std::env::set_var(name, value);
					(*name, previous)
				})
				.collect();
			Self(previous)
		}
	}
	impl Drop for EnvGuard {
		fn drop(&mut self) {
			for (name, value) in &self.0 {
				match value {
					Some(value) => std::env::set_var(name, value),
					None => std::env::remove_var(name),
				}
			}
		}
	}

	#[derive(Default)]
	struct Upstream {
		statuses: std::collections::HashMap<String, u16>,
		requests: Vec<(String, String)>,
		bodies: Vec<serde_json::Value>,
	}

	fn request(model: &str, previous: Option<&str>) -> CreateCompletionRequest {
		serde_json::from_value(
			serde_json::json!({"model": model, "input": "hello", "previous_completion_id": previous}),
		)
		.unwrap()
	}

	async fn mock_upstream() -> (std::net::SocketAddr, Arc<Mutex<Upstream>>, tokio::task::JoinHandle<()>) {
		use http_body_util::{BodyExt, Full};
		use hyper::service::service_fn;
		use hyper_util::rt::TokioIo;
		let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
		let address = listener.local_addr().unwrap();
		let upstream = Arc::new(Mutex::new(Upstream::default()));
		let state = upstream.clone();
		let server = tokio::spawn(async move {
			loop {
				let (socket, _) = listener.accept().await.unwrap();
				let state = state.clone();
				tokio::spawn(async move {
					hyper::server::conn::http1::Builder::new().serve_connection(TokioIo::new(socket), service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
                        let state = state.clone();
                        async move {
                            let path = req.uri().path().to_owned();
                            let auth = req.headers().get("Authorization").map(|h| h.to_str().unwrap().to_owned()).unwrap_or_default();
                            let body = req.collect().await.unwrap().to_bytes();
                            assert!(!String::from_utf8_lossy(&body).contains("owner-private"));
                            let status = {
                                let mut state = state.lock().unwrap();
                                state.requests.push((path.clone(), auth.clone()));
								state.bodies.push(serde_json::from_slice(&body).unwrap());
                                state.statuses.get(&path).copied().unwrap_or(200)
                            };
                            let body = if status == 200 {
                                serde_json::json!({"id":"mock", "object":"chat.completion", "created":1_700_000_000, "model":"glm-5", "choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
                                    "usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"cost":0.01}})
                            } else {
                                serde_json::json!({"error":{"message": format!("Rejected {auth} {}", "x".repeat(350))}})
                            };
                            Ok::<_, std::convert::Infallible>(hyper::Response::builder().status(status)
                                .header("content-type", "application/json")
                                .body(Full::new(bytes::Bytes::from(body.to_string()))).unwrap())
                        }
                    })).await.unwrap();
				});
			}
		});
		(address, upstream, server)
	}

	struct ChainHarness {
		engine: ProxyEngine,
		storage: Arc<SqliteStorage>,
		key: ApiKey,
		upstream: Arc<Mutex<Upstream>>,
		server: tokio::task::JoinHandle<()>,
		_env: EnvGuard,
	}

	impl ChainHarness {
		async fn new(provider: &str, url_env: &'static str, key_env: &'static str) -> Self {
			let (address, upstream, server) = mock_upstream().await;
			// Each test uses a different provider so process-wide overrides cannot race.
			let env = EnvGuard::set(&[(url_env, format!("http://{address}")), (key_env, "mock-key".into())]);
			let config: Config = toml::from_str(&format!("[models]\nchain = [\"{provider}:m\"]")).unwrap();
			let storage = Arc::new(SqliteStorage::new(":memory:").unwrap());
			let key = storage.create_api_key("key", None, None, None).unwrap();
			let credentials = Arc::new(Credentials::new(storage.clone(), None).unwrap());
			let limiter = Arc::new(RwLock::new(Arc::new(ProviderLimiter::from_config(&config))));
			let config = Arc::new(RwLock::new(Arc::new(config)));
			let engine = ProxyEngine::new(storage.clone(), config, limiter, 1, credentials);
			Self {
				engine,
				storage,
				key,
				upstream,
				server,
				_env: env,
			}
		}
	}

	impl Drop for ChainHarness {
		fn drop(&mut self) {
			self.server.abort();
		}
	}

	#[tokio::test]
	async fn unknown_previous_completion_id_uses_inline_input_without_dangling_parent() {
		let h = ChainHarness::new("cerebras", "CEREBRAS_API_URL", "CEREBRAS_API_KEY").await;
		let (response, _) = h
			.engine
			.process(request("chain", Some("missing")), &h.key, None, None)
			.await
			.unwrap();
		let stored = h.storage.get_completion(&response.id).unwrap().unwrap();
		assert_eq!(stored.previous_completion_id, None);
		let upstream = h.upstream.lock().unwrap();
		assert_eq!(upstream.bodies.len(), 1);
		assert_eq!(
			upstream.bodies[0]["messages"],
			serde_json::json!([{"role":"user", "content":"hello"}])
		);
	}

	#[tokio::test]
	async fn broken_previous_completion_chain_fails_before_upstream() {
		let h = ChainHarness::new("deepseek", "DEEPSEEK_API_URL", "DEEPSEEK_API_KEY").await;
		let (prior, _) = h
			.engine
			.process(request("chain", None), &h.key, None, None)
			.await
			.unwrap();
		let mut broken = h.storage.get_completion(&prior.id).unwrap().unwrap();
		broken.id = "broken".into();
		broken.previous_completion_id = Some("missing".into());
		h.storage.store_completion(&broken).unwrap();
		h.upstream.lock().unwrap().requests.clear();

		let result = h
			.engine
			.process(request("chain", Some("broken")), &h.key, None, None)
			.await;
		assert!(
			h.upstream.lock().unwrap().requests.is_empty(),
			"a broken chain must never reach the provider"
		);
		let Err(error) = result else {
			panic!("a broken chain must fail the request");
		};
		assert_eq!(error.to_string(), "loading the chain of previous_completion_id broken");
		assert!(format!("{error:#}").contains("Chain broken"));
	}

	#[tokio::test]
	async fn known_previous_completion_chain_replays_before_new_input() {
		let h = ChainHarness::new("moonshot", "MOONSHOT_API_URL", "MOONSHOT_API_KEY").await;
		let mut first = request("chain", None);
		first.input = Input::Text("prior question".into());
		let (prior, _) = h.engine.process(first, &h.key, None, None).await.unwrap();
		let mut next = request("chain", Some(&prior.id));
		next.input = Input::Text("follow-up".into());
		let (response, _) = h.engine.process(next, &h.key, None, None).await.unwrap();
		let stored = h.storage.get_completion(&response.id).unwrap().unwrap();
		assert_eq!(stored.previous_completion_id.as_deref(), Some(prior.id.as_str()));
		let upstream = h.upstream.lock().unwrap();
		assert_eq!(upstream.bodies.len(), 2);
		assert_eq!(
			upstream.bodies[1]["messages"],
			serde_json::json!([
				{"role":"user", "content":"prior question"},
				{"role":"assistant", "content":"hello"},
				{"role":"user", "content":"follow-up"}
			])
		);
	}

	#[tokio::test]
	async fn routing_dispatches_owner_keys_fails_over_and_persists_byok() {
		let (address, upstream, server) = mock_upstream().await;
		let _env = EnvGuard::set(&[
			("ZAI_API_URL", format!("http://{address}/zai")),
			("OLLAMA_API_URL", format!("http://{address}/ollama")),
			("LOCAL_API_URL", format!("http://{address}/platform")),
			("ZAI_API_KEY", "platform-zai".into()),
			("OLLAMA_API_KEY", "platform-ollama".into()),
		]);
		let config: Config = toml::from_str(
			r#"
            [server]
            api_key = "master"
            failover_on_error = false
            [models]
            face = ["local:glm-5", "ollama:glm-5", "zai:glm-5"]
            platform = ["local:glm-5"]
            [auto]
            default = "face"
        "#,
		)
		.unwrap();
		let storage = Arc::new(SqliteStorage::new(":memory:").unwrap());
		let key = storage
			.create_api_key("account key", None, Some("account"), None)
			.unwrap();
		let encryption_key = base64::engine::general_purpose::STANDARD.encode([1; 32]);
		let credentials = Arc::new(Credentials::new(storage.clone(), Some(&encryption_key)).unwrap());
		let limiter = Arc::new(RwLock::new(Arc::new(ProviderLimiter::from_config(&config))));
		let config = Arc::new(RwLock::new(Arc::new(config)));
		let engine = ProxyEngine::new(storage.clone(), config.clone(), limiter, 1, credentials.clone());
		let (response, _) = engine.process(request("face", None), &key, None, None).await.unwrap();
		assert!(response.usage.byok.is_none());
		assert!(serde_json::to_value(&response.usage).unwrap().get("byok").is_none());
		assert!(!upstream.lock().unwrap().requests[0].1.contains("owner-private"));
		upstream.lock().unwrap().requests.clear();
		let (response, _) = engine
			.process(request("zai:glm-5", None), &key, None, None)
			.await
			.unwrap();
		assert!(response.usage.byok.is_none());
		assert_eq!(
			upstream.lock().unwrap().requests,
			[("/zai".into(), "Bearer platform-zai".into())]
		);
		for provider in ["zai", "ollama"] {
			credentials
				.update(
					"account",
					credentials
						.put("account", provider, &format!("owner-private-{provider}-key"))
						.unwrap(),
				)
				.unwrap();
		}
		upstream.lock().unwrap().requests.clear();
		let (response, _) = engine
			.process(request("zai:glm-5", None), &key, None, None)
			.await
			.unwrap();
		assert_eq!(response.usage.byok, Some(true));
		assert_eq!(
			storage.get_completion(&response.id).unwrap().unwrap().usage["byok"],
			true
		);
		assert_eq!(
			upstream.lock().unwrap().requests,
			[("/zai".into(), "Bearer owner-private-zai-key".into())]
		);
		for status in [401, 429, 503] {
			{
				let mut state = upstream.lock().unwrap();
				state.requests.clear();
				state.statuses.insert("/zai".into(), status);
			}
			let error = engine
				.process(request("zai:glm-5", None), &key, None, None)
				.await
				.err()
				.unwrap();
			assert_eq!(error.downcast_ref::<ByokFailure>().unwrap().status.as_u16(), status);
			assert_eq!(
				upstream.lock().unwrap().requests,
				[("/zai".into(), "Bearer owner-private-zai-key".into())]
			);
		}
		// Overrides must work without a platform environment key.
		std::env::remove_var("ZAI_API_KEY");
		std::env::remove_var("OLLAMA_API_KEY");
		let mut previous = None;
		for status in [401, 403, 429, 500, 503] {
			{
				let mut state = upstream.lock().unwrap();
				state.requests.clear();
				state.statuses.insert("/zai".into(), status);
			}
			let logs = CapturedLogs::default();
			let writer = logs.clone();
			let subscriber = tracing_subscriber::fmt()
				.with_max_level(tracing::Level::DEBUG)
				.with_ansi(false)
				.without_time()
				.with_writer(move || writer.clone())
				.finish();
			let (response, _) = engine
				.process(request("auto", previous.as_deref()), &key, None, None)
				.with_subscriber(subscriber)
				.await
				.unwrap();
			let captured = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
			if matches!(status, 429 | 500..=599) {
				assert!(
					captured.contains("API request attempt 1 failed"),
					"retry log was not captured: {captured}"
				);
				assert!(
					captured.contains("[redacted]"),
					"retry log did not show redaction: {captured}"
				);
			}
			for secret in ["owner-private-zai-key", "owner-private-ollama-key"] {
				assert!(!captured.contains(secret), "credential leaked in captured logs");
			}
			assert_eq!(response.usage.byok, Some(true));
			assert!(response.usage.cost.is_some());
			let stored = storage.get_completion(&response.id).unwrap().unwrap();
			assert_eq!(stored.usage["byok"], true);
			assert_eq!(stored.usage["cost"], serde_json::json!(response.usage.cost));
			assert!(!format!("{stored:?}").contains("owner-private"));
			previous = Some(response.id.clone());
			let chat = ChatCompletionResponse::from(response);
			assert_eq!(serde_json::to_value(chat.usage).unwrap()["byok"], true);
			assert_eq!(
				upstream.lock().unwrap().requests,
				[
					("/zai".into(), "Bearer owner-private-zai-key".into()),
					("/ollama".into(), "Bearer owner-private-ollama-key".into())
				]
			);
			let rows = storage.owner_credentials("account").unwrap();
			assert_eq!(rows[0].last_error.as_ref().unwrap().status, status);
			assert!(rows[0].last_error.as_ref().unwrap().message.chars().count() <= 300);
			assert!(!rows[0].last_error.as_ref().unwrap().message.contains("owner-private"));
			assert!(rows[1].last_used_at.is_some());
		}
		// Sticky replay and platform health/rate settings must not reorder the owner lanes.
		engine
			.provider_health
			.record_failure("zai", std::time::Duration::from_secs(100));
		engine
			.provider_health
			.record_failure("zai", std::time::Duration::from_secs(100));
		engine
			.provider_health
			.record_failure("zai", std::time::Duration::from_secs(100));
		{
			let mut state = upstream.lock().unwrap();
			state.requests.clear();
			state.statuses.insert("/zai".into(), 200);
		}
		engine
			.process(request("face", previous.as_deref()), &key, None, None)
			.await
			.unwrap();
		assert_eq!(upstream.lock().unwrap().requests[0].0, "/zai");
		for status in [401, 429, 503] {
			{
				let mut state = upstream.lock().unwrap();
				state.requests.clear();
				state.statuses.insert("/zai".into(), 401);
				state.statuses.insert("/ollama".into(), status);
			}
			let error = engine
				.process(request("face", None), &key, None, None)
				.await
				.err()
				.unwrap();
			let failure = error.downcast_ref::<ByokFailure>().unwrap();
			assert_eq!(failure.status.as_u16(), status);
			assert!(error.to_string().starts_with("Your Ollama Cloud key failed: "));
			assert!(!format!("{error:?}").contains("owner-private"));
			assert_eq!(upstream.lock().unwrap().requests.len(), 2);
			assert!(upstream
				.lock()
				.unwrap()
				.requests
				.iter()
				.all(|(path, _)| path != "/platform"));
		}
		upstream.lock().unwrap().statuses.insert("/zai".into(), 400);
		upstream.lock().unwrap().requests.clear();
		assert!(engine.process(request("face", None), &key, None, None).await.is_err());
		assert_eq!(upstream.lock().unwrap().requests.len(), 1);
		credentials
			.update("account", CredentialChange::Order(vec!["ollama".into(), "zai".into()]))
			.unwrap();
		upstream.lock().unwrap().statuses.clear();
		upstream.lock().unwrap().requests.clear();
		engine.process(request("face", None), &key, None, None).await.unwrap();
		assert_eq!(upstream.lock().unwrap().requests[0].0, "/ollama");
		let (response, _) = engine
			.process(request("platform", None), &key, None, None)
			.await
			.unwrap();
		assert!(response.usage.byok.is_none(), "uncovered faces retain platform routing");
		let other_key = storage
			.create_api_key("other owner", None, Some("other"), None)
			.unwrap();
		let (response, _) = engine
			.process(request("platform", None), &other_key, None, None)
			.await
			.unwrap();
		assert!(response.usage.byok.is_none());
		let restricted_key = storage
			.create_api_key("denied", Some(&[]), Some("account"), None)
			.unwrap();
		assert!(engine
			.process(request("face", None), &restricted_key, None, None)
			.await
			.is_err());
		credentials
			.update(
				"account",
				CredentialChange::Put {
					provider: "zai".into(),
					secret: "corrupt".into(),
					key_hint: "...hint".into(),
				},
			)
			.unwrap();
		upstream.lock().unwrap().requests.clear();
		let error = engine
			.process(request("face", None), &key, None, None)
			.await
			.err()
			.unwrap();
		assert!(error.to_string().contains("Cannot decrypt owner credential"));
		assert!(upstream.lock().unwrap().requests.is_empty());
		credentials.update("account", CredentialChange::Clear).unwrap();
		let (response, _) = engine
			.process(request("platform", None), &key, None, None)
			.await
			.unwrap();
		assert!(response.usage.byok.is_none());
		server.abort();
	}

	#[tokio::test]
	async fn ordered_alias_prefers_its_first_lane_fails_over_and_keeps_the_session() {
		use http_body_util::{BodyExt, Full};
		use hyper::service::service_fn;
		use hyper_util::rt::TokioIo;
		let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
		let address = listener.local_addr().unwrap();
		let upstream = Arc::new(Mutex::new(Upstream::default()));
		let state = upstream.clone();
		let server = tokio::spawn(async move {
			loop {
				let (socket, _) = listener.accept().await.unwrap();
				let state = state.clone();
				tokio::spawn(async move {
					hyper::server::conn::http1::Builder::new()
						.serve_connection(
							TokioIo::new(socket),
							service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
								let state = state.clone();
								async move {
									let path = req.uri().path().to_owned();
									req.collect().await.unwrap();
									let status = {
										let mut state = state.lock().unwrap();
										state.requests.push((path.clone(), String::new()));
										state.statuses.get(&path).copied().unwrap_or(200)
									};
									let body = if status == 200 {
										serde_json::json!({"id":"mock", "object":"chat.completion", "created":1_700_000_000, "model":"m", "choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
											"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"cost":0.01}})
									} else {
										serde_json::json!({"error":{"message":"rejected"}})
									};
									Ok::<_, std::convert::Infallible>(
										hyper::Response::builder()
											.status(status)
											.header("content-type", "application/json")
											.body(Full::new(bytes::Bytes::from(body.to_string())))
											.unwrap(),
									)
								}
							}),
						)
						.await
						.unwrap();
				});
			}
		});
		// groq/fireworks: providers no other engine test points at a mock, so the
		// process-wide env this sets cannot race a parallel test.
		let _env = EnvGuard::set(&[
			("GROQ_API_URL", format!("http://{address}/primary")),
			("FIREWORKS_API_URL", format!("http://{address}/fallback")),
			("GROQ_API_KEY", "primary-key".into()),
			("FIREWORKS_API_KEY", "fallback-key".into()),
		]);
		let config: Config = toml::from_str(
			r#"
            [server]
            api_key = "master"
            failover_on_error = false
            [models]
            face = ["groq:m", "fireworks:m"]
            [routing.models]
            face = "ordered"
        "#,
		)
		.unwrap();
		let storage = Arc::new(SqliteStorage::new(":memory:").unwrap());
		let key = storage.create_api_key("key", None, None, None).unwrap();
		let encryption_key = base64::engine::general_purpose::STANDARD.encode([1; 32]);
		let credentials = Arc::new(Credentials::new(storage.clone(), Some(&encryption_key)).unwrap());
		let limiter = Arc::new(RwLock::new(Arc::new(ProviderLimiter::from_config(&config))));
		let config = Arc::new(RwLock::new(Arc::new(config)));
		let engine = ProxyEngine::new(storage, config, limiter, 1, credentials);
		let paths = || -> Vec<String> {
			let mut state = upstream.lock().unwrap();
			state.requests.drain(..).map(|(path, _)| path).collect()
		};

		// New conversations always start on the first lane; rotation would put
		// one of eight on the fallback with probability 1 - 2^-8.
		for _ in 0..8 {
			engine.process(request("face", None), &key, None, None).await.unwrap();
		}
		assert_eq!(paths(), vec!["/primary"; 8]);

		// The primary cannot serve (revoked key): the same call moves on, with
		// failover_on_error off.
		upstream.lock().unwrap().statuses.insert("/primary".into(), 401);
		let (moved, _) = engine.process(request("face", None), &key, None, None).await.unwrap();
		assert_eq!(paths(), vec!["/primary", "/fallback"]);

		// Once the primary recovers, the moved conversation stays where its cache
		// is, and a new one goes back to the primary.
		upstream.lock().unwrap().statuses.clear();
		engine
			.process(request("face", Some(&moved.id)), &key, None, None)
			.await
			.unwrap();
		assert_eq!(paths(), vec!["/fallback"]);
		engine.process(request("face", None), &key, None, None).await.unwrap();
		assert_eq!(paths(), vec!["/primary"]);

		// A 400 is the request's own fault: no lane would take it, so no failover.
		upstream.lock().unwrap().statuses.insert("/primary".into(), 400);
		assert!(engine.process(request("face", None), &key, None, None).await.is_err());
		assert_eq!(paths(), vec!["/primary"]);
		server.abort();
	}

	#[test]
	fn captured_output_items_become_a_stored_only_item() {
		let captured =
			serde_json::json!({"provider": "anthropic", "items": [{"type": "thinking", "signature": "sig"}]});
		let exchange_response = serde_json::json!({"output_items": captured});
		assert_eq!(
			stored_provider_items(&exchange_response),
			Some(serde_json::json!({
				"type": "provider_items",
				"provider": "anthropic",
				"items": [{"type": "thinking", "signature": "sig"}],
			}))
		);
		assert_eq!(stored_provider_items(&serde_json::json!({})), None);
		assert_eq!(stored_provider_items(&serde_json::json!({"output_items": null})), None);
	}

	#[tokio::test]
	async fn chain_rebuild_hands_stored_provider_items_back_on_the_assistant_message() {
		// Building the harness is enough: reconstruct_output never calls the provider.
		let h = ChainHarness::new("together", "TOGETHER_API_URL", "TOGETHER_API_KEY").await;
		let items = serde_json::json!([{"type": "thinking", "thinking": "hmm", "signature": "sig"}, {"type": "text", "text": "hello"}]);
		let stored_output = serde_json::json!([
			{"type": "reasoning", "id": "rsn_1", "content": [{"type": "output_text", "text": "hmm"}]},
			{"type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
			{"type": "provider_items", "provider": "anthropic", "items": items},
		]);
		let mut messages = Vec::new();
		h.engine.reconstruct_output(&stored_output, &mut messages);
		assert_eq!(messages.len(), 1);
		assert_eq!(messages[0].content, "hello");
		assert!(messages[0].thinking.is_some());
		assert_eq!(
			messages[0].provider_items,
			Some(serde_json::json!({"provider": "anthropic", "items": items}))
		);
	}

	#[tokio::test]
	async fn completion_without_captured_output_items_stores_no_provider_items() {
		let h = ChainHarness::new("nvidia", "NVIDIA_API_URL", "NVIDIA_API_KEY").await;
		let (response, _) = h
			.engine
			.process(request("chain", None), &h.key, None, None)
			.await
			.unwrap();
		let stored = h.storage.get_completion(&response.id).unwrap().unwrap();
		assert_eq!(stored.output, serde_json::to_value(&response.output).unwrap());
		assert!(stored
			.output
			.as_array()
			.unwrap()
			.iter()
			.all(|item| item["type"] != PROVIDER_ITEMS_TYPE));
	}
}
