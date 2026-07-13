use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use octolib::embedding::{create_embedding_provider_from_parts, InputType};
use octolib::llm::{
    chat_completion_enforced, ChatCompletionParams, FunctionDefinition, ImageAttachment, ImageData,
    Message, OutputFormat, ProviderFactory, ReasoningEffort, ResponseMode, SourceType,
    StructuredOutputRequest, ThinkingBlock, VideoAttachment, VideoData,
};
use uuid::Uuid;

use crate::api::types::*;
use crate::config::Config;
use crate::proxy::limiter::{OwnerLimiter, ProviderLimiter, ProviderRateTracker, OWNER_QUEUE_WAIT};
use crate::storage::{ApiKey, Storage, StoredCompletion, StoredEmbedding};

/// Marker added to model-restriction errors so the HTTP layer can map them
/// to `403 Forbidden` instead of the generic 400/500.
pub const MODEL_FORBIDDEN_MARKER: &str = "model_not_allowed_for_key";
/// Marker for owner-budget exhaustion — the handler maps it to HTTP 429.
pub const OWNER_LIMIT_MARKER: &str = "owner_concurrency_exhausted";

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

/// Core proxy engine that processes requests through octolib providers.
///
/// `config` and `limiter` are live handles swapped on SIGHUP reload (see
/// `main`), so each request reads a fresh snapshot instead of a value frozen
/// at startup.
pub struct ProxyEngine {
    storage: Arc<dyn Storage>,
    config: crate::Live<Config>,
    limiter: crate::Live<ProviderLimiter>,
    /// Per-owner in-flight budgets (keys sharing `ApiKey::owner`). Plain Arc,
    /// not `Live`: state comes from the key rows, not config, and must survive
    /// SIGHUP so in-flight accounting is never reset by a reload.
    owner_limiter: Arc<OwnerLimiter>,
    /// Per-provider request/token windows. Plain Arc for the same reason as
    /// `owner_limiter`: a SIGHUP reload must update the LIMITS (read from the
    /// live config at admission) without zeroing the day's counters.
    rate_tracker: Arc<ProviderRateTracker>,
}

impl ProxyEngine {
    pub fn new(
        storage: Arc<dyn Storage>,
        config: crate::Live<Config>,
        limiter: crate::Live<ProviderLimiter>,
    ) -> Self {
        Self {
            storage,
            config,
            limiter,
            owner_limiter: Arc::new(OwnerLimiter::new()),
            rate_tracker: Arc::new(ProviderRateTracker::new()),
        }
    }

    /// Take a slot in the key's shared owner budget, or fail with the 429
    /// marker after `OWNER_QUEUE_WAIT`. `None` = key is ungrouped/unlimited —
    /// hold the returned permit (if any) for the whole upstream call.
    async fn acquire_owner_slot(
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
            .acquire(owner, capacity, OWNER_QUEUE_WAIT)
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
    fn limiter(&self) -> Arc<ProviderLimiter> {
        self.limiter.read().unwrap().clone()
    }

    /// Process a create completion request, attributing usage to the given API key.
    /// Returns the response plus the upstream provider call duration — the latter
    /// drives `octohub_completion_duration_seconds` so the histogram reflects
    /// provider latency only, excluding our own auth/parse/storage overhead.
    pub async fn process(
        &self,
        req: CreateCompletionRequest,
        api_key: &ApiKey,
    ) -> Result<(CreateCompletionResponse, std::time::Duration)> {
        ensure_model_allowed(api_key, &req.model)?;
        // Owner budget slot — held for the WHOLE request (queue + upstream +
        // storage), completions and embeddings drain the same budget.
        let _owner_slot = self.acquire_owner_slot(api_key).await?;

        // Snapshot config + limiter once so this request keeps a consistent view
        // even if SIGHUP swaps them mid-flight.
        let config = self.config();
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

        if let Some(ref prev_cmpl_id) = req.previous_completion_id {
            // Unknown IDs are tolerated — the client may be migrating from a stateless
            // provider (Anthropic, etc.) where they pass the full history inline in
            // `input`. Hard-failing would break that workflow. Mirrors OpenAI's
            // guidance: "retry with full input context and previous_response_id null."
            let chain_start = std::time::Instant::now();
            let storage = self.storage.clone();
            let prev_id = prev_cmpl_id.clone();
            let chain_result =
                tokio::task::spawn_blocking(move || storage.walk_chain(&prev_id)).await?;
            let chain_ms = chain_start.elapsed().as_millis() as u64;
            tracing::Span::current().record("chain_ms", chain_ms);

            let chain = match chain_result {
                Ok(chain) => {
                    resolved_prev_id = Some(prev_cmpl_id.clone());
                    chain
                }
                Err(err) => {
                    tracing::warn!(
                        previous_completion_id = %prev_cmpl_id,
                        error = %err,
                        "Unknown previous_completion_id — falling back to inline input only",
                    );
                    Vec::new()
                }
            };

            // Inherit session_id from the most recent (last) chain entry.
            inherited_session_id = chain.last().map(|c| c.session_id.clone());

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

        // 4. Resolve provider and model via config. Candidates whose rate
        //    windows are exhausted are skipped; all exhausted → 429 with
        //    Retry-After (see `pick_admitted`).
        let candidates = config
            .model_candidates(&req.model)
            .with_context(|| format!("Failed to resolve model '{}'", req.model))?;
        let (provider_name, resolved_model) =
            pick_admitted(&self.rate_tracker, &config, candidates, &req.model)?;

        // Record provider in request span
        tracing::Span::current().record("provider", provider_name.as_str());

        // 5. Get provider instance
        let provider = ProviderFactory::create_provider(&provider_name)
            .with_context(|| format!("Provider '{}' not available", provider_name))?;

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

        // Sampling: forward temperature + top_p straight from the client.
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

        if let Some(tools) = tools {
            params.tools = Some(tools);
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
        let queue_timeout =
            std::time::Duration::from_secs(config.server.provider_queue_timeout_secs);
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
        let provider_response = tokio::time::timeout(
            upstream_timeout,
            chat_completion_enforced(provider.as_ref(), params),
        )
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
        drop(_permit);

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
            input_tokens: exchange_usage.as_ref().map(|u| u.input_tokens).unwrap_or(0),
            output_tokens: exchange_usage
                .as_ref()
                .map(|u| u.output_tokens)
                .unwrap_or(0),
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
        self.rate_tracker.record_tokens(
            &provider_name,
            usage
                .total_tokens
                .max(usage.input_tokens + usage.output_tokens),
        );

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
        let session_id =
            inherited_session_id.unwrap_or_else(|| format!("sess_{}", Uuid::new_v4().simple()));

        let stored = StoredCompletion {
            id: completion_id,
            api_key_id: api_key.id,
            session_id,
            previous_completion_id: resolved_prev_id,
            input_model: req.model.clone(),
            resolved_model,
            provider: provider.name().to_string(),
            input: serde_json::to_value(&req.input).unwrap_or(serde_json::Value::Null),
            output: serde_json::to_value(&output).unwrap_or(serde_json::Value::Null),
            instructions: req.instructions.as_ref().map(|i| i.text()),
            exchange: serde_json::json!({
                "request": provider_response.exchange.request,
                "response": provider_response.exchange.response,
            }),
            usage: serde_json::to_value(&usage).unwrap_or(serde_json::Value::Null),
            created_at: now,
        };
        let store_start = std::time::Instant::now();
        let storage = self.storage.clone();
        tokio::task::spawn_blocking(move || storage.store_completion(&stored)).await??;
        tracing::Span::current().record("store_ms", store_start.elapsed().as_millis() as u64);

        Ok((response, upstream_duration))
    }

    /// Process an embedding request, attributing usage to the given API key.
    /// Returns the response, the upstream call duration (for the histogram),
    /// the resolved provider name (so callers can label metrics correctly),
    /// and the approximate input token count.
    pub async fn process_embedding(
        &self,
        req: CreateEmbeddingRequest,
        api_key: &ApiKey,
    ) -> Result<EmbeddingOutcome> {
        ensure_model_allowed(api_key, &req.model)?;
        // Same shared owner budget as completions (see `process`).
        let _owner_slot = self.acquire_owner_slot(api_key).await?;

        // Snapshot config + limiter once (see `process`).
        let config = self.config();
        let limiter = self.limiter();

        let start = std::time::Instant::now();

        // 1. Resolve provider and model — same rate-window admission as
        //    completions (both drain the same provider windows).
        let candidates = config
            .embedding_model_candidates(&req.model)
            .with_context(|| format!("Failed to resolve embedding model '{}'", req.model))?;
        let (provider_name, resolved_model) =
            pick_admitted(&self.rate_tracker, &config, candidates, &req.model)?;

        // Record provider in request span
        tracing::Span::current().record("provider", provider_name.as_str());

        // 2. Parse provider type and create provider
        let provider_model = format!("{}:{}", provider_name, resolved_model);
        let (provider_type, model_name) =
            octolib::embedding::parse_provider_model(&provider_model)?;
        let provider = create_embedding_provider_from_parts(&provider_type, &model_name).await?;

        // 3. Generate embeddings — gated by the same per-provider concurrency
        // limit as completions. Permit released as soon as the call returns.
        let texts: Vec<String> = match &req.input {
            EmbeddingInput::Single(s) => vec![s.clone()],
            EmbeddingInput::Batch(v) => v.clone(),
        };

        let queue_start = std::time::Instant::now();
        let queue_timeout =
            std::time::Duration::from_secs(config.server.provider_queue_timeout_secs);
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
        drop(_permit);

        // Embeddings drain the same provider token windows as completions.
        self.rate_tracker
            .record_tokens(&provider_name, usage_calc.input_tokens);

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

            for item in items {
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
                messages.push(msg);
            }
        }
    }
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

    msg
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
                let mut msg = Message::tool(output, call_id, "function");
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
                let args_value: serde_json::Value =
                    serde_json::from_str(arguments).unwrap_or(serde_json::json!({}));
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
/// Pick the first candidate whose provider rate windows admit a request
/// (counting it against the winner's windows). Exhausted candidates are
/// skipped; when ALL are exhausted the caller gets [`RateLimitedError`]
/// carrying the soonest retry among them.
fn pick_admitted(
    tracker: &ProviderRateTracker,
    config: &Config,
    candidates: Vec<(String, String)>,
    model: &str,
) -> Result<(String, String)> {
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

fn ensure_model_allowed(api_key: &ApiKey, model: &str) -> Result<()> {
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
            providers,
            logging: Default::default(),
            metrics: Default::default(),
        }
    }

    fn candidate(provider: &str, model: &str) -> (String, String) {
        (provider.to_string(), model.to_string())
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

        let picked = pick_admitted(
            &tracker,
            &config,
            vec![
                candidate("moonshot", "kimi-k2"),
                candidate("ollama", "kimi-k2"),
            ],
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

        let err = pick_admitted(
            &tracker,
            &config,
            vec![candidate("moonshot", "kimi-k2")],
            "kimi-k2",
        )
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
        assert!(pick_admitted(&tracker, &config, candidates.clone(), "kimi-k2").is_ok());
        assert!(
            pick_admitted(&tracker, &config, candidates, "kimi-k2").is_err(),
            "admission must have consumed the rpm=1 budget"
        );
    }

    #[test]
    fn pick_admitted_ignores_providers_without_config() {
        let tracker = ProviderRateTracker::new();
        let config = config_with_provider("moonshot", ProviderConfig::default());

        // Unlisted provider and listed-but-unlimited provider both admit.
        for provider in ["ollama", "moonshot"] {
            let picked =
                pick_admitted(&tracker, &config, vec![candidate(provider, "m")], "m").unwrap();
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
        assert!(
            messages[1].tool_calls.is_some(),
            "assistant must carry tool_calls"
        );
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
        assert!(
            !messages[1].cached,
            "unmarked tool result must not be cached"
        );
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
