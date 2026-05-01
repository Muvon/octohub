use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use octolib::embedding::{create_embedding_provider_from_parts, InputType};
use octolib::llm::{
    ChatCompletionParams, FunctionDefinition, Message, ProviderFactory, ThinkingBlock,
};
use uuid::Uuid;

use crate::api::types::*;
use crate::config::Config;
use crate::storage::{Storage, StoredCompletion, StoredEmbedding};

/// Core proxy engine that processes requests through octolib providers
pub struct ProxyEngine {
    storage: Arc<dyn Storage>,
    config: Arc<Config>,
}

impl ProxyEngine {
    pub fn new(storage: Arc<dyn Storage>, config: Arc<Config>) -> Self {
        Self { storage, config }
    }

    /// Process a create completion request, attributing usage to the given API key
    pub async fn process(
        &self,
        req: CreateCompletionRequest,
        api_key_id: i64,
    ) -> Result<CreateCompletionResponse> {
        // 1. Build conversation history from chain
        let mut messages: Vec<Message> = Vec::new();

        // Live system message from req.instructions has priority over chain-stored
        // instructions because the request carries the current cache_control marker.
        let mut system_msg: Option<Message> = req.instructions.as_ref().map(content_to_system);

        if let Some(ref prev_cmpl_id) = req.previous_completion_id {
            let chain = self
                .storage
                .walk_chain(prev_cmpl_id)
                .with_context(|| format!("Failed to walk chain from '{}'", prev_cmpl_id))?;

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
                for item in items {
                    match item {
                        InputItem::Message { role, content } => {
                            messages.push(content_to_message(role, content));
                        }
                        InputItem::FunctionCallOutput { call_id, output } => {
                            messages.push(Message::tool(output, call_id, "function"));
                        }
                    }
                }
            }
        }

        // 4. Resolve provider and model via config
        let (provider_name, resolved_model) = self
            .config
            .resolve_model(&req.model)
            .with_context(|| format!("Failed to resolve model '{}'", req.model))?;

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
                    cache_control: None,
                })
                .collect::<Vec<_>>()
        });

        let mut params = ChatCompletionParams::new(
            &messages,
            &resolved_model,
            req.temperature,
            1.0,
            50,
            req.max_output_tokens,
        );

        if let Some(tools) = tools {
            params.tools = Some(tools);
        }

        // 7. Call provider
        let provider_response = provider
            .chat_completion(params)
            .await
            .with_context(|| format!("Provider '{}' chat_completion failed", provider.name()))?;

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

        let response = CreateCompletionResponse {
            id: completion_id.clone(),
            object: "completion",
            model: resolved_model.clone(),
            output: output.clone(),
            usage: usage.clone(),
            created_at: now,
        };

        // 9. Store for observability
        let session_id = if let Some(ref prev_id) = req.previous_completion_id {
            self.storage
                .get_session_id(prev_id)?
                .unwrap_or_else(|| format!("sess_{}", Uuid::new_v4().simple()))
        } else {
            format!("sess_{}", Uuid::new_v4().simple())
        };

        let stored = StoredCompletion {
            id: completion_id,
            api_key_id,
            session_id,
            previous_completion_id: req.previous_completion_id.clone(),
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
        self.storage.store_completion(&stored)?;

        Ok(response)
    }

    /// Process an embedding request, attributing usage to the given API key
    pub async fn process_embedding(
        &self,
        req: CreateEmbeddingRequest,
        api_key_id: i64,
    ) -> Result<CreateEmbeddingResponse> {
        let start = std::time::Instant::now();

        // 1. Resolve provider and model
        let (provider_name, resolved_model) = self
            .config
            .resolve_embedding_model(&req.model)
            .with_context(|| format!("Failed to resolve embedding model '{}'", req.model))?;

        // 2. Parse provider type and create provider
        let provider_model = format!("{}:{}", provider_name, resolved_model);
        let (provider_type, model_name) =
            octolib::embedding::parse_provider_model(&provider_model)?;
        let provider = create_embedding_provider_from_parts(&provider_type, &model_name).await?;

        // 3. Generate embeddings
        let texts: Vec<String> = match &req.input {
            EmbeddingInput::Single(s) => vec![s.clone()],
            EmbeddingInput::Batch(v) => v.clone(),
        };
        let embeddings = provider
            .generate_embeddings_batch(texts.clone(), InputType::None)
            .await
            .with_context(|| {
                format!(
                    "Embedding provider '{}' failed for model '{}'",
                    provider_name, resolved_model
                )
            })?;

        let elapsed_ms = start.elapsed().as_millis() as u64;

        // 4. Build response (simple: just the embedding(s))
        let response = match &req.input {
            EmbeddingInput::Single(_) => {
                CreateEmbeddingResponse::Single(embeddings.into_iter().next().unwrap_or_default())
            }
            EmbeddingInput::Batch(_) => CreateEmbeddingResponse::Batch(embeddings),
        };

        // Approximate token count from input text length (rough: 1 token ≈ 4 chars)
        let approx_tokens = texts.iter().map(|t| t.len() as u64).sum::<u64>() / 4;
        let usage = serde_json::json!({
            "input_tokens": approx_tokens,
            "total_tokens": approx_tokens,
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
            api_key_id,
            input_model: req.model.clone(),
            resolved_model,
            provider: provider_name,
            input: serde_json::to_value(&req.input).unwrap_or(serde_json::Value::Null),
            usage: usage.clone(),
            created_at: now,
        };
        self.storage.store_embedding(&stored)?;

        Ok(response)
    }

    /// Reconstruct input messages from stored JSON
    fn reconstruct_input(&self, input: &serde_json::Value, messages: &mut Vec<Message>) {
        // Input can be a string or array of items
        if let Some(text) = input.as_str() {
            messages.push(Message::user(text));
        } else if let Some(items) = input.as_array() {
            push_items(items, messages);
        }
        // Handle {"Text": "..."} from serde serialization of Input::Text
        else if let Some(text) = input.get("Text").and_then(|v| v.as_str()) {
            messages.push(Message::user(text));
        }
        // Handle {"Items": [...]} from serde serialization of Input::Items
        else if let Some(items) = input.get("Items").and_then(|v| v.as_array()) {
            push_items(items, messages);
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
/// from either `ContentValue::Text` or `ContentValue::Parts` and propagating
/// any `cache_control` markers as `cached`/`cache_ttl` flags.
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
        msg.cache_ttl = content.cache_ttl();
    }
    msg
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

/// Deserialize each stored input item and append it to the message list.
/// Silently skips items that fail to parse (forward-compat with new types).
fn push_items(items: &[serde_json::Value], messages: &mut Vec<Message>) {
    for item in items {
        let Ok(input_item) = serde_json::from_value::<InputItem>(item.clone()) else {
            continue;
        };
        match input_item {
            InputItem::Message { role, content } => {
                messages.push(content_to_message(&role, &content));
            }
            InputItem::FunctionCallOutput { call_id, output } => {
                messages.push(Message::tool(&output, &call_id, "function"));
            }
        }
    }
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
