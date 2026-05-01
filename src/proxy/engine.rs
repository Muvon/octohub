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

        // Tracks whether the supplied previous_completion_id actually resolved.
        // Unknown IDs are accepted (stateless-provider migration path) but must
        // not be persisted — that would create dangling chains forever.
        let mut resolved_prev_id: Option<String> = None;

        if let Some(ref prev_cmpl_id) = req.previous_completion_id {
            // Unknown IDs are tolerated — the client may be migrating from a stateless
            // provider (Anthropic, etc.) where they pass the full history inline in
            // `input`. Hard-failing would break that workflow. Mirrors OpenAI's
            // guidance: "retry with full input context and previous_response_id null."
            let chain = match self.storage.walk_chain(prev_cmpl_id) {
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

        // 9. Store for observability. Use resolved_prev_id (None if chain didn't
        // resolve) so we never persist a link to an unknown ID.
        let session_id = if let Some(ref prev_id) = resolved_prev_id {
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
    }
    msg
}

/// Build the system message from request `instructions`, preserving cache markers.
fn content_to_system(content: &ContentValue) -> Message {
    let mut msg = Message::system(&content.text());
    if content.is_cached() {
        msg.cached = true;
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
            InputItem::FunctionCallOutput { call_id, output } => {
                messages.push(Message::tool(output, call_id, "function"));
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
fn push_stored_items(items: &[serde_json::Value], messages: &mut Vec<Message>) {
    let typed: Vec<InputItem> = items
        .iter()
        .filter_map(|v| serde_json::from_value(v.clone()).ok())
        .collect();
    push_items(&typed, messages);
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
