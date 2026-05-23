use serde::{Deserialize, Serialize};

/// Request to create a completion (OpenAI Responses API compatible)
#[derive(Debug, Deserialize)]
pub struct CreateCompletionRequest {
    /// Model identifier in "provider:model" format (e.g., "openai:gpt-4o")
    pub model: String,

    /// Input content - either a simple string or array of input items
    pub input: Input,

    /// System instructions — plain string or array of typed text parts.
    /// Array form carries optional `cache_control` markers per Responses API spec.
    #[serde(default)]
    pub instructions: Option<ContentValue>,

    /// Previous completion ID for multi-turn conversations
    #[serde(default)]
    pub previous_completion_id: Option<String>,

    /// Sampling temperature (0.0 to 2.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p nucleus sampling (0.0 to 1.0). Proxied straight to the upstream
    /// provider so client-controlled sampling actually takes effect.
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Maximum output tokens (0 = provider default)
    #[serde(default)]
    pub max_output_tokens: u32,

    /// Reasoning effort hint for thinking-capable models. Accepted values:
    /// "low" | "medium" | "high" | "xhigh" | "max". Unknown values are
    /// ignored (provider default applies). Proxied through unchanged.
    #[serde(default)]
    pub reasoning_effort: Option<String>,

    /// Structured output format request (Responses API `text` field).
    /// Carries either `{format: {type: "json_object"}}` or a JSON schema.
    #[serde(default)]
    pub text: Option<TextConfig>,

    /// Tool definitions for function calling
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

/// Responses-API `text` configuration object — only `format` is meaningful
/// to the proxy today; future fields (e.g. verbosity) can be added without
/// breaking the wire shape.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub format: TextFormat,
}

/// Output format selector. `json_object` requests free-form JSON; `json_schema`
/// pins the response to a schema and optionally enforces strict adherence.
///
/// Note: the wire shape also carries a `name` string (octolib hardcodes
/// "response_schema") — we accept and ignore it via `#[serde(default)]` on
/// any extra fields, since octolib's `StructuredOutputRequest` has no name
/// slot and nothing downstream uses it.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextFormat {
    /// Free-form JSON output ("type": "json_object")
    JsonObject,
    /// JSON conforming to a schema ("type": "json_schema")
    JsonSchema {
        schema: serde_json::Value,
        #[serde(default)]
        strict: bool,
    },
}

/// Input can be a simple string or an array of typed items
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Input {
    /// Simple text input (becomes a user message)
    Text(String),
    /// Array of typed input items
    Items(Vec<InputItem>),
}

/// Individual input item in an array input
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputItem {
    /// A conversation message. `content` is either a plain string or
    /// an array of typed parts carrying optional `cache_control`.
    #[serde(rename = "message")]
    Message { role: String, content: ContentValue },
    /// Output from a function call (tool result)
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
    /// A prior assistant tool call replayed by the client (used when migrating
    /// mid-conversation from a stateless provider that recorded tool_calls).
    /// Mirrors `OutputItem::FunctionCall` so clients can copy items verbatim.
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        /// JSON-encoded arguments string (same shape as OutputItem::FunctionCall).
        arguments: String,
    },
    /// A prior assistant reasoning/thinking block replayed by the client.
    /// Required by DeepSeek when the prior assistant turn produced tool_calls.
    /// Mirrors `OutputItem::Reasoning`.
    #[serde(rename = "reasoning")]
    Reasoning { content: Vec<ContentPart> },
}

/// A structured content part used in message content or instructions.
/// Per Responses API spec: `type` is "text"/"input_text" (text), "input_image"
/// (image), or "input_video" (video). For image/video parts, `image_url` or
/// `video_url` carries either an `https://` URL or a `data:<mime>;base64,...`
/// data URI; `text` is omitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPartInput {
    #[serde(rename = "type")]
    pub part_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<serde_json::Value>,
}

/// Either a plain string or an array of structured content parts.
/// Used for both message `content` and system `instructions`. Array form
/// allows clients to attach `cache_control` markers (ephemeral caching).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContentValue {
    Text(String),
    Parts(Vec<ContentPartInput>),
}

impl ContentValue {
    /// Concatenated text payload (parts joined without separator).
    /// Non-text parts (image/video) contribute nothing here.
    pub fn text(&self) -> String {
        match self {
            ContentValue::Text(s) => s.clone(),
            ContentValue::Parts(parts) => parts
                .iter()
                .filter_map(|p| p.text.as_deref())
                .collect::<String>(),
        }
    }

    /// `image_url` values from any `input_image` parts, in order.
    /// Each is either an `https://` URL or a `data:<mime>;base64,...` data URI.
    pub fn image_urls(&self) -> Vec<String> {
        match self {
            ContentValue::Parts(parts) => parts
                .iter()
                .filter(|p| p.part_type == "input_image")
                .filter_map(|p| p.image_url.clone())
                .collect(),
            ContentValue::Text(_) => Vec::new(),
        }
    }

    /// `video_url` values from any `input_video` parts, in order.
    pub fn video_urls(&self) -> Vec<String> {
        match self {
            ContentValue::Parts(parts) => parts
                .iter()
                .filter(|p| p.part_type == "input_video")
                .filter_map(|p| p.video_url.clone())
                .collect(),
            ContentValue::Text(_) => Vec::new(),
        }
    }

    /// True when any part carries a `cache_control` marker.
    pub fn is_cached(&self) -> bool {
        matches!(self, ContentValue::Parts(parts) if parts.iter().any(|p| p.cache_control.is_some()))
    }

    /// First `cache_control.ttl` value found, if any (e.g. "1h").
    pub fn cache_ttl(&self) -> Option<String> {
        match self {
            ContentValue::Parts(parts) => parts.iter().find_map(|p| {
                p.cache_control
                    .as_ref()
                    .and_then(|cc| cc.get("ttl"))
                    .and_then(|v| v.as_str())
                    .map(String::from)
            }),
            ContentValue::Text(_) => None,
        }
    }
}

/// Tool definition for function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool type (always "function" for now)
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function name
    pub name: String,
    /// Function description
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema for parameters
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
    /// Optional ephemeral cache marker. Clients attach this to the last tool
    /// to anchor a cache boundary covering the entire tools section. We are a
    /// proxy: pass it through to the upstream provider unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<serde_json::Value>,
}

/// Completion response object.
///
/// Mirrors `octolib::llm::ProviderResponse`. Fields that vary per request
/// (`structured_output`, `finish_reason`) are skipped from serialization
/// when absent so the wire shape stays compact. The proxy is a transparent
/// pass-through: every meaningful field returned by the upstream provider
/// is surfaced here verbatim — the client must not have to re-derive
/// anything from `output` text.
#[derive(Debug, Serialize)]
pub struct CreateCompletionResponse {
    /// Unique completion ID
    pub id: String,
    /// Object type (always "completion")
    pub object: &'static str,
    /// Model used
    pub model: String,
    /// Provider name that handled this request
    pub provider: String,
    /// Output items
    pub output: Vec<OutputItem>,
    /// Token usage
    pub usage: Usage,
    /// Parsed structured output when the request attached a JSON schema and
    /// the upstream provider produced schema-conforming JSON. Mirrored
    /// straight from `ProviderResponse.structured_output` — clients SHOULD
    /// prefer this over re-parsing `output[].content[].text`, since the
    /// upstream may have validated against the schema server-side.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_output: Option<serde_json::Value>,
    /// Why the upstream model stopped generating, when the provider reports
    /// it. Values are provider-specific (`stop`, `length`, `tool_calls`,
    /// `content_filter`, etc.) — surfaced as-is, no normalisation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Unix timestamp of creation
    pub created_at: u64,
}

/// Output item in a completion
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    /// Text message output
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
        content: Vec<ContentPart>,
    },
    /// Function call output
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
    /// Reasoning / thinking output (e.g. DeepSeek R1, Claude thinking).
    /// Must be replayed to providers that require thinking continuity
    /// (DeepSeek rejects assistant turns with tool_calls if `reasoning_content`
    /// from the previous turn is missing).
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        content: Vec<ContentPart>,
    },
}

/// Content part within a message output
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content
    #[serde(rename = "output_text")]
    OutputText { text: String },
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
    /// Tokens spent on internal reasoning (DeepSeek R1, Claude thinking, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    pub total_tokens: u64,
    /// Cost in USD
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    /// Request time in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_time_ms: Option<u64>,
}

// ── Embedding types ──────────────────────────────────────────────────

/// Request to create embeddings
#[derive(Debug, Deserialize)]
pub struct CreateEmbeddingRequest {
    /// Model identifier (e.g., "voyage:voyage-3.5" or mapped name)
    pub model: String,
    /// Input text(s) to embed
    pub input: EmbeddingInput,
}

/// Embedding input: single string or array of strings
#[derive(Debug, Clone)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

impl<'de> Deserialize<'de> for EmbeddingInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::String(s) => Ok(EmbeddingInput::Single(s)),
            serde_json::Value::Array(arr) => {
                let strings: Result<Vec<String>, _> = arr
                    .into_iter()
                    .map(|v| {
                        v.as_str()
                            .map(|s| s.to_string())
                            .ok_or_else(|| serde::de::Error::custom("expected string in array"))
                    })
                    .collect();
                Ok(EmbeddingInput::Batch(strings?))
            }
            _ => Err(serde::de::Error::custom(
                "input must be a string or array of strings",
            )),
        }
    }
}

impl Serialize for EmbeddingInput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            EmbeddingInput::Single(s) => serializer.serialize_str(s),
            EmbeddingInput::Batch(v) => v.serialize(serializer),
        }
    }
}

/// Response from embedding creation
/// Returns just the embedding(s):
/// - Single input: `[0.1, 0.2, ...]`
/// - Batch input: `[[0.1, ...], [0.2, ...]]`
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum CreateEmbeddingResponse {
    Single(Vec<f32>),
    Batch(Vec<Vec<f32>>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_text_input() {
        let json = r#"{
            "model": "openai:gpt-4o",
            "input": "Hello, world!"
        }"#;
        let req: CreateCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "openai:gpt-4o");
        assert!(matches!(req.input, Input::Text(ref s) if s == "Hello, world!"));
        assert!(req.previous_completion_id.is_none());
        assert_eq!(req.temperature, 1.0);
    }

    #[test]
    fn test_deserialize_array_input_with_messages() {
        let json = r#"{
            "model": "openai:gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": "What is Rust?"}
            ],
            "instructions": "You are helpful",
            "temperature": 0.7
        }"#;
        let req: CreateCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            req.instructions.as_ref().map(|i| i.text()).as_deref(),
            Some("You are helpful")
        );
        assert!(!req.instructions.as_ref().unwrap().is_cached());
        assert_eq!(req.temperature, 0.7);
        if let Input::Items(items) = &req.input {
            assert_eq!(items.len(), 1);
            match &items[0] {
                InputItem::Message { role, content } => {
                    assert_eq!(role, "user");
                    assert_eq!(content.text(), "What is Rust?");
                    assert!(!content.is_cached());
                }
                _ => panic!("expected Message"),
            }
        } else {
            panic!("Expected Items input");
        }
    }

    #[test]
    fn test_deserialize_cached_instructions_array() {
        // Reproduces the failure mode reported by octolib's OctoHub provider:
        // when system prompt is cached, instructions arrive as an array of
        // {type, text, cache_control} parts. Must deserialize cleanly.
        let json = r#"{
            "model": "octohub:glm-5.1",
            "input": "hihi",
            "instructions": [
                {"type": "text", "text": "You are an assistant.", "cache_control": {"type": "ephemeral"}}
            ]
        }"#;
        let req: CreateCompletionRequest = serde_json::from_str(json).unwrap();
        let instr = req.instructions.expect("instructions present");
        assert_eq!(instr.text(), "You are an assistant.");
        assert!(instr.is_cached());
        assert_eq!(instr.cache_ttl(), None);
    }

    #[test]
    fn test_deserialize_cached_message_content_array() {
        // Mirror failure mode for user message content: array of input_text parts.
        let json = r#"{
            "model": "octohub:glm-5.1",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Hello", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
                    ]
                }
            ]
        }"#;
        let req: CreateCompletionRequest = serde_json::from_str(json).unwrap();
        if let Input::Items(items) = &req.input {
            match &items[0] {
                InputItem::Message { role, content } => {
                    assert_eq!(role, "user");
                    assert_eq!(content.text(), "Hello");
                    assert!(content.is_cached());
                    assert_eq!(content.cache_ttl().as_deref(), Some("1h"));
                }
                _ => panic!("expected Message"),
            }
        } else {
            panic!("Expected Items input");
        }
    }

    #[test]
    fn test_deserialize_function_call_input() {
        // Client migrating from a stateless provider replays the prior assistant
        // tool call as an input item alongside the matching function_call_output.
        let json = r#"{
    		"model": "deepseek:deepseek-chat",
    		"input": [
    			{"type": "message", "role": "user", "content": "weather in Paris?"},
    			{"type": "reasoning", "content": [{"type": "output_text", "text": "Need to call get_weather"}]},
    			{"type": "function_call", "call_id": "call_xyz", "name": "get_weather", "arguments": "{\"location\":\"Paris\"}"},
    			{"type": "function_call_output", "call_id": "call_xyz", "output": "18°C cloudy"},
    			{"type": "message", "role": "user", "content": "and London?"}
    		]
    	}"#;
        let req: CreateCompletionRequest = serde_json::from_str(json).unwrap();
        let Input::Items(items) = &req.input else {
            panic!("Expected Items input");
        };
        assert_eq!(items.len(), 5);
        assert!(matches!(&items[1], InputItem::Reasoning { .. }));
        assert!(
            matches!(&items[2], InputItem::FunctionCall { call_id, name, arguments } if call_id == "call_xyz" && name == "get_weather" && arguments.contains("Paris"))
        );
    }

    #[test]
    fn test_deserialize_function_call_output_input() {
        let json = r#"{
            "model": "openai:gpt-4o",
            "input": [
                {"type": "function_call_output", "call_id": "call_abc123", "output": "72°F sunny"}
            ],
            "previous_completion_id": "cmpl_001"
        }"#;
        let req: CreateCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.previous_completion_id.as_deref(), Some("cmpl_001"));
        if let Input::Items(items) = &req.input {
            assert_eq!(items.len(), 1);
            assert!(
                matches!(&items[0], InputItem::FunctionCallOutput { call_id, output } if call_id == "call_abc123" && output == "72°F sunny")
            );
        } else {
            panic!("Expected Items input");
        }
    }

    #[test]
    fn test_deserialize_with_tools() {
        let json = r#"{
            "model": "openai:gpt-4o",
            "input": "What's the weather?",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                }
            ]
        }"#;
        let req: CreateCompletionRequest = serde_json::from_str(json).unwrap();
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
    }

    #[test]
    fn test_serialize_completion_message() {
        let resp = CreateCompletionResponse {
            id: "cmpl_001".to_string(),
            object: "completion",
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            output: vec![OutputItem::Message {
                id: "msg_001".to_string(),
                role: "assistant".to_string(),
                content: vec![ContentPart::OutputText {
                    text: "Hello!".to_string(),
                }],
            }],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_tokens: None,
                cache_write_tokens: None,
                reasoning_tokens: None,
                total_tokens: 15,
                cost: Some(0.0001),
                request_time_ms: Some(500),
            },
            structured_output: None,
            finish_reason: None,
            created_at: 1700000000,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["id"], "cmpl_001");
        assert_eq!(json["object"], "completion");
        assert_eq!(json["output"][0]["type"], "message");
        assert_eq!(json["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(json["output"][0]["content"][0]["text"], "Hello!");
        assert!(
            json.get("structured_output").is_none(),
            "absent structured_output must be skipped from serialization"
        );
        assert!(
            json.get("finish_reason").is_none(),
            "absent finish_reason must be skipped from serialization"
        );
    }

    #[test]
    fn test_serialize_completion_function_call() {
        let resp = CreateCompletionResponse {
            id: "cmpl_002".to_string(),
            object: "completion",
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            output: vec![OutputItem::FunctionCall {
                id: "fc_001".to_string(),
                call_id: "call_abc".to_string(),
                name: "get_weather".to_string(),
                arguments: r#"{"location":"NYC"}"#.to_string(),
            }],
            usage: Usage {
                input_tokens: 20,
                output_tokens: 10,
                cache_read_tokens: None,
                cache_write_tokens: None,
                reasoning_tokens: None,
                total_tokens: 30,
                cost: None,
                request_time_ms: None,
            },
            structured_output: None,
            finish_reason: Some("tool_calls".to_string()),
            created_at: 1700000000,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["output"][0]["type"], "function_call");
        assert_eq!(json["output"][0]["name"], "get_weather");
        assert_eq!(json["output"][0]["call_id"], "call_abc");
        assert_eq!(json["finish_reason"], "tool_calls");
    }

    #[test]
    fn test_serialize_completion_with_structured_output() {
        // When the upstream provider returns schema-validated JSON, the
        // proxy MUST mirror it under `structured_output` so clients can
        // consume it directly without re-parsing `output[].content[].text`.
        let resp = CreateCompletionResponse {
            id: "cmpl_003".to_string(),
            object: "completion",
            model: "gpt-5-nano".to_string(),
            provider: "openai".to_string(),
            output: vec![OutputItem::Message {
                id: "msg_003".to_string(),
                role: "assistant".to_string(),
                content: vec![ContentPart::OutputText {
                    text: r#"{"answer":42}"#.to_string(),
                }],
            }],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_tokens: None,
                cache_write_tokens: None,
                reasoning_tokens: None,
                total_tokens: 15,
                cost: None,
                request_time_ms: None,
            },
            structured_output: Some(serde_json::json!({"answer": 42})),
            finish_reason: Some("stop".to_string()),
            created_at: 1700000000,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["structured_output"]["answer"], 42);
        assert_eq!(json["finish_reason"], "stop");
    }
}
