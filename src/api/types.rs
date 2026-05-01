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

    /// Maximum output tokens (0 = provider default)
    #[serde(default)]
    pub max_output_tokens: u32,

    /// Tool definitions for function calling
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
}

fn default_temperature() -> f32 {
    1.0
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
/// Per Responses API spec: `type` is "text" (instructions) or "input_text" (user content).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPartInput {
    #[serde(rename = "type")]
    pub part_type: String,
    pub text: String,
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
    pub fn text(&self) -> String {
        match self {
            ContentValue::Text(s) => s.clone(),
            ContentValue::Parts(parts) => parts.iter().map(|p| p.text.as_str()).collect::<String>(),
        }
    }

    /// True when any part carries a `cache_control` marker.
    pub fn is_cached(&self) -> bool {
        matches!(self, ContentValue::Parts(parts) if parts.iter().any(|p| p.cache_control.is_some()))
    }

    /// First `cache_control.ttl` value found, if any (e.g. "1h").
    #[cfg(test)]
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
}

/// Completion response object
#[derive(Debug, Serialize)]
pub struct CreateCompletionResponse {
    /// Unique completion ID
    pub id: String,
    /// Object type (always "completion")
    pub object: &'static str,
    /// Model used
    pub model: String,
    /// Output items
    pub output: Vec<OutputItem>,
    /// Token usage
    pub usage: Usage,
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
            created_at: 1700000000,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["id"], "cmpl_001");
        assert_eq!(json["object"], "completion");
        assert_eq!(json["output"][0]["type"], "message");
        assert_eq!(json["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(json["output"][0]["content"][0]["text"], "Hello!");
    }

    #[test]
    fn test_serialize_completion_function_call() {
        let resp = CreateCompletionResponse {
            id: "cmpl_002".to_string(),
            object: "completion",
            model: "gpt-4o".to_string(),
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
            created_at: 1700000000,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["output"][0]["type"], "function_call");
        assert_eq!(json["output"][0]["name"], "get_weather");
        assert_eq!(json["output"][0]["call_id"], "call_abc");
    }
}
