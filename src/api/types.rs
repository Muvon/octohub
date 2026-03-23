use serde::{Deserialize, Serialize};

/// Request to create a completion (OpenAI Responses API compatible)
#[derive(Debug, Deserialize)]
pub struct CreateCompletionRequest {
    /// Model identifier in "provider:model" format (e.g., "openai:gpt-4o")
    pub model: String,

    /// Input content - either a simple string or array of input items
    pub input: Input,

    /// System instructions (optional, sent as system message)
    #[serde(default)]
    pub instructions: Option<String>,

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
    /// A conversation message
    #[serde(rename = "message")]
    Message { role: String, content: String },
    /// Output from a function call (tool result)
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
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
        assert_eq!(req.instructions.as_deref(), Some("You are helpful"));
        assert_eq!(req.temperature, 0.7);
        if let Input::Items(items) = &req.input {
            assert_eq!(items.len(), 1);
            assert!(
                matches!(&items[0], InputItem::Message { role, content } if role == "user" && content == "What is Rust?")
            );
        } else {
            panic!("Expected Items input");
        }
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
