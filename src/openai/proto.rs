use serde::{Deserialize, Serialize};
use serde_json::Value;

// ----------------------------
// Types sent to the server
// ----------------------------

#[derive(Clone, Debug, Serialize)]
pub struct ChatCompletionRequest {
  pub model: String,
  pub messages: Vec<Message>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub tools: Vec<ToolDef>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub stream_options: Option<StreamOptions>,
  pub stream: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct StreamOptions {
  pub include_usage: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolDef {
  pub r#type: &'static str,
  pub function: FunctionToolDef,
}

#[derive(Clone, Debug, Serialize)]
pub struct FunctionToolDef {
  pub name: String,
  pub description: String,
  pub parameters: Value,
}

// ----------------------------
// Message types (sent & received)
// ----------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
  System {
    content: String,
  },
  User {
    content: String,
  },
  Assistant {
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
  },
  Tool {
    tool_call_id: String,
    content: String,
  },
}

// ----------------------------
// Types received from the server
// ----------------------------

#[derive(Clone, Debug, Deserialize)]
pub struct ChatCompletionChunk {
  #[allow(dead_code)]
  pub id: String,
  pub choices: Vec<Choice>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Choice {
  pub delta: Delta,
  pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Delta {
  pub content: Option<String>,
  pub tool_calls: Option<Vec<ToolCall>>,
  pub reasoning_content: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub index: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub id: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub r#type: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub function: Option<FunctionToolCall>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionToolCall {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub name: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub arguments: Option<String>,
}
