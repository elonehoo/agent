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

#[cfg(test)]
mod tests {
  use super::*;
  use serde_json::json;

  #[test]
  fn test_serialize_system_message() {
    let msg = Message::System {
      content: "Hello".into(),
    };
    let json = serde_json::to_value(&msg).unwrap();
    assert_eq!(json["role"], "system");
    assert_eq!(json["content"], "Hello");
  }

  #[test]
  fn test_serialize_user_message() {
    let msg = Message::User {
      content: "Hi".into(),
    };
    let json = serde_json::to_value(&msg).unwrap();
    assert_eq!(json["role"], "user");
    assert_eq!(json["content"], "Hi");
  }

  #[test]
  fn test_serialize_assistant_message_with_content() {
    let msg = Message::Assistant {
      content: Some("Response".into()),
      tool_calls: None,
      reasoning_content: None,
    };
    let json = serde_json::to_value(&msg).unwrap();
    assert_eq!(json["role"], "assistant");
    assert_eq!(json["content"], "Response");
    assert!(json.get("tool_calls").is_none());
    assert!(json.get("reasoning_content").is_none());
  }

  #[test]
  fn test_serialize_assistant_message_with_tool_calls() {
    let msg = Message::Assistant {
      content: None,
      tool_calls: Some(vec![ToolCall {
        index: Some(0),
        id: Some("call_123".into()),
        r#type: Some("function".into()),
        function: Some(FunctionToolCall {
          name: Some("shell".into()),
          arguments: Some(r#"{"cmdline":"ls"}"#.into()),
        }),
      }]),
      reasoning_content: None,
    };
    let json = serde_json::to_value(&msg).unwrap();
    assert_eq!(json["role"], "assistant");
    assert!(json["content"].is_null());
    let tc = &json["tool_calls"][0];
    assert_eq!(tc["id"], "call_123");
    assert_eq!(tc["function"]["name"], "shell");
  }

  #[test]
  fn test_serialize_tool_message() {
    let msg = Message::Tool {
      tool_call_id: "call_123".into(),
      content: "result".into(),
    };
    let json = serde_json::to_value(&msg).unwrap();
    assert_eq!(json["role"], "tool");
    assert_eq!(json["tool_call_id"], "call_123");
    assert_eq!(json["content"], "result");
  }

  #[test]
  fn test_deserialize_messages_roundtrip() {
    let messages = vec![
      Message::System {
        content: "sys".into(),
      },
      Message::User {
        content: "user".into(),
      },
      Message::Assistant {
        content: Some("asst".into()),
        tool_calls: None,
        reasoning_content: Some("thinking...".into()),
      },
      Message::Tool {
        tool_call_id: "id1".into(),
        content: "ok".into(),
      },
    ];
    for msg in messages {
      let json = serde_json::to_string(&msg).unwrap();
      let deserialized: Message = serde_json::from_str(&json).unwrap();
      let json2 = serde_json::to_string(&deserialized).unwrap();
      assert_eq!(json, json2);
    }
  }

  #[test]
  fn test_deserialize_chat_completion_chunk() {
    let json = json!({
      "id": "chatcmpl-123",
      "choices": [{
        "delta": {
          "content": "Hello",
          "tool_calls": null,
          "reasoning_content": null
        },
        "finish_reason": null
      }]
    });
    let chunk: ChatCompletionChunk = serde_json::from_value(json).unwrap();
    assert_eq!(chunk.id, "chatcmpl-123");
    assert_eq!(chunk.choices.len(), 1);
    assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hello"));
    assert!(chunk.choices[0].finish_reason.is_none());
  }

  #[test]
  fn test_deserialize_chunk_with_finish_reason() {
    let json = json!({
      "id": "chatcmpl-456",
      "choices": [{
        "delta": {},
        "finish_reason": "stop"
      }]
    });
    let chunk: ChatCompletionChunk = serde_json::from_value(json).unwrap();
    assert_eq!(
      chunk.choices[0].finish_reason.as_deref(),
      Some("stop")
    );
  }

  #[test]
  fn test_deserialize_chunk_with_tool_calls() {
    let json = json!({
      "id": "chatcmpl-789",
      "choices": [{
        "delta": {
          "tool_calls": [{
            "index": 0,
            "id": "call_abc",
            "type": "function",
            "function": {
              "name": "shell",
              "arguments": "{\"cmdline\": \"ls\"}"
            }
          }]
        },
        "finish_reason": null
      }]
    });
    let chunk: ChatCompletionChunk = serde_json::from_value(json).unwrap();
    let tc = &chunk.choices[0].delta.tool_calls.as_ref().unwrap()[0];
    assert_eq!(tc.index, Some(0));
    assert_eq!(tc.id.as_deref(), Some("call_abc"));
    assert_eq!(
      tc.function.as_ref().unwrap().name.as_deref(),
      Some("shell")
    );
  }

  #[test]
  fn test_serialize_chat_completion_request() {
    let req = ChatCompletionRequest {
      model: "gpt-4o".into(),
      messages: vec![Message::User {
        content: "test".into(),
      }],
      tools: vec![],
      stream_options: Some(StreamOptions {
        include_usage: true,
      }),
      stream: true,
    };
    let json = serde_json::to_value(&req).unwrap();
    assert_eq!(json["model"], "gpt-4o");
    assert_eq!(json["stream"], true);
    // empty vec is skipped via skip_serializing_if
    assert!(json.get("tools").is_none());
    assert_eq!(json["messages"][0]["role"], "user");
  }

  #[test]
  fn test_serialize_request_with_tools() {
    let req = ChatCompletionRequest {
      model: "gpt-4o".into(),
      messages: vec![],
      tools: vec![ToolDef {
        r#type: "function",
        function: FunctionToolDef {
          name: "shell".into(),
          description: "Run a command".into(),
          parameters: json!({"type": "object"}),
        },
      }],
      stream_options: None,
      stream: true,
    };
    let json = serde_json::to_value(&req).unwrap();
    assert_eq!(json["tools"][0]["type"], "function");
    assert_eq!(json["tools"][0]["function"]["name"], "shell");
    assert!(json.get("stream_options").is_none());
  }
}
