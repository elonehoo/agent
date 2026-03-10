//! Tool call support.
//!
//! Provides the [`Tool`] trait for defining tools that the model can call,
//! and a [`ToolManager`] for managing tool registration and execution.

pub mod glob_tool;
pub mod read_file;
pub mod shell;

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::openai::proto::{FunctionToolDef, ToolDef};

/// The result of a tool execution.
pub type ToolResult = Result<String, String>;

/// A boxed future that returns a tool result.
type BoxFuture = Pin<Box<dyn Future<Output = ToolResult> + Send>>;

/// A tool that can be called by the model.
///
/// Implementations should be stateless — they may hold configuration
/// but should not maintain mutable internal state.
pub trait Tool: Send + Sync + 'static {
  /// Returns the name of the tool.
  fn name(&self) -> &str;

  /// Returns the description of the tool.
  fn description(&self) -> &str;

  /// Returns the JSON Schema for the tool's parameters.
  fn parameter_schema(&self) -> &Value;

  /// Executes the tool with the given JSON arguments.
  ///
  /// The arguments are a JSON value matching the parameter schema.
  /// Returns a boxed future resolving to the tool result.
  fn execute(&self, input: Value) -> BoxFuture;
}

/// Manages a set of tools and provides tool definitions for the model API.
pub struct ToolManager {
  tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolManager {
  pub fn new() -> Self {
    Self {
      tools: HashMap::new(),
    }
  }

  /// Registers a tool.
  pub fn register(&mut self, tool: impl Tool) {
    let name = tool.name().to_owned();
    self.tools.insert(name, Box::new(tool));
  }

  /// Returns tool definitions for the OpenAI API.
  pub fn definitions(&self) -> Vec<ToolDef> {
    self
      .tools
      .values()
      .map(|t| ToolDef {
        r#type: "function",
        function: FunctionToolDef {
          name: t.name().to_owned(),
          description: t.description().to_owned(),
          parameters: t.parameter_schema().clone(),
        },
      })
      .collect()
  }

  /// Executes a tool by name with the given arguments.
  ///
  /// Returns `None` if the tool is not found.
  pub fn execute(&self, name: &str, arguments: Value) -> Option<BoxFuture> {
    self.tools.get(name).map(|tool| tool.execute(arguments))
  }
}

impl Default for ToolManager {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  struct MockTool {
    schema: Value,
  }

  impl MockTool {
    fn new() -> Self {
      Self {
        schema: serde_json::json!({
          "type": "object",
          "properties": { "input": { "type": "string" } }
        }),
      }
    }
  }

  impl Tool for MockTool {
    fn name(&self) -> &str {
      "mock_tool"
    }
    fn description(&self) -> &str {
      "A mock tool for testing"
    }
    fn parameter_schema(&self) -> &Value {
      &self.schema
    }
    fn execute(&self, input: Value) -> BoxFuture {
      Box::pin(async move {
        let input_str = input
          .get("input")
          .and_then(|v| v.as_str())
          .unwrap_or("none");
        Ok(format!("mock result: {input_str}"))
      })
    }
  }

  #[test]
  fn test_new_manager_is_empty() {
    let mgr = ToolManager::new();
    assert!(mgr.definitions().is_empty());
  }

  #[test]
  fn test_register_and_definitions() {
    let mut mgr = ToolManager::new();
    mgr.register(MockTool::new());

    let defs = mgr.definitions();
    assert_eq!(defs.len(), 1);
    assert_eq!(defs[0].r#type, "function");
    assert_eq!(defs[0].function.name, "mock_tool");
    assert_eq!(defs[0].function.description, "A mock tool for testing");
  }

  #[tokio::test]
  async fn test_execute_found() {
    let mut mgr = ToolManager::new();
    mgr.register(MockTool::new());

    let fut = mgr.execute("mock_tool", serde_json::json!({"input": "hello"}));
    assert!(fut.is_some());
    let result = fut.unwrap().await;
    assert_eq!(result.unwrap(), "mock result: hello");
  }

  #[test]
  fn test_execute_not_found() {
    let mgr = ToolManager::new();
    assert!(mgr.execute("nonexistent", serde_json::json!({})).is_none());
  }

  #[test]
  fn test_register_overwrites_same_name() {
    let mut mgr = ToolManager::new();
    mgr.register(MockTool::new());
    mgr.register(MockTool::new());
    assert_eq!(mgr.definitions().len(), 1);
  }

  #[test]
  fn test_default_is_empty() {
    let mgr = ToolManager::default();
    assert!(mgr.definitions().is_empty());
  }
}
