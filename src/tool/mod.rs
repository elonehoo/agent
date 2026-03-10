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
