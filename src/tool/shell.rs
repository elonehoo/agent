//! A tool for running shell commands.

use std::env;
use std::future::Future;
use std::pin::Pin;

use schemars::{schema_for, JsonSchema};
use serde::Deserialize;
use serde_json::Value;
use tokio::process::Command;

use super::{Tool, ToolResult};

#[derive(Deserialize, JsonSchema)]
struct ShellParams {
  #[schemars(description = "The command line to run.")]
  cmdline: String,
}

/// A tool for running shell commands.
///
/// Runs arbitrary commands using the user's shell (from `$SHELL`
/// environment variable, or `/bin/sh` as fallback).
/// Returns collected stdout and stderr as the tool output.
pub struct ShellTool {
  schema: Value,
}

impl ShellTool {
  pub fn new() -> Self {
    Self {
      schema: schema_for!(ShellParams).to_value(),
    }
  }
}

impl Default for ShellTool {
  fn default() -> Self {
    Self::new()
  }
}

impl Tool for ShellTool {
  fn name(&self) -> &str {
    "shell"
  }

  fn description(&self) -> &str {
    r#"
Runs arbitrary commands like using a terminal.
The command line should be single line if possible.
Strings collected from stdout and stderr will be returned as the tool's output."#
  }

  fn parameter_schema(&self) -> &Value {
    &self.schema
  }

  fn execute(&self, input: Value) -> Pin<Box<dyn Future<Output = ToolResult> + Send>> {
    Box::pin(async move {
      let params: ShellParams =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {e}"))?;

      let shell = env::var_os("SHELL").unwrap_or_else(|| "/bin/sh".into());

      let output = Command::new(shell)
        .arg("-cl")
        .arg(&params.cmdline)
        .output()
        .await
        .map_err(|e| format!("{e}"))?;

      let mut result = String::new();
      if !output.stdout.is_empty() {
        result.push_str("==> STDOUT <==\n");
        result.push_str(&String::from_utf8_lossy(&output.stdout));
      }
      if !output.stderr.is_empty() {
        result.push_str("\n==> STDERR <==\n");
        result.push_str(&String::from_utf8_lossy(&output.stderr));
      }
      Ok(result)
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[tokio::test]
  async fn test_run_command() {
    let tool = ShellTool::new();
    let result = tool
      .execute(serde_json::json!({ "cmdline": "echo 'Hello, World!'" }))
      .await;
    assert_eq!(result.unwrap(), "==> STDOUT <==\nHello, World!\n");
  }
}
