//! A tool for finding files using glob patterns.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use schemars::{schema_for, JsonSchema};
use serde::Deserialize;
use serde_json::Value;
use tokio::task::spawn_blocking;

use super::{Tool, ToolResult};

#[derive(Deserialize, JsonSchema)]
struct GlobParams {
  #[schemars(description = "The glob pattern, must be relative to `path`.")]
  pattern: String,
  #[schemars(description = "Absolute path to search in.")]
  path: String,
}

/// A tool for finding files and directories using glob patterns.
///
/// Supports standard glob syntax like `*`, `?`, and `**` for recursive searches.
/// Returns up to 50 matching paths.
pub struct GlobTool {
  schema: Value,
}

impl GlobTool {
  pub fn new() -> Self {
    Self {
      schema: schema_for!(GlobParams).to_value(),
    }
  }
}

impl Default for GlobTool {
  fn default() -> Self {
    Self::new()
  }
}

impl Tool for GlobTool {
  fn name(&self) -> &str {
    "glob"
  }

  fn description(&self) -> &str {
    r#"
Find files and directories using glob patterns.
This tool supports standard glob syntax like *, ?, and ** for recursive searches."#
  }

  fn parameter_schema(&self) -> &Value {
    &self.schema
  }

  fn execute(&self, input: Value) -> Pin<Box<dyn Future<Output = ToolResult> + Send>> {
    Box::pin(async move {
      let params: GlobParams =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {e}"))?;

      if Path::new(&params.pattern).is_absolute() {
        return Err("`pattern` must be relative to `path`".to_string());
      }
      if !Path::new(&params.path).is_absolute() {
        return Err("`path` must be absolute".to_string());
      }

      let mut full_pattern = params.path;
      if !full_pattern.ends_with('/') {
        full_pattern.push('/');
      }
      full_pattern.push_str(&params.pattern);

      let pattern = glob::glob(&full_pattern).map_err(|e| format!("{e}"))?;

      spawn_blocking(move || {
        let mut result = String::new();
        for item in pattern.take(50).flatten() {
          result.push_str(&item.to_string_lossy());
          result.push('\n');
        }
        Ok(result)
      })
      .await
      .map_err(|e| format!("{e}"))?
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[tokio::test]
  async fn test_input_validation() {
    let tool = GlobTool::new();

    // Relative path should fail
    let result = tool
      .execute(serde_json::json!({
        "pattern": "*.rs",
        "path": "some/relative/path"
      }))
      .await;
    assert!(result.is_err());

    // Absolute pattern should fail
    let result = tool
      .execute(serde_json::json!({
        "pattern": "/*.*",
        "path": "/some/path"
      }))
      .await;
    assert!(result.is_err());

    // Valid pattern should succeed
    let result = tool
      .execute(serde_json::json!({
        "pattern": "*",
        "path": "/"
      }))
      .await;
    assert!(!result.unwrap().is_empty());
  }
}
