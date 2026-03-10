//! A tool for reading file content with line numbers.

use std::fs::File;
use std::future::Future;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::pin::Pin;

use schemars::{schema_for, JsonSchema};
use serde::Deserialize;
use serde_json::Value;
use tokio::task::spawn_blocking;

use super::{Tool, ToolResult};

const MAX_LINES: usize = 50;

#[derive(Deserialize, JsonSchema)]
struct ReadFileItem {
  #[schemars(description = "Absolute path to the file.")]
  path: String,
  #[schemars(description = "1-based start line to read from, default to 1.")]
  start_line: Option<usize>,
}

#[derive(Deserialize, JsonSchema)]
struct ReadFileParams {
  #[schemars(description = "Files to read.")]
  files: Vec<ReadFileItem>,
}

/// A tool for reading file content with line numbers.
///
/// Reads files from absolute paths and returns their contents
/// prefixed with line numbers. Returns up to 50 lines per file.
pub struct ReadFileTool {
  schema: Value,
}

impl ReadFileTool {
  pub fn new() -> Self {
    Self {
      schema: schema_for!(ReadFileParams).to_value(),
    }
  }
}

impl Default for ReadFileTool {
  fn default() -> Self {
    Self::new()
  }
}

impl Tool for ReadFileTool {
  fn name(&self) -> &str {
    "read_file"
  }

  fn description(&self) -> &str {
    r#"
Reads files from absolute paths and returns their contents prefixed with line numbers.
Each file includes a path and a 1-based start line, and returns up to 50 lines."#
  }

  fn parameter_schema(&self) -> &Value {
    &self.schema
  }

  fn execute(&self, input: Value) -> Pin<Box<dyn Future<Output = ToolResult> + Send>> {
    Box::pin(async move {
      let params: ReadFileParams =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {e}"))?;

      let mut result = String::new();
      for file in params.files {
        if !Path::new(&file.path).is_absolute() {
          return Err("`path` must be absolute".to_string());
        }

        let start_line = file.start_line.unwrap_or(1);
        if start_line == 0 {
          return Err("`start_line` must be 1-based".to_string());
        }

        let section = spawn_blocking(move || read_file_section(&file.path, start_line))
          .await
          .map_err(|e| format!("{e}"))??;

        if !result.is_empty() {
          result.push('\n');
        }
        result.push_str(&section);
      }
      Ok(result)
    })
  }
}

fn read_file_section(path: &str, start_line: usize) -> ToolResult {
  let file = File::open(path).map_err(|e| format!("{e}"))?;
  let reader = BufReader::new(file);
  let mut lines = Vec::new();

  for (index, line) in reader.lines().enumerate() {
    let line_no = index + 1;
    if line_no < start_line {
      continue;
    }
    let line = line.map_err(|e| format!("{e}"))?;
    lines.push(line);
    if lines.len() >= MAX_LINES {
      break;
    }
  }

  let mut result = String::new();
  result.push_str(&format!("==> {path} <==\n"));

  if !lines.is_empty() {
    let last_line_no = start_line + lines.len() - 1;
    let width = last_line_no.to_string().len();
    for (offset, line) in lines.into_iter().enumerate() {
      let line_no = start_line + offset;
      result.push_str(&format!("{line_no:>width$}: {line}\n"));
    }
  }

  Ok(result)
}

#[cfg(test)]
mod tests {
  use std::io::Cursor;

  use super::*;

  fn format_cursor_section(path: &str, data: &[u8], start_line: usize) -> ToolResult {
    let reader = std::io::BufReader::new(Cursor::new(data));
    let mut lines = Vec::new();
    for (index, line) in reader.lines().enumerate() {
      let line_no = index + 1;
      if line_no < start_line {
        continue;
      }
      let line = line.map_err(|e| format!("{e}"))?;
      lines.push(line);
      if lines.len() >= MAX_LINES {
        break;
      }
    }
    let mut result = String::new();
    result.push_str(&format!("==> {path} <==\n"));
    if !lines.is_empty() {
      let last_line_no = start_line + lines.len() - 1;
      let width = last_line_no.to_string().len();
      for (offset, line) in lines.into_iter().enumerate() {
        let line_no = start_line + offset;
        result.push_str(&format!("{line_no:>width$}: {line}\n"));
      }
    }
    Ok(result)
  }

  #[test]
  fn test_format_lines() {
    let input = b"first\nsecond\nthird\n";
    let output = format_cursor_section("/fake/path", input, 2).unwrap();
    let mut lines = output.lines();
    assert_eq!(lines.next().unwrap(), "==> /fake/path <==");
    assert_eq!(lines.next().unwrap(), "2: second");
    assert_eq!(lines.next().unwrap(), "3: third");
  }

  #[test]
  fn test_respects_limit() {
    let mut input = Vec::new();
    for _ in 0..(MAX_LINES + 10) {
      input.extend_from_slice(b"line\n");
    }
    let output = format_cursor_section("/fake/path", &input, 1).unwrap();
    let lines = output.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), MAX_LINES + 1); // +1 for header
  }
}
