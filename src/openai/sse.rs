use reqwest::Response;

/// A simple SSE (Server-Sent Events) stream parser.
///
/// Parses chunked HTTP responses into individual SSE events,
/// following the `data: ...` format used by OpenAI's streaming API.
pub struct SseStream {
  response: Response,
  buffer: String,
}

/// Tries to extract the next complete SSE event from the buffer.
///
/// Returns `Some(Some(data))` when an event with data is found,
/// `Some(None)` when `[DONE]` is encountered, or `None` when more data is needed.
fn try_extract_event(buffer: &mut String) -> Option<Option<String>> {
  loop {
    let pos = buffer.find("\n\n")?;
    let event_block = buffer[..pos].to_string();
    *buffer = buffer[pos + 2..].to_string();

    for line in event_block.lines() {
      if let Some(data) = line.strip_prefix("data: ") {
        let data = data.trim();
        if data == "[DONE]" {
          return Some(None);
        }
        if !data.is_empty() {
          return Some(Some(data.to_string()));
        }
      }
    }
    // No data line found in this block, try next event in buffer
  }
}

/// Parses any remaining data in the buffer when the stream ends without a trailing `\n\n`.
fn parse_remaining(buffer: &str) -> Option<String> {
  for line in buffer.lines() {
    if let Some(data) = line.strip_prefix("data: ") {
      let data = data.trim();
      if data == "[DONE]" || data.is_empty() {
        return None;
      }
      return Some(data.to_string());
    }
  }
  None
}

impl SseStream {
  pub fn new(response: Response) -> Self {
    Self {
      response,
      buffer: String::new(),
    }
  }

  /// Returns the next SSE event data as a string.
  ///
  /// Returns `Ok(Some(data))` for each event, `Ok(None)` when the stream
  /// ends (either via `[DONE]` sentinel or EOF), or `Err` on failure.
  pub async fn next_event(&mut self) -> Result<Option<String>, String> {
    loop {
      if let Some(result) = try_extract_event(&mut self.buffer) {
        return Ok(result);
      }

      match self.response.chunk().await {
        Ok(Some(chunk)) => {
          self.buffer.push_str(&String::from_utf8_lossy(&chunk));
        }
        Ok(None) => {
          if !self.buffer.is_empty() {
            let remaining = std::mem::take(&mut self.buffer);
            return Ok(parse_remaining(&remaining));
          }
          return Ok(None);
        }
        Err(err) => return Err(format!("{err}")),
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_extract_single_event() {
    let mut buf = "data: {\"id\":\"1\"}\n\n".to_string();
    let result = try_extract_event(&mut buf);
    assert_eq!(result, Some(Some("{\"id\":\"1\"}".to_string())));
    assert!(buf.is_empty());
  }

  #[test]
  fn test_extract_done_event() {
    let mut buf = "data: [DONE]\n\n".to_string();
    let result = try_extract_event(&mut buf);
    assert_eq!(result, Some(None));
  }

  #[test]
  fn test_extract_incomplete_buffer() {
    let mut buf = "data: partial".to_string();
    let result = try_extract_event(&mut buf);
    assert_eq!(result, None);
    assert_eq!(buf, "data: partial");
  }

  #[test]
  fn test_extract_multiple_events() {
    let mut buf = "data: first\n\ndata: second\n\n".to_string();

    let r1 = try_extract_event(&mut buf);
    assert_eq!(r1, Some(Some("first".to_string())));

    let r2 = try_extract_event(&mut buf);
    assert_eq!(r2, Some(Some("second".to_string())));

    let r3 = try_extract_event(&mut buf);
    assert_eq!(r3, None);
  }

  #[test]
  fn test_extract_skips_empty_data() {
    let mut buf = "data: \n\ndata: real\n\n".to_string();
    let result = try_extract_event(&mut buf);
    assert_eq!(result, Some(Some("real".to_string())));
  }

  #[test]
  fn test_extract_skips_non_data_lines() {
    let mut buf = "event: message\nid: 1\n\ndata: actual\n\n".to_string();
    let result = try_extract_event(&mut buf);
    assert_eq!(result, Some(Some("actual".to_string())));
  }

  #[test]
  fn test_extract_preserves_remaining_buffer() {
    let mut buf = "data: first\n\ndata: leftover".to_string();
    let result = try_extract_event(&mut buf);
    assert_eq!(result, Some(Some("first".to_string())));
    assert_eq!(buf, "data: leftover");
  }

  #[test]
  fn test_parse_remaining_with_data() {
    let result = parse_remaining("data: final chunk");
    assert_eq!(result, Some("final chunk".to_string()));
  }

  #[test]
  fn test_parse_remaining_with_done() {
    let result = parse_remaining("data: [DONE]");
    assert_eq!(result, None);
  }

  #[test]
  fn test_parse_remaining_empty() {
    let result = parse_remaining("");
    assert_eq!(result, None);
  }

  #[test]
  fn test_parse_remaining_no_data_prefix() {
    let result = parse_remaining("event: something");
    assert_eq!(result, None);
  }

  #[test]
  fn test_parse_remaining_empty_data() {
    let result = parse_remaining("data: ");
    assert_eq!(result, None);
  }
}
