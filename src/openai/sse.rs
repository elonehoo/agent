use reqwest::Response;

/// A simple SSE (Server-Sent Events) stream parser.
///
/// Parses chunked HTTP responses into individual SSE events,
/// following the `data: ...` format used by OpenAI's streaming API.
pub struct SseStream {
  response: Response,
  buffer: String,
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
      // Try to extract a complete event from the buffer
      if let Some(pos) = self.buffer.find("\n\n") {
        let event_block = self.buffer[..pos].to_string();
        self.buffer = self.buffer[pos + 2..].to_string();

        // Parse data lines from the event block
        for line in event_block.lines() {
          if let Some(data) = line.strip_prefix("data: ") {
            let data = data.trim();
            if data == "[DONE]" {
              return Ok(None);
            }
            if !data.is_empty() {
              return Ok(Some(data.to_string()));
            }
          }
        }
        // No data line found in this block, try next
        continue;
      }

      // Need more data from the HTTP response
      match self.response.chunk().await {
        Ok(Some(chunk)) => {
          self.buffer.push_str(&String::from_utf8_lossy(&chunk));
        }
        Ok(None) => {
          // Stream ended - process remaining buffer
          if !self.buffer.is_empty() {
            let remaining = std::mem::take(&mut self.buffer);
            for line in remaining.lines() {
              if let Some(data) = line.strip_prefix("data: ") {
                let data = data.trim();
                if data == "[DONE]" {
                  return Ok(None);
                }
                if !data.is_empty() {
                  return Ok(Some(data.to_string()));
                }
              }
            }
          }
          return Ok(None);
        }
        Err(err) => return Err(format!("{err}")),
      }
    }
  }
}
