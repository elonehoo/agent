//! Agent core logic.
//!
//! Implements the agent loop: receive user input → send to model →
//! execute tool calls → send results back → repeat until done.

use std::sync::Arc;
use std::time::Duration;

use backoff::backoff::Backoff;
use backoff::ExponentialBackoffBuilder;
use tokio::sync::mpsc;

use crate::openai::proto::{Message, ToolDef};
use crate::openai::{FinishReason, OpenAIClient};
use crate::tool::ToolManager;

type OnIdleFn = Arc<dyn Fn() + Send + Sync>;
type OnErrorFn = Arc<dyn Fn(String) + Send + Sync>;
type OnTranscriptFn = Arc<dyn Fn(&str, TranscriptSource) + Send + Sync>;

/// Where the transcript comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscriptSource {
  /// User input message.
  User,
  /// Assistant response message.
  Assistant,
}

/// Builder for constructing an [`Agent`].
pub struct AgentBuilder {
  client: OpenAIClient,
  tool_manager: ToolManager,
  system_prompt: Option<String>,
  on_idle: Option<OnIdleFn>,
  on_error: Option<OnErrorFn>,
  on_transcript: Option<OnTranscriptFn>,
}

impl AgentBuilder {
  /// Creates a new builder with the specified OpenAI client.
  pub fn new(client: OpenAIClient) -> Self {
    Self {
      client,
      tool_manager: ToolManager::new(),
      system_prompt: None,
      on_idle: None,
      on_error: None,
      on_transcript: None,
    }
  }

  /// Sets the system prompt for the agent.
  pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
    self.system_prompt = Some(prompt.into());
    self
  }

  /// Attaches a callback invoked when the agent becomes idle.
  pub fn on_idle(mut self, f: impl Fn() + Send + Sync + 'static) -> Self {
    self.on_idle = Some(Arc::new(f));
    self
  }

  /// Attaches a callback invoked when an error occurs.
  pub fn on_error(mut self, f: impl Fn(String) + Send + Sync + 'static) -> Self {
    self.on_error = Some(Arc::new(f));
    self
  }

  /// Attaches a callback invoked when a transcript event is generated.
  pub fn on_transcript(
    mut self,
    f: impl Fn(&str, TranscriptSource) + Send + Sync + 'static,
  ) -> Self {
    self.on_transcript = Some(Arc::new(f));
    self
  }

  /// Registers a built-in tool.
  pub fn with_tool(mut self, tool: impl crate::tool::Tool) -> Self {
    self.tool_manager.register(tool);
    self
  }

  /// Builds the agent and spawns the background processing task.
  pub fn build(self) -> Agent {
    let (tx, rx) = mpsc::unbounded_channel();

    let mut conversation = Vec::new();
    if let Some(prompt) = &self.system_prompt {
      conversation.push(Message::System {
        content: prompt.clone(),
      });
    }

    let tool_defs = self.tool_manager.definitions();

    let ctx = AgentContext {
      client: self.client,
      tool_manager: self.tool_manager,
      conversation,
      tool_defs,
      on_idle: self.on_idle,
      on_error: self.on_error,
      on_transcript: self.on_transcript,
    };

    tokio::spawn(agent_task(rx, ctx));

    Agent { tx }
  }
}

/// Internal state for the agent task.
struct AgentContext {
  client: OpenAIClient,
  tool_manager: ToolManager,
  conversation: Vec<Message>,
  tool_defs: Vec<ToolDef>,
  on_idle: Option<OnIdleFn>,
  on_error: Option<OnErrorFn>,
  on_transcript: Option<OnTranscriptFn>,
}

/// An agent instance.
///
/// The agent runs as a background tokio task. Sending messages
/// through this handle enqueues them for processing.
#[derive(Clone)]
pub struct Agent {
  tx: mpsc::UnboundedSender<String>,
}

impl Agent {
  /// Sends a message to the agent for processing.
  pub fn send_message(&self, message: &str) {
    self.tx.send(message.to_owned()).ok();
  }
}

/// The main agent task loop.
///
/// Receives user input from the channel, processes the full
/// agent loop (model → tools → model → ...), then waits for
/// the next input.
async fn agent_task(mut rx: mpsc::UnboundedReceiver<String>, mut ctx: AgentContext) {
  while let Some(input) = rx.recv().await {
    // Notify transcript for user input
    if let Some(on_transcript) = &ctx.on_transcript {
      on_transcript(&input, TranscriptSource::User);
    }

    // Add user message to conversation
    ctx.conversation.push(Message::User { content: input });

    // Run the agent loop (model request → tool calls → repeat)
    run_agent_loop(&mut ctx).await;

    // Notify idle
    if let Some(on_idle) = &ctx.on_idle {
      on_idle();
    }
  }
}

/// Runs the inner agent loop: sends requests to the model,
/// executes tool calls, and loops until the model stops requesting tools.
async fn run_agent_loop(ctx: &mut AgentContext) {
  let mut retry_backoff = ExponentialBackoffBuilder::default()
    .with_max_interval(Duration::from_secs(5 * 60))
    .with_max_elapsed_time(Some(Duration::from_secs(30 * 60)))
    .build();

  loop {
    // Clone the transcript callback for the model request
    let on_transcript = ctx.on_transcript.clone();
    let on_transcript_fn = move |text: String| {
      if let Some(on_transcript) = &on_transcript {
        on_transcript(&text, TranscriptSource::Assistant);
      }
    };

    let result = ctx
      .client
      .send_request(&ctx.conversation, &ctx.tool_defs, on_transcript_fn)
      .await;

    match result {
      Ok(response) => {
        retry_backoff.reset();

        // Add assistant message to conversation history
        ctx.conversation.push(response.raw_message);

        // Check if the model wants to call tools
        if response.finish_reason == FinishReason::ToolCalls && !response.tool_calls.is_empty() {
          // Execute all tool calls concurrently
          let mut handles = Vec::new();
          for tool_call in &response.tool_calls {
            let id = tool_call.id.clone();
            let name = tool_call.name.clone();
            if let Some(fut) = ctx.tool_manager.execute(&name, tool_call.arguments.clone()) {
              handles.push(tokio::spawn(async move {
                let result = fut.await;
                (id, result)
              }));
            } else {
              let id_clone = id;
              handles.push(tokio::spawn(async move {
                (id_clone, Err(format!("Tool '{name}' not found")))
              }));
            }
          }

          // Collect all tool results and add to conversation
          for handle in handles {
            if let Ok((id, result)) = handle.await {
              let content = match &result {
                Ok(s) => s.clone(),
                Err(e) => format!("Error: {e}"),
              };

              ctx.conversation.push(Message::Tool {
                tool_call_id: id,
                content,
              });
            }
          }

          // Continue the loop — send tool results back to model
          continue;
        }

        // No more tool calls — done with this turn
        break;
      }
      Err(err) => {
        if let Some(on_error) = &ctx.on_error {
          on_error(err);
        }

        // Retry with exponential backoff
        if let Some(timeout) = retry_backoff.next_backoff() {
          tokio::time::sleep(timeout).await;
          continue;
        }

        // Max retries exceeded — give up
        break;
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_transcript_source_equality() {
    assert_eq!(TranscriptSource::User, TranscriptSource::User);
    assert_eq!(TranscriptSource::Assistant, TranscriptSource::Assistant);
    assert_ne!(TranscriptSource::User, TranscriptSource::Assistant);
  }

  #[test]
  fn test_transcript_source_debug() {
    assert_eq!(format!("{:?}", TranscriptSource::User), "User");
    assert_eq!(format!("{:?}", TranscriptSource::Assistant), "Assistant");
  }

  #[tokio::test]
  async fn test_agent_builder_and_send() {
    let config = crate::openai::OpenAIConfigBuilder::with_api_key("sk-test").build();
    let client = crate::openai::OpenAIClient::new(config);

    let agent = AgentBuilder::new(client)
      .with_system_prompt("You are helpful.")
      .build();

    // Agent should accept messages without panicking
    agent.send_message("Hello");
  }

  #[tokio::test]
  async fn test_agent_builder_with_callbacks() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let idle_called = Arc::new(AtomicBool::new(false));
    let error_called = Arc::new(AtomicBool::new(false));
    let transcript_called = Arc::new(AtomicBool::new(false));

    let config = crate::openai::OpenAIConfigBuilder::with_api_key("sk-test").build();
    let client = crate::openai::OpenAIClient::new(config);

    let _idle = idle_called.clone();
    let _error = error_called.clone();
    let _transcript = transcript_called.clone();

    // Just verify the builder accepts all callbacks without panicking
    let _agent = AgentBuilder::new(client)
      .with_system_prompt("Test")
      .on_idle(move || {
        _idle.store(true, Ordering::SeqCst);
      })
      .on_error(move |_err| {
        _error.store(true, Ordering::SeqCst);
      })
      .on_transcript(move |_text, _source| {
        _transcript.store(true, Ordering::SeqCst);
      })
      .with_tool(crate::tool::shell::ShellTool::new())
      .build();
  }
}
