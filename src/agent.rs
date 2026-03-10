//! Agent core logic.
//!
//! Implements the agent loop: receive user input → send to model →
//! execute tool calls → send results back → repeat until done.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use backoff::backoff::Backoff;
use backoff::ExponentialBackoffBuilder;
use tokio::sync::mpsc;

use crate::openai::proto::{Message, ToolDef};
use crate::openai::{FinishReason, ModelResponse, OpenAIClient};
use crate::tool::ToolManager;

type OnIdleFn = Arc<dyn Fn() + Send + Sync>;
type OnErrorFn = Arc<dyn Fn(String) + Send + Sync>;
type OnTranscriptFn = Arc<dyn Fn(&str, TranscriptSource) + Send + Sync>;
type OnModelTranscriptFn = Arc<dyn Fn(String) + Send + Sync>;
type ModelRequestFuture<'a> =
  Pin<Box<dyn Future<Output = Result<ModelResponse, String>> + Send + 'a>>;

pub(crate) const DEFAULT_MAX_ITERATIONS: u32 = 8;
const DEFAULT_COMPACT_PROMPT: &str = r#"You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work."#;
const COMPACT_SUMMARY_PREFIX: &str = "Context checkpoint summary:";
const KEEP_RECENT_USER_MESSAGES: usize = 2;
const KEEP_RECENT_MESSAGES: usize = 8;
const APPROX_CHARS_PER_TOKEN: usize = 4;

trait ModelClient {
  fn send_request<'a>(
    &'a self,
    messages: &'a [Message],
    tools: &'a [ToolDef],
    on_transcript: OnModelTranscriptFn,
  ) -> ModelRequestFuture<'a>;
}

impl ModelClient for OpenAIClient {
  fn send_request<'a>(
    &'a self,
    messages: &'a [Message],
    tools: &'a [ToolDef],
    on_transcript: OnModelTranscriptFn,
  ) -> ModelRequestFuture<'a> {
    Box::pin(async move {
      OpenAIClient::send_request(self, messages, tools, move |text| on_transcript(text)).await
    })
  }
}

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
  auto_compact_token_limit: Option<u32>,
  compact_prompt: Option<String>,
  max_iterations: Option<u32>,
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
      auto_compact_token_limit: None,
      compact_prompt: None,
      max_iterations: Some(DEFAULT_MAX_ITERATIONS),
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

  /// Enables Codex-style automatic context compaction using an approximate token limit.
  pub fn with_auto_compact_token_limit(mut self, auto_compact_token_limit: u32) -> Self {
    self.auto_compact_token_limit = Some(auto_compact_token_limit);
    self
  }

  /// Overrides the prompt used to summarize history during compaction.
  pub fn with_compact_prompt(mut self, compact_prompt: impl Into<String>) -> Self {
    self.compact_prompt = Some(compact_prompt.into());
    self
  }

  /// Sets the maximum number of successful model rounds for a single user turn.
  pub fn with_max_iterations(mut self, max_iterations: u32) -> Self {
    self.max_iterations = Some(max_iterations);
    self
  }

  /// Disables the per-turn iteration limit.
  pub fn disable_iteration_limit(mut self) -> Self {
    self.max_iterations = None;
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
      auto_compact_token_limit: self.auto_compact_token_limit,
      compact_prompt: self.compact_prompt,
      max_iterations: self.max_iterations,
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
  auto_compact_token_limit: Option<u32>,
  compact_prompt: Option<String>,
  max_iterations: Option<u32>,
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
  run_agent_loop_with_client(
    &ctx.client,
    &ctx.tool_manager,
    &mut ctx.conversation,
    &ctx.tool_defs,
    ctx.on_error.clone(),
    ctx.on_transcript.clone(),
    ctx.auto_compact_token_limit,
    ctx.compact_prompt.as_deref(),
    ctx.max_iterations,
  )
  .await;
}

async fn run_agent_loop_with_client<C: ModelClient>(
  client: &C,
  tool_manager: &ToolManager,
  conversation: &mut Vec<Message>,
  tool_defs: &[ToolDef],
  on_error: Option<OnErrorFn>,
  on_transcript: Option<OnTranscriptFn>,
  auto_compact_token_limit: Option<u32>,
  compact_prompt: Option<&str>,
  max_iterations: Option<u32>,
) {
  let mut retry_backoff = ExponentialBackoffBuilder::default()
    .with_max_interval(Duration::from_secs(5 * 60))
    .with_max_elapsed_time(Some(Duration::from_secs(30 * 60)))
    .build();
  let mut successful_rounds = 0_u32;

  loop {
    maybe_auto_compact_conversation(
      client,
      conversation,
      on_error.clone(),
      auto_compact_token_limit,
      compact_prompt,
    )
    .await;

    // Clone the transcript callback for the model request
    let transcript_handler = on_transcript.clone();
    let on_transcript_fn: OnModelTranscriptFn = Arc::new(move |text: String| {
      if let Some(on_transcript) = &transcript_handler {
        on_transcript(&text, TranscriptSource::Assistant);
      }
    });

    let result = client
      .send_request(conversation, tool_defs, on_transcript_fn)
      .await;

    match result {
      Ok(response) => {
        retry_backoff.reset();
        successful_rounds += 1;

        if response.finish_reason == FinishReason::ToolCalls && !response.tool_calls.is_empty() {
          if let Some(limit) = max_iterations.filter(|limit| successful_rounds >= *limit) {
            if let Some(on_error) = &on_error {
              on_error(iteration_limit_error(limit));
            }
            break;
          }

          // Add assistant message to conversation history
          conversation.push(response.raw_message);

          // Execute all tool calls concurrently
          let mut handles = Vec::new();
          for tool_call in &response.tool_calls {
            let id = tool_call.id.clone();
            let name = tool_call.name.clone();
            if let Some(fut) = tool_manager.execute(&name, tool_call.arguments.clone()) {
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

              conversation.push(Message::Tool {
                tool_call_id: id,
                content,
              });
            }
          }

          // Continue the loop — send tool results back to model
          continue;
        }

        // No more tool calls — done with this turn
        conversation.push(response.raw_message);
        break;
      }
      Err(err) => {
        if let Some(on_error) = &on_error {
          on_error(err);
        }

        // Retry with exponential backoff
        if let Some(timeout) = retry_backoff.next_backoff() {
          sleep_for_retry(timeout).await;
          continue;
        }

        // Max retries exceeded — give up
        break;
      }
    }
  }
}

async fn maybe_auto_compact_conversation<C: ModelClient>(
  client: &C,
  conversation: &mut Vec<Message>,
  on_error: Option<OnErrorFn>,
  auto_compact_token_limit: Option<u32>,
  compact_prompt: Option<&str>,
) {
  let Some(limit) = auto_compact_token_limit.map(|limit| limit as usize) else {
    return;
  };

  if approximate_token_count(conversation) < limit {
    return;
  }

  let Some((system_message, messages_to_summarize, messages_to_keep)) =
    build_compaction_plan(conversation)
  else {
    return;
  };

  let request = vec![
    Message::System {
      content: compact_prompt.unwrap_or(DEFAULT_COMPACT_PROMPT).to_string(),
    },
    Message::User {
      content: format_compaction_input(&messages_to_summarize),
    },
  ];

  let no_transcript: OnModelTranscriptFn = Arc::new(|_text| {});
  let result = client.send_request(&request, &[], no_transcript).await;

  match result {
    Ok(response) => {
      let summary = compaction_summary_from_response(response);
      let mut compacted = Vec::new();

      if let Some(system_message) = system_message {
        compacted.push(system_message);
      }
      compacted.push(compaction_summary_message(summary));
      compacted.extend(messages_to_keep);

      *conversation = compacted;
    }
    Err(err) => {
      if let Some(on_error) = &on_error {
        on_error(format!("Automatic context compaction failed: {err}"));
      }
    }
  }
}

fn build_compaction_plan(
  conversation: &[Message],
) -> Option<(Option<Message>, Vec<Message>, Vec<Message>)> {
  let (system_message, history_start) = match conversation.first() {
    Some(Message::System { .. }) => (Some(conversation[0].clone()), 1),
    _ => (None, 0),
  };

  if conversation.len() <= history_start + 1 {
    return None;
  }

  let primary = collect_compaction_messages(
    conversation,
    history_start,
    KEEP_RECENT_USER_MESSAGES,
    KEEP_RECENT_MESSAGES,
  );
  if !primary.0.is_empty() {
    return Some((system_message, primary.0, primary.1));
  }

  let fallback = collect_compaction_messages(conversation, history_start, 1, 2);
  if !fallback.0.is_empty() {
    return Some((system_message, fallback.0, fallback.1));
  }

  None
}

fn collect_compaction_messages(
  conversation: &[Message],
  history_start: usize,
  keep_recent_user_messages: usize,
  keep_recent_messages: usize,
) -> (Vec<Message>, Vec<Message>) {
  let mut keep = vec![false; conversation.len()];
  if history_start == 1 {
    keep[0] = true;
  }

  let mut kept_user_messages = 0;
  for index in (history_start..conversation.len()).rev() {
    if matches!(conversation[index], Message::User { .. }) {
      keep[index] = true;
      kept_user_messages += 1;
      if kept_user_messages >= keep_recent_user_messages {
        break;
      }
    }
  }

  let fallback_start = conversation.len().saturating_sub(keep_recent_messages);
  for index in fallback_start.max(history_start)..conversation.len() {
    keep[index] = true;
  }

  for (index, message) in conversation.iter().enumerate().skip(history_start) {
    if is_compaction_summary_message(message) {
      keep[index] = false;
    }
  }

  let messages_to_summarize = conversation
    .iter()
    .enumerate()
    .skip(history_start)
    .filter(|(index, _)| !keep[*index])
    .map(|(_, message)| message.clone())
    .collect::<Vec<_>>();

  let messages_to_keep = conversation
    .iter()
    .enumerate()
    .skip(history_start)
    .filter(|(index, _)| keep[*index])
    .map(|(_, message)| message.clone())
    .collect::<Vec<_>>();

  (messages_to_summarize, messages_to_keep)
}

fn approximate_token_count(messages: &[Message]) -> usize {
  messages
    .iter()
    .map(|message| {
      let json = serde_json::to_string(message).unwrap_or_default();
      (json.len().saturating_add(APPROX_CHARS_PER_TOKEN - 1)) / APPROX_CHARS_PER_TOKEN
    })
    .sum()
}

fn format_compaction_input(messages: &[Message]) -> String {
  let mut sections = Vec::with_capacity(messages.len() + 1);
  sections.push(
    "Summarize the following older conversation history. More recent raw context is kept separately."
      .to_string(),
  );

  for message in messages {
    sections.push(format_compaction_message(message));
  }

  sections.join("\n\n")
}

fn format_compaction_message(message: &Message) -> String {
  match message {
    Message::System { content } => format!("system:\n{content}"),
    Message::User { content } => format!("user:\n{content}"),
    Message::Assistant {
      content,
      tool_calls,
      reasoning_content,
    } => {
      let mut parts = Vec::new();
      if let Some(content) = content {
        if !content.is_empty() {
          parts.push(format!("content:\n{content}"));
        }
      }
      if let Some(reasoning_content) = reasoning_content {
        if !reasoning_content.is_empty() {
          parts.push(format!("reasoning:\n{reasoning_content}"));
        }
      }
      if let Some(tool_calls) = tool_calls {
        for tool_call in tool_calls {
          let name = tool_call
            .function
            .as_ref()
            .and_then(|function| function.name.as_deref())
            .unwrap_or("unknown");
          let arguments = tool_call
            .function
            .as_ref()
            .and_then(|function| function.arguments.as_deref())
            .unwrap_or("{}");
          parts.push(format!("tool_call {name}:\n{arguments}"));
        }
      }
      if parts.is_empty() {
        "assistant:\n(no content)".to_string()
      } else {
        format!("assistant:\n{}", parts.join("\n\n"))
      }
    }
    Message::Tool {
      tool_call_id,
      content,
    } => format!("tool result ({tool_call_id}):\n{content}"),
  }
}

fn compaction_summary_from_response(response: ModelResponse) -> String {
  match response.raw_message {
    Message::Assistant {
      content: Some(content),
      ..
    } if !content.trim().is_empty() => content,
    _ if !response.transcript.trim().is_empty() => response.transcript,
    _ => "(no summary available)".to_string(),
  }
}

fn compaction_summary_message(summary: String) -> Message {
  Message::Assistant {
    content: Some(format!("{COMPACT_SUMMARY_PREFIX}\n{}", summary.trim())),
    tool_calls: None,
    reasoning_content: None,
  }
}

fn is_compaction_summary_message(message: &Message) -> bool {
  matches!(
    message,
    Message::Assistant {
      content: Some(content),
      tool_calls: None,
      ..
    } if content.starts_with(COMPACT_SUMMARY_PREFIX)
  )
}

fn iteration_limit_error(limit: u32) -> String {
  format!("Iteration limit exceeded for this turn (maxIterations={limit}).")
}

#[cfg(not(test))]
async fn sleep_for_retry(timeout: Duration) {
  tokio::time::sleep(timeout).await;
}

#[cfg(test)]
async fn sleep_for_retry(_timeout: Duration) {
  tokio::task::yield_now().await;
}

#[cfg(test)]
mod tests {
  use std::collections::VecDeque;
  use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
  use std::sync::Mutex;

  use serde_json::{json, Value};

  use super::*;
  use crate::openai::proto::{FunctionToolCall, ToolCall};
  use crate::openai::ToolCallRequest;
  use crate::tool::{Tool, ToolResult};

  struct MockModelClient {
    responses: Mutex<VecDeque<Result<ModelResponse, String>>>,
    requests: Mutex<Vec<(Vec<Message>, usize)>>,
  }

  impl MockModelClient {
    fn new(responses: Vec<Result<ModelResponse, String>>) -> Self {
      Self {
        responses: Mutex::new(VecDeque::from(responses)),
        requests: Mutex::new(Vec::new()),
      }
    }

    fn requests(&self) -> Vec<(Vec<Message>, usize)> {
      self
        .requests
        .lock()
        .expect("mock requests mutex poisoned")
        .clone()
    }
  }

  impl ModelClient for MockModelClient {
    fn send_request<'a>(
      &'a self,
      messages: &'a [Message],
      tools: &'a [ToolDef],
      _on_transcript: OnModelTranscriptFn,
    ) -> ModelRequestFuture<'a> {
      self
        .requests
        .lock()
        .expect("mock requests mutex poisoned")
        .push((messages.to_vec(), tools.len()));

      let response = self
        .responses
        .lock()
        .expect("mock responses mutex poisoned")
        .pop_front()
        .expect("mock response queue exhausted");

      Box::pin(async move { response })
    }
  }

  struct CountingTool {
    count: Arc<AtomicUsize>,
    schema: Value,
  }

  impl CountingTool {
    fn new(count: Arc<AtomicUsize>) -> Self {
      Self {
        count,
        schema: json!({
          "type": "object",
          "properties": {},
        }),
      }
    }
  }

  impl Tool for CountingTool {
    fn name(&self) -> &str {
      "counting_tool"
    }

    fn description(&self) -> &str {
      "Counts executions for tests."
    }

    fn parameter_schema(&self) -> &Value {
      &self.schema
    }

    fn execute(&self, _input: Value) -> Pin<Box<dyn Future<Output = ToolResult> + Send>> {
      let count = self.count.clone();
      Box::pin(async move {
        count.fetch_add(1, Ordering::SeqCst);
        Ok("counting tool result".to_string())
      })
    }
  }

  fn tool_call_response(id: &str, name: &str, arguments: Value) -> ModelResponse {
    let arguments_str = serde_json::to_string(&arguments).expect("serialize tool call args");
    ModelResponse {
      transcript: String::new(),
      raw_message: Message::Assistant {
        content: None,
        tool_calls: Some(vec![ToolCall {
          index: Some(0),
          id: Some(id.to_string()),
          r#type: Some("function".to_string()),
          function: Some(FunctionToolCall {
            name: Some(name.to_string()),
            arguments: Some(arguments_str),
          }),
        }]),
        reasoning_content: None,
      },
      tool_calls: vec![ToolCallRequest {
        id: id.to_string(),
        name: name.to_string(),
        arguments,
      }],
      finish_reason: FinishReason::ToolCalls,
    }
  }

  fn terminal_response(content: &str) -> ModelResponse {
    ModelResponse {
      transcript: content.to_string(),
      raw_message: Message::Assistant {
        content: Some(content.to_string()),
        tool_calls: None,
        reasoning_content: None,
      },
      tool_calls: Vec::new(),
      finish_reason: FinishReason::Stop,
    }
  }

  fn conversation_with_user() -> Vec<Message> {
    vec![Message::User {
      content: "hello".to_string(),
    }]
  }

  fn long_text(prefix: &str) -> String {
    format!("{prefix} {}", "detail ".repeat(80))
  }

  fn error_collector(storage: Arc<Mutex<Vec<String>>>) -> OnErrorFn {
    Arc::new(move |err| {
      storage
        .lock()
        .expect("error storage mutex poisoned")
        .push(err);
    })
  }

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

  #[tokio::test]
  async fn test_run_agent_loop_appends_terminal_response_within_budget() {
    let client = MockModelClient::new(vec![Ok(terminal_response("done"))]);
    let tool_manager = ToolManager::new();
    let tool_defs = tool_manager.definitions();
    let mut conversation = conversation_with_user();

    run_agent_loop_with_client(
      &client,
      &tool_manager,
      &mut conversation,
      &tool_defs,
      None,
      None,
      None,
      None,
      Some(1),
    )
    .await;

    assert_eq!(conversation.len(), 2);
    assert!(matches!(
      &conversation[1],
      Message::Assistant {
        content: Some(content),
        tool_calls: None,
        ..
      } if content == "done"
    ));
  }

  #[tokio::test]
  async fn test_run_agent_loop_executes_tool_calls_within_budget() {
    let tool_count = Arc::new(AtomicUsize::new(0));
    let mut tool_manager = ToolManager::new();
    tool_manager.register(CountingTool::new(tool_count.clone()));

    let client = MockModelClient::new(vec![
      Ok(tool_call_response("call_1", "counting_tool", json!({}))),
      Ok(terminal_response("all done")),
    ]);
    let tool_defs = tool_manager.definitions();
    let mut conversation = conversation_with_user();

    run_agent_loop_with_client(
      &client,
      &tool_manager,
      &mut conversation,
      &tool_defs,
      None,
      None,
      None,
      None,
      Some(2),
    )
    .await;

    assert_eq!(tool_count.load(Ordering::SeqCst), 1);
    assert_eq!(conversation.len(), 4);
    assert!(matches!(
      &conversation[1],
      Message::Assistant {
        tool_calls: Some(tool_calls),
        ..
      } if tool_calls.len() == 1
    ));
    assert!(matches!(
      &conversation[2],
      Message::Tool {
        tool_call_id,
        content,
      } if tool_call_id == "call_1" && content == "counting tool result"
    ));
    assert!(matches!(
      &conversation[3],
      Message::Assistant {
        content: Some(content),
        tool_calls: None,
        ..
      } if content == "all done"
    ));
  }

  #[tokio::test]
  async fn test_run_agent_loop_stops_at_iteration_limit_before_tool_execution() {
    let tool_count = Arc::new(AtomicUsize::new(0));
    let mut tool_manager = ToolManager::new();
    tool_manager.register(CountingTool::new(tool_count.clone()));

    let client = MockModelClient::new(vec![Ok(tool_call_response(
      "call_1",
      "counting_tool",
      json!({}),
    ))]);
    let tool_defs = tool_manager.definitions();
    let mut conversation = conversation_with_user();
    let errors = Arc::new(Mutex::new(Vec::new()));

    run_agent_loop_with_client(
      &client,
      &tool_manager,
      &mut conversation,
      &tool_defs,
      Some(error_collector(errors.clone())),
      None,
      None,
      None,
      Some(1),
    )
    .await;

    assert_eq!(tool_count.load(Ordering::SeqCst), 0);
    assert_eq!(conversation.len(), 1);
    assert!(matches!(
      &conversation[0],
      Message::User { content } if content == "hello"
    ));
    assert_eq!(
      errors
        .lock()
        .expect("error storage mutex poisoned")
        .as_slice(),
      [iteration_limit_error(1)]
    );
  }

  #[tokio::test]
  async fn test_run_agent_loop_retries_without_consuming_iteration_budget() {
    let tool_count = Arc::new(AtomicUsize::new(0));
    let mut tool_manager = ToolManager::new();
    tool_manager.register(CountingTool::new(tool_count.clone()));

    let client = MockModelClient::new(vec![
      Err("temporary failure".to_string()),
      Ok(tool_call_response("call_1", "counting_tool", json!({}))),
      Ok(terminal_response("done after retry")),
    ]);
    let tool_defs = tool_manager.definitions();
    let mut conversation = conversation_with_user();
    let errors = Arc::new(Mutex::new(Vec::new()));

    run_agent_loop_with_client(
      &client,
      &tool_manager,
      &mut conversation,
      &tool_defs,
      Some(error_collector(errors.clone())),
      None,
      None,
      None,
      Some(2),
    )
    .await;

    assert_eq!(tool_count.load(Ordering::SeqCst), 1);
    assert_eq!(
      errors
        .lock()
        .expect("error storage mutex poisoned")
        .as_slice(),
      ["temporary failure"]
    );
    assert_eq!(conversation.len(), 4);
    assert!(matches!(
      &conversation[3],
      Message::Assistant {
        content: Some(content),
        ..
      } if content == "done after retry"
    ));
  }

  #[tokio::test]
  async fn test_run_agent_loop_does_not_append_unfinished_tool_call_message_on_limit() {
    let tool_count = Arc::new(AtomicUsize::new(0));
    let mut tool_manager = ToolManager::new();
    tool_manager.register(CountingTool::new(tool_count));

    let client = MockModelClient::new(vec![Ok(tool_call_response(
      "call_2",
      "counting_tool",
      json!({}),
    ))]);
    let tool_defs = tool_manager.definitions();
    let mut conversation = vec![
      Message::User {
        content: "previous question".to_string(),
      },
      Message::Assistant {
        content: Some("previous answer".to_string()),
        tool_calls: None,
        reasoning_content: None,
      },
      Message::User {
        content: "current question".to_string(),
      },
    ];
    let errors = Arc::new(Mutex::new(Vec::new()));

    run_agent_loop_with_client(
      &client,
      &tool_manager,
      &mut conversation,
      &tool_defs,
      Some(error_collector(errors.clone())),
      None,
      None,
      None,
      Some(1),
    )
    .await;

    assert_eq!(conversation.len(), 3);
    assert!(matches!(
      conversation.last(),
      Some(Message::User { content }) if content == "current question"
    ));
    assert_eq!(
      errors
        .lock()
        .expect("error storage mutex poisoned")
        .as_slice(),
      [iteration_limit_error(1)]
    );
  }

  #[tokio::test]
  async fn test_auto_compact_replaces_older_messages_with_summary() {
    let client = MockModelClient::new(vec![Ok(terminal_response("checkpoint summary"))]);
    let mut conversation = vec![
      Message::System {
        content: "You are helpful.".to_string(),
      },
      Message::User {
        content: long_text("old question"),
      },
      Message::Assistant {
        content: Some(long_text("old answer")),
        tool_calls: None,
        reasoning_content: None,
      },
      Message::User {
        content: "recent question".to_string(),
      },
      Message::Assistant {
        content: Some("recent answer".to_string()),
        tool_calls: None,
        reasoning_content: None,
      },
    ];

    maybe_auto_compact_conversation(
      &client,
      &mut conversation,
      None,
      Some(20),
      Some("custom compact prompt"),
    )
    .await;

    let requests = client.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].1, 0);
    assert!(matches!(
      &requests[0].0[0],
      Message::System { content } if content == "custom compact prompt"
    ));
    assert_eq!(conversation.len(), 4);
    assert!(matches!(
      &conversation[0],
      Message::System { content } if content == "You are helpful."
    ));
    assert!(matches!(
      &conversation[1],
      Message::Assistant {
        content: Some(content),
        tool_calls: None,
        ..
      } if content == &format!("{COMPACT_SUMMARY_PREFIX}\ncheckpoint summary")
    ));
    assert!(matches!(
      &conversation[2],
      Message::User { content } if content == "recent question"
    ));
    assert!(matches!(
      &conversation[3],
      Message::Assistant {
        content: Some(content),
        ..
      } if content == "recent answer"
    ));
  }

  #[tokio::test]
  async fn test_auto_compact_failure_keeps_original_conversation() {
    let client = MockModelClient::new(vec![Err("compact failed".to_string())]);
    let original = vec![
      Message::User {
        content: long_text("old question"),
      },
      Message::Assistant {
        content: Some(long_text("old answer")),
        tool_calls: None,
        reasoning_content: None,
      },
      Message::User {
        content: "recent question".to_string(),
      },
    ];
    let mut conversation = original.clone();
    let errors = Arc::new(Mutex::new(Vec::new()));

    maybe_auto_compact_conversation(
      &client,
      &mut conversation,
      Some(error_collector(errors.clone())),
      Some(20),
      None,
    )
    .await;

    assert_eq!(conversation.len(), original.len());
    assert!(matches!(
      &conversation[0],
      Message::User { content } if content == &long_text("old question")
    ));
    assert!(matches!(
      &conversation[1],
      Message::Assistant {
        content: Some(content),
        ..
      } if content == &long_text("old answer")
    ));
    assert!(matches!(
      &conversation[2],
      Message::User { content } if content == "recent question"
    ));
    assert_eq!(
      errors
        .lock()
        .expect("error storage mutex poisoned")
        .as_slice(),
      ["Automatic context compaction failed: compact failed"]
    );
  }

  #[tokio::test]
  async fn test_auto_compact_does_not_consume_iteration_budget() {
    let tool_count = Arc::new(AtomicUsize::new(0));
    let mut tool_manager = ToolManager::new();
    tool_manager.register(CountingTool::new(tool_count.clone()));

    let client = MockModelClient::new(vec![
      Ok(terminal_response("checkpoint summary")),
      Ok(tool_call_response("call_1", "counting_tool", json!({}))),
      Ok(terminal_response("checkpoint summary 2")),
      Ok(terminal_response("done")),
    ]);
    let tool_defs = tool_manager.definitions();
    let mut conversation = vec![
      Message::System {
        content: "You are helpful.".to_string(),
      },
      Message::User {
        content: long_text("older question"),
      },
      Message::Assistant {
        content: Some(long_text("older answer")),
        tool_calls: None,
        reasoning_content: None,
      },
      Message::User {
        content: "current question".to_string(),
      },
    ];

    run_agent_loop_with_client(
      &client,
      &tool_manager,
      &mut conversation,
      &tool_defs,
      None,
      None,
      Some(20),
      None,
      Some(2),
    )
    .await;

    assert_eq!(tool_count.load(Ordering::SeqCst), 1);
    let requests = client.requests();
    assert_eq!(requests.len(), 4);
    assert_eq!(requests[0].1, 0);
    assert_eq!(requests[1].1, 1);
    assert_eq!(requests[2].1, 0);
    assert_eq!(requests[3].1, 1);
    assert!(matches!(
      &conversation[1],
      Message::Assistant {
        content: Some(content),
        ..
      } if content.starts_with(COMPACT_SUMMARY_PREFIX)
    ));
    assert!(matches!(
      conversation.last(),
      Some(Message::Assistant {
        content: Some(content),
        ..
      }) if content == "done"
    ));
  }
}
