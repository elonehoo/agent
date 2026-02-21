#![deny(clippy::all)]

mod agent;
mod openai;
mod tool;

use std::sync::OnceLock;

use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio::runtime::Runtime;

use agent::{AgentBuilder, TranscriptSource};
use openai::{OpenAIClient, OpenAIConfigBuilder};
use tool::glob_tool::GlobTool;
use tool::read_file::ReadFileTool;
use tool::shell::ShellTool;

static TOKIO_RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn tokio_runtime() -> &'static Runtime {
  TOKIO_RUNTIME.get_or_init(|| {
    tokio::runtime::Builder::new_multi_thread()
      .enable_all()
      .build()
      .expect("Failed to create tokio runtime")
  })
}

/// Configuration for creating an agent session.
#[napi(object)]
pub struct AgentConfig {
  /// OpenAI API key.
  pub api_key: String,
  /// Model name (e.g. "gpt-4o"). Defaults to "gpt-4o" if not specified.
  pub model: Option<String>,
  /// Base URL for the OpenAI-compatible API. Defaults to "https://api.openai.com/v1".
  pub base_url: Option<String>,
  /// System prompt for the agent.
  pub system_prompt: Option<String>,
}

/// Event emitted when the agent generates a transcript.
#[napi(object)]
pub struct TranscriptEvent {
  /// The transcript text content.
  pub text: String,
  /// Source of the transcript: "user" or "assistant".
  pub source: String,
}

/// A chat session with an AI agent.
///
/// The session holds a fully configured agent that supports tool calls
/// (shell, glob, read_file) and interacts with an OpenAI-compatible LLM.
///
/// The agent implements a simple loop: the model receives user input and
/// may output tool-call requests. The agent executes tools and sends the
/// results back to the model. The loop continues until there are no more
/// tool-call requests.
///
/// Built-in tools:
/// - **shell**: Runs arbitrary shell commands
/// - **glob**: Finds files using glob patterns
/// - **read_file**: Reads file content with line numbers
#[napi]
pub struct AgentSession {
  agent: agent::Agent,
}

#[napi]
impl AgentSession {
  /// Creates a new agent session.
  ///
  /// @param config - Configuration for the agent (API key, model, etc.).
  /// @param onIdle - Optional callback invoked when the agent becomes idle (finished processing).
  /// @param onError - Optional callback invoked when an error occurs.
  /// @param onTranscript - Optional callback invoked when a transcript event is generated.
  #[napi(constructor)]
  pub fn new(
    config: AgentConfig,
    on_idle: Option<ThreadsafeFunction<()>>,
    on_error: Option<ThreadsafeFunction<String>>,
    on_transcript: Option<ThreadsafeFunction<TranscriptEvent>>,
  ) -> napi::Result<Self> {
    // Enter the tokio runtime so that internal tokio::spawn calls work
    let _guard = tokio_runtime().enter();

    // Build OpenAI client
    let mut config_builder = OpenAIConfigBuilder::with_api_key(&config.api_key);
    if let Some(model) = &config.model {
      config_builder = config_builder.with_model(model);
    }
    if let Some(base_url) = &config.base_url {
      config_builder = config_builder.with_base_url(base_url);
    }
    let openai_config = config_builder.build();
    let client = OpenAIClient::new(openai_config);

    // Build the agent with callbacks and tools
    let mut builder = AgentBuilder::new(client);

    if let Some(system_prompt) = &config.system_prompt {
      builder = builder.with_system_prompt(system_prompt);
    }

    if let Some(on_idle_fn) = on_idle {
      builder = builder.on_idle(move || {
        on_idle_fn.call(Ok(()), ThreadsafeFunctionCallMode::NonBlocking);
      });
    }

    if let Some(on_error_fn) = on_error {
      builder = builder.on_error(move |err| {
        on_error_fn.call(Ok(err), ThreadsafeFunctionCallMode::NonBlocking);
      });
    }

    if let Some(on_transcript_fn) = on_transcript {
      builder = builder.on_transcript(move |text, source| {
        let source_str = match source {
          TranscriptSource::User => "user",
          TranscriptSource::Assistant => "assistant",
        };
        let event = TranscriptEvent {
          text: text.to_string(),
          source: source_str.to_string(),
        };
        on_transcript_fn.call(Ok(event), ThreadsafeFunctionCallMode::NonBlocking);
      });
    }

    // Register built-in tools
    builder = builder
      .with_tool(ShellTool::new())
      .with_tool(GlobTool::new())
      .with_tool(ReadFileTool::new());

    let agent = builder.build();

    Ok(Self { agent })
  }

  /// Sends a message to the agent for processing.
  ///
  /// The agent will process the message, potentially make tool calls,
  /// and generate transcript events via the onTranscript callback.
  /// When processing is complete, the onIdle callback will be invoked.
  #[napi]
  pub fn send_message(&self, message: String) {
    self.agent.send_message(&message);
  }
}
