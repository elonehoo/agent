# @elonehoo/agent

A lightweight embedded AI agent framework for Node.js — powered by Rust and [NAPI-RS](https://napi.rs).

Inspired by [little-agent](https://github.com/unixzii/little-agent). Works with any OpenAI-compatible LLM.

## Features

- **Rust Core** — High-performance native addon via NAPI-RS, no child process overhead
- **OpenAI-Compatible** — Supports OpenAI, Azure, local models, or any compatible endpoint
- **Built-in Tools** — Shell execution, glob file search, and file reading out of the box
- **Streaming Transcripts** — Real-time assistant output via callbacks
- **Agent Loop** — Automatic `tool_call → execute → respond` loop until task completion
- **Cross-Platform** — Prebuilt binaries for macOS (x64/arm64), Linux (x64), and Windows (x64)

## Install

```bash
npm install @elonehoo/agent
```

## Quick Start

```typescript
import { AgentSession } from '@elonehoo/agent'

const session = new AgentSession(
  {
    apiKey: process.env.OPENAI_API_KEY!,
    model: 'gpt-4o',
    baseUrl: 'https://api.openai.com/v1',
    systemPrompt: 'You are a helpful coding assistant.',
  },
  () => console.log('\nDone.'),
  (_err, error) => console.error('Error:', error),
  (_err, event) => {
    if (event.source === 'assistant') {
      process.stdout.write(event.text)
    }
  },
)

session.sendMessage('List all TypeScript files in the current directory')
```

See [examples/basic.ts](examples/basic.ts) for a full example, or [examples/chat.ts](examples/chat.ts) for an interactive multi-turn REPL.

## API

### `AgentConfig`

| Property | Type | Default | Description |
|---|---|---|---|
| `apiKey` | `string` | — | OpenAI API key (required) |
| `model` | `string` | `"gpt-4o"` | Model name |
| `baseUrl` | `string` | `"https://api.openai.com/v1"` | API base URL |
| `systemPrompt` | `string` | — | System prompt for the agent |

### `AgentSession`

```typescript
class AgentSession {
  constructor(
    config: AgentConfig,
    onIdle?: (err: Error | null) => void,
    onError?: (err: Error | null, error: string) => void,
    onTranscript?: (err: Error | null, event: TranscriptEvent) => void,
  )

  /** Send a user message. Triggers the agent loop. */
  sendMessage(message: string): void
}
```

### `TranscriptEvent`

```typescript
interface TranscriptEvent {
  text: string    // Streamed text content
  source: string  // "user" | "assistant"
}
```

## Built-in Tools

| Tool | Description |
|---|---|
| `shell` | Runs shell commands and returns stdout/stderr |
| `glob` | Finds files via glob patterns (`*`, `?`, `**`) |
| `read_file` | Reads file content with line numbers |

## Build from Source

Prerequisites: Rust (latest stable), Node.js >= 16, pnpm

```bash
pnpm install
pnpm build
```

## Test

```bash
pnpm test
```

## License

[MIT](LICENSE)
