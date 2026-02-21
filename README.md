# `@elonehoo/agent`

> A lightweight embedded agent framework for Node.js, powered by Rust and [NAPI-RS](https://napi.rs). Based on [little-agent](https://github.com/unixzii/little-agent) — supports OpenAI-compatible LLMs with built-in tools.

## Features

- **High Performance** — Core logic written in Rust, compiled to native Node.js addon
- **OpenAI-Compatible** — Works with any OpenAI-compatible API (OpenAI, Azure, local models, etc.)
- **Built-in Tools** — Shell execution, glob file search, and file reading out of the box
- **Streaming** — Real-time transcript events via callbacks
- **Agent Loop** — Automatic tool call → result → model loop until task completion

## Install

```bash
npm install @elonehoo/agent
```

## Usage

```typescript
import { AgentSession } from '@elonehoo/agent'

const session = new AgentSession(
  {
    apiKey: process.env.OPENAI_API_KEY!,
    model: 'gpt-4o',
    // baseUrl: 'https://api.openai.com/v1',  // optional
    systemPrompt: 'You are a helpful coding assistant.',
  },
  // onIdle — called when the agent finishes processing
  (_err) => {
    console.log('Agent is idle.')
  },
  // onError — called when an error occurs
  (_err, error) => {
    console.error('Error:', error)
  },
  // onTranscript — called for each transcript event
  (_err, event) => {
    const prefix = event.source === 'assistant' ? '🤖' : '👤'
    process.stdout.write(`${prefix} ${event.text}`)
  },
)

session.sendMessage('List all TypeScript files in the current directory')
```

## API

### `AgentConfig`

```typescript
interface AgentConfig {
  apiKey: string        // OpenAI API key
  model?: string        // Model name (default: "gpt-5.2")
  baseUrl?: string      // API base URL (default: "https://api.openai.com/v1")
  systemPrompt?: string // System prompt for the agent
}
```

### `AgentSession`

```typescript
class AgentSession {
  constructor(
    config: AgentConfig,
    onIdle?: (err: Error | null) => void,
    onError?: (err: Error | null, error: string) => void,
    onTranscript?: (err: Error | null, event: TranscriptEvent) => void,
  )

  sendMessage(message: string): void
}
```

### `TranscriptEvent`

```typescript
interface TranscriptEvent {
  text: string    // The transcript text
  source: string  // "user" or "assistant"
}
```

## Built-in Tools

| Tool | Description |
|------|-------------|
| `shell` | Runs arbitrary shell commands and returns stdout/stderr |
| `glob` | Finds files using glob patterns (supports `*`, `?`, `**`) |
| `read_file` | Reads file content with line numbers (up to 50 lines per read) |

## Build from Source

```bash
npm install
npm run build
```

## License

MIT

With GitHub Actions, each commit and pull request will be built and tested automatically in [`node@20`, `@node22`] x [`macOS`, `Linux`, `Windows`] matrix. You will never be afraid of the native addon broken in these platforms.

### Release

Release native package is very difficult in old days. Native packages may ask developers who use it to install `build toolchain` like `gcc/llvm`, `node-gyp` or something more.

With `GitHub actions`, we can easily prebuild a `binary` for major platforms. And with `N-API`, we should never be afraid of **ABI Compatible**.

The other problem is how to deliver prebuild `binary` to users. Downloading it in `postinstall` script is a common way that most packages do it right now. The problem with this solution is it introduced many other packages to download binary that has not been used by `runtime codes`. The other problem is some users may not easily download the binary from `GitHub/CDN` if they are behind a private network (But in most cases, they have a private NPM mirror).

In this package, we choose a better way to solve this problem. We release different `npm packages` for different platforms. And add it to `optionalDependencies` before releasing the `Major` package to npm.

`NPM` will choose which native package should download from `registry` automatically. You can see [npm](./npm) dir for details. And you can also run `yarn add @napi-rs/package-template` to see how it works.

## Develop requirements

- Install the latest `Rust`
- Install `Node.js@10+` which fully supported `Node-API`
- Install `yarn@1.x`

## Test in local

- yarn
- yarn build
- yarn test

And you will see:

```bash
$ ava --verbose

  ✔ sync function from native code
  ✔ sleep function from native code (201ms)
  ─

  2 tests passed
✨  Done in 1.12s.
```

## Release package

Ensure you have set your **NPM_TOKEN** in the `GitHub` project setting.

In `Settings -> Secrets`, add **NPM_TOKEN** into it.

When you want to release the package:

```bash
npm version [<newversion> | major | minor | patch | premajor | preminor | prepatch | prerelease [--preid=<prerelease-id>] | from-git]

git push
```

GitHub actions will do the rest job for you.

> WARN: Don't run `npm publish` manually.
