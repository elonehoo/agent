import test from 'ava'

import { AgentSession } from '../index'
import type { AgentConfig, TranscriptEvent } from '../index'

// ===== 构造函数测试 =====

test('AgentSession — 完整配置构造', (t) => {
  const config: AgentConfig = {
    apiKey: 'test-key',
    model: 'gpt-4o',
    baseUrl: 'https://api.openai.com/v1',
    systemPrompt: 'You are a helpful assistant.',
  }

  const session = new AgentSession(config)
  t.truthy(session)
  t.is(typeof session.sendMessage, 'function')
})

test('AgentSession — 仅必填字段构造', (t) => {
  const config: AgentConfig = {
    apiKey: 'test-key',
  }

  const session = new AgentSession(config)
  t.truthy(session)
})

test('AgentSession — 带所有回调构造', (t) => {
  const config: AgentConfig = {
    apiKey: 'test-key',
  }

  const transcripts: TranscriptEvent[] = []

  const session = new AgentSession(
    config,
    (_err) => {
      // on idle
    },
    (_err, _error) => {
      // on error
    },
    (_err, event) => {
      transcripts.push(event)
    },
  )
  t.truthy(session)
})

test('AgentSession — 部分回调 (只传 onIdle)', (t) => {
  const session = new AgentSession({ apiKey: 'test-key' }, (_err) => {
    // on idle only
  })
  t.truthy(session)
})

test('AgentSession — 回调传 null', (t) => {
  const session = new AgentSession({ apiKey: 'test-key' }, null, null, null)
  t.truthy(session)
})

test('AgentSession — 回调传 undefined', (t) => {
  const session = new AgentSession({ apiKey: 'test-key' }, undefined, undefined, undefined)
  t.truthy(session)
})

// ===== sendMessage 测试 =====

test('AgentSession.sendMessage — 方法存在且可调用', (t) => {
  const session = new AgentSession({ apiKey: 'test-key' })
  t.notThrows(() => {
    // 虽然 API key 无效会导致 onError 回调，但 sendMessage 本身不会抛出
    session.sendMessage('hello')
  })
})

test('AgentSession.sendMessage — 空字符串不会崩溃', (t) => {
  const session = new AgentSession({ apiKey: 'test-key' })
  t.notThrows(() => {
    session.sendMessage('')
  })
})

test('AgentSession.sendMessage — 长文本不会崩溃', (t) => {
  const session = new AgentSession({ apiKey: 'test-key' })
  const longText = 'a'.repeat(10000)
  t.notThrows(() => {
    session.sendMessage(longText)
  })
})

// ===== 自定义 baseUrl 测试 =====

test('AgentSession — 支持自定义 baseUrl', (t) => {
  const session = new AgentSession({
    apiKey: 'test-key',
    baseUrl: 'http://localhost:8080/v1',
    model: 'local-model',
  })
  t.truthy(session)
})

// ===== 集成测试 (需要真实 API key) =====

const INTEGRATION = process.env.OPENAI_API_KEY ? test : test.skip

INTEGRATION('集成测试 — 基本对话', async (t) => {
  const transcripts: TranscriptEvent[] = []

  await new Promise<void>((resolve) => {
    const session = new AgentSession(
      {
        apiKey: process.env.OPENAI_API_KEY!,
        model: process.env.OPENAI_MODEL || 'gpt-4o',
        baseUrl: process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1',
        systemPrompt: 'Reply with exactly: HELLO',
      },
      (_err) => {
        // on idle — agent 处理完毕
        resolve()
      },
      (_err, error) => {
        t.fail(`Unexpected error: ${error}`)
        resolve()
      },
      (_err, event) => {
        transcripts.push(event)
      },
    )

    session.sendMessage('Say hello')
  })

  // 应该有 user 和 assistant 两种 transcript
  const userTranscripts = transcripts.filter((e) => e.source === 'user')
  const assistantTranscripts = transcripts.filter((e) => e.source === 'assistant')

  t.true(userTranscripts.length > 0, '应该有 user transcript')
  t.true(assistantTranscripts.length > 0, '应该有 assistant transcript')

  // user transcript 应该包含 "Say hello"
  const userText = userTranscripts.map((e) => e.text).join('')
  t.true(userText.includes('Say hello'))

  // assistant 应该回复了 HELLO
  const assistantText = assistantTranscripts.map((e) => e.text).join('')
  t.true(assistantText.includes('HELLO'))
})

INTEGRATION('集成测试 — 工具调用 (shell)', async (t) => {
  const transcripts: TranscriptEvent[] = []

  await new Promise<void>((resolve) => {
    const session = new AgentSession(
      {
        apiKey: process.env.OPENAI_API_KEY!,
        model: process.env.OPENAI_MODEL || 'gpt-4o',
        baseUrl: process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1',
        systemPrompt: 'You are a helpful assistant. Use tools when needed. Be concise.',
      },
      (_err) => resolve(),
      (_err, error) => {
        t.fail(`Unexpected error: ${error}`)
        resolve()
      },
      (_err, event) => {
        transcripts.push(event)
      },
    )

    session.sendMessage('Run "echo AGENT_TEST_OK" and tell me the output.')
  })

  const assistantText = transcripts
    .filter((e) => e.source === 'assistant')
    .map((e) => e.text)
    .join('')

  t.true(assistantText.includes('AGENT_TEST_OK'), `Assistant response should contain the echo output: ${assistantText}`)
})
