import { describe, it, expect } from 'vitest'

import { AgentSession } from '../index'
import type { AgentConfig, TranscriptEvent } from '../index'

// ===== 构造函数测试 =====

describe('AgentSession', () => {
  it('完整配置构造', () => {
    const config: AgentConfig = {
      apiKey: 'test-key',
      model: 'gpt-4o',
      baseUrl: 'https://api.openai.com/v1',
      systemPrompt: 'You are a helpful assistant.',
    }

    const session = new AgentSession(config)
    expect(session).toBeTruthy()
    expect(typeof session.sendMessage).toBe('function')
  })

  it('仅必填字段构造', () => {
    const config: AgentConfig = {
      apiKey: 'test-key',
    }

    const session = new AgentSession(config)
    expect(session).toBeTruthy()
  })

  it('带所有回调构造', () => {
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
    expect(session).toBeTruthy()
  })

  it('部分回调 (只传 onIdle)', () => {
    const session = new AgentSession({ apiKey: 'test-key' }, (_err) => {
      // on idle only
    })
    expect(session).toBeTruthy()
  })

  it('回调传 null', () => {
    const session = new AgentSession({ apiKey: 'test-key' }, null, null, null)
    expect(session).toBeTruthy()
  })

  it('回调传 undefined', () => {
    const session = new AgentSession({ apiKey: 'test-key' }, undefined, undefined, undefined)
    expect(session).toBeTruthy()
  })
})

// ===== sendMessage 测试 =====

describe('AgentSession.sendMessage', () => {
  it('方法存在且可调用', () => {
    const session = new AgentSession({ apiKey: 'test-key' })
    expect(() => {
      session.sendMessage('hello')
    }).not.toThrow()
  })

  it('空字符串不会崩溃', () => {
    const session = new AgentSession({ apiKey: 'test-key' })
    expect(() => {
      session.sendMessage('')
    }).not.toThrow()
  })

  it('长文本不会崩溃', () => {
    const session = new AgentSession({ apiKey: 'test-key' })
    const longText = 'a'.repeat(10000)
    expect(() => {
      session.sendMessage(longText)
    }).not.toThrow()
  })
})

// ===== 自定义 baseUrl 测试 =====

describe('自定义 baseUrl', () => {
  it('支持自定义 baseUrl', () => {
    const session = new AgentSession({
      apiKey: 'test-key',
      baseUrl: 'http://localhost:8080/v1',
      model: 'local-model',
    })
    expect(session).toBeTruthy()
  })
})

// ===== 集成测试 (需要真实 API key) =====

const runIntegration = !!process.env.OPENAI_API_KEY

describe.skipIf(!runIntegration)('集成测试', () => {
  it('基本对话', async () => {
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
          resolve()
        },
        (_err, error) => {
          expect.unreachable(`Unexpected error: ${error}`)
          resolve()
        },
        (_err, event) => {
          transcripts.push(event)
        },
      )

      session.sendMessage('Say hello')
    })

    const userTranscripts = transcripts.filter((e) => e.source === 'user')
    const assistantTranscripts = transcripts.filter((e) => e.source === 'assistant')

    expect(userTranscripts.length).toBeGreaterThan(0)
    expect(assistantTranscripts.length).toBeGreaterThan(0)

    const userText = userTranscripts.map((e) => e.text).join('')
    expect(userText).toContain('Say hello')

    const assistantText = assistantTranscripts.map((e) => e.text).join('')
    expect(assistantText).toContain('HELLO')
  })

  it('工具调用 (shell)', async () => {
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
          expect.unreachable(`Unexpected error: ${error}`)
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

    expect(assistantText).toContain('AGENT_TEST_OK')
  })
})
