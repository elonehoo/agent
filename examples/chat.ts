/**
 * 多轮对话示例
 *
 * 运行方式:
 *   export OPENAI_API_KEY="你的API Key"
 *   node --import @oxc-node/core/register examples/chat.ts
 */

import * as readline from 'node:readline'
import { AgentSession } from '../index'
import type { AgentConfig } from '../index'

const apiKey = process.env.OPENAI_API_KEY
if (!apiKey) {
  console.error('请设置环境变量 OPENAI_API_KEY')
  process.exit(1)
}

const config: AgentConfig = {
  apiKey,
  model: process.env.OPENAI_MODEL || 'gpt-4o',
  baseUrl: process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1',
  maxIterations: 12,
  systemPrompt: `你是一个有帮助的编程助手，可以使用以下工具：
- shell: 运行 shell 命令
- glob: 用 glob 模式查找文件
- read_file: 读取文件内容
请用中文回复。`,
}

let waitingForIdle: (() => void) | null = null

const session = new AgentSession(
  config,
  (_err) => {
    console.log('\n')
    if (waitingForIdle) {
      waitingForIdle()
      waitingForIdle = null
    }
  },
  (_err, error) => {
    console.error('\n[错误]', error)
  },
  (_err, event) => {
    if (event.source === 'assistant') {
      process.stdout.write(event.text)
    }
  },
)

function waitForIdle(): Promise<void> {
  return new Promise((resolve) => {
    waitingForIdle = resolve
  })
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
})

async function main() {
  console.log('🤖 Agent 已就绪，输入你的问题（输入 exit 退出）')
  console.log('   内置工具: shell, glob, read_file\n')

  const askQuestion = () => {
    rl.question('👤 > ', async (input) => {
      const trimmed = input.trim()
      if (trimmed === 'exit' || trimmed === 'quit') {
        console.log('再见！')
        rl.close()
        process.exit(0)
      }
      if (!trimmed) {
        askQuestion()
        return
      }

      console.log()
      session.sendMessage(trimmed)
      await waitForIdle()
      askQuestion()
    })
  }

  askQuestion()
}

main()
