/**
 * @elonehoo/agent 使用示例
 *
 * 运行方式:
 *   export OPENAI_API_KEY="你的API Key"
 *   export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选
 *   export OPENAI_MODEL="gpt-4o"                        # 可选
 *   node --import @oxc-node/core/register examples/basic.ts
 */

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
  systemPrompt: '你是一个有帮助的编程助手。请用中文回复。',
}

const session = new AgentSession(
  config,
  // onIdle — agent 处理完毕时调用
  (_err) => {
    console.log('\n--- Agent 处理完毕 ---\n')
  },
  // onError — 发生错误时调用
  (_err, error) => {
    console.error('\n[错误]', error)
  },
  // onTranscript — 每次生成文本片段时调用
  (_err, event) => {
    if (event.source === 'user') {
      console.log(`\n👤 用户: ${event.text}`)
    } else {
      // assistant 的消息是流式的，一个字一个字输出
      process.stdout.write(event.text)
    }
  },
)

// 发送消息
session.sendMessage('请列出当前目录下的所有 .ts 文件')
