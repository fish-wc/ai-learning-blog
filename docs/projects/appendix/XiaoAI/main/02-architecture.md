# 02 · 总体架构

> 目标：读完本章，你能在白板上画出这个系统的组件图，并说清"消息从哪来、算力在哪、状态放哪"。
> 本章以官方文档 `docs/concepts/architecture.md`、`docs/concepts/agent-loop.md` 为骨架，补充代码入口位置。

## 1. 一张图看懂

```
                         ┌────────────────────────────────────────────┐
  聊天渠道（外部服务）     │            Gateway 进程（控制面）             │
  ┌──────────┐           │  ┌─────────┐  ┌──────────┐  ┌───────────┐ │
  │ Telegram  │◄─────────┼─►│ Channel │─►│ Routing  │─►│  Session  │ │
  │ WhatsApp  │  webhook/ │  │ 适配层   │  │ 路由/配对 │  │  会话管理  │ │
  │ Feishu …  │  长连接    │  └─────────┘  └──────────┘  └─────┬─────┘ │
  └──────────┘           │                                    ▼        │
                          │   ┌─────────────────────────────────────┐ │
  控制端客户端             │   │        Agent 运行时（消息 → 回复）      │ │
  ┌──────────┐  WebSocket │   │  上下文组装 → LLM 推理 → 工具执行循环   │ │
  │ CLI/TUI  │◄──────────►│   │  → 流式回复 → 会话持久化（含压缩）      │ │
  │ Web UI   │            │   └───────────────┬─────────────────────┘ │
  │ macOS App│            │                   ▼                        │
  └──────────┘           │   工具层：shell/文件/浏览器/MCP/消息发送 …    │
                          └────────────────────────────────────────────┘
  设备节点（可选）                    ▲ WebSocket（role: node）
  iOS / Android / macOS 节点 ────────┘  提供 camera/canvas/location 等能力
```

三个要记住的结论：

1. **Gateway 是唯一常驻进程**：所有渠道连接、客户端连接、会话状态都归它管；一台机器跑一个。
2. **控制端（CLI/UI/App）只是客户端**：它们通过 WebSocket 连 Gateway，本身不跑模型。
3. **Agent 运行时在 Gateway 进程内执行**（"embedded"）：`runEmbeddedPiAgent`（`src/agents/…`）把底层 pi-agent-core 循环嵌进来跑。

## 2. 组件清单

### 2.1 Gateway（`src/gateway/`）

- 维护与各渠道（WhatsApp 用 Baileys、Telegram 用 grammY……）的连接。
- 暴露 **类型化的 WebSocket API**：请求（`health`/`status`/`send`/`agent`/`system-presence`…）、响应、服务端推送事件（`agent`/`chat`/`presence`/`health`/`heartbeat`/`cron`）。
- 入站帧按 **JSON Schema 校验**（TypeBox 定义 → 生成 JSON Schema → 生成 Swift 模型，见 `pnpm protocol:gen`）。
- 同端口（默认 `127.0.0.1:18789`）还服务 HTTP：Canvas 宿主（`/__xiaoaiclaw__/canvas/`、`/__xiaoaiclaw__/a2ui/`）等。

### 2.2 客户端（`src/cli/`、`src/tui/`、`ui/`、`apps/`）

- 每个客户端一条 WS 连接；首帧必须是 `connect` 握手。
- 幂等键（idempotency key）：`send`/`agent` 等有副作用的方法必须带，服务端短时去重 —— 这是安全重试的前提。
- `ui/` 是 Web 控制台（Vite）；`apps/` 是 macOS/iOS/Android 原生客户端（协议模型由代码生成）。

### 2.3 Nodes（设备节点）

- iOS/Android/macOS/headless 设备以 `role: node` 接入同一个 WS 服务，在 `connect` 声明能力（`canvas.*`、`camera.*`、`screen.record`、`location.get` …）。
- 设备配对是 **设备级** 的：新设备 ID 需要审批，之后拿设备令牌连接。

### 2.4 通道层（`src/channels/`、`src/routing/`、`src/pairing/` + `extensions/*`）

- 渠道适配把各家消息格式规整为统一消息对象；`src/routing/` 决定消息进哪个会话/agent；`src/pairing/` 管配对与准入（allowlist）。
- **注意**：除了 `src/telegram` 等内置通道，还有 `extensions/` 里的 feishu、msteams、matrix 等 —— 改公共逻辑（路由/配对/allowlist）时必须考虑 **全部** 通道。

### 2.5 Agent 运行时（`src/agents/`，全仓最大的目录）

一次 agent run（agentic loop）= **收消息 → 组装上下文 → 模型推理 → 工具执行 → 流式回复 → 持久化**。细节在第 3、4 章；先记住入口链：

```
Gateway RPC(agent)  →  agentCommand  →  runEmbeddedPiAgent  →  subscribeEmbeddedPiSession
      │                    │                   │                       │
   校验参数、          解析模型/默认值、     per-session + 全局队列串行、   把底层事件映射为
   立即返回 runId     加载 skills 快照      超时中止、返回用量           lifecycle/assistant/tool 流
```

### 2.6 扩展体系（`src/plugin-sdk/`、`src/plugins/`、`src/hooks/`、`extensions/`、`plugins-local/`）

- **Plugin hooks** 是扩展的核心形态：`before_model_resolve`、`before_prompt_build`、`before_tool_call`/`after_tool_call`、`message_received`/`message_sending`/`message_sent`、`session_start`/`session_end` …（完整清单见 `docs/concepts/agent-loop.md` "Hook points"）。
- **Gateway hooks**（内部钩子）：`agent:bootstrap`、`/new`、`/reset` 等命令事件触发脚本。
- 第 5 章细讲。

## 3. 线协议（WS）速览

```
连接后首帧必须是 connect：
  {type:"req", id, method:"connect", params:{ auth:{token?}, device:{...}, challenge签名... }}

之后：
  请求 → 响应：{type:"req", id, method, params} → {type:"res", id, ok, payload|error}
  服务端推送：{type:"event", event, payload, seq?, stateVersion?}
```

- 鉴权：设了 `XIAOAICLAW_GATEWAY_TOKEN`（或 `--token`）时 `connect.params.auth.token` 必须匹配。
- 配对与本地信任：所有连接带 **设备身份**；新设备需配对审批后签发设备令牌；**本机/回环连接可自动批准**，非本机必须显式批准；`connect.challenge` nonce 必须签名（v3 签名还绑定 `platform`+`deviceFamily`）。详见 `docs/channels/pairing.md`、`docs/gateway/security.md`。
- 事件不回放：断线重连后客户端要主动刷新（`stateVersion`/快照）。

## 4. 会话模型（理解多用户与隔离的关键）

- 消息按来源路由会话：DM 默认共享一个会话；群/房间各自隔离；cron 每次新会话；webhook 每 hook 隔离（`docs/concepts/session.md`）。
- **多用户场景必须开 DM 隔离**：`session.dmScope: "per-channel-peer"`（推荐），否则所有私聊共享上下文（隐私事故）。可用 `session.identityLinks` 把同一个人的多渠道身份关联。
- 会话生命周期：默认每天凌晨 4 点重置、可配闲置重置（`session.reset.idleMinutes`）、聊天里 `/new` `/reset` 手动重置（`/new <model>` 还能换模型）。
- 状态归 Gateway 所有：`~/.xiaoaiclaw/agents/<agentId>/sessions/sessions.json` + 每会话一个 `<sessionId>.jsonl` transcript。**[二开] SaaS 形态改为 Redis/JFS 存储**，见第 8 章。
- 排队与并发（`docs/concepts/queue.md`）：入站自动回复经 **lane-aware FIFO 队列** 串行 —— `session:<key>` lane 保证单会话同时只有 1 个 run，`main` 全局 lane 限制总并发（`agents.defaults.maxConcurrent`，main 默认 4、subagent 8）。队列模式按渠道可配：`collect`（默认，合并积压成一轮）/ `steer`（插话打断）/ `followup` / `steer-backlog` / `interrupt`。

## 5. 模型与推理

- 模型供应商大多是 `extensions/` 里的独立包（openai、anthropic、google、ollama、moonshot、volcengine、xai …），统一实现 provider 接口。
- 模型选择 + 鉴权 profile 轮换 + 故障转移见 `docs/concepts/models.md`、`docs/concepts/model-failover.md`。
- 思考量/推理级别（thinking）、流式输出、上下文压缩（compaction）都有专门文档：`docs/concepts/streaming.md`、`docs/concepts/compaction.md`、`docs/concepts/context.md`。

## 6. 部署形态

| 形态 | 说明 | 入口 |
|---|---|---|
| 本机开发 | `pnpm gateway:watch` 热重载跑 Gateway | `package.json` scripts |
| 单机安装 | `xiaoaiclaw onboard --install-daemon` 装 launchd/systemd 服务 | `README.md` |
| Docker | 官方镜像 + compose | `Dockerfile*`、`docker-compose.yml` |
| **[二开] K8s SaaS** | 多副本、多用户、云端记忆 | `xiaoaiclaw-ops/k8s/`、`xiaoaiclaw-ops/Dockerfile`（第 8 章） |

## 7. 动手试试

```bash
pnpm gateway:dev        # 起一个不加载渠道的 Gateway（XIAOAICLAW_SKIP_CHANNELS=1）
pnpm tui                # 另开终端起 TUI 客户端，观察 WS 握手日志
pnpm xiaoaiclaw status  # 看会话存储路径与最近活动
```

思考题：

1. 客户端断线重连后，如何拿到"我错过了什么"？（提示：事件不回放 + 快照）
2. 为什么单会话要串行，而多会话可以并行？如果反过来会出什么问题？

## 8. 深入阅读

- `docs/concepts/architecture.md`（Gateway 架构与不变量）
- `docs/concepts/agent-loop.md`（Agent 循环与 hook 点全表）
- `docs/concepts/queue.md`（排队语义）、`docs/concepts/session.md`（会话隔离）
- `docs/gateway/protocol.md`（WS 协议契约）
- 下一章：[03-message-lifecycle.md](03-message-lifecycle.md)（把本章的图变成可跟踪的代码链路）
