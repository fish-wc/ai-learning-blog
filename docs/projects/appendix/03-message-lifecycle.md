# 03 · 一条消息的生命周期（核心章节）

> 目标：跟着一条聊天消息从"渠道收到"走到"回复发出"，把每个跳板的**关键文件**记住。
> 建议：开着 IDE 对照读；行号会漂移，以符号名为准。

## 0. 总览（六个跳板）

```
Hop A  渠道收到事件（extensions/<channel>）          —— 防抖合帧 + 准入闸门
Hop B  配对检查（src/pairing）                       —— 用户配对 / 设备配对
Hop C  路由 + 会话解析（src/routing、src/config/sessions）—— 决定 sessionKey
Hop D  入站派发（src/auto-reply）                    —— 队列/斜杠命令/去重
Hop E  回复生成（get-reply → agent-runner → pi run）  —— 真正跑模型与工具
Hop F  出站投递（route-reply → 渠道 outbound）        —— 回到用户所在渠道
```

## 1. 进程入口（Gateway 怎么起来的）

| 文件 | 职责 |
|---|---|
| `xiaoaiclaw.mjs`（根目录） | 薄 shim：校验 Node ≥ 22.12、设 `UV_THREADPOOL_SIZE`、开 compile cache，动态 import `dist/entry.js` |
| `src/entry.ts`（`runMainOrRootHelp:196`） | 真正的 bootstrap：进程标题、env/argv 规整（Windows argv 修正、`--profile`/`--container`），再懒加载 CLI |
| `src/cli/run-main.ts:119 runCli` | Commander CLI：`route.ts` 快速路由热命令，否则 `program/build-program.ts` 建全量命令树 |
| `src/cli/gateway-cli/run.ts`、`run-loop.ts:29 runGatewayLoop` | `xiaoaiclaw gateway` 子命令：解析 `--port/--bind/--token/--auth`，**守护循环**（`acquireGatewayLock`、信号处理、进程内重启、等待嵌入式 run 收尾） |
| `src/gateway/server.impl.ts:426 startGatewayServer` | Gateway 单体启动：配置快照、死状态断言、插件自动启用、secrets 快照 → HTTP/WS 服务、channel manager、node registry、exec 审批、cron、sidecars |
| `src/daemon/service.ts:113 startGatewayService` | daemon **不在进程内跑 gateway**，而是装成 OS 服务（systemd/launchd/schtasks），最终执行 `xiaoaiclaw gateway run` |

## 2. Hop A：渠道收到事件

渠道是 **插件**（`extensions/<id>/`），经 `src/channels/plugins/bundled.ts`（jiti 加载 `xiaoaiclaw.plugin.json`）进入注册表 `src/channels/plugins/registry.ts`；插件契约 `ChannelPlugin`（`src/channels/plugins/types.plugin.ts:81`）定义了 `config/pairing/security/groups/mentions/outbound/streaming/threading/allowlist/bindings/agentTools/gatewayMethods` 等能力适配器。

以 Telegram 为例：

1. `extensions/telegram/src/bot-handlers.runtime.ts:102 registerTelegramHandlers` —— grammY 事件接线（message / callback_query / reaction / migrate）。
2. **防抖合帧**：文本碎片、媒体组按策略合并（核心策略 `src/channels/inbound-debounce-policy.ts`，插件默认 `defaults.queue.debounceMs`）。
3. **准入闸门**：`resolveTelegramEventAuthorization`（同文件 `:671`）→ DM/群访问控制（`dm-access.ts`、`group-access.ts`）→ allowlist（`src/channels/allow-from.ts`、`allowlist-match.ts`）。
4. 每条消息的处理管线：`extensions/telegram/src/bot-message-dispatch.ts:160 dispatchTelegramMessage`。

## 3. Hop B：配对（pairing）

- **渠道用户配对**：`src/pairing/pairing-store.ts`（`PairingRequest`、审批、allow-from store）、`setup-code.ts`（配对码编解码）、`pairing-challenge.ts`；渠道适配器经 `src/channels/plugins/pairing.ts` 接入。
- **设备/节点配对**（手机 App 等 node）：`extensions/device-pair`（QR + bootstrap token 流）+ gateway 方法 `device.pair.*` / `device.token.*`（`src/gateway/server-methods/devices.ts`）+ `node.pair.*`（`src/gateway/node-registry.ts`）；挑战/签名核心在 `src/gateway/device-auth.ts`（`buildDeviceAuthPayloadV3`）。

## 4. Hop C：路由 + 会话解析

- **路由**：`src/routing/resolve-route.ts:631 resolveAgentRoute(input) → ResolvedAgentRoute {agentId, channel, accountId, sessionKey, mainSessionKey, matchedBy}`；binding 优先级 `peer → … → account → channel → default`（`src/routing/bindings.ts`）。
- **sessionKey**：`src/routing/session-key.ts`（`buildAgentPeerSessionKey`、`resolveAgentIdFromSessionKey`、`resolveSaasUserIdFromSessionKey` —— **[二开] SaaS 用户挂在 `:u-<uid>` 后缀**）。Telegram 的线程会话键在 `extensions/telegram/src/conversation-route.ts`。
- **消息信封 `MsgContext`**（`src/auto-reply/templating.ts:23`）—— **全链路核心数据结构**：`Body`、`BodyForAgent`、`From/To`、`SessionKey`、`AccountId`、`ReplyTo*`、`MediaPath(s)`、`ChatType`、`WasMentioned`，以及 SaaS 字段（`SaasClientContext`、`SaasUserRole`、`MemorySwitch`、`DeviceRouteOverride`…）。`finalizeInboundContext` 后得 `FinalizedMsgContext`。
- **会话记录**：`src/channels/session.ts:45 recordInboundSession` → `src/config/sessions/inbound.runtime.js`。
- **会话模型 `SessionEntry`**（`src/config/sessions/types.ts:75`）：`sessionId/sessionFile/systemSent/abortedLastRun/abortCutoffMessageSid`、per-session 覆盖（`modelOverride`、`authProfileOverride`、`thinkingLevel`、`queueMode`）、token/成本核算、`spawnDepth`、`deliveryContext`。存储经可插拔后端 `src/config/sessions/backend/`（**FS `sessions.json` 或 Redis `ClawSessionStore`**，二开 SaaS 用 Redis）。

## 5. Hop D：入站派发与排队

```
渠道插件 → src/plugin-sdk/inbound-reply-dispatch.ts（recordInboundSessionAndDispatchReply）
  → src/auto-reply/dispatch.ts:40 dispatchInboundMessage
      （SaaS 模式经 SaasUserQueue（src/saas/user-queue.ts）按 {userId}:{groupId} 串行）
  → dispatch-reply-from-config.ts:218 dispatchReplyFromConfig
      （去重、hooks、斜杠命令检测 src/auto-reply/command-detection.ts、中止文本检测）
  → ReplyDispatcher（src/auto-reply/reply/reply-dispatcher.ts）
      （ReplyDispatchKind = "tool" | "block" | "final"；块流式缓冲、typing 控制）
```

排队语义（collect/steer/followup…）见第 2 章 §4 与 `docs/concepts/queue.md`。

## 6. Hop E：回复生成（模型与工具真正跑起来的地方）

```
get-reply.ts:136 getReplyFromConfig
  解析 agent（resolveSessionAgentId）、模型（resolveDefaultModel）、
  SaaS 工作区（src/saas/workspace.ts）、typing、媒体/链接理解
→ get-reply-run.ts:285 runPreparedReply
  组装本轮全部上下文：群介绍、inbound-meta prompt、记忆能力提示、输出语言、声纹 prompt、
  信封选项、队列/typing 策略 …… （:1017 调 runReplyAgent）
→ agent-runner.ts:66 runReplyAgent
  per-run 会话核算、fallback 提示、块回复管线、followup/queue 策略、提醒守卫
→ agent-runner-execution.ts:151 runAgentTurnWithFallback   ← 模型故障转移重试
→ src/agents/pi-embedded-runner/run.ts:149 runEmbeddedPiAgent
  per-session lane + 全局 lane 串行、加载插件、解析模型/鉴权、before_model_resolve、超时
→ run/attempt.ts:1273 runEmbeddedAttempt   ← 一次完整 attempt（流式、工具循环、压缩触发）
```

事件流桥接：`src/agents/pi-embedded-subscribe*.ts` 把 pi 事件转成 `GetReplyOptions` 回调（`onPartialReply`/`onBlockReply`/`onToolResult`/`onModelSelected`，类型 `src/auto-reply/types.ts:28`），产出 `ReplyPayload`（同文件 `:95`：text/mediaUrl/interactive/replyToId/isReasoning…）。

> 工具循环、技能注入、压缩等细节见 [04-agents-tools-skills.md](04-agents-tools-skills.md)。

## 7. Hop F：出站投递

- `src/auto-reply/reply/route-reply.ts routeReply(payload, channel, to, …)` —— **按 `OriginatingChannel/OriginatingTo` 路由**（不是 session.lastChannel！），解析渠道插件，规整 payload（`normalize-reply.ts`），经 `src/infra/outbound/deliver-runtime.ts` 投递。
- 渠道侧 outbound 适配器（Telegram：`extensions/telegram/src/bot/delivery.ts`、`delivery.send.ts`）：流式编辑/删除 lane、状态回应（`src/channels/status-reactions.ts`）。
- 运行状态广播：`src/channels/run-state-machine.ts`（busy/activeRuns 状态补丁）。

**铁律**（`AGENTS.md`）：外部消息面只发**最终回复**，绝不发流式半成品 —— 块流式只对支持的内部面生效。

## 8. Gateway 控制面（CLI/TUI/UI 怎么进来的）

- **协议**：`src/gateway/protocol/`（TypeBox 帧定义 + Ajv 校验）：`ConnectParams`（client 信息、role、caps、device 签名、auth{token,bootstrapToken,deviceToken,password}）、`RequestFrame`/`ResponseFrame`/`EventFrame`、`HelloOk`。
- **HTTP 多路复用**（`src/gateway/server-http.ts`）：WS upgrade、`/hooks` webhook（token 门控）、Control UI 静态站（`ui/`）、Slack/飞书 webhook、embeddings、canvas WS。
- **WS 调度**：`src/gateway/server/ws-connection.ts`（握手发 `connect.challenge`、验 `ConnectParams`）→ `src/gateway/server-methods.ts`（`authorizeGatewayMethod`：role → scope → 限流 → 审计）。
- **RPC 方法**：`src/gateway/server-methods/`（`server-methods-list.ts:130 listGatewayMethods`）：`chat.send`、**`chat.send.saas`（:4066）**、**`channel.send.saas`（:4714）**、`chat.abort`、`sessions`、`send`、`agents`、`config`、`cron`、`devices`、`nodes`、`skills`、`models`、`health`…。处理器契约 `GatewayRequestHandler`，共享上下文 `GatewayRequestContext`（`server-methods/types.ts:36`，含 `chatAbortControllers`、`agentRunSeq`、广播函数等）。
- **客户端侧**：CLI `src/cli/gateway-rpc.ts → src/gateway/call.ts → src/gateway/client.ts`（WS + 设备身份 + challenge 签名 + pending-request 表）；TUI `src/tui/gateway-chat.ts`；UI 在 `ui/`。
- **鉴权**（`src/gateway/auth.ts`）：模式 `none | token | password | trusted-proxy`；WS 连接可凭 token / password / tailscale whois / 设备令牌 / bootstrap token；限流 `auth-rate-limit.ts`、浏览器 origin 检查。

## 9. Run、中止与工具循环熔断

**运行注册**：`src/agents/pi-embedded-runner/runs.ts`（`setActiveEmbeddedRun`、`abortEmbeddedPiRun:78`、`queueEmbeddedPiMessage`，按 sessionId 键控）。

**中止有四层**：

1. **自然语言停止**：`src/auto-reply/reply/abort-primitives.ts`（`isAbortRequestText`）+ `src/gateway/chat-abort.ts isChatStopCommandText`；
2. **渠道 `/stop` 快路径**：`src/auto-reply/reply/abort.ts:321 tryFastAbortFromMessage`；截断语义 `abort-cutoff.ts`（把 `SessionEntry.abortCutoffMessageSid` 持久化，**跳过停止前积压的排队消息**）；`abort-last-run-cache.ts` 记录被中止的半成品（`SessionEntry.abortedLastRun`）；
3. **Gateway RPC `chat.abort`**（`server-methods/chat.ts:2281`）：操作 `ChatAbortControllerEntry`（`src/gateway/chat-abort.ts:6`）并广播 `broadcastChatAborted`；
4. **run 级**：`abortEmbeddedPiRun(sessionId)` 取消 AbortController；`pi-embedded-runner/abort.ts isRunnerAbortError` 分类中止错误。

**工具循环熔断**（per-run、runId 隔离）：

- 检测核心 `src/agents/tool-loop-detection.ts`：`detectToolCallLoop`（约 `:378`），历史窗口 30，阈值 **warning@10 / critical@20 / 全局熔断@30**（`:36` 常量，另有"连续 3 次 critical 即中止"去抖）；检测器：`generic_repeat`、`known_poll_no_progress`、`ping_pong`、`global_circuit_breaker`；工具调用哈希 = 工具名 + 参数稳定 JSON 摘要。
- 接入点 `src/agents/pi-tools.before-tool-call.ts`（每个工具调用前过一遍）：critical 时返回 `{blocked:true, reason}` —— 给模型一个 CRITICAL 工具结果让它自我纠正，并回调 `ctx.onCriticalLoop`。
- 中止去抖在 `run/attempt.ts:1562-1662 handleCriticalLoop`：**连续 3 次 blocked critical 才真正 abort run**。诊断日志 `src/logging/diagnostic.ts`。

> 经典事故复盘（理解熔断为什么这么设计）：`xiaoaiclaw-ops/incidents/p4t-incident-20260409-tool-loop.md`。

## 10. 关键类型速查表

| 概念 | 类型 | 位置 |
|---|---|---|
| 消息信封 | `MsgContext` / `FinalizedMsgContext` | `src/auto-reply/templating.ts:23,254` |
| run 选项/回调 | `GetReplyOptions` | `src/auto-reply/types.ts:28` |
| 回复输出 | `ReplyPayload` | `src/auto-reply/types.ts:95` |
| 回复派发器 | `ReplyDispatcher`、`ReplyDispatchKind` | `src/auto-reply/reply/reply-dispatcher.ts` |
| 会话模型 | `SessionEntry`、`ClawSessionStore` | `src/config/sessions/types.ts:75`、`backend/types.ts:116` |
| 路由 | `ResolvedAgentRoute` | `src/routing/resolve-route.ts:39` |
| 渠道插件 | `ChannelPlugin` | `src/channels/plugins/types.plugin.ts:81` |
| 线协议帧 | `ConnectParams`/`RequestFrame`/`EventFrame` | `src/gateway/protocol/schema/` |
| RPC 契约 | `GatewayRequestHandler`、`GatewayRequestContext` | `src/gateway/server-methods/types.ts:148,36` |
| 中止注册 | `ChatAbortControllerEntry` | `src/gateway/chat-abort.ts:6` |
| 循环熔断配置 | `ToolLoopDetectionConfig` | `src/config/types.tools.ts:154` |

## 11. 三条新手代码走读路线（强烈建议照着走一遍）

**A. 一条消息端到端（telegram 为例）**：
`src/channels/plugins/types.plugin.ts` → `extensions/telegram/src/bot-handlers.runtime.ts`（准入闸门 `:671`）→ `bot-message-context.ts` + `recordInboundSession` → `src/routing/resolve-route.ts` + `src/auto-reply/templating.ts`（`MsgContext`）→ `bot-message-dispatch.ts:160` → `src/auto-reply/dispatch.ts` → `get-reply.ts` → `get-reply-run.ts` → `agent-runner.ts` → `route-reply.ts` + `extensions/telegram/src/bot/delivery.ts`。

**B. Gateway 控制面**：
`src/cli/gateway-cli/run-loop.ts` → `src/gateway/server.impl.ts:426`（启动顺序）→ `protocol/schema/frames.ts` → `server/ws-connection.ts` → `server-methods.ts` + `server-methods/types.ts` → `auth.ts` + `method-scopes.ts` → `src/gateway/client.ts`（客户端侧）。

**C. Run / 中止 / 熔断**：
`src/auto-reply/types.ts`（runId/abortSignal）→ `pi-embedded-runner/run.ts` → `run/attempt.ts` → `tool-loop-detection.ts` + `pi-tools.before-tool-call.ts`（阈值与拦截）→ `attempt.ts:1562`（去抖中止）→ `runs.ts` + `server-methods/chat.ts`（`chat.abort`）→ `abort.ts`/`abort-cutoff.ts`（`/stop` 语义）。

## 12. 深入阅读

- `docs/concepts/agent-loop.md`（生命周期事件全表）
- `docs/gateway/protocol.md`（WS 协议契约）、`docs/concepts/queue.md`、`docs/concepts/session.md`
- 下一章：[04-agents-tools-skills.md](04-agents-tools-skills.md)（Hop E 内部的完整展开）
