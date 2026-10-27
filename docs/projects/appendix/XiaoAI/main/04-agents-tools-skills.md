# 04 · Agent 运行时、工具与 Skill

> 目标：读完本章，你知道一次 agent run 在代码里怎么跑、工具怎么注册和管控、Skill 是什么怎么写。
> 这是全书技术密度最高的一章，建议对照代码读。

## 0. 一句话架构

XiaoAIClaw **把 Pi agent SDK**（`@mariozechner/pi-agent-core` / `pi-ai` / `pi-coding-agent`，v0.64，带本地补丁 `patches/`）**嵌进进程**：

- **Pi 拥有 agentic loop 本身**：模型流式推理 + 工具调用循环；
- **XiaoAIClaw 拥有 loop 周围的一切**：会话/工作区管理、工具套件组装、Skills、插件注册表、模型/鉴权故障转移、压缩策略、消息投递。

官方集成文档：`docs/pi.md`（注意其中依赖版本略旧，以 `package.json` 为准）。

## 1. Agent 是什么

一个 agent 就是配置 `agents.list` 里的一条（id、模型、thinking 默认值、skills 过滤、工具策略、沙箱、身份、心跳、群聊、subagents …），由 `src/agents/agent-scope.ts` 解析（`resolveAgentConfig`、`resolveSessionAgentIds` 等）。

- 每个 agent 有独立目录与工作区：会话 JSONL 在 `~/.xiaoaiclaw/agents/<agentId>/sessions/<SessionId>.jsonl`（`src/agents/session-dirs.ts`）。
- 工作区 bootstrap 文件 `AGENTS.md`、`SOUL.md`、`TOOLS.md`、`BOOTSTRAP.md`、`IDENTITY.md`、`USER.md` 在首轮注入（`src/agents/bootstrap-files.ts`）。
- 模型目录按 agent 生成：`ensureXiaoAIClawModelsJson` 写 `<agentDir>/models.json`（`src/agents/models-config.ts`）。
- 人格/身份：`src/agents/identity.ts`；**[二开]** SaaS 预置 agent 模板：`src/saas/preset-agent-templates.ts`。

## 2. Run 的两条入口链

### 链 A：消息自动回复（聊天渠道）

```
src/auto-reply/reply/get-reply.ts:136 getReplyFromConfig
  → get-reply-run.ts:285 runPreparedReply
  → agent-runner.ts:66 runReplyAgent
  → agent-runner-execution.ts:151 runAgentTurnWithFallback   ← 模型故障转移重试循环
  → runEmbeddedPiAgent（:450 调用）
  → agent-runner-payloads.ts buildReplyPayloads（回复塑形）
```

配套模块（都在 `src/auto-reply/reply/`）：排队/插话（`queue.ts`、`queue-policy.ts`）、块流式（`block-streaming.ts`）、typing 指示（`typing.ts`）、记忆刷写（`agent-runner-memory.ts`）、followup（`followup-runner.ts`）。

> 这条链（get-reply-run / agent-runner-execution / attempt.ts 等）正是 `AGENTS.md` 列出的 **观测性受保护面** —— 改动需走 review checklist（见第 7 章）。

### 链 B：Gateway RPC `agent` / CLI 显式调用

```
src/agents/agent-command.ts:900 agentCommand（本地可信入口；:920 agentCommandFromIngress 强制显式信任标志）
  → src/agents/command/attempt-execution.ts:416 runAgentAttempt
  → runEmbeddedPiAgent（:547）或 runCliAgent（:462，外部 CLI 后端如 claude-cli/codex）
  → delivery.ts 投递
```

其他入口：`src/commands/agent.ts`（CLI）、`src/cron/isolated-agent/run.ts`（cron 定时任务）。

## 3. 核心执行：`src/agents/pi-embedded-runner/`

| 文件 | 职责 |
|---|---|
| `run.ts:149 runEmbeddedPiAgent` | 总入口：per-session lane + 全局 lane 串行（`lanes.ts`）、解析工作区、**加载运行时插件**（`src/agents/runtime-plugins.ts` → `src/plugins/loader.ts`）、解析 provider/model、触发 `before_model_resolve`、超时/中止管控、返回 payload 与用量 |
| `run/attempt.ts:1273 runEmbeddedAttempt` | **一次完整 attempt（约 4k 行，全仓核心）**：会话文件初始化 → skills prompt（`:1390`）→ 工具套件（`:1614`）→ `before_prompt_build` 钩子（`:1027`）→ `createAgentSession`（`:2255`，来自 pi-coding-agent）→ 工具拆分（`:2225` builtIn vs custom）→ system prompt 覆盖 → 跑 pi loop |
| `pi-embedded-subscribe.ts:39 subscribeEmbeddedPiSession` | 把 pi 事件桥接为 XiaoAIClaw 事件流：`assistant`（增量）/ `tool`（start/update/end）/ `lifecycle`（start/end/error） |
| `runs.ts` | 运行注册表、`abortEmbeddedPiRun`、`queueEmbeddedPiMessage` |
| `compact.ts:1605 compactEmbeddedPiSession` | 压缩（配合 `src/agents/compaction.ts` 摘要生成、标识符保留、`repairToolUseResultPairing`）；溢出触发压缩、后台压缩、超时/重试聚合 |
| `model.ts:672 resolveModel` | 模型解析（`ModelRegistry`） |
| `auth-controller.ts`、`failover-observation.ts`、`llm-idle-timeout.ts` | 鉴权控制、故障转移观察、空闲超时 |

排障辅助：`attempt.tool-call-argument-repair.ts`（工具参数修复）、`attempt.stop-reason-recovery.ts`（异常 stop reason 恢复）、`tool-call-normalization.ts`。

## 4. 模型、鉴权与故障转移

- **真正的 LLM 流式调用**：`streamSimple`（`@mariozechner/pi-ai`）；工具循环在 pi-agent-core 里 —— 我们 **包装工具** 而不是拥有循环。
- **模型引用格式**：`provider/model`（按第一个 `/` 拆分），见 `docs/concepts/agent.md`。
- **Provider 怪癖由 provider 插件自己管**（`docs/concepts/model-providers.md`）：`prepareExtraParams`、`wrapStreamFn`、`formatApiKey`、`refreshOAuth`、`resolveDynamicModel`、`capabilities` … 接线在 `pi-embedded-runner/extra-params.ts` 与各 `*-stream-wrappers.ts`（anthropic/google/openai/minimax/moonshot/zai/proxy/bedrock）。
- **鉴权 profile**（API key + OAuth，轮换 + 冷却）：`src/agents/auth-profiles/`（`store.ts` 存 `~/.xiaoaiclaw/agents/<agentId>/agent/auth-profiles.json`；`order.ts` 排序；`usage.ts` 标失败/冷却；`session-override.ts` 会话级钉住）。
- **两级故障转移**（`docs/concepts/model-failover.md`）：① 同 provider 内鉴权 profile 轮换（OAuth 优先、会话粘性、指数冷却 1m→5m→25m→1h、计费问题禁用）；② 模型级 fallback：`agents.defaults.model.fallbacks`，`src/agents/model-fallback.ts:590 runWithModelFallback`（图像模型另有 `:898`）。
- **CLI 推理后端**（把 claude-cli/codex 当 provider）：`src/agents/cli-runner.ts` + `cli-runner/*`。
- 用量核算：`src/agents/usage.ts`、`request-metrics-writer.ts`；payload 日志：`anthropic-payload-log.ts`、`llm-payload-log.ts`。

## 5. 工具系统

### 5.1 工具都在哪

- **单工具实现**：`src/agents/tools/*`（一文件一工具，`create*Tool()` 工厂）。代表性：`message`（所有渠道统一的发送/编辑/回应工具）、`sessions_list`/`sessions_history`/`sessions_send`/`sessions_spawn`/`session_fetch`/`session_status`、`subagents`、`skill_search`/`skill_read`、`cron`/`reminder`、`web_fetch`/`web_search`、`file_*`、`script_run`、`canvas`、`image_generate`、`tts`、`nodes`、`gateway`，**[二开]** `xiaoai_web_call`、`xiaoai_image_fetch`、`fds_*` 等。
- **核心 fs/shell 工具**（`read`/`write`/`edit`/`exec`/`process`/`apply_patch`）来自 pi-coding-agent（`codingTools`），被 XiaoAIClaw 重新包装：`src/agents/bash-tools*.ts`（PTY、job control、`exec-security.ts` 命令拦截）、`pi-tools.read.ts`（工作区根护栏）、`apply-patch.ts`。
- **静态目录与 profile**：`src/agents/tool-catalog.ts`（profile：`minimal|coding|messaging|full`）。⚠️ 常见坑：`messaging` profile **不含 memory 工具**（新装"失忆"，见第 8 章）。

### 5.2 注册与类型（写工具前必读）

- 工具类型：`src/agents/tools/common.ts:9 AnyAgentTool = AgentTool<any,unknown> & {ownerOnly?, displaySummary?}`，`AgentTool` 来自 pi-agent-core（`name/label/description/parameters(JSON Schema)/execute`）。
- **Schema 用 TypeBox**，护栏（根 `AGENTS.md` 强制）：**禁止 `Type.Union`/`anyOf`/`oneOf`/`allOf`** —— 因为部分 provider 拒绝 anyOf；字符串枚举用 `src/agents/schema/typebox.ts:15 stringEnum` / `:31 optionalStringEnum`。
- **Provider 怪癖规整**：`src/agents/pi-tools.schema.ts:70 normalizeToolParameterSchema` + `schema/clean-for-gemini.ts`、`clean-for-xai.ts`、Claude 兼容补丁、`stripUnsupportedSchemaKeywords`。

### 5.3 组装与管控（`src/agents/pi-tools.ts:223 createXiaoAIClawCodingTools`）

构建本次 run 的完整工具清单，顺序大致为：

1. **策略解析**：`pi-tools.policy.ts`（有效工具策略/群聊策略）；
2. **fs 策略 + 隐形 denylist**：`tool-fs-policy.ts`、`fs-denylist.ts`；
3. **沙箱变体**：`sandbox/*`；按消息 provider 的门控（`TOOL_DENY_BY_MESSAGE_PROVIDER`）；
4. **包装器栈**：`wrapToolWithAbortSignal`、`wrapToolWithBeforeToolCallHook`、`wrapToolWithFsDenylist`、`wrapToolWorkspaceRootGuard`、`wrapToolParamNormalization`；
5. **扩展工具**：`src/agents/xiaoaiclaw-tools.ts:78 createXiaoAIClawTools` 加 XiaoAIClaw 专属工具 + **插件工具**（`resolvePluginTools`）+ 渠道工具（`channel-tools.ts`）。

安全与防呆机制：

- **工具循环熔断**：`tool-loop-detection.ts`（哈希识别重复工具调用；warning@10 / critical@20 / 熔断@30，critical 可中止 run）；
- **执行审批**：`tool-approval.ts:127 createApprovalRegistration`（allow-once / allow-always / deny）；
- 结果治理：`session-tool-result-guard*.ts`（转录裁剪）、`tool-images.ts`（图片脱敏）、`transcript-policy.ts`、`tool-error.ts`。

### 5.4 执行与工具循环

循环本身在 pi-agent-core；XiaoAIClaw 插在工具边界：

- `pi-tool-definition-adapter.ts`：pi `ToolDefinition` ↔ `AgentTool` 适配、参数序列化、`after_tool_call` 行为；
- `pi-tools.before-tool-call.ts:143 runBeforeToolCallHook` / `:449 wrapToolWithBeforeToolCallHook`：插件 `before_tool_call` 钩子（可改参数或 `{block:true}` 拦截）；
- 工具事件经 `pi-embedded-subscribe.handlers.tools.ts` 上到 `tool` 流。

## 6. Skill 体系

**Skill = 一个目录 + `SKILL.md`**（YAML frontmatter：`name`、`description`、可选 `metadata`、`user-invocable`、`command-dispatch: tool` + `command-tool`），兼容 AgentSkills 规范。仓库内置约 60 个（`skills/`）。

- **加载与优先级**（`src/agents/skills/`，facade `src/agents/skills.ts`）：
  `skills.load.extraDirs`（配置）< 内置 `skills/` < `~/.xiaoaiclaw/skills` < `~/.agents/skills` < `<workspace>/.agents/skills` < `<workspace>/skills`（**后者覆盖前者**）。
  合并/同步/快照：`skills/workspace.ts`（`buildWorkspaceSkillSnapshot`、`buildWorkspaceSkillsPrompt`、`resolveSkillsPromptForRun`）；frontmatter 解析 `skills/frontmatter.ts`；门控 `skills/filter.ts` + `skills/config.ts`（`requires.bins/env/config`）；**分设备过滤** `applyDeviceSkillFilter`；独占技能 ACL `skills/ownership.ts`。
- **四种触发方式**：
  1. 每次 run 把 skills prompt 注入 system prompt（`resolveSkillsPromptForRun`，attempt.ts:1390）；
  2. 模型按需读取：`skill_search` / `skill_read` 工具；
  3. 用户斜杠命令：`src/auto-reply/skill-commands.ts`（`command-dispatch: tool` 时 **绕过 LLM 直接派发**）；
  4. 插件随包 skills：manifest `skills: ["./skills"]`（如 `extensions/feishu/xiaoaiclaw.plugin.json`）。
- **[二开] SaaS skill 存储后端**：`skills/backend/{filesystem-skill-store,mysql-skill-store}.ts`；安装/更新走 ClawHub（`skills-install*.ts`）；热更新 `skills/refresh.ts`。
- **[二开] 设备端业务 skills**（iot-control、music-search-play、vqa …31 个）在 `xiaoaiclaw-ops/skills/`，经 `skills.load.extraDirs` 加载 —— 详见第 8 章。

### 写一个 Skill（最小例子）

```markdown
---
name: my-skill
description: 做 XX 事情的技能（这句话决定模型何时选它，要写清"何时用"）
metadata:
  xiaoaiclaw:
    requires:
      bins: [ffmpeg]
---

# my-skill

1. 第一步……
2. 第二步……
```

放进 `<workspace>/skills/my-skill/SKILL.md` 即生效；可选 `command-dispatch: tool` 注册斜杠命令。规范：`docs/tools/skills.md`、创作指南 `docs/tools/creating-skills.md`、二开开发指南 `xiaoaiclaw-ops/skill-dev-guide.md`。

## 7. MCP 三种角色

1. **Agent run 的 MCP 客户端**（外接 MCP server 当工具）：配置 `mcp.servers`（`src/config/mcp-config.ts`；CLI `xiaoaiclaw mcp servers set`）；传输层 `src/agents/mcp-transport*.ts`（stdio/SSE/streamable-http）；按会话实例化（`pi-bundle-mcp-runtime.ts`），工具物化进 run（`pi-bundle-mcp-materialize.ts:62`）。
2. **把 XiaoAIClaw 渠道暴露为 MCP server**（让 Claude Code 等用你的会话）：`src/mcp/channel-server.ts`（`xiaoaiclaw mcp serve`），工具 `conversations_list`、`conversation_get` …
3. **把插件工具暴露给 ACP 会话**（Claude Code）：`src/mcp/plugin-tools-serve.ts`。

## 8. 动手试试

1. 跑一个 agent run 并打开 verbose，对照本章把 `runEmbeddedPiAgent → runEmbeddedAttempt` 的日志对上号。
2. 用 `xiaoaiclaw sessions --json` + `/context list` 看会话与 system prompt 组成。
3. 练习 A：写一个最小 skill（比如"生成周报模板"）放进 `<workspace>/skills`，聊天里触发它。
4. 练习 B：给某个工具加一个参数（走 `stringEnum`，体会 schema 护栏），补 `*.test.ts`，跑 `pnpm test`。
5. 观察工具循环熔断：故意让工具陷入重复调用，看 warning → critical → 熔断的日志（可参考 `xiaoaiclaw-ops/incidents/p4t-incident-20260409-tool-loop.md`）。

## 9. 深入阅读

- `docs/concepts/agent.md`、`docs/concepts/agent-loop.md`（词汇表与全循环）
- `docs/pi.md`（Pi 集成与 `src/agents` 文件地图）
- `docs/concepts/model-providers.md`、`docs/concepts/model-failover.md`
- `docs/concepts/session-tool.md`（跨会话工具）
- `docs/tools/skills.md`、`docs/tools/creating-skills.md`
- 下一章：[05-plugins-extensions.md](05-plugins-extensions.md)
