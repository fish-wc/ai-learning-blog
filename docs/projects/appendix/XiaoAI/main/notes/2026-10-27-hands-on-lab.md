# 10-hands-on 实操全程记录（2026-10-27）

> 本文是第 10 章 6 个实验的实操记录，包含所有踩坑与修复过程。
> 环境结论在文末「复现路径」，照着做可以跳过所有踩坑。

## 实验 1：跑起来并观察 ✅

**目标**：`pnpm install && pnpm build` → `pnpm gateway:dev` → `pnpm tui` → `pnpm xiaoaiclaw status`

**踩坑 1：pnpm 未安装**

- 现象：`pnpm: 未找到命令`
- 修复：`corepack pnpm` 可用（自动读 package.json 锁定的 10.32.1），把
  `~/.cache/node/corepack/v1/pnpm/10.32.1/bin/pnpm.cjs` 软链到 `~/.local/bin/pnpm`，
  并在 `~/.bashrc` 加 `export PATH="$HOME/.local/bin:$PATH"`

**踩坑 2：dev 模式端口与 CLI 默认端口不一致**

- 现象：`pnpm xiaoaiclaw status` 显示 Gateway unreachable（它连 18789，dev Gateway 在 19001）
- 原因：`--dev` 是全局 CLI 标志，`gateway:dev` 脚本带了它，但后续 CLI 命令不继承
- 修复：**所有 CLI 命令都要带 `--dev`**：`pnpm xiaoaiclaw --dev status`

**踩坑 3：TUI 握手 pairing required**

- 现象：TUI 连接被拒：`GatewayClientRequestError: pairing required`，日志显示
  `scope-upgrade operator.read → operator.admin`
- 原因：设备自动配对只批了 operator.read；TUI 需要全部 operator 权限，升级请求被挂起
- 修复：`pnpm xiaoaiclaw --dev devices approve --latest`（批准后配对表更新为全 scope）
- **学习点**：这就是「Pairing（配对）」机制——设备级审批 + 令牌。WS 握手时序：
  连接 → token 认证 → 设备身份 → scope 检查 →（不满足则挂起 repair 请求）

**踩坑 4：模型 API key 缺失**

- 现象：`No API key found for provider "anthropic"`
- 排查链：`live-auth-keys.ts` 读 `ANTHROPIC_API_KEY` env → 不存在；
  auth-profiles.json 放错目录（CLI embedded 用 `agents/main/agent`，agent 运行时用 `agents/dev/agent`）
- 修复（3 步）：
  1. `~/.xiaoaiclaw-dev/agents/{main,dev}/agent/auth-profiles.json` 写入占位 key
     （本地 mify 代理会注入真实认证）
  2. config 的 `models.providers.anthropic.baseUrl` 指向 `http://127.0.0.1:18080/anthropic`
  3. `models: []` 空数组会被 merge 逻辑保留隐式模型列表（`mergeProviderModels` 兜底）
- **学习点**：pi-ai SDK 用 `model.baseUrl`（models.json 里的），不读 ANTHROPIC_BASE_URL env！

**会话文件观察**

- 位置：`~/.xiaoaiclaw-dev/agents/dev/sessions/<sessionId>.jsonl`（懒创建，第一条消息后才生成）
- 结构：`session` 头 → `model_change` → `thinking_level_change` → user/assistant 消息对
  （assistant content 里 toolCall / text 混排，usage 带 token 统计）

## 实验 2：跟踪一条消息的完整链路 ✅

**链路对代码**（CLI `agent` 命令视角）：

```
CLI 发起 → WS connect（token+pairing）→ gateway chat/agent method
  → dispatchInboundMessage（saas 队列判断, src/auto-reply/dispatch.ts）
  → lane 队列（session 串行化, lane=session:agent:dev:main）
  → runReplyAgent（src/auto-reply/reply/agent-runner.ts）
  → runEmbeddedPiAgent（src/agents/pi-embedded-runner/run.ts）
  → attempt（故障转移粒度）→ 工具循环（pi-agent-core）
  → 回复 → session 落盘 → WS 回包
```

**sessionKey 归因**：我们的会话键是 `agent:dev:main`

- 格式：`agent:{agentId}:{channel}:{groupId}:u-{userId}`（SaaS 渠道）
- CLI/TUI 无渠道，退化为 `agent:{agentId}:{mainKey}`
- 解析代码：`src/sessions/session-key-utils.ts:23 parseAgentSessionKey`

**连发 3 条消息观察队列**（并行发 3 条 CLI 消息）：

- 日志证据：`lane wait exceeded: lane=session:agent:dev:main waitedMs=9132 queueAhead=1`
- 语义：**同一 sessionKey 的 run 互斥串行**，第二条等了 9 秒（queueAhead=1）
- 回复顺序严格 1→2→3，没有乱序
- collect 模式（多渠道默认）：在 run 进行中的新消息会合并进下一次 run

## 实验 3：写一个 Skill ✅

**创建**：`~/.xiaoaiclaw/workspace-dev/skills/learning-notes/SKILL.md`

- frontmatter：name + description + metadata.xiaoaiclaw.emoji
- 验证：`pnpm xiaoaiclaw --dev skills list` 显示 `✓ ready · 来源 xiaoaiclaw-workspace`

**六级加载来源**（`src/agents/skills/workspace.ts:594`，优先级从高到低）：

1. workspace `<ws>/skills` 与 `<ws>/.agents/skills`
2. `~/.xiaoaiclaw/skills`（managed）
3. `~/.agents/skills`
4. hotload 目录（SaaS）
5. plugin 提供的 skills
6. `skills.load.extraDirs` + bundled（仓库 `skills/`）

**实验：description 含糊化**

- 改成 `description: 一个工具。` → 模型不再选它（笔记写去了别处）
- 恢复清晰描述 → 触发正常
- **结论**：description 是模型触发 skill 的唯一线索，要写「何时用」而非「是什么」

**实验：requires.bins 不满足**

- 加 `requires.bins: ["definitely-not-installed-bin"]`
- `skills list` 状态变 `△ needs setup`；代码上 `shouldIncludeSkill` → `evaluateRuntimeEligibility`
  返回 false，skill 被排除出模型 prompt（模型根本看不到它）

## 实验 4：写一个 hook 插件 ✅

**创建**：`learning/exec-audit-plugin/`（hook-only，约 30 行）

- `xiaoaiclaw.plugin.json`：`{"id": "exec-audit", "configSchema": {...}}`
  ⚠️ **configSchema 必填**，没有它 Gateway 启动直接报 manifest invalid
- `index.js`：`definePluginEntry` + `api.on("before_tool_call", ...)` 拦截 exec 写审计日志
- 配置：config `plugins.load.paths: [".../exec-audit-plugin"]` → 重启 Gateway

**验证**：让模型执行 `echo hello-from-exec-test`，日志出现：

```
[exec-audit] {"ts":"...","runId":"7545950a-...","tool":"exec","command":"echo hello-from-exec-test"}
```

返回 `{block: true, blockReason}` 可拦截；返回 void 放行（审计场景）。

**边界 lint**：`pnpm lint:extensions:no-plugin-sdk-internal` /
`pnpm lint:plugins:no-extension-imports` 均通过（外部插件只能 import
`xiaoaiclaw/plugin-sdk` 公开面）

## 实验 5：可观测性定位慢回复 ✅

**开启 cacheTrace**（config `diagnostics.cacheTrace`，按 ops 文档推荐）：

```json
{"enabled": true, "includeMessages": false, "includePrompt": true, "includeSystem": false}
```

输出：`~/.xiaoaiclaw-dev/logs/cache-trace.jsonl`，记录 stage 序列
（session:loaded → sanitized → limited → prompt:before → stream:context → session:after）

**31 秒 run 的归因**（transcript 时间戳 + cache-trace stage）：

| 阶段                                 | 耗时  | 占比 |
| ------------------------------------ | ----- | ---- |
| CLI 冷启动 + 配置预热 + session 加载 | 13.6s | 44%  |
| 第 1 次 LLM 调用（→ 发起 exec）     | 5.9s  | 19%  |
| exec 工具执行（date 命令）           | 0.05s | 0.2% |
| 第 2 次 LLM 调用（→ 最终回复）      | 11.4s | 37%  |
| 回复投递                             | 0.2s  | 0.6% |

结论：慢在两次 LLM 调用（TTFT 在模型侧），工具和网关都很快。
CLI 冷启动 44% 解释了上游为什么做 OPT-2（models-config 缓存）。

**Grafana 面板 ↔ 链路映射**（`xiaoaiclaw-ops/monitoring/grafana/xiaoaiclaw-observability.json`）：

- TTFT 首包时延 → LLM 调用阶段；排队等待时延 → lane 队列；
  tool.invoke 时延 → 工具循环；agent.run 时延 → runReplyAgent 全程；
  compaction → 上下文压缩；queue_depth → 入站队列

## 实验 6：SaaS 会话存储排障推演 ✅

**写 session 状态的所有位置**（docs/xiaoai/session-sync-redis-jfs-double-write.md）：

1. `.jsonl` transcript：agent run 的文件型 SessionManager（persist=true）
2. `sessions.json` 元数据：三处裸 `updateSessionStore`（subagent-spawn / orphan-recovery / cron）
3. Redis `msgs:{sid}`：权威存储（每次都写）
4. maskUid 目录：`ensureSaasUserDir`（每请求建，hotload 模式已短路）

**跨天上下文分裂先查什么**：

- 三个数据源的 sessionId 是否对齐（sessions.json 的 UUID vs jsonl/Redis 的 hex）
- JFS 是否落后 Redis（双写非每次发生，Redis 权威）
- jsonl header 的 cwd 是否错位（跨用户隔离 bug）

**rebase 检查 grep**（upstream-tracker.md）实测：

- `resolveSaasUserIdFromSessionKey` in session-utils.ts = 2 ✅
- `resolveSaasAgentSessionsDir` in session.ts = 2 ✅
- 发现：tracker 里 web_search 检查命令的文件路径已过时
  （`src/web/provider-runtime-shared.ts` → 现为 `src/web-search/runtime.ts`）

## 复现路径（跳过踩坑版）

```bash
# 0. 环境（一次性）
corepack pnpm --version   # 确认 10.32.1 可用；不可用则装
# pnpm 软链 + PATH 见实验 1 踩坑 1

# 1. 启动
pnpm install && pnpm build
pnpm gateway:dev          # 终端 1，等 "listening on ws://127.0.0.1:19001"

# 2. TUI（终端 2）—— 首次会 pairing required，另开终端批准：
pnpm xiaoaiclaw --dev devices approve --latest
pnpm xiaoaiclaw --dev tui

# 3. 发消息测试（CLI 方式）
pnpm xiaoaiclaw --dev agent --agent dev --message "hello"

# 4. 看会话文件
ls ~/.xiaoaiclaw-dev/agents/dev/sessions/
python3 -m json.tool --json-lines <任意 .jsonl>

# 5. skill / 插件 / 观测 见上文各实验
```

## 遗留问题

- [ ] TUI 的完整交互界面没深入玩（本次只验证了连接与状态栏）
- [ ] cache-trace 的 includeMessages=true 全量模式未试（磁盘开销大）
- [ ] OTel/Jaeger 全链路 trace 没启动（需要 `xiaoaiclaw-ops/o11y/manage.sh start`）
- [ ] upstream-tracker 的 web_search 检查命令文件路径过时，值得给 ops 同学提个醒
