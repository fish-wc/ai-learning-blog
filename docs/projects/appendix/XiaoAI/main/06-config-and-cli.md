# 06 · 配置体系、CLI 与运行时状态

> 目标：读完本章，你知道配置从哪来、优先级怎么算、CLI 命令怎么加、磁盘上每个文件是干什么的。

## 1. 配置系统（`src/config/`）

### 1.1 格式与位置

- **格式**：单个 **JSON5** 文件（允许注释与尾逗号）。IO 入口：`src/config/io.ts`（`loadConfig`、`parseConfigJson5`、`writeConfigFile`）、`src/config/mutate.ts`。
- **位置**（`src/config/paths.ts`）：

| 项 | 解析顺序 |
|---|---|
| 状态目录 | `XIAOAICLAW_STATE_DIR` → `~/.xiaoaiclaw`（旧 `~/.clawdbot` 仍兼容；`XIAOAICLAW_HOME` 可整体改 homedir） |
| 配置文件 | `XIAOAICLAW_CONFIG_PATH` → `<stateDir>/xiaoaiclaw.json` |
| 端口等常量 | `DEFAULT_GATEWAY_PORT = 18789`；OAuth：`XIAOAICLAW_OAUTH_DIR` → `<stateDir>/credentials/oauth.json` |

- **严格校验**：未知键/类型错误会让 **Gateway 拒绝启动**（根层级只额外允许 `$schema`），此时仅 `doctor`/`logs`/`health`/`status` 可用。校验用 Zod（`src/config/zod-schema.*.ts`）。

### 1.2 ⚠️ 配置改动的"双写"铁律

`src/config/AGENTS.md`：**每个配置字段必须同时加进 `types.*.ts` 和配对的 `zod-schema.*.ts`**（如 `types.agent-defaults.ts` ↔ `zod-schema.agent-defaults.ts`）。`pnpm tsgo` **不会**发现漏掉的 Zod 条目 —— 运行时直接 `Unrecognized key` 拒启。改完 schema 冒烟：`node xiaoaiclaw.mjs gateway`。

生成物（CI 有 `--check` 校验）：`src/config/schema.base.generated.ts`（`pnpm config:schema:gen`）、渠道配置元数据（`config:channels:gen`）、配置文档基线（`config:docs:gen` → `docs/xiaoai/config-doc/`）。

### 1.3 配置结构速览

根类型 `XiaoAIClawConfig`（`src/config/types.xiaoaiclaw.ts:76`）涵盖：`meta, auth, env, diagnostics, logging, cli, browser, ui, secrets, skills, plugins, models, agents, tools, sandbox, bindings, media, messages, commands, approvals, session, web, channels, cron, hooks, discovery, canvasHost, talk, gateway, memory, mcp, saas, nacos, apollo, startup`。

- 常用：`agents.defaults.model.{primary,fallbacks}`；`channels.<provider>.{enabled,botToken,dmPolicy: pairing|allowlist|open|disabled,allowFrom}`。
- 权威参考（**当字典用，不要线性读**）：`docs/gateway/configuration-reference.md`；快速上手 `docs/gateway/configuration.md`、`configuration-examples.md`。
- **[二开]** SaaS 配置文档集：`docs/xiaoai/config-doc/`（saas-core/session/security/concurrency/skill-preload/tools/logging/observability/cron/model）。

### 1.4 优先级与分层（面试常考 😄）

**环境变量优先级**（`docs/help/environment.md`）：

```
process env > .env（当前目录）> ~/.xiaoaiclaw/.env > 配置 env 块（不覆盖）> 可选 login-shell 导入（只补缺失键）
```

- 配置值里可用 `${VAR}` 替换（缺变量=加载报错；`$${VAR}` 转义）—— `src/config/env-substitution.ts`；回写配置时 `${ENV}` 引用被保留（`env-preserve.ts`）。
- **SecretRef**（`{source:"env"|"file"|"exec", provider, id}`）是明文/`${ENV}` 之外的第三种凭据写法，见 §3.3。
- **运行时覆盖**：`src/config/runtime-overrides.ts`（`setConfigOverride`，受 managed-lock 管控）叠在文件之上。
- **CLI 标志**：`--profile`/`--log-level`/`--verbose` 等在 `src/cli/program/preaction.ts` 处理；`--log-level` 写入 `XIAOAICLAW_LOG_LEVEL` 从而压过 env。
- **`$include`**：拆分配置文件（`src/config/includes.ts`，防缺失/循环）。
- **写入安全**：原子 rename、`config-audit.jsonl` + `config-health.json` 审计、备份轮转（`backup-rotation.ts`）、原型污染防护、快照脱敏（`redact-snapshot.ts`）。

### 1.5 热更新（文件监听）与 [二开] Apollo 热配置

**文件热重载**（`gateway.reload.{mode,debounceMs}`）：

| 模式 | 行为 |
|---|---|
| `hybrid`（默认） | 安全项热生效，关键项自动重启 |
| `hot` / `restart` / `off` | 全热 / 全重启 / 关闭 |

- **热生效**：channels、agent/agents/models/routing、hooks/cron/heartbeat、session/messages、tools/browser/skills/audio/talk、ui/logging/identity/bindings。
- **需重启**：`gateway.*`（端口/绑定/鉴权/TLS）、discovery/canvasHost/plugins。
- **编程式**：RPC `config.get` / `config.apply`（全量替换）/ `config.patch`（JSON merge-patch：对象合并、`null` 删键、数组整体替换）；限流 3 次/60s，重启 30s 冷却合并。

**Apollo 集成**（`src/infra/apollo/*`；设计 `docs/xiaoai/feature-apollo-config-integration.md`、`feature-apollo-hot-config.md`）：

- `xiaoaiclaw.json` 是基底；Apollo 里 `xiaoaiclaw.` 前缀的 key 按点路径**整节点覆盖**（`xiaoaiclaw.models` 覆盖全部 models，`xiaoaiclaw.agents.defaults.model` 只覆盖 model）；细粒度 key 优先于粗粒度。
- `ConfigBridge` 合并 + Zod 校验后 `setRuntimeConfigSnapshot`；60s 长轮询 + 90s 全量刷新（releaseKey 304）。
- 两级缓存：L1 内存 + L2 磁盘（`apollo.cachePath`，**仅校验通过才写**；灰度响应不写）。⚠️ `apollo.dumpPath` 是**只写调试转储**，恢复数据源是 L2 缓存（`src/infra/apollo/AGENTS.md`）。
- 降级：初始化 Zod 失败 → 回放 L2 → 重新合并校验；仍失败则 Apollo 置空、Gateway 正常启动。

## 2. CLI（`src/cli/`、`src/commands/`）

- 入口链：`xiaoaiclaw.mjs` → `src/entry.ts` → `src/cli/run-main.ts:119 runCli`（Commander）。**命令懒注册**（`src/cli/program/register-lazy-command.ts`：先占位、首次调用才 import 真模块）以保启动速度 —— 启动耗时有 bench 预算（`pnpm test:startup:bench`）。
- 命令注册表：`src/cli/program/command-registry.ts`（核心命令）+ `register.subclis.ts` / `subcli-descriptors.ts`（子 CLI）。

### 2.1 命令面（挑重点）

- **核心**：`setup`、`onboard`（交互式向导）、`config`（get/set/unset/file/validate）、`doctor`（体检 + `--fix`）、`dashboard`、`backup`、`message`、`agent`（单轮 agent 调用）、`agents`、`status`、`health`、`sessions`、`tasks`、`mcp`。
- **子 CLI**：`gateway`（`src/cli/gateway-cli/`：run/call/discover/dev）、`daemon`、`logs`、`models`、`approvals`、`nodes`/`devices`、`sandbox`、`tui`、`cron`、`hooks`、`webhooks`、`pairing`、`plugins`、`channels`、`security`、`secrets`、`skills`、`update`、`completion` …

### 2.2 怎么加一个命令（标准四步）

1. 在 `src/commands/<name>.ts` 实现；
2. 在 `src/cli/program/` 加 `register.<name>.ts`（或并入现有 registrar，如 `register.maintenance.ts`）；
3. 在 `command-registry.ts` 的 `coreEntries` 加条目（注意 `hasSubcommands` 标记）；子 CLI 则进 `register.subclis.ts` + `subcli-descriptors.ts`；
4. 补 colocated 测试 + `docs/cli/<name>.md` 文档页。

规范提醒（`AGENTS.md`）：进度条用 `src/cli/progress.ts`、表格 `src/terminal/table.ts`、颜色 `src/terminal/palette.ts`，不要手搓。

### 2.3 TUI 与 Web 控制台

- **TUI**（`src/tui/`）：WS 连 Gateway 的终端客户端（基于 `pi-tui`），`pnpm tui`；支持会话操作、流式组装、本地 shell。
- **Control UI**（`ui/`，Lit + Vite）：由 Gateway 托管（默认 `http://127.0.0.1:18789`），`xiaoaiclaw dashboard` 打开；含 Config（schema 驱动表单 + JSON 编辑）、Logs（RPC tail）、会话/聊天。
- **角色分工**：Gateway 拥有全部状态；TUI/UI 只是薄客户端（"All session state is owned by the gateway"）。原生客户端在 `apps/`。

## 3. 运行时磁盘状态（`~/.xiaoaiclaw`）

权威表格：`docs/help/faq.md`（"Where things live on disk"）。

```
~/.xiaoaiclaw/                      # 状态目录（XIAOAICLAW_STATE_DIR 可改；--profile 形态为 ~/.xiaoaiclaw-<profile>/）
├── xiaoaiclaw.json                 # 主配置（JSON5）
├── .env                            # 全局 env 兜底（不覆盖已有）
├── credentials/                    # oauth.json、渠道凭据（whatsapp/<accountId>/creds.json）、pairing/allowFrom
├── agents/<agentId>/agent/         # agentDir（可被 agents.list[].agentDir 覆盖；重复会报 DuplicateAgentDirError）
│   ├── auth-profiles.json          # OAuth + API key（可含 SecretRef）
│   └── …
├── agents/<agentId>/sessions/
│   ├── sessions.json               # 会话元数据 store
│   └── <sessionId>.jsonl           # transcript（还有 .acp-stream.jsonl 等变体）
├── secrets.json                    # file 型 SecretRef 的载荷（可选）
├── logs/                           # 如 commands.log（command-logger hook）
├── skills/                         # 共享技能目录
└── sandboxes/                      # 沙箱工作区

~/.xiaoaiclaw/workspace/            # 工作区（agents.defaults.workspace；XIAOAICLAW_PROFILE 时为 workspace-<profile>）
├── AGENTS.md SOUL.md USER.md IDENTITY.md TOOLS.md HEARTBEAT.md BOOTSTRAP.md
├── MEMORY.md  memory/YYYY-MM-DD.md   # 记忆（真相源是 Markdown）
├── skills/   canvas/
```

- 会话 store 也可为 **Redis**（`session.store: "filesystem"|"redis"`）；**[二开]** SaaS 用 Redis/JFS + `saas.saasWorkspaceDir`（见第 8 章）。
- 备份：`xiaoaiclaw backup`（状态目录含密钥 → **备份要加密**）。
- 破除误解：`src/global-state.ts` 只是两个进程级开关（`verbose`/`yes`），**不是**状态存储。

### 3.3 Secrets（`src/secrets/`）

- **SecretRef 契约**：`{source:"env"|"file"|"exec", provider, id}`（env：id=环境变量名；file：id=文件内 JSON Pointer；exec：provider 命令）。参考 `docs/gateway/secrets.md`、`docs/reference/secretref-credential-surface.md`。
- **运行时模型**：激活时**急切解析**成内存快照；有效引用解析不了则**启动失败**；重载是原子换快照（全成功或保持上一份好的）—— secret provider 故障不落在热路径上。
- **有效面过滤**：禁用渠道/工具上的引用被忽略（诊断码 `SECRETS_REF_IGNORED_INACTIVE_SURFACE`）。
- CLI：`xiaoaiclaw secrets`（含只读 `secrets audit`、`plan`/`apply`）。

## 4. 动手试试

```bash
pnpm xiaoaiclaw config validate        # 校验配置
pnpm xiaoaiclaw config get agents.defaults.model
pnpm xiaoaiclaw doctor --fix           # 体检 + 修复
pnpm xiaoaiclaw logs --follow          # tail 日志（下一章细讲）
```

1. 故意在 `xiaoaiclaw.json` 里写一个不存在的键，启动 gateway 看严格校验报错 —— 体会"配置即 schema"。
2. 用 `config.patch` RPC 改一个热生效项（如 `logging.level`）和一个需重启项（如 `gateway.port`），观察 `hybrid` 模式行为差异。
3. 找到你的 `sessions.json` 和一个 `.jsonl` transcript，对照 §3 的地图确认每个路径的含义。

## 5. 深入阅读

**配置**：`docs/gateway/configuration.md` → `src/config/types.xiaoaiclaw.ts`（根形状）→ `src/config/paths.ts` + `io.ts` → `src/config/AGENTS.md`（双写规则）→ `docs/gateway/configuration-reference.md`（字典）→ `docs/help/environment.md` → `docs/gateway/secrets.md` → `docs/xiaoai/feature-apollo-config-integration.md` + `src/infra/apollo/AGENTS.md` → `docs/xiaoai/config-doc/README.md`

**CLI**：`src/cli/program/command-registry.ts` → `register-lazy-command.ts` → 挑一条纵向切片（`register.maintenance.ts` → `src/commands/doctor.ts`）→ `docs/cli/index.md`、`docs/web/control-ui.md`、`docs/web/tui.md`

**运行时状态**：`docs/help/faq.md` → `docs/concepts/agent-workspace.md` → `docs/concepts/session.md` → `docs/install/migrating.md` → `src/config/agent-dirs.ts`

下一章：[07-observability.md](07-observability.md)
