# 01 · 项目全景与仓库地图

> 目标：读完本章，你能回答"这个项目是干什么的、代码都在哪、哪些是上游的、哪些是我们自己的"。

## 1. 项目定位

**XiaoAIClaw**（`xiaoaiclaw` CLI）是一个 **自托管的个人 AI 助手**：

- 你在日常使用的聊天软件（WhatsApp、Telegram、Slack、Discord、飞书、微信、iMessage、Teams……共 20+ 渠道）里给它发消息；
- 它在你自己的机器上运行一个 **Gateway（网关）** 进程，接收消息后交给 **Agent**（背后是各家大模型）执行任务（读写文件、跑命令、查网页、调用内部工具……）；
- 结果回送到你所在的渠道，还能语音对话、控制 macOS/iOS/Android 设备节点、渲染实时 Canvas。

一句话：**"Gateway 只是控制面，产品是那个助手本身"**（`README.md`）。

### 命名小史（看到奇怪的包名别慌）

上游项目经历了改名：Warelay → Clawdbot → Moltbot → OpenClaw；本仓库的公开名改为 **XiaoAIClaw**。所以你会看到：

- `packages/clawdbot`、`packages/moltbot`：历史改名留下的兼容 stub 包；
- 代码里偶尔出现 `claw`、`molt`、`lobster`（🦞 龙虾是项目吉祥物）等词根，都是历史遗留，无实际含义；
- 趣味背景见 `docs/start/lore.md`。

## 2. 上游 vs 二开：最重要的心智模型

本仓库是 **上游开源项目的二次开发**。改动遵守一条铁律：**上游目录尽量不动，二开内容集中在独立目录**，这样 rebase 上游更新时冲突最小。

| 归属 | 位置 | 内容 |
|---|---|---|
| **[上游]** | `src/` | 核心运行时：CLI、Gateway、Agent、配置、通道等全部主逻辑 |
| **[上游]** | `extensions/` | 90 个 workspace 扩展包（+ `shared` 公共库）：渠道（telegram/feishu/…）、模型供应商（openai/anthropic/…）、能力（browser/memory-core/…） |
| **[上游]** | `docs/` | 官方文档站源码（Mintlify），概念 / 安装 / 渠道 / 工具手册 |
| **[上游]** | `ui/`、`apps/` | Web 控制台；macOS / iOS / Android 原生客户端 |
| **[上游]** | `skills/` | 内置 Skill 包（可被 Agent 加载的"技能说明书"） |
| **[二开]** | `xiaoaiclaw-ops/` | **二开大本营**：部署（Docker/K8s）、运维 runbook、安全审计、自研 skill、上游 workaround 跟踪 |
| **[二开]** | `docs/xiaoai/` | 二开设计文档、bugfix 记录、性能/可观测性分析 |
| **[二开]** | `src/saas/` | SaaS 多用户场景接入代码（与上游共存于 `src/`，是少数侵入上游目录的点） |
| **[二开]** | `plugins-local/` | 本地插件：`memory-cloud`（云端记忆）、`system-control` |

规则细节（提交规范、镜像复制规则、可变更的观测面清单等）见根目录 `AGENTS.md`；二开运维规范见 `xiaoaiclaw-ops/AGENTS.md`。

## 3. 仓库地图（按目录）

```
xiaoaiclaw/
├── src/                  # [上游] 核心运行时（TypeScript ESM），约 5900 个 .ts 文件
│   ├── cli/  commands/   #   CLI 入口与子命令（onboard/gateway/agent/message/doctor…）
│   ├── gateway/          #   Gateway 进程：WS 协议、HTTP、RPC 方法、server-methods
│   ├── agents/           #   Agent 运行时：模型调用、工具循环、会话、压缩（最大的目录）
│   ├── channels/ routing/ pairing/  # 消息通道抽象、路由、配对鉴权
│   ├── config/           #   配置加载与 schema（含 Apollo 热配置接入）
│   ├── plugin-sdk/ plugins/ hooks/  # 插件 SDK 与内置插件、钩子
│   ├── mcp/  acp/        #   MCP 与 ACP 协议接入
│   ├── sessions/ chat/   #   会话与聊天记录
│   ├── logging/ infra/   #   日志、基础设施（otel、redis 等封装）
│   ├── saas/             #   [二开] SaaS 多用户接入
│   └── …（media/ tts/ cron/ security/ secrets/ tui/ …）
├── extensions/           # [上游] 93 个扩展包（渠道、模型供应商、工具能力）
├── packages/             # [上游] 共享包：memory-host-sdk、plugin-package-contract、改名 stub
├── ui/  apps/            # [上游] Web 控制台；macOS/iOS/Android 客户端
├── skills/               # [上游] 内置技能包（markdown 为主的技能说明书）
├── plugins-local/        # [二开] memory-cloud、system-control
├── xiaoaiclaw-ops/       # [二开] 部署/运维/安全/自研（详见第 8 章）
├── docs/                 # [上游] 文档站；docs/xiaoai/ 是 [二开] 设计与分析文档
├── test/  test-fixtures/ # 少量非 colocate 测试与夹具
├── scripts/  ci/  git-hooks/  # 构建、代码生成、CI、提交钩子
├── openspec/             # 规格变更流程（changes/ 存提案）
├── AGENTS.md             # 开发者规范（CLAUDE.md 指向它）
└── package.json          # pnpm monorepo 根，全部脚本入口
```

### 规模感（截至 2026-10）

- `src/` ≈ 5900 个 TS 文件，`extensions/` ≈ 2900 个，`ui/` ≈ 230 个。
- `src/` 内最大的几个目录：`agents/`（≈1300 文件）、`infra/`（≈600）、`gateway/`（≈530）、`commands/`（≈450）、`auto-reply/`（≈420）、`config/`（≈340）、`plugin-sdk/`（≈340）。
- **结论：不可能通读。** 正确姿势是"地图 + 主链路 + 按需下钻"，这正是本教程的结构。

### workspace 结构

`pnpm-workspace.yaml` 定义了 monorepo 成员：根包 `xiaoaiclaw`（发 npm 的 CLI 本体）、`ui/`、`packages/*`、`extensions/*`。每个 extension 是独立 npm 包，**插件专属依赖写在各自 `package.json` 里**，不要污染根包。

## 4. 两条产品线场景

同一个代码基座支撑两种形态（后面第 8 章展开）：

1. **[上游] 个人单机**：`xiaoaiclaw onboard` 在自己电脑/服务器上装好 Gateway，绑定自己的聊天账号，单用户。
2. **[二开] 公司 SaaS / 设备（音箱等）**：多用户、K8s 部署、云端记忆、模型路由、全链路可观测 —— 代码上主要通过 `src/saas/` + `plugins-local/` + `xiaoaiclaw-ops/` 扩展。

学习时先掌握第 1 种（它是主链路、文档最全），再看第 2 种在哪些点上做了增强。

## 5. 术语先行（详细版见第 10 章术语表）

| 名词 | 一句话解释 |
|---|---|
| **Gateway** | 常驻进程，管理所有消息渠道与客户端连接，是"控制面" |
| **Channel** | 一个消息渠道的实现（telegram / feishu / web …） |
| **Agent / run** | 一次"收消息→思考→调工具→回复"的执行；run 是单次运行实例 |
| **Tool** | 模型可调用的原子能力（shell、读文件、发消息…） |
| **Skill** | 放在工作区里的 markdown 技能说明书，运行时注入 prompt |
| **Plugin / Extension** | 扩展代码；extension 是 workspace 独立包形态的插件 |
| **Session** | 一段对话的上下文与历史（存于 `~/.xiaoaiclaw/…`，二开 SaaS 用 Redis/JFS） |
| **Compaction** | 长对话自动摘要压缩 |
| **pi-agent-core** | 上游使用的底层 agent 循环运行时（嵌入式执行） |

## 6. 动手试试

```bash
pnpm install          # 安装依赖（较久）
pnpm build            # 构建（包含一串代码生成步骤，见 09 章）
pnpm xiaoaiclaw --help        # 看看 CLI 全貌
pnpm xiaoaiclaw doctor        # 环境自检
```

- 打开 `src/entry.ts`，看 CLI 是怎么启动的（后面第 6 章细讲）。
- 打开 `docs/concepts/architecture.md`，把架构图看一遍。

## 7. 深入阅读

- `README.md` —— 项目总介绍、安装与快速开始
- `AGENTS.md` —— **开发者规范，进仓库第一件事**
- `docs/concepts/architecture.md` —— Gateway 架构（第 2 章据此展开）
- `docs/concepts/agent-loop.md` —— Agent 循环（第 3、4 章据此展开）
- `xiaoaiclaw-ops/README.md` —— 二开全景
- 下一章：[02-architecture.md](02-architecture.md)
