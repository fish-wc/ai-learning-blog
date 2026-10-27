# 10 · 动手实验、自查清单与术语表

> 目标：把前几章的知识变成"手上功夫"。每个实验都标注了对应章节与预期产出。
> 原则：**跑过一次胜过读十页**；每个实验尽量留下笔记（哪怕三行）。

## A. 阶段实验

### 实验 1（入门，~1h）——跑起来并观察

对应 [01](01-project-overview.md)、[02](02-architecture.md)。

```bash
pnpm install && pnpm build
pnpm gateway:dev          # 不加载渠道的 Gateway
pnpm tui                  # 另开终端，用 TUI 客户端连上
pnpm xiaoaiclaw status
```

- [ ] 画一张图：TUI 与 Gateway 之间 WS 握手发生了什么（对照 `docs/concepts/architecture.md` 的时序图）。
- [ ] 找到 `~/.xiaoaiclaw/` 下的会话文件，用编辑器打开一个 `.jsonl` 看 transcript 长什么样。

### 实验 2（核心，~3h）——跟踪一条消息的完整链路

对应 [03](03-message-lifecycle.md)。

- [ ] 开 verbose 日志，在聊天渠道（或 SaaS TUI）发一条消息，把日志与代码链路逐段对上：通道接收 → 路由 → 队列 → `runReplyAgent` → `runEmbeddedPiAgent` → attempt → 工具循环 → 回复投递。
- [ ] 找出"这条消息被路由进哪个 sessionKey"并解释为什么。
- [ ] 故意连发 3 条消息，观察 `collect` 队列模式的日志（对照 `docs/concepts/queue.md`）。

### 实验 3（体系，~2h）——写一个 Skill

对应 [04](04-agents-tools-skills.md)。

- [ ] 写一个解决你真实小需求的 skill（周报模板 / 代码评审清单 / 查询脚本），放入 `<workspace>/skills`。
- [ ] 故意把 `description` 写得含糊，观察模型是否还选它 —— 体会"description 决定触发"。
- [ ] 给它加 `requires.bins`，观察不满足条件时的行为。

### 实验 4（扩展，~3h）——写一个 hook 插件

对应 [05](05-plugins-extensions.md)。

- [ ] 写一个 hook-only 插件：`before_tool_call` 拦截 `exec` 并打审计日志；或 `before_prompt_build` 注入一段上下文。
- [ ] 用 `plugins.load.paths` 加载并验证。
- [ ] 跑一遍边界 lint（`pnpm lint:extensions:*`、`lint:plugins:*`），理解 SDK 边界。

### 实验 5（排障，~2h）——用可观测性定位一次慢回复

对应 [07](07-observability.md)、[08](08-saas-and-ops.md)。

- [ ] 按 `xiaoaiclaw-ops/logging-guide.md` 打开模型交互记录（cache-trace / anthropic-payload）。
- [ ] 挑一次真实 run，从 trace 里回答：首 token 慢在哪？工具占了多少时间？（参考 `docs/xiaoai/knowledge/ttft-attribution-analysis.md` 的归因方法）
- [ ] 对照 Grafana 看板（`xiaoaiclaw-ops/monitoring/grafana/`）说出每个面板对应链路的哪一段。

### 实验 6（二开，~半天）——SaaS 会话存储排障推演

对应 [08](08-saas-and-ops.md)。

- [ ] 精读 `docs/xiaoai/session-sync-redis-jfs-double-write.md`，列出该系统 **所有会写 session 状态的位置**。
- [ ] 推演：如果"用户跨天后上下文分裂"，按文档的根因分类你会先查哪里？
- [ ] 读 `xiaoaiclaw-ops/upstream-tracker.md`，找出"上游同步中最容易丢的二开逻辑"及其 grep 检查命令。

## B. 自查清单（学完能不能答出来）

**概念层**

1. Gateway、Channel、Agent run、Session、Tool、Skill、Plugin 各是什么，边界在哪？
2. 上游目录与二开目录分别有哪些？为什么这样分？
3. 消息入站后到生成回复，中间经过哪几个大阶段？
4. 一个 sessionKey 由哪些部分组成？多用户隔离靠什么？
5. 工具循环为什么会熔断？阈值与处置是什么？

**代码层**

6. `runEmbeddedPiAgent` 和 `runEmbeddedAttempt` 的分工？
7. 工具的 TypeBox schema 为什么禁 `Type.Union`？
8. 插件的 `before_tool_call` 返回 `{block:false}` 会发生什么？
9. Skill 的六级加载来源与优先级？
10. 模型故障转移的两级机制分别在哪个文件实现？

**运维层（二开同学必答）**

11. SaaS 三条 RPC 路径及各自的调用方？
12. `session.storeMode=redis` 时 key 的命名空间与类型？如何回滚？
13. 镜像里 extensions 为什么只保留 7 个？合上游时要检查什么？
14. 观测性受保护面包括哪些文件？改动时要走什么流程？
15. "小爱 APP 发消息无回复"按 SOP 从哪三步开始查？

## C. 术语表

| 术语 | 解释 |
|---|---|
| **Gateway** | 常驻控制面进程：管理渠道连接、客户端 WS、路由、会话与 agent 调度 |
| **Node（节点）** | 以 `role: node` 接入的设备（iOS/Android/macOS），提供摄像头/定位/画布等能力 |
| **Channel（渠道）** | 一个消息平台的适配实现（telegram/feishu/web…），内置或扩展包形态 |
| **Pairing（配对）** | 客户端/设备接入 Gateway 的设备级审批与令牌机制 |
| **sessionKey** | 会话路由键；SaaS 格式 `agent:{agentId}:{channel}:{groupId}:u-{userId}` |
| **Transcript** | 会话消息流水（`<sessionId>.jsonl`），含 assistant/tool 消息 |
| **Agent run / attempt** | 一次"消息→回复"的执行；attempt 是 run 内一次完整尝试（含故障转移重试） |
| **Agentic loop / pi loop** | 模型推理↔工具调用的循环，由 pi-agent-core 执行 |
| **pi-agent-core / pi-ai / pi-coding-agent** | 嵌入的上游 agent SDK 三件套（loop / 模型流式 / 编码工具） |
| **Tool / tool profile** | 模型可调用能力；profile（minimal/coding/messaging/full）决定工具集合 |
| **Tool loop guard** | 工具循环熔断：重复调用哈希检测，warning/critical/熔断三级 |
| **Skill** | `SKILL.md` 技能说明书，运行时注入 prompt 或按需读取 |
| **Skill snapshot** | 某次 run 使用的 skills 冻结快照（热更新注意点见 ops 文档） |
| **Plugin / Extension** | 扩展代码；extension = workspace 独立包形态的 bundled plugin |
| **plugin-sdk** | 插件作者的公开 API 面（子路径导出 + baseline 锁面） |
| **Activation boundary** | 插件运行时只经 SDK facade 进入的隔离约束 |
| **Compaction** | 长对话摘要压缩（含 Memory Flush、溢出触发、后台压缩） |
| **Auth profile** | 一份模型鉴权凭据（API key/OAuth），支持轮换与冷却 |
| **Model failover** | 两级故障转移：鉴权 profile 轮换 → 模型 fallbacks |
| **Queue modes** | 入站消息排队语义：collect/steer/followup/steer-backlog/interrupt |
| **Bootstrap files** | 首轮注入的工作区文件（AGENTS.md/SOUL.md/IDENTITY.md/USER.md…） |
| **[二开] SaaS / maskUid** | 多租户形态；用户脱敏 ID，全链路隔离键 |
| **[二开] ai-claw-proxy** | Java 接入层服务，SaaS RPC 的调用方 |
| **[二开] 快系统** | 与助手并存的另一业务系统，会注入/共享 session（见 session-sync 文档） |
| **[二开] JFS / Redis store** | 会话存储后端（JuiceFS 文件 / P4T Redis，`mibrain:claw:` 前缀） |
| **[二开] Memory-cloud** | 云端记忆插件（`plugins-local/memory-cloud`），LongTerm/Daily 记忆 |
| **[二开] device-model-routing** | 按设备类型（Phone/SoundBox/TV/Automotive…）改写模型与 thinking |
| **Observability 受保护面** | 触碰需走 OTel review checklist 的核心文件清单（见 `AGENTS.md`） |

## D. 学完之后的进阶路径

1. **成为某个子系统的"本地专家"**：从第 3 章的主链路挑一段（如工具系统 / 会话存储 / 渠道），精读其目录 + 测试，写一份自己的"该子系统 20 问"。
2. **修一个 `xiaoaiclaw-ops/upstream-tracker.md` 里的上游 bug**：完整走一遍"定位 → 上游 PR → 同步纪律"。
3. **做一次性能归因**：按 `docs/xiaoai/knowledge/` 的归因方法论（TTFT / 工具 IO / 首包）产出一份分析报告。
4. **参与上游同步**：跟一次每周同步（`branch-strategy.md` 附录 A），这是理解全仓最快的方式。
