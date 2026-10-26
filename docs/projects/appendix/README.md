# XiaoAIClaw 项目学习教程

> 面向新加入本仓库的同学，目标是让你在 **1～2 周内** 建立对项目的整体认知，并能独立定位代码、上手开发与排查问题。
> 本目录是"学习路线图 + 代码导读"，不替代权威文档；每章末尾都有 **"深入阅读"** 指向 `docs/`、`xiaoaiclaw-ops/` 中的原始文档。

## 这个项目是什么（30 秒版）

**XiaoAIClaw** 是一个跑在自己设备上的 **个人 AI 助手**：你在 WhatsApp / Telegram / Slack / 飞书 / 微信 等已有的聊天渠道上给它发消息，它调用大模型 + 工具（Shell、文件、浏览器、日程……）帮你干活并回复。

本仓库基于上游开源项目二次开发（下称"二开"）：

- **上游代码**：`src/`、`extensions/`、`docs/`、`ui/` 等 —— 个人单机场景的完整助手。
- **二开专属**：`xiaoaiclaw-ops/`、`docs/xiaoai/`、`src/saas/` 等 —— 面向公司 SaaS / 音箱等多用户场景的架构、部署、运维、可观测性与自研能力。

详见仓库根的 `AGENTS.md`（开发者必读）。

## 章节索引与学习顺序

| 顺序 | 章节 | 你会学到 | 建议用时 |
|---|---|---|---|
| 0 | [01-project-overview.md](01-project-overview.md) | 项目定位、上游 vs 二开、仓库地图、规模 | 半天 |
| 1 | [02-architecture.md](02-architecture.md) | 总体架构：Gateway / 通道 / Agent / 客户端 | 半天 |
| 2 | [03-message-lifecycle.md](03-message-lifecycle.md) | 一条消息从进入到回复的完整链路（核心） | 1 天 |
| 3 | [04-agents-tools-skills.md](04-agents-tools-skills.md) | Agent 运行时、工具系统、Skill 机制 | 1 天 |
| 4 | [05-plugins-extensions.md](05-plugins-extensions.md) | 插件 / extension / 各类扩展点 | 半天～1 天 |
| 5 | [06-config-and-cli.md](06-config-and-cli.md) | 配置体系、CLI、运行时磁盘状态 | 半天 |
| 6 | [07-observability.md](07-observability.md) | 日志、OTel 链路与指标、排障工具 | 半天 |
| 7 | [08-saas-and-ops.md](08-saas-and-ops.md) | 二开 SaaS 架构、部署运维、会话/记忆体系 | 1 天 |
| 8 | [09-dev-workflow.md](09-dev-workflow.md) | 构建、测试、CI、上游同步与贡献规范 | 半天 |
| 9 | [10-hands-on.md](10-hands-on.md) | 动手实验、自查清单、术语表 | 持续 |

**推荐阅读方式**：

1. 先通读 01、02 建立"地图感"，**不要**一上来就钻代码细节。
2. 第 3 章（消息生命周期）是全书核心：它把各子系统串成一条线。读完后再按需下钻。
3. 每章的 **"动手试试"** 小节尽量真的跑一遍 —— 这个项目跑起来不难，跑过一次胜过读十页。
4. 遇到陌生名词查 [10-hands-on.md](10-hands-on.md) 末尾的术语表。

## 目标读者

- 刚接触本仓库的后端 / 全端 / 算法工程同学。
- 假设你会 TypeScript、git、命令行；不要求了解上游项目。
- 如果你只负责二开 SaaS 侧，读完 01、02、03 后可直接跳到 08、07、09。

## 约定

- 文中代码位置一律写成 **仓库根相对路径**（如 `src/gateway/x.ts`），可直接在 IDE 中全局搜索定位。
- 文件后的 `:数字` 表示大致行号，随代码演进会漂移，以符号名为准。
- 标注 **[二开]** 的内容是小米侧专属；标注 **[上游]** 的是开源项目自带能力。
