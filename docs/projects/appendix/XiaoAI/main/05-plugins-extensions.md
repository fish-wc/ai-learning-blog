# 05 · 插件与扩展体系

> 目标：读完本章，你能分清 plugin / extension / SDK 三者的关系，知道扩展点都在哪，并能照着模板写出自己的扩展。
> 前置：[04-agents-tools-skills.md](04-agents-tools-skills.md)（工具与 run 的概念）。

## 1. 三个容易混淆的东西

| 概念                                | 位置                | 是什么                                                                                               |
| ----------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------- |
| **插件宿主（plugin host）**   | `src/plugins/`    | 上游核心：发现、manifest 校验、启停、加载（jiti，进程内）、注册表、钩子、安装/市场                   |
| **插件 SDK（给插件作者用）**  | `src/plugin-sdk/` | 窄而稳的公开 API 面：`xiaoaiclaw/plugin-sdk/<subpath>` 子路径导出 + API baseline 锁面              |
| **扩展包（bundled plugins）** | `extensions/*`    | ~90 个 workspace 独立包：模型 provider、聊天渠道、能力组件                                           |
| **[二开] 本地插件**           | `plugins-local/`  | `memory-cloud`（云端记忆）、`system-control`（IoT/系统工具门控），经 `plugins.load.paths` 加载 |

还有一个历史遗留：`src/extensionAPI.ts`（32 行）只是**废弃兼容 shim**（会打警告），真正的 API 是插件入口的 `register(api)` 回调。新代码不要用它。

## 2. 插件长什么样

每个扩展包 = 目录 + 三件套：

1. **`xiaoaiclaw.plugin.json`** —— 声明式 manifest（**manifest-first**：不执行插件代码就能完成配置校验）。含 id、`providers`/`channels`/`cliBackends`、`providerAuthEnvVars`、`skills`、`configSchema` 等。
2. **`package.json` 的 `"xiaoaiclaw"` 块** —— 入口、渠道元数据、安装/打包/发布配置。
3. **`index.ts`** —— 调 `definePluginEntry` / `defineChannelPluginEntry`，在 `register(api)` 里注册能力。

扩展包三大家族：

- **模型 provider**：`anthropic`、`openai`、`google`、`ollama`、`moonshot`、`volcengine`、`xai`、`deepseek`、`github-copilot`、`xiaomi` …
- **聊天渠道**：`feishu`、`msteams`、`telegram`、`slack`、`discord`、`matrix`、`whatsapp`、`qqbot`、`line` …
- **能力组件**：`memory-core`/`memory-lancedb`、`browser`、`image-generation-core`、`speech-core`、`media-understanding-core`、`diagnostics-otel`（OTel 观测）、`llm-task`、`personal-task`、`voice-call`、`thread-ownership`、`diffs`、`custom-prompt`、`openshell`、`acpx` …

好例子：`extensions/anthropic/`（manifest + `stream-wrappers.ts` + `api.ts`，看 provider 怎么包模型怪癖）、`extensions/feishu/`（`index.ts` 用 `api.registerTool` 注册 bitable/chat/docx/drive/wiki 工具）。

## 3. 插件 API 面（`register(api)` 能做什么）

```
api.registerProvider(...)                 # 模型 provider（catalog、鉴权、流包装）
api.registerChannel(...)                  # 聊天渠道
api.registerTool(tool, {optional})        # Agent 工具（有 pluginToolAllowlist 门控）
api.registerCommand / registerCli / registerCliBackend
api.registerHook / on(hookName)           # 生命周期钩子
api.registerHttpRoute / registerGatewayMethod / registerService / registerInteractiveHandler
api.registerSpeechProvider / registerMediaUnderstandingProvider / registerImageGenerationProvider
api.registerWebSearchProvider / registerContextEngine / registerMemory*
```

完整表格：`docs/plugins/sdk-overview.md`。**决策语义要记牢**：钩子返回 `{block:true}` / `{cancel:true}` 是终局的、停止后续低优先级处理器；`{block:false}` 是 **no-op，不会解除之前的 block**。

## 4. 插件钩子：在 agent 生命周期里"插一脚"

按执行时机（详见 `docs/concepts/agent-loop.md` "Hook points"）：

| 钩子                                                                                     | 时机 / 用途                                                                                                 |
| ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `before_model_resolve`                                                                 | 会话建立前（无 messages），**确定性**改写 provider/model                                              |
| `before_prompt_build`                                                                  | 加载会话后，注入 `prependContext` / `systemPrompt` / `prependSystemContext` / `appendSystemContext` |
| `before_agent_reply`                                                                   | LLM 调用前，插件可"截胡"返回合成回复或让本轮静默                                                            |
| `before_tool_call` / `after_tool_call`                                               | 拦截/改写工具参数与结果                                                                                     |
| `tool_result_persist`                                                                  | 写入 transcript 前**同步**变换工具结果                                                                |
| `before_compaction` / `after_compaction`                                             | 观察/标注压缩                                                                                               |
| `before_install`                                                                       | 安装 skill/plugin 前审查，可拦截                                                                            |
| `message_received` / `message_sending` / `message_sent`                            | 出入站消息                                                                                                  |
| `session_start` / `session_end`、`gateway_start` / `gateway_stop`、`agent_end` | 生命周期边界                                                                                                |

另有 **Gateway hooks**（内部钩子，脚本型）：`agent:bootstrap`（增删 bootstrap 上下文文件）、`/new`、`/reset` 等命令事件（`docs/automation/hooks.md`）。

注册表与运行器：`src/plugins/registry.ts`（注册记录类型：tool/command/http route/channel/provider/cliBackend/speech/mediaUnderstanding/imageGeneration/webFetch/webSearch/hook/service）、`src/plugins/hooks.ts:186 createHookRunner`（类型化钩子执行器）。

## 5. 加载管线与激活边界

```
发现（src/plugins/discovery.ts:894 discoverXiaoAIClawPlugins）
   来源：bundled / 全局 / workspace / plugins.load.paths（带路径安全检查 + 拒绝硬链接）
 → 启停与校验（src/plugins/config-state.ts；独占槽位如 memory）
 → jiti 进程内加载 + register(api)（src/plugins/loader.ts:437/916 → registry.ts）
 → 各消费面取用：
     providers  → 模型目录/鉴权（models-config.ts、auth-profiles）+ 流包装
     channels   → src/channels/routing + 共享 message 工具动作
     tools      → resolvePluginTools（src/plugins/tools.ts:72）并入 createXiaoAIClawTools
     skills     → resolvePluginSkillDirs 并入 skill 快照
     hooks      → createHookRunner → before_* 各钩子
```

**激活边界（activation boundary）**：bundled 插件运行时只能经 plugin-sdk facade 门面进入（`src/plugin-sdk/facade-runtime.ts` 是唯一入口），`src/plugin-activation-boundary.test.ts` 锁死 —— 导入核心模块不得连带拉起插件运行时。SDK 导出面由 `sdk-alias.ts` + `api-baseline.ts` 校验（`pnpm plugin-sdk:check-exports` 等）。

**import 边界 lint**（`package.json` scripts 里的 `lint:extensions:*`、`lint:plugins:*`）强制：extension 不得相对路径引用包外、不得直接引 `src/` 内部模块；plugin 不得引 extension 内部实现等。违反会在 `pnpm check` 挂掉。

## 6. 扩展点速查："我想……"

| 我想…              | 怎么做                                                                                                                                                         | 参考                                                                |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 加一个 Agent 工具   | `src/agents/tools/` 写工厂函数（TypeBox schema + `stringEnum`），在 `xiaoaiclaw-tools.ts` 注册 + `tool-catalog.ts` 加条目；或插件 `api.registerTool` | `src/agents/tools/runtime-tools.ts`、`sessions-history-tool.ts` |
| 写一个 Skill        | 目录 + `SKILL.md`，放 `<workspace>/skills` 或随插件发                                                                                                      | `docs/tools/skills.md`                                            |
| 写一个插件          | `definePluginEntry` + manifest；只 import `xiaoaiclaw/plugin-sdk/*`；用钩子做生命周期拦截                                                                  | `docs/plugins/building-plugins.md`                                |
| 接一个模型 provider | `api.registerProvider({catalog, prepareExtraParams, wrapStreamFn, formatApiKey, refreshOAuth, …})`                                                          | `extensions/anthropic/`、`docs/concepts/model-providers.md`     |
| 接一个聊天渠道      | `defineChannelPluginEntry` + `buildChannelConfigSchema`                                                                                                    | `extensions/feishu/`、`docs/plugins/sdk-channel-plugins.md`     |
| 外接 MCP server     | 配置 `mcp.servers` 或 `xiaoaiclaw mcp servers set`                                                                                                         | 第 4 章 §7                                                         |
| 拦截/改写 prompt    | `before_prompt_build`                                                                                                                                        | `docs/plugins/sdk-overview.md`                                    |
| 做工具安全门控      | `before_tool_call`（`{block:true}`）                                                                                                                       | 参考 `plugins-local/system-control`                               |

## 7. 测试与规范

- 插件测试工具：`src/plugin-sdk/testing.ts`，跑 `pnpm test:extensions`（`vitest.extensions.config.ts`）。
- 契约测试：`pnpm test:contracts:plugins`、`test:contracts:channels`（防上下游接口漂移）。
- 边界规范（各自目录的 `AGENTS.md`）：`src/plugin-sdk/AGENTS.md`、`extensions/AGENTS.md`。
- 版本同步：`pnpm plugins:sync`；发布检查 `release:plugins:npm:*`。

## 8. 动手试试

1. 读一个最小插件的全链路：`extensions/anthropic/`（manifest → 入口 → 流包装 → 注册）。
2. 写一个 hook-only 插件：在 `before_tool_call` 里拦截某个工具并打日志；用 `plugins.load.paths` 加载，跑 `pnpm gateway:dev` 验证。
3. 跑 `pnpm lint:extensions:no-src-outside-plugin-sdk` 等边界检查，故意违规一次看报错信息（理解约束在护什么）。

## 9. 深入阅读

- `docs/plugins/architecture.md`（插件形态与 4 层管线）、`docs/plugins/sdk-overview.md`（API 全表）
- `docs/plugins/building-plugins.md`、`sdk-provider-plugins.md`、`sdk-channel-plugins.md`、`manifest.md`、`sdk-migration.md`
- 下一章：[06-config-and-cli.md](06-config-and-cli.md)
