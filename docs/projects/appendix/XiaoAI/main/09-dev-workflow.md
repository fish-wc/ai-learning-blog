# 09 · 开发工作流：构建、测试、CI、上游同步

> 目标：读完本章，你能跑起本地开发循环、写出符合本仓库规范的改动，并知道怎么跟上游更新。
> 事实来源：根 `package.json` scripts、`AGENTS.md`、`xiaoaiclaw-ops/branch-strategy.md`、`xiaoaiclaw-ops/upstream-tracker.md`。

## 1. 环境与安装

- 运行时：**Node 24（推荐）或 22.16+**（`AGENTS.md` 要求 Node 22+ 且保持 Node/Bun 两条路径可用）。
- 包管理：**pnpm**（monorepo）；执行 TS 脚本优先 **Bun**（`bun <file.ts>` / `bunx`）。
- 安装：`pnpm install`。若报 `vitest not found`、`node_modules` 缺失等，先 `pnpm install` 再重试。

```bash
pnpm install
pnpm ui:build        # 首次会自动装 UI 依赖
pnpm build
pnpm xiaoaiclaw onboard --install-daemon   # 可选：装本地 daemon
```

## 2. 日常开发循环

| 我想…… | 命令 |
|---|---|
| 起 Gateway 并热重载 | `pnpm gateway:watch` |
| 起 Gateway（跳过渠道加载） | `pnpm gateway:dev` |
| 跑 CLI | `pnpm xiaoaiclaw ...` 或 `pnpm dev` |
| 起 TUI | `pnpm tui`（dev profile：`pnpm tui:dev`） |
| 起 Web 控制台 | `pnpm ui:dev` |
| 类型检查 | `pnpm tsgo` |
| lint + 格式 + 一批边界检查 | `pnpm check` |
| 自动修格式 | `pnpm format:fix`（oxfmt） |
| 跑测试 | `pnpm test` |
| 内存紧张时跑测试 | `XIAOAICLAW_TEST_PROFILE=low XIAOAICLAW_TEST_SERIAL_GATEWAY=1 pnpm test` |

> `pnpm check` 不只是 lint：它还串了 `check:no-conflict-markers`、webhook 鉴权顺序、pairing 鉴权范围等 **安全边界检查**（脚本在 `scripts/`，如 `check-webhook-auth-body-order.mjs`）。提交前务必跑。

## 3. 构建流水线（`pnpm build` 里发生了什么）

`build` 是一长串 codegen + 产物组装（`scripts/tsdown-build.mjs` 主导），理解顺序即可，不必一次看懂：

1. `canvas:a2ui:bundle` —— 打包 Canvas A2UI 资源；
2. `tsdown-build` + `runtime-postbuild` + `build-stamp` —— 编译、运行时后处理、写入构建戳；
3. `build:plugin-sdk:dts` + `check-plugin-sdk-exports` —— 生成插件 SDK 类型声明并校验导出（**保证插件 API 面稳定**）；
4. `copy-hook-metadata` / `copy-export-html-templates` / `write-build-info` / `write-cli-startup-metadata` / `write-cli-compat` —— 拷贝元数据、生成 CLI 启动与兼容信息。

衍生命令：`pnpm build:docker`（跳过插件 SDK dts，容器构建用）、`pnpm protocol:gen`（TypeBox → JSON Schema → Swift 模型，`pnpm protocol:check` 校验生成物无漂移）。

## 4. 测试体系（Vitest，多配置分流）

仓库有多个 vitest 配置文件，按测试面拆分（并发/内存都有讲究）：

| 配置 | 覆盖面 | 命令 |
|---|---|---|
| `vitest.unit.config.ts` | 单元（含覆盖率门槛，见下方说明） | `pnpm test` / `pnpm test:fast` / `pnpm test:coverage` |
| `vitest.gateway.config.ts` | Gateway 协议/鉴权等 | `pnpm test:gateway` |
| `vitest.contracts.config.ts` | 插件/通道契约测试 | `pnpm test:contracts` |
| `vitest.extensions.config.ts` | extensions 包 | `pnpm test:extensions` |
| `vitest.e2e.config.ts` | 端到端 | `pnpm test:e2e` |
| `vitest.live.config.ts` | 真实外部依赖（需 `XIAOAICLAW_LIVE_TEST=1`） | `pnpm test:live` |
| Docker 系列 | `scripts/e2e/*-docker.sh` | `pnpm test:docker:all` |

约定（`AGENTS.md`）：

- 测试与源码 **同目录 colocate**：`*.test.ts`，e2e 用 `*.e2e.test.ts`；
- 测试里用 **实例级 stub**，禁止原型污染；
- 触逻辑的改动，push 前必须 `pnpm test`。
- 只测改动影响面：`pnpm test:changed`（对比 `origin/main`）。

**关于覆盖率门槛（精确版）**：`AGENTS.md` 写的目标是 lines/branches/functions/statements 70%，但**当前实际强制执行**的是 `coverage-thresholds.cjs` 里的 **per-file 阈值**（由真实全量覆盖率向下取 5 的倍数生成）：默认 30，热目录 `src/saas|agents|gateway|auto-reply|infra` 的文件 60（lines/statements），个别文件最高 90、少数豁免为 0（文件内注释说明原因）。该文件被 `vitest.config.ts` 与 `ci/test.sh` 共用。另外 GitLab MR 卡口要求**增量覆盖率 ≥ 30%**（见 `xiaoaiclaw-ops/branch-strategy.md`）。

其他质量门（CI 会跑）：死代码扫描 `pnpm deadcode:knip`、重复代码 `pnpm dup:check`、单文件行数上限 `pnpm check:loc --max 500`、CLI 启动耗时/内存预算 `test:startup:*`、协议生成物一致性 `pnpm protocol:check`。

## 5. 编码规范要点（全文见 `AGENTS.md`）

- TypeScript ESM 严格类型；**禁止 `@ts-nocheck`、禁止关掉 `no-explicit-any`**。
- 文件保持精练（目标 ~500 行）；tricky 逻辑加简短注释。
- 命名：产品标题用 **XiaoAIClaw**；CLI/包/路径/配置键用 `xiaoaiclaw`。
- 工具 schema 约束：**不用 `Type.Union`/`anyOf`**；字符串枚举用 `stringEnum`/`optionalStringEnum`；可选用 `Type.Optional(...)` 而非 `| null`。
- **镜像复制规则**：代码里写 `// mirror xxx.ts` 之类"照搬另一文件"的逻辑时，**必须先读参考文件**再写，防止复制出参考文件已修复的 bug。
- CLI 进度条统一用 `src/cli/progress.ts`；表格/ANSI 用 `src/terminal/table.ts`；颜色用 `src/terminal/palette.ts`。
- **绝不向外部消息面发送流式/半成品回复**，只发最终回复。

## 6. 提交与 MR

- 提交信息：简洁、动词开头（如 `CLI: add verbose flag to send`）；相关改动打包，别夹带无关重构。
- 依赖：`pnpm.patchedDependencies` 必须锁死版本（禁 `^`/`~`）；**打依赖补丁需显式批准**。
- **观测面保护**：改动涉及 gateway message-handler、server-methods、chat、get-reply-run、agent-runner-execution、attempt.ts、handlers.tools、handlers.compaction、diagnostics-otel 时，必须走 `docs/xiaoai/feature-enhance-otel-traces.md` §6 的 **观测性 review checklist**（受保护面 + run contract 测试）。详见第 7 章。
- Git 多 agent 安全规约（本仓库常用 AI 协作开发）：不随便 `git stash`、不随便切分支/worktree、"push" 才 pull --rebase、"commit" 只圈自己的改动（见 `AGENTS.md` "Git Notes"）。

## 7. [二开] 与上游同步

上游更新合入是本仓库的常态工作，流程要点：

1. **分支策略**见 `xiaoaiclaw-ops/branch-strategy.md`；上游 rebase 时 `AGENTS.md` 预期冲突 —— 用我们的版本覆盖。
2. `xiaoaiclaw-ops/` **不会**与上游冲突 —— 这就是二开内容都放这里的原因。
3. 合入上游后检查 `xiaoaiclaw-ops/upstream-tracker.md`：确认哪些已打的 workaround 可以移除。
4. 上游新增 extension 时，评估是否加入 `xiaoaiclaw-ops/Dockerfile` 的保留列表。
5. 上游变更日志/同步历史在 `xiaoaiclaw-ops/sync-history/`。

## 8. 动手试试

```bash
pnpm check && pnpm test:fast        # 提交前的标准组合
pnpm test:changed                   # 只跑受影响测试
pnpm protocol:check                 # 生成物一致性检查（会 git diff --exit-code）
```

小实验（练手 MR 级别）：给某个内置工具加一个参数 —— 走一遍"改 schema → 补测试 → `pnpm check` → `pnpm test`"，体会约束（如 schema 禁 Union）是怎么在 lint 里强制执行的。

## 9. 深入阅读

- `AGENTS.md`（权威规范）、`CONTRIBUTING.md`
- `xiaoaiclaw-ops/branch-strategy.md`、`xiaoaiclaw-ops/upstream-tracker.md`
- `ci-unit-test.md`、`ci/` 目录（CI 流水线细节）
- `docs/xiaoai/feature-enhance-otel-traces.md` §6（观测面 checklist）
