# 07 · 日志与可观测性

> 目标：读完本章，你知道日志开关在哪、一次请求在 trace 里长什么样、改观测代码要守什么规矩。
> 本仓库的可观测性是 **[二开] 重点投入方向**，也是改动风险最高的面之一。

## 1. 日志（`src/logging/`）

**最佳单一文档：`xiaoaiclaw-ops/logging-guide.md`**（架构 + 全部开关 + 优先级，628 行）。这里只给地图。

### 1.1 架构

- 基于 **tslog**，**双通道**：文件与控制台各有独立级别。
- 关键源码：`src/logging/logger.ts`（初始化 + 文件滚动）、`console.ts`（控制台接管）、`subsystem.ts`（`SubsystemLogger`，主 API，子系统前缀 `[gateway]`）、`levels.ts`（silent>fatal>error>warn>info>debug>trace）、`src/logger.ts`（`logInfo/logWarn/logError` 便捷封装）、`access-log.ts`（stdout 访问日志）、`redact*.ts`（脱敏）。

### 1.2 文件日志

- JSONL 格式；默认目录 `XIAOAICLAW_LOG_DIR` → `/tmp/xiaoaiclaw`（生产设 `/home/work/log`，见 `xiaoaiclaw-ops/k8s/configmap.yaml`）。
- **滚动 + 分片**（`docs/xiaoai/feature-log-rolling-sharding.md`）：按小时（默认）或天滚动，`maxShardBytes` 默认 200MB（片内再分片）、`maxFileBytes` 默认 500MB（超限抑制写入并告警），旧文件 24h 清理。

### 1.3 级别优先级（记这张表）

| 通道 | 优先级（高 → 低） |
|---|---|
| 文件 | `XIAOAICLAW_LOG_LEVEL`（或 `--log-level`，启动时写入该 env）> `logging.level` > `info` |
| 控制台 | `XIAOAICLAW_LOG_LEVEL` > `--verbose/--debug`（强制 debug）> `logging.consoleLevel` > `info` |

注意：**`--verbose` 从不影响文件日志**。控制台样式 `logging.consoleStyle: pretty|compact|json`；脱敏 `logging.redactSensitive: off|tools` + `redactPatterns`（工具侧打码：留前 6 后 4）。

### 1.4 怎么读日志

```bash
pnpm xiaoaiclaw logs --follow [--json|--plain]   # 经 RPC logs.tail
pnpm xiaoaiclaw channels logs --channel feishu   # 单渠道
```

Control UI 的 Logs 标签页；Gateway WS 调用日志开关 `--ws-log auto|compact|full`（正常模式只打错误和 ≥50ms 的慢调用）。

## 2. Diagnostics 事件与 OTel 总览

- **Diagnostics = 进程内结构化事件**（`src/infra/diagnostic-events.ts` 目录：`model.usage`、`webhook.*`、`message.queued/processed`、`queue.lane.*`、`session.stuck`、`run.attempt`、`tool.invocation`、`tool.loop`、`skill.invoked`、`memory.embedding.*`…），喂给 exporter，**不替代日志**。
- **Exporter 是插件**：`extensions/diagnostics-otel/`（`src/service.ts`、`span-descriptions.ts`）。启用：`plugins.entries.diagnostics-otel.enabled` + `diagnostics.enabled` + `diagnostics.otel.{enabled,endpoint,protocol,serviceName,traces,metrics,logs,sampleRate,…}`；env 支持 `OTEL_EXPORTER_OTLP_ENDPOINT` 等。
- 诊断开关：`diagnostics.flags: [...]` / `XIAOAICLAW_DIAGNOSTICS`。

## 3. [二开] Trace（`docs/xiaoai/feature-enhance-otel-traces.md`，必读）

### 3.1 Span 树（一个请求一棵）

```
gateway.request（root，trace_id 来自 ws.request_id / saas.request_id）
├── message.queued（+ saas.global.queue）
├── agent.run
│   ├── session.load / skill.load / bootstrap.read / agent.session.create / system_prompt.build
│   ├── saas.session.queue / saas.lane.queue          ← [二开] 排队等待
│   ├── model.usage ×N                                 ← 每次 LLM 调用恰好一个
│   ├── tool.invoke ×K / tool.loop / skill.invoke
│   └── compaction / lifecycle.error.grace / memory.embedding.query
└── memory.flush
独立 span：webhook.processed/error、session.stuck、memory.embedding.fallback
独立 trace：shadow.llm.benchmark → shadow.llm.call       ← 影子模型对比
```

- 22 种 span 及完整属性表见文档 §1.3/§1.4；属性前缀 `xiaoaiclaw.*`，如 `xiaoaiclaw.tokens.*`、`xiaoaiclaw.model.first_token_ms`、`xiaoaiclaw.model.model_request_id`（= `X-Model-Request-Id` 响应头，**日志↔trace 对账用**）、`xiaoaiclaw.queue_wait_ms`（= `sum(qb.*)` 排队分解）。
- **契约（contract tests 锁定）**：`agent.run` 由 `message.processed` 收尾（`chat.send` 类请求**不在** `ws.request.completed` 收）；`model.usage` 必须是 `agent.run` 的子 span，每次 LLM 调用一个（由 `run/attempt.ts` 的流包装处发出）。

### 3.2 配置

`diagnostics.otel.traces: true | {exportIdsAsAttributes, includePromptAndCompletion, llmInputThreshold(默认 100000), spanTimeoutMs(300000), excludeRequestTypes}`。平台侧经 OTLP 接 SkyWalking（`docs/xiaoai/feature-enhance-observability-to-platform.md`）。

### 3.3 ⚠️ 观测性受保护面与 Review Checklist

**这可能是本仓库最重要的一条开发纪律**（根 `AGENTS.md` 强制；由来见文档 §6.3，一次 revert 曾因未走 checklist 丢失观测逻辑）。

受保护的发射点：

| 文件 | 事件 |
|---|---|
| `src/gateway/server/ws-connection/message-handler.ts` | `ws.request.received` |
| `src/auto-reply/reply/agent-runner-execution.ts` | `run.attempt` |
| `src/agents/pi-embedded-runner/run/attempt.ts` | `model.usage` |
| `src/agents/pi-embedded-subscribe.handlers.tools.ts` | `tool.invocation` |
| `src/agents/pi-embedded-subscribe.handlers.compaction.ts` | compaction 事件 |
| `src/logging/diagnostic.ts` | `message.processed` |

requestId 链：`server-methods.ts` → `server-methods/chat.ts`（`setRequestIdForRun`）→ `get-reply-run.ts` → `attempt.ts`。

**触碰上述文件的大 MR / revert 必须**：diff 检查发射点与 requestId 链 → 跑契约测试（`pnpm test -- src/gateway/request-id-trace.contract.test.ts`、`extensions/diagnostics-otel/src/service.test.ts`、`src/agents/pi-embedded-subscribe.handlers.tools.test.ts`）→ 验证 SaaS 路径上 `model.usage` 仍在 `agent.run` 之下。

## 4. [二开] Metrics（`docs/xiaoai/feature-enhance-otel-metrics.md`）

- 配置：`diagnostics.otel.metrics: true | {modules: {agent,model,tool,skill,compaction}}`。
- **当前命名规范**：下划线、无产品前缀 —— `model_invoke_total`、`model_invoke_duration_ms`、`model_first_token_ms`、`model_input_tokens`/`model_output_tokens`、`model_cache_*`、`agent_run_total`、`tool_invoke_total/errors_total/duration_ms`、`skill_invoke_*`、`request_llm_calls_count`/`request_tool_calls_count`、`message_queue_wait_ms`、`compaction_*`、`session_transcript_chars`、`process_heap_used_bytes`/`process_rss_bytes`、`gc_pause_ms`。
  （注意：上游 `docs/logging.md` 里 `xiaoaiclaw.tokens` 之类是**旧上游命名**，以本节为准；命名差异速查 `xiaoaiclaw-ops/o11y/metrics-name-mismatch.md`。）
- 看板设计：`docs/xiaoai/grafana-dashboard-design.md`（可用性/延迟/QPS/组件/特性五维；单指标 + `outcome` 标签原则）；JSON 在 `xiaoaiclaw-ops/o11y/grafana/` 与 `xiaoaiclaw-ops/monitoring/grafana/`。
- **本地全家桶**（Jaeger + Prometheus + Grafana）：

```bash
cd xiaoaiclaw-ops/o11y && ./manage.sh start
# otel-collector :4318/:4317 · Jaeger :16686 · Prometheus :9090 · Grafana :3000
```

## 5. 排障方法论（[二开] 知识库精华）

| 问题 | 方法论 |
|---|---|
| 首 token / 首包慢 | `docs/xiaoai/knowledge/ttft-attribution-analysis.md`、`user-first-package-speak-latency-*`（分解到 queue_wait / LLM / TTS） |
| 工具调用慢 | `docs/xiaoai/knowledge/tool-io-attribution-analysis.md`、`web-call-tool-attribution-analysis.md` |
| trace 数据字典 | `docs/xiaoai/knowledge/claw-trace-data-dictionary.md` |
| 全链路延迟架构分析 | `docs/xiaoai/analysis/arch-latency-analysis-*.md` |
| 线上无回复 | `xiaoaiclaw-ops/incidents/debug-sop-proxy-no-response.md`（SOP + 根因速查卡 + 指标阈值） |
| 指标失联类事故 | `docs/xiaoai/issue-metrics-blackout-prod-20261014.md` |

模型交互深挖（`xiaoaiclaw-ops/logging-guide.md` §）：`cache-trace.jsonl`、`anthropic-payload.jsonl`、`commands.log`。

## 6. 动手试试

1. 起本地 o11y 全家桶 + `pnpm gateway:dev`，发一条消息，在 Jaeger 里找到这棵 span 树，说出每层对应第 3 章链路的哪一步。
2. 从 trace 取 `xiaoaiclaw.model.model_request_id`，去日志里把同一次 LLM 调用的原始 payload 找出来 —— 体会 log↔trace 对账。
3. 读 `docs/xiaoai/feature-enhance-otel-traces.md` §6.3（那次 revert 事故），对照根 `AGENTS.md` 的 commit 规则，想想如果你来设计防线还会加什么。
4. 算一次"排队等待占比"：用 `xiaoaiclaw.queue_wait_ms = sum(qb.*)` 验证分解求和的自洽性。

## 7. 深入阅读

`xiaoaiclaw-ops/logging-guide.md` → `docs/logging.md` → `docs/xiaoai/feature-enhance-otel-traces.md`（§1 span 树 → §2 配置 → §5 事件↔span 映射 → **§6 checklist**）→ `feature-enhance-otel-metrics.md` → `feature-enhance-observability-to-platform.md` → `grafana-dashboard-design.md` → `xiaoaiclaw-ops/o11y/AGENTS.md`

相关章节：[03-message-lifecycle.md](03-message-lifecycle.md)（span 树对应的代码链路）、[08-saas-and-ops.md](08-saas-and-ops.md)（监控看板与事故 SOP）
