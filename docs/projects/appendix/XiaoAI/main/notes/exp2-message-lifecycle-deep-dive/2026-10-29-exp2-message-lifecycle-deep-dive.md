# 实验 2 补课：一条消息的完整链路逐段详解

> 本文回答实验 2 的三个检查点。所有日志来自 **2026-10-29 上午的实测**（gateway 日志
> `/tmp/xiaoaiclaw/xiaoaiclaw-2026-10-29-11.log`），每一段链路都给出了代码出处。
> 读完你应该能独立回答：
>
> 1. 消息从进入到回复离开，经过哪几站？每站干什么？
> 2. 为什么我们的消息被路由进 `agent:dev:main` 这个 sessionKey？
> 3. 连发 3 条消息时系统内部发生了什么？"collect" 到底 collect 了什么？

---

## 先建立总览：一张地铁线路图

把整条链路想成一条地铁线，消息是乘客，共 **8 站**：

```
[客户端]                [网关]                                         [模型]
   │                      │                                             │
   │  ① req:agent         │                                             │
   ├─────────────────────►│ ② 参数校验 + 幂等去重                        │
   │                      │ ③ sessionKey 推导（路由）                    │
   │                      │ ④ 队列模式决策（steer/collect/...）           │
   │                      │ ⑤ runReplyAgent（编排层）                    │
   │                      │ ⑥ runEmbeddedPiAgent（两层 lane 排队）        │
   │                      │ ⑦ runEmbeddedAttempt（一次尝试）              │
   │                      │    └─ 工具循环 ──────────────────────────────►│
   │                      │       ◄─────────────────────────────────────┤
   │  ⑧ res:agent final   │ ⑨ 回复整理 + session 落盘 + 回包              │
   ◄──────────────────────┤                                             │
```

先记住各站的"站长"（负责的函数），后面逐站参观：

| 站           | 代码位置                                                                       | 职责                                       |
| ------------ | ------------------------------------------------------------------------------ | ------------------------------------------ |
| ① 客户端    | `src/commands/agent.ts`（CLI）                                               | 组装请求经 WS 发 `agent` method          |
| ② 网关入口  | `src/gateway/server-methods/agent.ts:281`                                    | 校验参数、幂等去重                         |
| ③ 路由      | `src/config/sessions/main-session.ts:36`                                     | 推导 sessionKey                            |
| ④ 队列决策  | `src/auto-reply/reply/queue/settings.ts:32`                                  | 决定 steer/followup/collect                |
| ⑤ 编排      | `src/auto-reply/reply/agent-runner.ts:66` `runReplyAgent`                  | 打字提示、压缩预检、记忆 flush、调用执行层 |
| ⑥ 两层 lane | `src/agents/pi-embedded-runner/run.ts:149`                                   | session lane + global lane 串行化          |
| ⑦ 尝试      | `src/agents/pi-embedded-runner/run/attempt.ts:1273` `runEmbeddedAttempt`   | 组装 session、skills、工具，启动 pi 循环   |
| 工具循环     | `node_modules/@mariozechner/pi-agent-core/dist/agent-loop.js`（`runLoop`） | LLM↔工具 反复迭代直到无 toolCall          |
| ⑨ 回复      | `dispatch-from-config.ts` → session 落盘 → WS `res`                      | 最终回复回客户端                           |

---

## 第 1 站：客户端发起（CLI 视角）

我们执行的命令：

```bash
pnpm xiaoaiclaw --dev agent --agent dev --message "链路追踪：用exec执行 echo pipeline-marker-1 然后只回复ok"
```

CLI 做的事：

1. 连上 dev Gateway（19001，握手过程见[实验 1 补课](2026-10-27-exp1-ws-handshake-and-transcript.md)）
2. 组装 `agent` method 请求：`{message, agentId: "dev", idempotencyKey, ...}`
3. 等待 `res`（最终回复）

注意：**CLI 不直接调模型**。它只是一位乘客，把消息递给网关就坐下等了。

## 第 2 站：网关收到 —— 校验 + 幂等去重

`src/gateway/server-methods/agent.ts:281` 开始的 `agent:` handler：

```ts
agent: async ({ params, ... }) => {
  if (!validateAgentParams(p)) { /* 参数不合法直接拒 */ }
  const cached = context.dedupe.get(`agent:${idem}`);   // 幂等去重
  ...
```

两个值得记住的防御：

- **参数校验**（TypeBox schema）—— 不合法的请求在门口就拒
- **幂等去重**（`context.dedupe.get("agent:" + idempotencyKey)`）—— 网络重试导致的重复请求，
  返回缓存的上次结果，不会跑两遍模型

## 第 3 站：路由 —— 为什么是 `agent:dev:main`？

这是检查点 2 的答案。sessionKey 的推导在
`src/gateway/server-methods/agent.ts:461-481`：

```ts
let requestedSessionKey =
  requestedSessionKeyRaw ??                       // ① 显式传了就用它
  resolveExplicitAgentSessionKey({ cfg, agentId }); // ② 否则从 agentId 推导
```

我们没传 `--session-key`，走 ②。`resolveExplicitAgentSessionKey`
（`src/config/sessions/main-session.ts:36`）的实现：

```ts
export function resolveExplicitAgentSessionKey(params) {
  const agentId = params.agentId?.trim();       // "dev"（来自 --agent dev）
  if (!agentId) return undefined;
  return resolveAgentMainSessionKey({ cfg, agentId });
}
// 最终：buildMainSessionKey(agentId, mainKey)
//      = `agent:${normalizeAgentId("dev")}:${normalizeMainKey(undefined)}`
//      = "agent:dev:main"
```

**为什么退化成三段格式？** 完整的 SaaS 格式是
`agent:{agentId}:{channel}:{groupId}:u-{userId}`（五段，见术语表）——渠道消息需要
"哪个渠道 + 哪个群 + 哪个用户"三要素隔离会话。而 CLI/TUI 是**直连网关的本地客户端**：

- 没有 channel（不经 telegram/feishu 等外部渠道）
- 没有 groupId（不在群里）
- 没有 userId（本地单用户）
  所以只剩 `agentId` + 默认 mainKey，即 `agent:dev:main`。

解析侧的代码（教程里提到的 `parseAgentSessionKey`，
`src/sessions/session-key-utils.ts:23`）：

```ts
const parts = raw.split(":").filter(Boolean);
if (parts[0] !== "agent") return null;
const agentId = parts[1];              // "dev"
const rest = parts.slice(2).join(":"); // "main"
return { agentId, rest };
```

它只认第一段必须是 `agent`，后面整体当 rest —— 所以三段五段都兼容。
配合 `resolveAgentIdFromSessionKey`（`src/routing/session-key.ts:73`）就能从
sessionKey 反查 agentId，这就是"会话按 agent 隔离"的实现基础。

**多用户隔离靠什么？**（自查清单第 4 题的答案也在这）：SaaS 模式下 sessionKey 末尾的
`u-{userId}`（maskUid）就是隔离键 —— 不同用户的同名 agent 会落到不同的
`agent:dev:telegram:g1:u-10086` 与 `agent:dev:telegram:g1:u-10010`，物理上就是两个
session 文件，互不可见。（实验 6 里"跨用户 cwd 错位"bug 的严重性也源于此 ——
sessionKey 与文件目录的映射错了，就是数据越权。）

## 第 4 站：队列模式决策

拿到 sessionKey 后，`get-reply-run.ts:786` 解析队列设置：

```ts
const resolvedQueue = resolveQueueSettings({ cfg, channel, sessionEntry, inlineMode, ... });
```

`resolveQueueSettings`（`src/auto-reply/reply/queue/settings.ts:32`）的优先级链
（**从高到低**）：

```
inlineMode（本条消息内联指定，如 /queue 指令）
  → sessionEntry.queueMode（会话级持久化覆盖）
  → queueCfg.byChannel[channel]（按渠道配置）
  → queueCfg.mode（全局配置）
  → defaultQueueModeForChannel()  ← 兜底，永远返回 "collect"
```

我们的 dev 配置没设 `messages.queue`，所以**走到兜底 = collect**。
这回答了检查点 3 的第一半：我们观察到的一切行为，都是 collect 模式的行为。

`get-reply-run.ts:850-856` 把模式翻译成两个布尔值：

```ts
const shouldSteer = mode === "steer" || mode === "steer-backlog";
const shouldFollowup = mode === "followup" || mode === "collect" || mode === "steer-backlog";
```

collect → `shouldSteer=false, shouldFollowup=true`。这两个值会传给下一站。

## 第 5 站：`runReplyAgent` —— 编排层

`src/auto-reply/reply/agent-runner.ts:66`。拿到 `shouldSteer/shouldFollowup/isActive`
后，先做一次**分流**（agent-runner.ts:209-250，这是 collect 语义的核心）：

```ts
if (shouldSteer && isStreaming) { ... }          // steer：注入进行中的 run（我们不走这）

const activeRunQueueAction = resolveActiveRunQueueAction({
  isActive,          // 该 session 是否有 run 正在跑
  shouldFollowup,    // collect=true
  queueMode,         // "collect"
});

if (activeRunQueueAction === "enqueue-followup") {
  enqueueFollowupRun(queueKey, followupRun, resolvedQueue, ...);  // ★ 挂到 followup 队列
  return undefined;   // ★ 本条消息的处理到此结束！
}
// 没有 active run 时才继续往下走，真正启动模型
```

**collect 的本质就在这**：如果当前 session 已有 run 在跑，新消息**不启动新 run**，
只是把 `followupRun`（消息 + 全部路由上下文打包）塞进 per-session 的 followup 队列，
然后立刻返回。等 active run 结束后，队列 drain 时用 `buildCollectPrompt`
（`src/utils/queue-helpers.ts:217`）把**积压的多条合并成一个 prompt**：

```ts
function buildCollectPrompt({ title, items, renderItem }) {
  const blocks = [title, ...items.map(renderItem)];
  return blocks.join("\n\n");      // 多条消息拼成一段带编号的合并 prompt
}
```

这就是队列文档（`docs/concepts/queue.md`）那句 "coalesce all queued messages into a
single followup turn" 的代码实现。**模型只被叫醒一次，一次看完所有积压消息** ——
省 token、避免并发写坏 session。

排队的阈值也有讲究：`debounceMs=1000`（默认，等消息流安静 1 秒再 drain，防止
"继续/继续"被拆成两轮）、`cap=20`（最多积压 20 条）、`drop="summarize"`（溢出时摘要）。

## 第 6 站：`runEmbeddedPiAgent` —— 两层 lane 排队

没有 active run 时（或 followup 队列 drain 后），进入
`src/agents/pi-embedded-runner/run.ts:149`。开头就是**两层嵌套排队**
（run.ts:169-186）：

```ts
const sessionLane = resolveSessionLane(sessionKey);   // "session:agent:dev:main"
return enqueueSession(() => {                          // 第一层：session lane
  return enqueueGlobal(async () => {                   // 第二层：global lane ("main")
    ... // 真正的 run 逻辑在这
  });
});
```

lane 的实现（`src/process/command-queue.ts`，`drainLane`/`pump`）：

- 每个 lane 是一个 FIFO 队列 + 并发上限（默认 1）
- **session lane**（`session:agent:dev:main`）：保证同一会话同时只有一个 run
  —— 保护 session 文件不被并发写坏
- **global lane**（`main`）：进程级总闸，默认并发 4（`agents.defaults.maxConcurrent`
  可调）—— 防止打爆模型配额

排队超 2 秒会打 warn 日志（command-queue.ts:187）：

```ts
diag.warn(`lane wait exceeded: lane=${lane} waitedMs=${waitedMs} queueAhead=${state.queue.length}`);
```

**这行就是我们连发实验里看到的证据**，第 9 节详细分析。

## 第 7 站：`runEmbeddedAttempt` —— 一次完整尝试

过了两层 lane，进入 `src/agents/pi-embedded-runner/run/attempt.ts:1273`。它做的是
"把一切准备好"：

1. 解析沙箱/workspace（`resolveSandboxContext`）
2. 加载 skills（`resolveEmbeddedRunSkillEntries` + 设备过滤）
3. 打开 session（`SessionManager`，读 `.jsonl`，redis 模式走 backend 分支）
4. `createAgentSession`（attempt.ts:2255）—— 创建 pi 的 agent session：
   挂工具（builtInTools + customTools）、模型、authStorage
5. 最后把 prompt 送进去（attempt.ts:3710）：

```ts
await abortable(activeSession.prompt(effectivePrompt, ...));
```

**attempt vs run 的分工**（自查清单第 6 题）：一个 **run** 是"消息→回复"的完整执行，
由 `runEmbeddedPiAgent` 负责（含排队、workspace、fallback 循环）；一个 **attempt** 是
run 内的**一次尝试** —— 模型故障转移时 run 会换 profile/model 再起一个 attempt
（run.ts:612 的 `runEmbeddedAttempt` 调用在循环里）。attempt 是"试一次"的粒度，
run 是"办成这件事"的粒度。

## 第 8 站：工具循环（pi-agent-core 的 runLoop）

`node_modules/@mariozechner/pi-agent-core/dist/agent-loop.js` 的 `runLoop`，
结构是双层 while：

```js
while (true) {                                   // 外层：处理 follow-up 消息
    let hasMoreToolCalls = true;
    while (hasMoreToolCalls || pendingMessages.length > 0) {   // 内层：LLM↔工具
        // 1. 注入 pending 消息（steering 的消息在这里插队）
        // 2. 流式调 LLM → assistant 消息
        const message = await streamAssistantResponse(...);
        // 3. 有 toolCall 就执行工具，结果作为 toolResult 追加
        const toolCalls = message.content.filter((c) => c.type === "toolCall");
        hasMoreToolCalls = toolCalls.length > 0;
        if (hasMoreToolCalls) {
            toolResults.push(...(await executeToolCalls(...)));
        }
        // stopReason=="end_turn" 且无 toolCall → 内层退出
    }
    // 外层检查 followup 队列，有就继续，没有 → agent_end
}
```

一轮循环 = 一次模型调用。模型说"我要用 exec"（`stopReason: "toolUse"`）→
执行工具 → 结果喂回去 → 模型再说 → … 直到模型纯文本收尾（`end_turn`）。
这就是术语表里 "Agentic loop / pi loop" 的真身。

工具调用本身还过我们实验 4 写的 hook（`before_tool_call`），审计日志就是那时打的。

## 第 9 站：回复投递 + session 落盘

模型流式输出 → `onPartialReply`/`onBlockReply` 回调一路上传（渠道侧可分段投递）→
最终文本经 `normalizeStreamingText` 清洗（剥 HEARTBEAT_OK 等内部 token）→
session 落盘（transcript append）→ 网关回 `res:agent final` → CLI 打印。

---

## 实测对照：把日志逐段对上链路

**实验命令**（2026-10-29 11:15:47 发出，26 秒后收到回复）：

```bash
pnpm xiaoaiclaw --dev agent --agent dev --message "链路追踪：用exec执行 echo pipeline-marker-1 然后只回复ok"
```

**网关日志**（`/tmp/xiaoaiclaw/xiaoaiclaw-2026-10-29-11.log`，已按时间排列，
右侧是我加的链路标注）：

```
11:16:04.571 ⇄ res ✓ agent 56ms runId=8d2a7b7d… conn=5f16212a…      ← 站②：method 收到（56ms 完成参数阶段）
11:16:04.665 [OPT-4] resolveModelAsync: success provider=anthropic    ← 站③④之间：解析默认模型
11:16:04.679 [OPT-5] repairSessionFileIfNeeded: repaired=false         ← 站⑦前奏：校验 session 文件
11:16:04.687 [OPT-5] SessionManager.open: entryCount=49                ← 站⑦：打开 session（49 条历史）
11:16:08.774 [exec-audit] {"runId":"8d2a7b7d…","tool":"exec",         ← 站⑧：工具循环第一轮，
              "command":"echo pipeline-marker-1"}                         模型发起了 exec，hook 拦截审计
11:16:08.780 exec saas guard: check … command="echo pipeline-marker-1" ← 站⑧：exec 工具自身的安全检查
11:16:13.644 session-store-lock wait=458ms path=…/sessions.json        ← 站⑨：写 session 元数据（短暂锁等待）
11:16:13.727 <concise_summary>ok</concise_summary>                    ← 站⑨：最终回复（两行是双通道日志）
11:16:13.729 client disconnected … durationMs=9518                    ← CLI 拿到回复断开
```

**逐段对时间轴**（总 26s，其中 CLI 冷启动 ~17s）：

| 窗口                     | 耗时  | 对应链路段                                                         |
| ------------------------ | ----- | ------------------------------------------------------------------ |
| 11:15:47 → 11:16:04     | ~17s  | CLI 进程启动 + models-config 预热（OPT-2 日志可见 elapsedMs=4267） |
| 11:16:04.5 → 11:16:04.7 | ~0.2s | 站②③④：校验/路由/队列决策 + 模型解析 + session 打开             |
| 11:16:04.7 → 11:16:08.8 | 4.1s  | 站⑦⑧ 第一轮：prompt 组装 → LLM 返回 toolCall                    |
| 11:16:08.8 附近          | 瞬时  | exec 执行 `echo`（本地命令，毫秒级）                             |
| 11:16:08.8 → 11:16:13.6 | 4.8s  | 站⑧ 第二轮：toolResult 喂回 → LLM 生成最终文本                   |
| 11:16:13.6 → 11:16:13.7 | ~0.1s | 站⑨：落盘 + 回包                                                  |

看到规律了吗：**耗时全在两次 LLM 调用（4.1s + 4.8s），工具本身瞬间完成**。
网关的处理（校验/路由/开 session）加起来不到 0.2 秒。这正是实验 5 归因方法
（TTFT 拆解）的结论来源。

顺带一提 `runId=8d2a7b7d…` —— 术语表里 "Agent run" 的实例 ID，从站②的日志
一路带到站⑧的审计行，是排查时串联整条链路的**线索钥匙**（grep 它就能捞出
这次 run 的全部日志）。

---

## 检查点 3 详解：连发 3 条消息实测

**实验方法**（模拟消息几乎同时到达）：

```bash
for i in 1 2 3; do
  (pnpm xiaoaiclaw --dev agent --agent dev --message "连发测试${i}：请只回复数字${i}" &)
done
```

**网关日志关键行**：

```
11:16:46 ⇄ res ✓ agent 73ms  runId=bb5759bf…   ← 三条几乎同时到达（46-47 行三条 res ✓）
11:16:46 ⇄ res ✓ agent 166ms runId=ff7d262f…
11:16:46 ⇄ res ✓ agent 212ms runId=318935aa…
11:16:46 [OPT-5] SessionManager.open: entryCount=53          ← 第一条率先进入 run
11:16:53 lane wait exceeded: lane=session:agent:dev:main     ← ★ 第二条的证据：
            waitedMs=7439 queueAhead=1                          在 session lane 排队等了 7.4 秒
11:16:54 <concise_summary>2</concise_summary>                ← 第二条回复
11:17:08 lane wait exceeded: lane=session:agent:dev:main     ← ★ 第三条：
            waitedMs=21569 queueAhead=0                         等了 21.6 秒（前面两条累计）
11:17:08 <concise_summary>1</concise_summary>                ← 第三条回复
```

**这段日志讲了什么**：

1. 三条消息同一秒到达，网关都正常接收（三条 `res ✓ agent`，参数阶段各自 ~100ms）
2. 第一条立刻开始 run（entryCount=53，session 被打开）
3. 第二、三条撞上 **session lane 互斥**（`session:agent:dev:main` 并发=1），
   排队等前面跑完 —— `queueAhead=1` 说明前面还有 1 条，`waitedMs` 是纯等待时间
4. 回复顺序严格按获得 lane 的顺序：先到的先跑

**transcript 里的写入顺序**（`~/.xiaoaiclaw-dev/agents/dev/sessions/23dfc…jsonl`）：

```
03:16:46 连发测试2：请只回复数字2    ← 先进入 run 的是"测试2"（OS 调度随机）
03:16:54 连发测试1：请只回复数字1
03:17:08 连发测试3：请只回复数字3
```

（注意顺序是 2→1→3：三个后台进程谁先抢到 WS 不是确定的，但这正说明
**lane 保证了"谁先入队谁先跑、同一时刻只有一条在跑"**，入队顺序本身不受控。）

**为什么这次看到的是"排队"而不是"合并"？** —— 微妙但重要的区别：

- 我们的 3 条消息来自**3 个独立 CLI 进程**，每条都有自己的 WS 连接和 runId，
  它们竞争同一个 session lane，行为是**串行排队**（lane 语义）
- collect 的"合并"发生在**另一个层面**：当一条消息到达时 session 已有 active run，
  它会进 followup 队列被**合并成下一个 prompt**（第 5 站的 enqueueFollowupRun +
  buildCollectPrompt）。我们的场景里三条几乎同时到，第一条启动 run 之前另外两条
  也已经过了"isActive 检查"（都还没 active run），所以全走了 lane 排队路径
- 想看到真正的 collect 合并：用 TUI 在一条回复还没出来时**继续在同一个输入框**连发
  （同连接的消息会命中 active run → enqueue-followup → 合并 prompt）
- 队列文档里那句 "If messages target different channels/threads, they drain
  individually"（不同渠道/线程的消息各自 drain）解释了为什么合并要按路由分组 ——
  合并的是"同一个投递目的地"的消息

**五种种队列模式速记**（`src/config/types.queue.ts` + `docs/concepts/queue.md`）：

| 模式                  | 行为                                      | 一句话记忆       |
| --------------------- | ----------------------------------------- | ---------------- |
| `collect`（默认）   | 等 active run 结束，积压合并成一个 prompt | "攒一起说"       |
| `followup`          | 同 collect 但不合并，逐条排队下一轮       | "排队一个一个说" |
| `steer`             | 立即注入进行中的 run（工具边界处打断）    | "插话"           |
| `steer-backlog`     | 插话 + 保留消息给下一轮                   | "插话且留底"     |
| `interrupt`（遗留） | 直接中止 active run，用最新消息重跑       | "抢话筒"         |

---

## 自查：你能回答吗？

1. `runReplyAgent` 和 `runEmbeddedAttempt` 分工？（编排 vs 单次尝试；run 含 fallback 循环）
2. session lane 和 global lane 各防什么？（并发写坏 session / 打爆模型配额）
3. collect 模式下消息什么时候不启动新 run？（session 已有 active run 时，
   进 followup 队列等合并）
4. `lane wait exceeded` 日志出现在哪一层、什么条件打？（command-queue.ts drainLane，
   等待超 2 秒）
5. 排查一次 run 的所有日志，用什么串联？（runId，从 res ✓ agent 到工具审计行都在带它）
6. CLI 消息为什么 sessionKey 是三段格式？（无 channel/groupId/userId 可用，
   退化为 `agent:{agentId}:main`）

## 动手验证清单（15 分钟）

```bash
# 1. 发一条带工具的消息，拿到 runId
pnpm xiaoaiclaw --dev agent --agent dev --message "用exec执行 date 然后告诉我时间"

# 2. 用 runId 串联整条链路的日志（把 <runId> 换成上面输出的）
grep "<runId>" /tmp/xiaoaiclaw/xiaoaiclaw-$(date +%Y-%m-%d-%H).log

# 3. 看 lane 排队（连发 3 条）
for i in 1 2 3; do (pnpm xiaoaiclaw --dev agent --agent dev --message "测试${i} 回复数字${i}" > /tmp/b$i.log 2>&1 &); done
grep "lane wait exceeded" /tmp/xiaoaiclaw/xiaoaiclaw-$(date +%Y-%m-%d-%H).log | tail -3

# 4. 验证"路由键 vs 文件键"两层设计（实测结论）
#    CLI 只有 --session-id 选项（没有 --session-key）。经网关跑一遍：
pnpm xiaoaiclaw --dev agent --agent dev --session-id my-test-session --message "hi"
python3 -m json.tool ~/.xiaoaiclaw-dev/agents/dev/sessions/sessions.json
#    实测：消息仍写进 agent:dev:main 对应的旧 sessionId 文件 —— 网关路径下
#    已有 entry 的 sessionId 优先（agent.ts:554 `entry?.sessionId ?? generateSecureHex()`），
#    传入的 id 只影响"按 id 反查 entry"（session.ts:72 反向匹配）。
#    结论：sessionKey 是路由键（决定进哪个会话），sessionId 是文件键（决定写哪个文件），
#    两者通过 sessions.json 映射，路由键说了算。

# 5. 想看"真正的 collect 合并"：开 TUI，在一条回复没出来前继续在输入框连发
pnpm xiaoaiclaw --dev tui
#    （同连接消息会命中 active run → 进 followup 队列 → 合并成下一个 prompt）

# 6. 切换队列模式观察差异（会话级覆盖）：TUI 里发 /queue steer 再连发两条
```

## 相关阅读

- 队列设计文档：`docs/concepts/queue.md`（89 行，值得全读）
- 链路核心文件：`src/auto-reply/reply/get-reply-run.ts`（协调中枢，1045 行）
- lane 实现：`src/process/command-queue.ts`
- 工具循环：`node_modules/@mariozechner/pi-agent-core/dist/agent-loop.js`（`runLoop` 函数）
- 实验 1 补课（握手 + transcript）：[2026-10-27-exp1-ws-handshake-and-transcript.md](2026-10-27-exp1-ws-handshake-and-transcript.md)
- 实验 5 的耗时归因方法：[2026-10-27-hands-on-lab.md](2026-10-27-hands-on-lab.md) 实验 5 一节
