# 增量学习路线：从 50 行玩具到 8 站网关

> 这是 [why-8-stations](../2026-10-29-exp2-why-8-stations.md) 的配套实操。
> 7 个文件，每个都能独立运行（`node stepN-xxx.cjs`），零依赖、mock 模型。
>
> **玩法**：按顺序跑，每跑完一个对照"运行观察点"和"思考题"。
> 想看增量 diff：`diff step(N-1)*.cjs stepN*.cjs` —— 每步新增的代码都有
> 【增量】标注，老代码原样保留（这正是"分层是长出来的"的直观体现）。

## 路线图

| 步骤 | 文件 | 解决的问题 | 长出的层 | 对应真实代码 |
| --- | --- | --- | --- | --- |
| 0 | [step0-toy-bot.cjs](step0-toy-bot.cjs) | （起点）能收能回就行 | 无 —— 一切开始的地方 | 无（反面教材） |
| 1 | [step1-attempt-layer.cjs](step1-attempt-layer.cjs) | 模型挂了进程就崩 | attempt 层 + 故障转移 | `run/attempt.ts` + `model-fallback.ts` |
| 2 | [step2-queue-layer.cjs](step2-queue-layer.cjs) | 连发 3 条互相踩踏 | SessionLane + collect 合并 | `command-queue.ts` + `buildCollectPrompt` |
| 3 | [step3-routing-layer.cjs](step3-routing-layer.cjs) | 没记忆、没隔离 | sessionKey 路由 + 按 key 分道 | `main-session.ts` + `lanes.ts` |
| 4 | [step4-session-file-layer.cjs](step4-session-file-layer.cjs) | 重启失忆、文件写坏 | JSONL + parentId 链 + compaction | `SessionManager`（工程红线的由来） |
| 5 | [step5-tool-loop-layer.cjs](step5-tool-loop-layer.cjs) | 模型不会做事 / 死循环烧钱 | 工具循环 + 熔断器 + hook | pi-agent-core `runLoop` + `tool-loop-detection.ts` |
| 6 | [step6-gateway-layer.cjs](step6-gateway-layer.cjs) | 重试重复计费、流程散乱 | 校验 + 幂等 + 编排层（总装） | `agent.ts` 入口 + `runReplyAgent` |

## 一口气跑完

```bash
cd learning/learning1027/main/notes/exp2-message-lifecycle-deep-dive/example
node step0-toy-bot.cjs
node step1-attempt-layer.cjs
node step2-queue-layer.cjs
node step3-routing-layer.cjs
rm -rf /tmp/toy-sessions        # step4/5/6 各有独立目录，这行可选
node step4-session-file-layer.cjs   # 跑两次！第二次演示"重启恢复"
node step5-tool-loop-layer.cjs
node step6-gateway-layer.cjs
```

## 每步的"命运对照"（跑之前别看，跑完对答案）

- **step1**：主模型挂 2 次 → 自动切备用 → 第二条消息时主模型恢复
- **step2**：3 条消息合并成 1 次模型调用（551ms 而非 3×400ms）
- **step3**：小明第二次提问模型"记得 2 句话"；换用户/换渠道都是 0 句
- **step4**：聊到第 5 句触发压缩，上下文从 122 字降到 27 字；重跑读回全部历史
- **step5**：三种结局 —— 正常 2 轮 / 抽风后自救 4 轮 / 死循环被熔断（进程不死）
- **step6**：畸形请求门口拒绝 / 并发重试只跑一次 / 全链路 8 站走通

## 与真实系统的终极对照

```
step6 总装后的 toy 管线                真实 xiaoaiclaw
─────────────────────────────      ─────────────────────────────
agentMethod() 入口              ↔   站①② gateway agent method（agent.ts:281）
validateParams                  ↔   站②a validateAgentParams
DedupeCache(进行中+缓存)         ↔   站②b context.dedupe
（简化：单会话）                 ↔   站③ sessionKey 推导（main-session.ts:36）
runReply 编排                   ↔   站⑤ runReplyAgent（agent-runner.ts:66）
  ├─ onTyping                   ↔     打字提示
  └─ 压缩预检                    ↔     runPreflightCompactionIfNeeded
SessionLane.enqueue             ↔   站⑥ 两层 lane（run.ts:152 起）
runLoop + ToolLoopGuard         ↔   站⑦⑧ attempt + 工具循环 + 熔断
SessionManager(JSONL/链)        ↔   站⑨ SessionManager + 落盘
```

## 建议的实验姿势

1. **先跑后读**：每个文件先跑，看输出对不对得上文件头的问题描述
2. **改坏它**：注释掉某个关键行再跑（比如 step5 注释掉 `guard.check`）
   —— 亲眼看"没有这层防护会发生什么"，比读十遍注释印象深刻
3. **diff 相邻两步**：`diff step2*.cjs step3*.cjs` —— 只看【增量】部分
4. 回到真实代码：每步文件头的"对应真实代码"给了文件:行号，
   带着 toy 里的直觉去读真实实现，会发现"哦原来是这个的放大版"
