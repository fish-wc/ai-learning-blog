// ============================================================
// 第 1 步：模型挂了怎么办 —— 长出「attempt 层」
// ============================================================
//
// 【增量】相对第 0 步，本文件新增：
//   1. mockModel 支持注入故障（failNTimes）—— 让"模型挂"可以复现
//   2. runAttempt()   —— 一次"尝试"（真实代码: run/attempt.ts 的 runEmbeddedAttempt）
//   3. runWithFailover() —— 失败后的重试策略（真实代码: model-fallback.ts）
//      策略：先用主模型试 → 挂了换凭据(profile)再试 → 再挂换备用模型
//
// 【对应真实代码】
//   attempt 层      → src/agents/pi-embedded-runner/run/attempt.ts:1273
//   故障转移两级    → src/agents/model-fallback.ts（runWithModelFallback）
//                     第一级: auth profile 轮换（换 key）
//                     第二级: 模型 fallbacks（换模型）
//
// 用法：
//   node step1-attempt-layer.cjs
// ============================================================

const { EventEmitter } = require("events");

// ---------- mock 模型【增量：支持故障注入】 ----------
// 真实世界里模型挂 = API 500 / 限流 429 / key 失效。
// 这里用计数器模拟"前 N 次调用必挂"。
function makeMockModel({ name, failNTimes = 0, replyPrefix }) {
  let calls = 0;
  return {
    name,
    async create({ messages }) {
      calls += 1;
      if (calls <= failNTimes) {
        // 模拟一次 500 错误（真实: Error: status 500 overloaded）
        const err = new Error(`${name} 第${calls}次调用失败(模拟500)`);
        err.statusCode = 500;
        throw err;
      }
      const last = messages[messages.length - 1];
      return {
        content: [{ type: "text", text: `[${replyPrefix}] 你说的是: ${last.content}` }],
      };
    },
  };
}

// 【增量】模型矩阵：主模型 + 两个备用（fallbacks）
// 真实配置对应 agents.defaults.model.fallbacks
const modelPool = [
  makeMockModel({ name: "opus", failNTimes: 2, replyPrefix: "opus" }), // 前2次挂
  makeMockModel({ name: "sonnet", failNTimes: 0, replyPrefix: "sonnet" }),
];

// ============================================================
// 【增量】attempt 层 —— 核心新增概念
// ============================================================
//
// 一次 attempt = 一次"从调模型到拿到结果"的完整尝试。
// 注意它自己是"哑的"：失败就往上抛，不重试。
// 重试的智慧在它外面的 runWithFailover 里。
//
async function runAttempt({ model, messages, attemptNo }) {
  console.log(`   ├─ attempt#${attemptNo} 调用模型 ${model.name}...`);
  try {
    const reply = await model.create({ messages });
    console.log(`   ├─ attempt#${attemptNo} 成功 ✓`);
    return reply;
  } catch (err) {
    console.log(`   ├─ attempt#${attemptNo} 失败 ✗ (${err.message})`);
    throw err; // 失败上抛，由外层决定怎么办
  }
}

// ============================================================
// 【增量】故障转移 —— attempt 的"外循环"
// ============================================================
//
// 真实代码 model-fallback.ts 的 runWithModelFallback 就是这个结构：
//   for (candidate of 候选列表) { try { return run(candidate) } catch { 换下一个 } }
//
// 现实中候选列表的第一级是"同模型换凭据"（auth profile 轮换），
// 第二级才是"换模型"。这里简化成"换模型"，语义一致。
//
async function runWithFailover({ messages }) {
  for (let i = 0; i < modelPool.length; i++) {
    try {
      return await runAttempt({ model: modelPool[i], messages, attemptNo: i + 1 });
    } catch (err) {
      const next = modelPool[i + 1];
      if (next) {
        console.log(`   └─ 切换备用模型: ${modelPool[i].name} → ${next.name}`);
      } else {
        // 所有候选都挂：返回错误而不是抛出 —— 进程不能死
        return { content: [{ type: "text", text: `[error] 全部模型失败: ${err.message}` }] };
      }
    }
  }
}

// ---------- 网关（与第 0 步相同，核心处理函数换成走 failover） ----------
const gateway = new EventEmitter();

gateway.on("message", async (text) => {
  console.log(`→ 收到消息: ${text}`);
  // 【增量】不再直接调 mockModel.create，而是经 runWithFailover 编排
  const reply = await runWithFailover({
    messages: [{ role: "user", content: text }],
  });
  console.log(`↩ 回复: ${reply.content[0].text}\n`);
});

// ---------- 模拟用户 ----------
(async () => {
  console.log("=== 第 1 步：attempt 层（模型挂了怎么办）===\n");
  // 主模型 opus 设置了 failNTimes=2：
  //   第1条消息: attempt#1(opus)挂 → attempt#2(sonnet)成功
  //   第2条消息: attempt#3(opus)已过失败期 → 直接成功
  await new Promise((r) => setTimeout(r, 0));
  gateway.emit("message", "第一条：模型正在挂的时候发来的");
  await new Promise((r) => setTimeout(r, 50));
  gateway.emit("message", "第二条：模型恢复了");
})();

// ============================================================
// 运行观察点：
//   node step1-attempt-layer.cjs
//
// 预期输出（注意 attempt 的切换链）：
//   → 收到消息: 第一条...
//      ├─ attempt#1 调用模型 opus...
//      ├─ attempt#1 失败 ✗ (opus 第1次调用失败(模拟500))
//      └─ 切换备用模型: opus → sonnet
//      ├─ attempt#2 调用模型 sonnet...
//      ├─ attempt#2 成功 ✓
//   ↩ 回复: [sonnet] 你说的是: 第一条...
//
// 对比第 0 步：同样的故障在第 0 步会让进程直接崩（unhandled error）。
// 现在它只是一行日志 + 一次切换。
//
// 思考：
//   1. 如果两个模型都挂，进程死了吗？（没有 —— 返回了 [error] 文本）
//   2. "attempt" 和 "run" 的关系？（run=办成这件事，attempt=试一次；
//      一个 run 可能包含多个 attempt —— 这就是术语表 Agent run/attempt 的含义）
//   3. 为什么 attempt 不自己重试？（职责分离：attempt 只管试，
//      重试策略集中在一处才好统一调整 —— 比如换冷却时间）
// ============================================================
