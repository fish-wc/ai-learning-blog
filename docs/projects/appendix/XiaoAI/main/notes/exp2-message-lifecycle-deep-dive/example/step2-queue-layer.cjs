// ============================================================
// 第 2 步：用户连发 3 条怎么办 —— 长出「队列层」
// ============================================================
//
// 【增量】相对第 1 步，本文件新增：
//   1. SessionLane     —— 会话内串行队列（并发=1）
//   2. collect 合并    —— 排队期间到达的消息合并成一个 prompt
//   3. 等待告警日志    —— 对应真实代码的 "lane wait exceeded"
//
// 【对应真实代码】
//   lane 实现          → src/process/command-queue.ts（drainLane/pump）
//   session lane 拼接  → src/agents/pi-embedded-runner/lanes.ts:3（"session:"+key）
//   collect 合并 prompt→ src/utils/queue-helpers.ts:217（buildCollectPrompt）
//   等待告警           → command-queue.ts:187 "lane wait exceeded: waitedMs=... queueAhead=..."
//   队列模式决策       → src/auto-reply/reply/queue/settings.ts:32（默认 collect）
//
// 用法：
//   node step2-queue-layer.cjs
// ============================================================

const { EventEmitter } = require("events");

// ---------- mock 模型（与第 1 步相同，去掉故障注入简化聚焦） ----------
const mockModel = {
  async create({ messages }) {
    // 【增量】模拟模型慢：400ms —— 给并发问题留出暴露窗口
    await new Promise((r) => setTimeout(r, 400));
    const last = messages[messages.length - 1];
    const turnCount = messages.length; // 第3步会用到：消息越多说明上下文越长
    return {
      content: [{ type: "text", text: `[model] 你说的是: ${last.content}` }],
      usage: { input_tokens: 10 * turnCount, output_tokens: 10 },
    };
  },
};

// ============================================================
// 【增量】SessionLane —— 会话内的串行队列
// ============================================================
//
// 核心不变量：同一会话，同一时刻，最多一个 run 在跑。
// 为什么必须这样：如果两个 run 并发，它们会同时读写同一份
// 会话文件（第4步会看到这个文件）—— 交错写入 = 文件损坏。
//
// 真实代码 command-queue.ts 的结构就是：每 lane 一个 FIFO 数组 +
// 一个"泵"(pump)，泵从队头取任务，跑完一个再取下一个。
//
class SessionLane {
  constructor(laneKey) {
    this.laneKey = laneKey;
    this.queue = []; // FIFO: [{task, enqueuedAt}]
    this.active = false;
  }

  // 入队。task 是 async 函数。
  enqueue(task) {
    return new Promise((resolve, reject) => {
      this.queue.push({
        task,
        enqueuedAt: Date.now(),
        resolve,
        reject,
      });
      this.#pump(); // 有新任务就尝试泵
    });
  }

  // 泵：只要没有 active 任务且队列非空，就取队头执行
  async #pump() {
    if (this.active) return; // 【互斥核心】有人在跑，等着
    this.active = true;
    while (this.queue.length > 0) {
      const entry = this.queue.shift();
      const waitedMs = Date.now() - entry.enqueuedAt;
      // 【增量】等待告警 —— 真实代码 command-queue.ts:187 的同款日志
      if (waitedMs > 100) {
        console.log(
          `   ⏳ lane wait exceeded: lane=${this.laneKey} waitedMs=${waitedMs} queueAhead=${this.queue.length}`,
        );
      }
      try {
        entry.resolve(await entry.task());
      } catch (err) {
        entry.reject(err);
      }
    }
    this.active = false;
  }
}

// ============================================================
// 【增量】collect 合并 —— 排队的消息不白排
// ============================================================
//
// 真实场景：用户手速快连发"查天气""北京的""明天也要"。
// collect 语义：第二条到达时若已有 run 在跑，不立刻排队逐条跑，
// 而是攒着；等 run 结束，把攒的合并成一个 prompt 一次问模型。
// （真实代码: buildCollectPrompt 把多条消息拼成带编号的合并文本）
//
// 收益：模型只叫醒一次（省 token 省延迟），且模型能同时看到三条
// 的完整意图，回复质量反而更高。
//
class CollectBuffer {
  constructor({ debounceMs = 150 }) {
    this.debounceMs = debounceMs; // 攒批窗口：静默这么久才算"说完了"
    this.items = [];
    this.timer = null;
    this.resolve = null;
  }

  // 放入一条消息；返回"合并后的 prompt"（等 debounce 静默后 resolve）
  add(text) {
    return new Promise((resolve) => {
      this.items.push(text);
      this.resolve = resolve;
      // 每来一条就重置计时器（"用户还在说"）—— 对应 debounceMs 语义
      clearTimeout(this.timer);
      this.timer = setTimeout(() => {
        const merged = this.#buildCollectPrompt();
        this.items = [];
        resolve(merged);
      }, this.debounceMs);
    });
  }

  // 【增量】对应真实代码 queue-helpers.ts:217 buildCollectPrompt
  #buildCollectPrompt() {
    if (this.items.length === 1) return this.items[0];
    const blocks = this.items.map((t, i) => `[消息${i + 1}] ${t}`);
    return `(用户在短时间内连发了${this.items.length}条消息，请合并理解)\n${blocks.join("\n")}`;
  }
}

// ---------- 网关与处理（沿用第 1 步的 attempt 结构，简化） ----------
const gateway = new EventEmitter();

// 【增量】每个会话一个 lane（真实代码按 sessionKey 分道，第3步引入 key）
const lane = new SessionLane("session:agent:demo:main");
const collect = new CollectBuffer({ debounceMs: 150 });

gateway.on("message", async (text) => {
  const t0 = Date.now();
  // 入口先进 collect 攒批 → 合并 prompt → 进 lane 排队 → 跑
  const prompt = await collect.add(text);
  const reply = await lane.enqueue(async () => {
    console.log(`   ▶ run 开始 (prompt 前30字: ${prompt.slice(0, 30)}...)`);
    const r = await mockModel.create({ messages: [{ role: "user", content: prompt }] });
    return r;
  });
  console.log(`↩ 回复: ${reply.content[0].text} (总耗时${Date.now() - t0}ms)\n`);
});

// ---------- 模拟用户连发 3 条 ----------
(async () => {
  console.log("=== 第 2 步：队列层（连发 3 条怎么办）===\n");
  console.log("用户 100ms 内连发 3 条...\n");

  // 不 await —— 模拟三条几乎同时到达
  gateway.emit("message", "帮我查天气");
  await new Promise((r) => setTimeout(r, 60));
  gateway.emit("message", "北京的");
  await new Promise((r) => setTimeout(r, 60));
  gateway.emit("message", "顺便明天也查");

  await new Promise((r) => setTimeout(r, 2000));
})();

// ============================================================
// 运行观察点：
//   node step2-queue-layer.cjs
//
// 预期输出：
//   用户 100ms 内连发 3 条...
//   → (3条进入 collect 攒批，150ms 静默后合并)
//   ▶ run 开始 (prompt 前30字: (用户在短时间内连发了3条消息...)
//   ↩ 回复: [model] 你说的是: (用户在短时间内连发了3条消息... (总耗时~600ms)
//
// 关键数字：3 条消息只触发 **1 次**模型调用（400ms 慢模型），
// 而不是 3 次串行 1200ms，更不是 3 次并发互相踩踏。
//
// 再做一个实验（改代码体验）：
//   把 CollectBuffer 的 debounceMs 改成 0 → 每条独立跑，
//   你会看到 3 次 "run 开始" + 2 条 "lane wait exceeded" 日志
//   —— 这就是我们实测里看到的真实行为（教程第9站）
//
// 思考：
//   1. lane 并发=1 保护的到底是什么？（下一份会话文件的写安全）
//   2. collect 的代价是什么？（第二条要等第一条 + debounce，有延迟）
//   3. 为什么 debounce 要"每来一条就重置"？（用户还在说就别急着问模型，
//      否则"查天气/北京的"会被拆成两次无意义调用）
// ============================================================
