// ============================================================
// 第 5 步：模型要用工具怎么办 —— 长出「工具循环层」
// ============================================================
//
// 【增量】相对第 4 步，本文件新增：
//   1. 工具注册表        —— exec/date 两个可执行工具
//   2. runLoop           —— LLM↔工具 循环（pi-agent-core 的同款结构）
//   3. 工具循环熔断器    —— 四种探测器的简化版（真实事故换来的刹车）
//   4. before_tool_call   —— 我们实验4写的 hook 的最小版
//
// 【对应真实代码】
//   循环本体          → node_modules/@mariozechner/pi-agent-core/dist/agent-loop.js
//                        的 runLoop（双层 while：内层工具、外层 followup）
//   熔断器            → src/agents/tool-loop-detection.ts
//                        （WARNING=10 / CRITICAL=20 / GLOBAL=30，
//                         CIRCUIT_BREAKER_ABORT_AFTER_BLOCKS=3）
//   真实事故          → git show 9d139e9a63（熔断把 run 杀太狠 → 设备空回复，
//                        修复=先 block 三次回灌提示让模型自救，再 abort）
//   hook              → src/plugins/types.ts 的 before_tool_call（实验4实测过）
//
// 用法：
//   node step5-tool-loop-layer.cjs
// ============================================================

const fs = require("fs");
const path = require("path");
const { EventEmitter } = require("events");
const { createHash } = require("crypto");

// ============================================================
// 【增量】工具注册表 —— 模型可以"动手"的能力清单
// ============================================================
// 真实系统里 exec/read/write/web_search 都在这里注册。
// 工具的本质：名字 + 参数 + 一个会返回结果的函数。
const TOOLS = {
  exec: {
    description: "执行一个 shell 命令",
    async run(args) {
      // 教学 mock：不真跑 shell，假装执行并返回
      return { ok: true, output: `(exec: ${args.command} → 输出mock)` };
    },
  },
  date: {
    description: "获取当前时间",
    async run() {
      return { ok: true, output: new Date().toLocaleString("zh-CN") };
    },
  },
};

// ============================================================
// 【增量】mock 模型 —— 会"抽风"的模型（关键教学装置）
// ============================================================
// 通过 failMode 控制三种行为：
//   "normal"         正常：要么直接回文本，要么发起合理的工具调用
//   "loop_forever"   抽风1：永远重复调同一个工具（generic_repeat）
//   "recover"        抽风2：死循环几次后收到熔断回灌提示，改邪归正
//
// 模型看到的消息里若有 [TOOL_RESULT] 标记，说明上一轮工具结果回来了。
function makeMockModel({ failMode = "normal" } = {}) {
  let calls = 0;
  return {
    async create({ messages, tools }) {
      await new Promise((r) => setTimeout(r, 80));
      calls += 1;
      const last = messages[messages.length - 1];
      const lastText = typeof last.content === "string" ? last.content : "";
      const isToolResult = lastText.includes("[TOOL_RESULT]");

      // ---- 抽风模式：永远要调 exec，无视结果 ----
      if (failMode === "loop_forever") {
        return {
          content: [
            { type: "toolCall", id: `call_${calls}`, name: "exec", arguments: { command: "date" } },
          ],
          stopReason: "toolUse",
        };
      }
      // ---- 自救模式：被回灌 [WARNING]/[CRITICAL] 提示后改说人话 ----
      if (failMode === "recover" && (lastText.includes("[CRITICAL]") || lastText.includes("[WARNING]"))) {
        return {
          content: [{ type: "text", text: "抱歉，我意识到自己陷入重复了。已停止，改用文字直接回复。" }],
          stopReason: "end_turn",
        };
      }
      if (failMode === "recover" && calls <= 9) {
        return {
          content: [
            { type: "toolCall", id: `call_${calls}`, name: "exec", arguments: { command: "date" } },
          ],
          stopReason: "toolUse",
        };
      }

      // ---- 正常模式 ----
      // 第一轮：用户问时间 → 发起 date 工具调用
      if (!isToolResult && /几点|时间/.test(lastText)) {
        return {
          content: [{ type: "toolCall", id: `call_${calls}`, name: "date", arguments: {} }],
          stopReason: "toolUse",
        };
      }
      // 收到工具结果 → 总结回复
      if (isToolResult) {
        return {
          content: [{ type: "text", text: `根据工具结果：${lastText.replace("[TOOL_RESULT] ", "").slice(0, 40)}` }],
          stopReason: "end_turn",
        };
      }
      return {
        content: [{ type: "text", text: `[model] 你说的是: ${lastText}` }],
        stopReason: "end_turn",
      };
    },
  };
}

// ============================================================
// 【增量】工具循环熔断器 —— 真实事故换来的刹车
// ============================================================
// 对应 tool-loop-detection.ts 的简化版，保留核心语义：
//   - 对每次工具调用算哈希，历史里重复次数 = 死循环证据
//   - WARNING（>=3次同哈希）：回灌提示，让模型意识到重复
//   - CRITICAL（>=5次）：回灌强提示（真实: 先 block 三次给机会自救）
//   - ABORT（>=8次）：放弃整个 run（真实: global_circuit_breaker=30）
//
// 为什么不立刻杀？—— 真实事故（9d139e9a63）：critical 一命中就 abort，
// 模型一个字没输出，设备端收到空回复。修复后：先 block+回灌几次，
// 给模型自救机会，实在不行才 abort。
//
class ToolLoopGuard {
  constructor() {
    this.history = []; // [{hash, toolName}]
    this.blocks = 0; // critical 后已 block 的次数
  }
  #hash(call) {
    return createHash("sha256").update(`${call.name}:${JSON.stringify(call.arguments)}`).digest("hex").slice(0, 12);
  }
  /** 返回 null=放行；返回 string=拦截原因（回灌给模型） */
  check(toolCall) {
    const h = this.#hash(toolCall);
    this.history.push({ hash: h, toolName: toolCall.name });
    const repeatCount = this.history.filter((x) => x.hash === h).length;

    if (repeatCount >= 8) {
      return { abort: true, reason: `全局熔断: 工具${toolCall.name}重复${repeatCount}次，放弃run` };
    }
    if (repeatCount >= 5) {
      this.blocks += 1;
      // 真实语义: CIRCUIT_BREAKER_ABORT_AFTER_BLOCKS=3 —— block 三次后才 abort
      return {
        abort: this.blocks > 3,
        reason: `[CRITICAL] 检测到死循环(第${repeatCount}次重复)。请立即停止重复调用，直接用文字回复用户。`,
      };
    }
    if (repeatCount >= 3) {
      return { abort: false, reason: `[WARNING] 你已重复调用 ${toolCall.name} ${repeatCount} 次，请换个思路。` };
    }
    return null; // 放行
  }
}

// ============================================================
// 【增量】before_tool_call hook —— 实验4插件的最小版
// ============================================================
// 真实系统里我们用 30 行插件挂了审计日志（learning/exec-audit-plugin）。
// hook 在工具执行【前】拦截，可以：放行 / 改参数 / block。
//
const hooks = {
  beforeToolCall: null, // 可注入 (toolCall) => void | {block, reason}
};

// ============================================================
// 【增量】runLoop —— 工具循环本体（pi-agent-core 同款结构）
// ============================================================
//
//      ┌─────────────────────────────────────────┐
//      │  内层 while：有 toolCall 就继续           │
//      │    ① 流式调模型 → assistant 消息          │
//      │    ② content 里有 toolCall？             │
//      │       ├─ 是 → 跑工具 → 结果回灌 ──┐      │
//      │       └─ 否 → 内层退出            │      │
//      │    ←──────────────────────────────┘      │
//      │  外层：检查 followup 队列，有则继续        │
//      └─────────────────────────────────────────┘
//
async function runLoop({ model, messages, tools, guard, maxTurns = 20 }) {
  const convo = [...messages];
  for (let turn = 0; turn < maxTurns; turn++) {
    // ① 调模型
    const reply = await model.create({ messages: convo, tools });
    const toolCalls = reply.content.filter((c) => c.type === "toolCall");

    // ② 没有 toolCall → 模型说完话了，返回最终回复
    if (toolCalls.length === 0) {
      convo.push({ role: "assistant", content: reply.content[0].text });
      return { text: reply.content[0].text, turns: turn + 1, convo };
    }

    // ③ 有 toolCall → 逐个执行（先过 hook 和熔断器）
    convo.push({
      role: "assistant",
      content: `[发起工具调用: ${toolCalls.map((t) => t.name).join(", ")}]`,
    });
    const results = [];
    let aborted = null;
    for (const tc of toolCalls) {
      // hook 拦截（实验4的审计插件挂这里）
      const hookResult = hooks.beforeToolCall?.(tc);
      if (hookResult?.block) {
        results.push(`[TOOL_RESULT] [BLOCKED by hook] ${hookResult.reason}`);
        continue;
      }
      // 熔断器拦截
      const verdict = guard.check(tc);
      if (verdict) {
        if (verdict.abort) {
          aborted = verdict.reason;
          break;
        }
        // block 但不 abort：回灌提示，让模型自救（真实事故的教训）
        results.push(`[TOOL_RESULT] ${verdict.reason}`);
        continue;
      }
      // 放行：真正执行工具
      const out = await tools[tc.name].run(tc.arguments);
      results.push(`[TOOL_RESULT] ${out.output}`);
    }
    if (aborted) {
      return { text: `[run aborted] ${aborted}`, turns: turn + 1, convo, aborted: true };
    }
    // ④ 工具结果作为下一轮的"user"消息回灌（真实: toolResult 挂 user 侧）
    convo.push({ role: "user", content: results.join("\n") });
  }
  return { text: "[超过最大轮数]", turns: maxTurns, convo };
}

// ---------- SessionLane / SessionManager（沿用第 4 步，精简） ----------
class SessionLane {
  constructor() { this.queue = []; this.active = false; }
  enqueue(task) {
    return new Promise((resolve, reject) => {
      this.queue.push({ task, resolve, reject });
      this.#pump();
    });
  }
  async #pump() {
    if (this.active) return;
    this.active = true;
    while (this.queue.length > 0) {
      const e = this.queue.shift();
      try { e.resolve(await e.task()); } catch (err) { e.reject(err); }
    }
    this.active = false;
  }
}

const SESSIONS_DIR = "/tmp/toy-sessions-step5";
class SessionManager {
  constructor(filePath) {
    this.filePath = filePath;
    this.entries = [];
    this.leafId = null;
    if (fs.existsSync(filePath)) {
      for (const line of fs.readFileSync(filePath, "utf8").split("\n")) {
        if (!line.trim()) continue;
        const e = JSON.parse(line);
        this.entries.push(e);
        if (e.id) this.leafId = e.id;
      }
    }
  }
  append(type, payload) {
    const entry = {
      type, id: Math.random().toString(16).slice(2, 10),
      parentId: this.leafId, timestamp: new Date().toISOString(), ...payload,
    };
    fs.appendFileSync(this.filePath, JSON.stringify(entry) + "\n");
    this.entries.push(entry);
    this.leafId = entry.id;
    return entry;
  }
  getHistory() {
    const chain = [];
    const byId = new Map(this.entries.map((e) => [e.id, e]));
    let cur = this.leafId;
    while (cur) { const e = byId.get(cur); if (!e) break; chain.unshift(e); cur = e.parentId; }
    return chain.filter((e) => e.type === "message").map((e) => ({ role: e.message.role, content: e.message.content }));
  }
}

// ---------- 网关 ----------
const gateway = new EventEmitter();
const lane = new SessionLane();
fs.mkdirSync(SESSIONS_DIR, { recursive: true });
const session = new SessionManager(path.join(SESSIONS_DIR, "demo.jsonl"));

// 【增量】每 run 一个新 guard（真实修复: 熔断计数按 runId 隔离，
// 跨 run 累加会误伤正常多轮复用 —— 事故复盘里的第1层根因）
gateway.on("message", async (text, { failMode } = {}) => {
  const model = makeMockModel({ failMode });
  const guard = new ToolLoopGuard(); // per-run
  const result = await lane.enqueue(async () => {
    session.append("message", { message: { role: "user", content: text } });
    const r = await runLoop({ model, messages: session.getHistory(), tools: TOOLS, guard });
    session.append("message", { message: { role: "assistant", content: r.text } });
    return r;
  });
  console.log(`→ "${text}" (${failMode ?? "normal"}模式)`);
  console.log(`↩ ${result.text}`);
  console.log(`   [工具循环共 ${result.turns} 轮${result.aborted ? "，被熔断中止" : ""}]\n`);
});

// ---------- 演示三种命运 ----------
(async () => {
  console.log("=== 第 5 步：工具循环层（模型会做事，且装了刹车）===\n");

  // A. 正常：问时间 → 模型调 date 工具 → 拿结果 → 总结回复
  gateway.emit("message", "现在几点了？");
  await new Promise((r) => setTimeout(r, 500));

  // B. 自救：模型抽风循环，但收到 WARNING/CRITICAL 回灌后改邪归正（真实修复的语义）
  gateway.emit("message", "现在几点了？(我会抽风但能自救)", { failMode: "recover" });
  await new Promise((r) => setTimeout(r, 1500));

  // C. 熔断：模型彻底死循环，guard 最终放弃 run
  gateway.emit("message", "现在几点了？(我会无限抽风)", { failMode: "loop_forever" });
  await new Promise((r) => setTimeout(r, 2500));
})();

// ============================================================
// 运行观察点：
//   node step5-tool-loop-layer.cjs
//
// 预期三个结局：
//   A 正常:   2 轮（发起date → 拿结果总结），回复"根据工具结果: 2026/…"
//   B 自救:   前4轮重复调exec → WARNING(第3次) → CRITICAL(第5次)回灌
//             → 模型看到 [CRITICAL] 提示后改说人话 → 正常收尾
//             ★ 这就是真实事故修复后的行为：给模型自救机会
//   C 熔断:   永远重复 → block 三次后 abort → "[run aborted] 全局熔断…"
//             ★ 进程没崩！下一条消息照常处理（隔离性）
//
// 思考：
//   1. 没有 guard 的 C 场景会怎样？（无限循环 = 无限烧钱，
//      用户永远等不到回复 —— 这就是熔断器存在的全部理由）
//   2. 为什么 WARNING 回灌的提示要放进 [TOOL_RESULT] 而不是直接 abort？
//      （把"你重复了"作为工具结果喂回去，模型有机会理解并纠正 ——
//       真实事故里直接 abort 导致设备端空回复）
//   3. guard 为什么 per-run 新建？（正常业务里同工具多次调用是合理的，
//      跨 run 累加会把"每天查一次天气"误判成死循环 —— 事故第1层根因）
//   4. hook 和 guard 的区别？（hook 是用户/插件的扩展点，管业务规则；
//      guard 是系统内置安全阀，管资源安全 —— 两者都在工具执行前）
// ============================================================
