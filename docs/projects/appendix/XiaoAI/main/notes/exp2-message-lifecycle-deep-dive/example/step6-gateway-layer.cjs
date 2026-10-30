// ============================================================
// 第 6 步：守门与编排 —— 长出「网关层」（校验 + 幂等 + 编排）
// ============================================================
//
// 【增量】相对第 5 步，本文件新增：
//   1. 参数校验器       —— 畸形请求在门口就拒
//   2. 幂等去重缓存     —— 网络重试不会跑两遍模型
//   3. 编排层 runReply  —— 统筹打字提示/压缩预检/记忆/执行（管家的站）
//
// 至此，前面 5 步的所有零件被"总装"成一条完整管线 ——
// 这就是真实 xiaoaiclaw 的样子（8 站全齐）。
//
// 【对应真实代码】
//   参数校验          → src/gateway/server-methods/agent.ts:281
//                        （validateAgentParams，TypeBox schema）
//   幂等去重          → agent.ts:358 context.dedupe.get(`agent:${idem}`)
//   编排层            → src/auto-reply/reply/agent-runner.ts:66 runReplyAgent
//                        （打字提示/压缩预检/记忆flush/流式分段都在这统筹）
//   全链路总装        → 真实系统 8 站对照见本文末尾的映射表
//
// 用法：
//   node step6-gateway-layer.cjs
// ============================================================

const fs = require("fs");
const path = require("path");
const { EventEmitter } = require("events");
const { createHash } = require("crypto");

// ============================================================
// 【增量】站②：参数校验器 —— 守门员的第一道岗
// ============================================================
// 真实系统用 TypeBox schema 校验。这里手写最小版：
// 消息必须是非空字符串，长度合理。
function validateParams(p) {
  const errors = [];
  if (typeof p.message !== "string" || !p.message.trim()) {
    errors.push("message: 必须为非空字符串");
  }
  if (p.message && p.message.length > 500) {
    errors.push(`message: 长度${p.message.length}超过500`);
  }
  if (p.idempotencyKey == null) {
    errors.push("idempotencyKey: 必填（幂等键）");
  }
  if (p.route && typeof p.route !== "object") {
    errors.push("route: 必须为对象");
  }
  return { ok: errors.length === 0, errors };
}

// ============================================================
// 【增量】站②：幂等去重缓存 —— 让网络重试无害
// ============================================================
// 场景：客户端超时重发同一条消息（网络抖动很常见）。
// 没有 dedupe：同一条消息跑两遍模型（双倍花钱、用户收到两条回复）。
// 有 dedupe：第二次直接返回缓存结果，模型只跑一次。
//
// 真实代码: agent.ts 用 `agent:${idempotencyKey}` 作 key，
// 短 TTL 的 Map 缓存。
class DedupeCache {
  constructor(ttlMs = 10_000) {
    this.map = new Map(); // key → {result, ts} | {pending: Promise}
    this.ttlMs = ttlMs;
  }
  // 【修复竞态】get 可能返回"进行中"的 Promise —— 重试撞上尚未完成的
  // 首次请求时，等待同一个 Promise 而不是再跑一遍。
  // 真实系统同样要处理这个（进行中请求的占位/dedupe entry 先落 ACK）。
  get(key) {
    const hit = this.map.get(key);
    if (!hit) return undefined;
    if (hit.pending) return hit; // {pending}
    if (Date.now() - hit.ts > this.ttlMs) {
      this.map.delete(key);
      return undefined;
    }
    return hit;
  }
  setPending(key, pending) {
    this.map.set(key, { pending });
  }
  settle(key, result) {
    this.map.set(key, { result, ts: Date.now() });
  }
}

// ============================================================
// 【增量】站⑤：编排层 —— 管家的站
// ============================================================
// 对应真实 runReplyAgent。它自己不调模型，统筹这些事：
//   1. 打字中提示（用户体验：让用户知道在处理）
//   2. 压缩预检（历史太长先压，别把 50 万 token 塞给模型）
//   3. 压力测试标注（教学用，模拟系统繁忙时的排队反馈）
//   4. 调用执行层（lane + attempt + 工具循环 —— 前5步的零件）
//   5. 回复清洗（剥内部 token，别把 [TOOL_RESULT] 泄给用户）
//
async function runReply({ text, session, lane, model, tools, onTyping }) {
  // ① 打字提示（真实: typing indicator，渠道支持时立即反馈）
  onTyping?.();

  // ② 压缩预检（对应真实 runPreflightCompactionIfNeeded）
  const history = session.getHistory();
  if (history.length > 8) {
    console.log("   [编排] 历史过长，先压缩...");
    session.append("compaction", {
      message: { role: "assistant", content: `(自动摘要: 此前${history.length}条)` },
    });
  }

  // ③ 执行（进 lane 排队 → 工具循环）
  return lane.enqueue(async () => {
    const guard = new ToolLoopGuard(); // per-run（第5步的教训）
    session.append("message", { message: { role: "user", content: text } });
    const r = await runLoop({ model, messages: session.getHistory(), tools, guard });
    session.append("message", { message: { role: "assistant", content: r.text } });
    return r;
  });
}

// ============================================================
// 以下为前 5 步的零件（SessionLane/SessionManager/ToolLoopGuard/
// runLoop/mock 模型），此处原样组装，标注从略 —— 它们的故事
// 各自的第 N 步文件里。
// ============================================================
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

const SESSIONS_DIR = "/tmp/toy-sessions-step6";
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
    const lastCompaction = chain.map((e) => e.type).lastIndexOf("compaction");
    const active = lastCompaction === -1 ? chain : chain.slice(lastCompaction);
    return active.filter((e) => e.type === "message").map((e) => ({ role: e.message.role, content: e.message.content }));
  }
}

class ToolLoopGuard {
  constructor() { this.history = []; this.blocks = 0; }
  #hash(call) {
    return createHash("sha256").update(`${call.name}:${JSON.stringify(call.arguments)}`).digest("hex").slice(0, 12);
  }
  check(toolCall) {
    const h = this.#hash(toolCall);
    this.history.push({ hash: h, toolName: toolCall.name });
    const n = this.history.filter((x) => x.hash === h).length;
    if (n >= 8) return { abort: true, reason: `全局熔断: 重复${n}次` };
    if (n >= 5) {
      this.blocks += 1;
      return { abort: this.blocks > 3, reason: `[CRITICAL] 死循环(第${n}次)。立即停止，用文字回复。` };
    }
    if (n >= 3) return { abort: false, reason: `[WARNING] 已重复调用 ${toolCall.name} ${n} 次。` };
    return null;
  }
}

const TOOLS = {
  exec: {
    async run(args) { return { ok: true, output: `(exec: ${args.command} → mock输出)` }; },
  },
  date: {
    async run() { return { ok: true, output: new Date().toLocaleString("zh-CN") }; },
  },
};

const mockModel = {
  async create({ messages }) {
    await new Promise((r) => setTimeout(r, 120));
    const last = messages[messages.length - 1];
    const lastText = typeof last.content === "string" ? last.content : "";
    const isToolResult = lastText.includes("[TOOL_RESULT]");
    if (!isToolResult && /几点|时间/.test(lastText)) {
      return {
        content: [{ type: "toolCall", id: `call_${Date.now()}`, name: "date", arguments: {} }],
        stopReason: "toolUse",
      };
    }
    if (isToolResult) {
      return {
        content: [{ type: "text", text: `根据工具结果：${lastText.replace("[TOOL_RESULT] ", "").slice(0, 40)}` }],
        stopReason: "end_turn",
      };
    }
    return { content: [{ type: "text", text: `[model] 你说的是: ${lastText}` }], stopReason: "end_turn" };
  },
};

async function runLoop({ model, messages, tools, guard, maxTurns = 20 }) {
  const convo = [...messages];
  for (let turn = 0; turn < maxTurns; turn++) {
    const reply = await model.create({ messages: convo, tools });
    const toolCalls = reply.content.filter((c) => c.type === "toolCall");
    if (toolCalls.length === 0) {
      convo.push({ role: "assistant", content: reply.content[0].text });
      return { text: reply.content[0].text, turns: turn + 1, convo };
    }
    convo.push({ role: "assistant", content: `[发起工具调用: ${toolCalls.map((t) => t.name).join(", ")}]` });
    const results = [];
    let aborted = null;
    for (const tc of toolCalls) {
      const verdict = guard.check(tc);
      if (verdict) {
        if (verdict.abort) { aborted = verdict.reason; break; }
        results.push(`[TOOL_RESULT] ${verdict.reason}`);
        continue;
      }
      const out = await tools[tc.name].run(tc.arguments);
      results.push(`[TOOL_RESULT] ${out.output}`);
    }
    if (aborted) return { text: `[run aborted] ${aborted}`, turns: turn + 1, convo, aborted: true };
    convo.push({ role: "user", content: results.join("\n") });
  }
  return { text: "[超过最大轮数]", turns: maxTurns, convo };
}

// ============================================================
// 【增量】网关 —— 守门员上岗，全链路总装
// ============================================================
const gateway = new EventEmitter();
const dedupe = new DedupeCache(10_000);
fs.mkdirSync(SESSIONS_DIR, { recursive: true });
const session = new SessionManager(path.join(SESSIONS_DIR, "demo.jsonl"));
const lane = new SessionLane();

// 【增量】agent method 入口 —— 完整走一遍 8 站
// （教学注意：真实网关的 method handler 由 WS 帧触发并有 res 回包；
//  这里把它抽成可 await 的函数，方便演示脚本控制时序）
async function agentMethod(params) {
  const t0 = Date.now();

  // ---- 站②a 参数校验：畸形请求止步于此 ----
  const v = validateParams(params);
  if (!v.ok) {
    console.log(`✗ 校验拒绝: ${v.errors.join("; ")}`);
    return { ok: false, error: `invalid params: ${v.errors.join("; ")}` };
  }

  // ---- 站②b 幂等去重：重试无害（含"进行中"的并发重试） ----
  const idemKey = `agent:${params.idempotencyKey}`;
  const cached = dedupe.get(idemKey);
  if (cached?.pending) {
    // 首次请求还在跑：等它，复用它的结果（不再起第二个 run）
    console.log(`↺ 幂等命中(进行中): 等待首次请求完成后复用结果`);
    return cached.pending;
  }
  if (cached?.result) {
    console.log(`↺ 幂等命中(缓存${Date.now() - cached.ts}ms前): 直接返回，不再调模型`);
    return cached.result;
  }
  // 占位：立刻登记"进行中"，之后撞上同一 key 的重试都会等这个 Promise
  const runPromise = (async () => {
    // ---- 站③ 路由（简化：单会话；真实按 route 推导 sessionKey 分道） ----
    const route = params.route ?? {};
    console.log(`→ [${route.userId ?? "cli"}] "${params.message.slice(0, 24)}" (idem=${params.idempotencyKey.slice(0, 8)})`);

    // ---- 站④⑤⑥⑦⑧：编排层统筹，内部含 lane 排队/工具循环/熔断 ----
    const result = await runReply({
      text: params.message,
      session,
      lane,
      model: mockModel,
      tools: TOOLS,
      onTyping: () => console.log("   ⌨️ (打字中提示)"),
    });

    // ---- 站⑨ 回复清洗 + 回包（对应真实 normalizeStreamingText） ----
    const cleaned = result.text.replace(/\[TOOL_RESULT\][^\n]*/g, "").trim();
    const payload = { ok: !result.aborted, text: cleaned, runId: `run_${Math.random().toString(16).slice(2, 8)}` };
    dedupe.settle(idemKey, payload); // 从"进行中"转为"已完成"

    console.log(`↩ ${payload.text} (${Date.now() - t0}ms, 工具循环${result.turns}轮)\n`);
    return payload;
  })();
  dedupe.setPending(idemKey, runPromise);
  return runPromise;
}

// ---------- 演示：守门员的价值 ----------
(async () => {
  console.log("=== 第 6 步：网关守门层（校验/幂等/编排）===\n");

  // A. 畸形请求：门口就拒（第0步会怎样？——直接炸到核心逻辑）
  console.log("[A] 畸形请求：");
  await agentMethod({ message: "", idempotencyKey: "bad-1" });
  await agentMethod({ message: "x".repeat(600), idempotencyKey: "bad-2" });
  await agentMethod({ message: "没有幂等键的消息" });

  // B. 网络重试：客户端超时，几乎同时重发同一 idempotencyKey
  //    （注意：两条都不 await —— 模拟并发到达，撞上"进行中"占位）
  console.log("\n[B] 并发重试（同 idem key 几乎同时到达两条）：");
  const p1 = agentMethod({ message: "现在几点了？", idempotencyKey: "retry-demo-001" });
  const p2 = agentMethod({ message: "现在几点了？", idempotencyKey: "retry-demo-001" }); // ← 重试
  await Promise.all([p1, p2]);

  // B2. 完成后的重试：命中已完成的缓存
  console.log("\n[B2] 完成后的重试（命中缓存）：");
  await agentMethod({ message: "现在几点了？", idempotencyKey: "retry-demo-001" });

  // C. 正常流程：走完整 8 站
  console.log("\n[C] 正常消息（走完整管线）：");
  await agentMethod({ message: "再问一次几点了", idempotencyKey: "normal-001" });
})();

// ============================================================
// 运行观察点：
//   node step6-gateway-layer.cjs
//
// 预期：
//   [A] 三条畸形请求全部被拒 —— 注意模型一次都没被调用
//       （守门员的意义：垃圾止步于门口，不浪费核心资源）
//   [B] 两条并发到达：只有一条真正起 run（"→ [cli]"只打印一次），
//       另一条 "↺ 幂等命中(进行中)" —— 等待并复用同一个 Promise
//   [B2] 完成后重试："↺ 幂等命中(缓存…ms前)" —— 模型没跑
//   [C] 全链路: 校验→路由→编排(打字提示→压缩预检)→lane→工具循环→清洗回包
//
// ============ 终极对照表：toy 6 步 ↔ 真实系统 8 站 ============
//
//   step1  attempt 层        ↔ 站⑦ runEmbeddedAttempt + model-fallback
//   step2  队列层            ↔ 站④ queue/settings + 站⑥ command-queue 两层lane
//   step3  路由层            ↔ 站③ sessionKey 推导/解析/分道
//   step4  session 文件层     ↔ 站⑨ SessionManager/JSONL/parentId/compaction
//   step5  工具循环层        ↔ 站⑧ pi-agent-core runLoop + 熔断器
//   step6  网关守门层        ↔ 站② validate+dedupe + 站⑤ runReplyAgent 编排
//   （站① 入口 = 我们的 EventEmitter gateway 本身）
//
// 思考：
//   1. 校验为什么要放在最外层？（越早拒绝越便宜 —— 畸形请求
//      穿透到编排层才会浪费排队/压缩/会话打开的开销）
//   2. dedupe 的 TTL 为什么短（真实10s量级）？（幂等键只需要覆盖
//      "网络重试窗口"，长了反而让用户重发的合法消息被吞）
//   3. 编排层为什么不自己调模型？（单一职责：它管时序和准备，
//      执行交给专门的层 —— 这样改执行策略不动编排，反之亦然）
// ============================================================
