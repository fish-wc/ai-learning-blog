// ============================================================
// 第 3 步：消息从哪来、往哪回 —— 长出「路由层（sessionKey）」
// ============================================================
//
// 【增量】相对第 2 步，本文件新增：
//   1. buildSessionKey()  —— 从消息上下文推导 sessionKey
//   2. lane 按 sessionKey 分道 —— 不同用户/渠道互不排队
//   3. 会话上下文注入 —— 模型每次能看到该用户的历史（有记忆了！）
//
// 【对应真实代码】
//   sessionKey 格式    → agent:{agentId}:{channel}:{groupId}:u-{userId}
//                        （五段格式，SaaS 多租户隔离键是最后的 u-{userId}）
//   key 推导           → src/config/sessions/main-session.ts:36（CLI 退化三段）
//   key 解析           → src/sessions/session-key-utils.ts:23（parseAgentSessionKey）
//   按 key 分 lane     → src/agents/pi-embedded-runner/lanes.ts:3
//                        （resolveSessionLane: "session:" + key）
//
// 用法：
//   node step3-routing-layer.cjs
// ============================================================

const { EventEmitter } = require("events");

// ---------- mock 模型（与第 2 步相同） ----------
const mockModel = {
  async create({ messages }) {
    await new Promise((r) => setTimeout(r, 300));
    const last = messages[messages.length - 1];
    const historyCount = messages.length - 1; // 除当前条外的历史条数
    return {
      content: [
        {
          type: "text",
          text: `[model] (记得你之前说过${historyCount}句话) 你说的是: ${last.content}`,
        },
      ],
    };
  },
};

// ============================================================
// 【增量】路由层：sessionKey 的构造与解析
// ============================================================
//
// sessionKey 回答的唯一问题："这条消息属于哪个会话？"
//
// 真实格式（五段）：
//   agent:{agentId}:{channel}:{groupId}:u-{userId}
//     │        │         │         │         │
//   哪个助手  哪个渠道   哪个群    私聊为空   哪个用户 ← 隔离键
//
// 群聊示例:  agent:dev:telegram:g123:u-10086
// 私聊示例:  agent:dev:telegram:-:u-10086     （groupId 空占位）
// CLI/TUI:   agent:dev:main                    （无渠道/群/用户 → 退化三段）
//
function buildSessionKey({ agentId, channel, groupId, userId }) {
  const g = groupId ?? "-"; // 群id缺失时占位（私聊）
  const u = userId != null ? `:u-${userId}` : ""; // 无用户身份（CLI）则省略
  return `agent:${agentId}:${channel}:${g}${u}`;
}

// 解析（对应真实 parseAgentSessionKey 的双段返回 {agentId, rest}）
function parseSessionKey(key) {
  const parts = key.split(":").filter(Boolean);
  return { agentId: parts[1], rest: parts.slice(2).join(":") };
}

// ============================================================
// 【增量】会话注册表：sessionKey → 该会话的（历史 + lane）
// ============================================================
//
// 这就是"记忆"的雏形：每个 key 一份 messages 数组。
// 模型每次调用时注入该 key 的全部历史 —— 同一个用户连续说话，
// 模型"记得"之前的内容；不同用户的数组天然隔离。
//
// 真实代码里历史的持久化是第 4 步的事（session 文件），
// 这一步先放在内存里，但"按 key 隔离"的结构已经就位。
//
const sessions = new Map(); // sessionKey → { messages: [], lane: SessionLane }

function getOrCreateSession(sessionKey) {
  if (!sessions.has(sessionKey)) {
    sessions.set(sessionKey, {
      messages: [], // 会话历史
      lane: new SessionLane(`session:${sessionKey}`), // 【增量】每会话独立 lane
    });
  }
  return sessions.get(sessionKey);
}

// ---------- SessionLane（与第 2 步相同） ----------
class SessionLane {
  constructor(laneKey) {
    this.laneKey = laneKey;
    this.queue = [];
    this.active = false;
  }
  enqueue(task) {
    return new Promise((resolve, reject) => {
      this.queue.push({ task, enqueuedAt: Date.now(), resolve, reject });
      this.#pump();
    });
  }
  async #pump() {
    if (this.active) return;
    this.active = true;
    while (this.queue.length > 0) {
      const entry = this.queue.shift();
      try {
        entry.resolve(await entry.task());
      } catch (err) {
        entry.reject(err);
      }
    }
    this.active = false;
  }
}

// ---------- 网关 ----------
const gateway = new EventEmitter();

// 【增量】消息入口现在带"路由上下文"（谁、从哪、哪个群）
// 真实世界里这些来自渠道适配层（telegram 的 from.id、飞书的 sender 等）
gateway.on("message", async (text, route) => {
  // ① 推导 sessionKey —— 每条消息先上"户口"
  const sessionKey = buildSessionKey(route);
  console.log(`→ [${route.userId ?? "cli"}@${route.channel}] "${text}"`);
  console.log(`   路由到 sessionKey: ${sessionKey}`);

  // ② 找到（或创建）该 key 的会话
  const session = getOrCreateSession(sessionKey);

  // ③ 进该会话自己的 lane（不同会话互不阻塞）
  const reply = await session.lane.enqueue(async () => {
    // 【增量】注入完整历史 —— 这就是"有记忆"的来源
    session.messages.push({ role: "user", content: text });
    const r = await mockModel.create({ messages: [...session.messages] });
    session.messages.push({ role: "assistant", content: r.content[0].text });
    return r;
  });
  console.log(`↩ ${reply.content[0].text}\n`);
});

// ---------- 模拟多用户多渠道 ----------
(async () => {
  console.log("=== 第 3 步：路由层（谁的消息进哪个会话）===\n");

  // 同一 telegram 用户的两条消息 → 同一个 key → 有记忆
  gateway.emit("message", "我叫小明", { agentId: "dev", channel: "telegram", userId: 10086 });
  await new Promise((r) => setTimeout(r, 400));
  gateway.emit("message", "我叫什么？", { agentId: "dev", channel: "telegram", userId: 10086 });
  await new Promise((r) => setTimeout(r, 400));

  // 另一个用户 → 不同 key → 互相看不见
  gateway.emit("message", "我叫什么？", { agentId: "dev", channel: "telegram", userId: 10010 });
  await new Promise((r) => setTimeout(r, 400));

  // 同一用户换渠道 → 也是不同 key（跨渠道不共享记忆）
  gateway.emit("message", "我叫什么？", { agentId: "dev", channel: "feishu", userId: 10086 });
  await new Promise((r) => setTimeout(r, 600));

  // 打印最终会话表
  console.log("=== 会话注册表（sessionKey → 历史条数）===");
  for (const [key, s] of sessions) {
    console.log(`  ${key}  (${s.messages.length} 条消息)`);
  }
})();

// ============================================================
// 运行观察点：
//   node step3-routing-layer.cjs
//
// 预期输出重点：
//   1. 小明的第二条"我叫什么" → 模型回复 "(记得你之前说过1句话)"
//      —— 同 key 历史注入生效，有记忆了！
//   2. 用户10010 问"我叫什么" → "(记得你之前说过0句话)"
//      —— 看不到小明的记忆，隔离生效！
//   3. 小明换到飞书问 → 同样 0 句话
//      —— 跨渠道隔离（设计如此：渠道是隔离维度之一）
//
// 思考：
//   1. 如果把 userId 从 key 里去掉会发生什么？（两个用户共享一份
//      历史 → A 能看到 B 的对话 —— 这就是"数据越权"，为什么
//      u-{userId} 是 SaaS 多租户的命根子）
//   2. 不同 key 的 lane 是并发的 —— 这解释了真实系统为什么
//      "不同会话可以并行，同一会话必须串行"
//   3. 为什么 channel 也要进 key？（同一人在不同渠道的对话
//      语境完全不同，混在一起反而干扰模型）
// ============================================================
