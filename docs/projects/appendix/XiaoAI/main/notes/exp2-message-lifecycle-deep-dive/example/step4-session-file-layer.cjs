// ============================================================
// 第 4 步：记忆存在哪、怎么不被写坏 —— 长出「session 文件层」
// ============================================================
//
// 【增量】相对第 3 步，本文件新增：
//   1. JSONL 持久化       —— append-only 会话文件（重启不丢）
//   2. parentId 链        —— 每条记录指向上一条（链表而非列表）
//   3. SessionManager     —— 唯一合法的写入入口（工程红线的原因）
//   4. 简化版 compaction  —— 历史太长时摘要压缩（链式结构的用途）
//
// 【对应真实代码】
//   会话文件路径      → ~/.xiaoaiclaw-dev/agents/<agentId>/sessions/<sessionId>.jsonl
//   懒创建            → 第一条消息处理完才建文件（我们实验1实测过）
//   parentId 链/DAG   → pi-agent-core 的 transcript 结构；
//                        工程红线：必须走 SessionManager.appendMessage()，
//                        裸写 JSONL 会切断 leaf 路径，破坏 compaction/历史回放
//                        （依据: src/gateway/server-methods/CLAUDE.md）
//   compaction        → 长对话摘要压缩（docs/concepts/compaction.md）
//
// 用法：
//   node step4-session-file-layer.cjs            （第一次跑：建文件）
//   node step4-session-file-layer.cjs            （再跑一次：读回历史！）
//   cat /tmp/toy-sessions/demo.jsonl             （亲眼看文件内容）
// ============================================================

const fs = require("fs");
const path = require("path");
const { EventEmitter } = require("events");

// 【增量】会话文件目录（真实: ~/.xiaoaiclaw-dev/agents/dev/sessions/）
const SESSIONS_DIR = "/tmp/toy-sessions";

// ---------- mock 模型 ----------
const mockModel = {
  async create({ messages }) {
    await new Promise((r) => setTimeout(r, 150));
    const last = messages[messages.length - 1];
    const historyChars = messages
      .slice(0, -1)
      .reduce((n, m) => n + m.content.length, 0);
    return {
      content: [
        { type: "text", text: `[model|上下文${historyChars}字] 你说的是: ${last.content}` },
      ],
    };
  },
  // 【增量】mock 的摘要模型：把长历史压成一句话（真实: 用 LLM 生成摘要）
  async summarize(texts) {
    return `(摘要: 此前聊了${texts.length}句，主题是"${texts[0].slice(0, 8)}…")`;
  },
};

// ============================================================
// 【增量】SessionManager —— 会话文件的唯一写入入口
// ============================================================
//
// 为什么所有写入必须走它？
//   1. 它负责维护 parentId 链 —— 绕过它写 = 可能出现"没有 parent 的
//      孤儿记录"，链从此断掉
//   2. 它负责 fsync 语义 —— 一行一个完整 JSON，写一半崩了最多丢一行
//   3. 它负责 leaf 指针 —— compaction 分支依赖 leaf 找到"当前链头"
//
// 真实红线（来自 src/gateway/server-methods/CLAUDE.md 原话）：
//   "never append Pi type:message entries via raw JSONL writes
//    (missing parentId can sever the leaf path and break compaction/history)"
//
class SessionManager {
  constructor(filePath) {
    this.filePath = filePath;
    this.entries = []; // 全部记录（内存态）
    this.leafId = null; // 当前链头
    this.#load();
  }

  // 启动时读回（重启不丢的关键）
  #load() {
    if (!fs.existsSync(this.filePath)) return; // 懒创建：没文件就没有会话
    for (const line of fs.readFileSync(this.filePath, "utf8").split("\n")) {
      if (!line.trim()) continue;
      const e = JSON.parse(line);
      this.entries.push(e);
      if (e.id) this.leafId = e.id; // 文件最后一行就是链头
    }
  }

  // 唯一合法的追加入口（对应真实 SessionManager.appendMessage）
  append(type, payload) {
    const entry = {
      type, // "session" | "message" | "compaction" ...
      id: Math.random().toString(16).slice(2, 10), // 模拟真实: randomBytes hex
      parentId: this.leafId, // 【增量】链的关键：指向上一条
      timestamp: new Date().toISOString(),
      ...payload,
    };
    // append-only：只追加，永不改写旧行（崩溃安全的核心）
    fs.appendFileSync(this.filePath, JSON.stringify(entry) + "\n");
    this.entries.push(entry);
    this.leafId = entry.id; // 链头前移
    return entry;
  }

  // 重建"当前链"：从 leaf 沿 parentId 回溯（compaction 后旧链自然被绕开）
  getActiveChain() {
    const byId = new Map(this.entries.map((e) => [e.id, e]));
    const chain = [];
    let cur = this.leafId;
    while (cur) {
      const e = byId.get(cur);
      if (!e) break;
      chain.unshift(e);
      cur = e.parentId;
    }
    return chain;
  }

  // 历史消息（仅 message 类型、仅当前链上的 —— 压缩过的旧链不计入）
  getHistory() {
    const chain = this.getActiveChain();
    // 【修复】找到链上最后一次 compaction，只有它之后的消息才算"活跃历史"
    // （真实 compaction 语义：摘要替代之前的所有内容）
    const lastCompactionIdx = chain.map((e) => e.type).lastIndexOf("compaction");
    const active = lastCompactionIdx === -1 ? chain : chain.slice(lastCompactionIdx);
    return active
      .filter((e) => e.type !== "session")
      .map((e) => ({ role: e.message.role, content: e.message.content }));
  }
}

// ============================================================
// 【增量】简化版 compaction —— 为什么必须是链不是列表
// ============================================================
//
// 场景：聊了 500 轮，每次把全部历史塞给模型 → 又贵又超窗口。
// 做法：生成一条"摘要记录"，让它成为新的链头。旧消息链**不删**
// （审计/回滚还要用），但 getActiveChain 从新链头回溯，
// 旧链自然"掉出"当前上下文。
//
// 这就是 parentId 链的杀手级用途：压缩 = 换链头，不动任何旧数据。
// 列表结构做不到（要么删数据、要么全量保留）。
//
async function compactIfNeeded(session, threshold = 6) {
  const msgs = session.getHistory();
  if (msgs.length <= threshold) return false;

  const userTexts = msgs.filter((m) => m.role === "user").map((m) => m.content);
  const summary = await mockModel.summarize(userTexts);

  // 写入一条 compaction 记录 —— 它成为新链头
  session.append("compaction", {
    message: { role: "assistant", content: summary },
    compactedCount: msgs.length, // 记录压掉了几条（真实系统也记，用于审计）
  });
  console.log(`   🗜 compaction: ${msgs.length} 条 → 1 条摘要 "${summary}"`);
  return true;
}

// ---------- SessionLane（与第 3 步相同，略去注释） ----------
class SessionLane {
  constructor(laneKey) {
    this.laneKey = laneKey;
    this.queue = [];
    this.active = false;
  }
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
      try {
        e.resolve(await e.task());
      } catch (err) {
        e.reject(err);
      }
    }
    this.active = false;
  }
}

// ---------- 网关（路由沿用第 3 步，历史改为走 SessionManager） ----------
const gateway = new EventEmitter();
const lane = new SessionLane("session:agent:demo:main");
fs.mkdirSync(SESSIONS_DIR, { recursive: true });

// 【增量】注意：这里不是每次 new —— 文件存在则读回（重启恢复）
const session = new SessionManager(path.join(SESSIONS_DIR, "demo.jsonl"));

// 懒创建提示：第一次跑文件不存在，首条消息处理时才有内容
if (session.entries.length === 0) {
  console.log("(新会话：文件为空，将懒创建)\n");
} else {
  console.log(`(重启恢复：从文件读回 ${session.entries.length} 条记录)\n`);
}

gateway.on("message", async (text) => {
  const reply = await lane.enqueue(async () => {
    await compactIfNeeded(session); // 【增量】先检查要不要压缩
    session.append("message", { message: { role: "user", content: text } });
    const r = await mockModel.create({ messages: session.getHistory() });
    session.append("message", { message: { role: "assistant", content: r.content[0].text } });
    return r;
  });
  console.log(`→ "${text}"\n↩ ${reply.content[0].text}\n`);
});

// ---------- 模拟 ----------
(async () => {
  console.log("=== 第 4 步：session 文件层（持久化 + 链 + 压缩）===\n");

  // 发 8 条消息（超过阈值 6，会触发一次 compaction）
  const round = process.env.ROUND || "1";
  for (const t of [
    "第1句：我在学架构",
    "第2句：今天是第4步",
    "第3句：马上要触发压缩",
    "第4句：继续聊",
    "第5句：继续聊",
    "第6句：继续聊",
    "第7句：继续聊",
    `第8句（本轮=${round}）：到这里应该压缩过了`,
  ]) {
    gateway.emit("message", t);
    await new Promise((r) => setTimeout(r, 220));
  }

  // 打印文件与链的状态
  console.log("=== 文件内容（cat /tmp/toy-sessions/demo.jsonl）===");
  for (const line of fs.readFileSync(session.filePath, "utf8").split("\n")) {
    if (!line.trim()) continue;
    const e = JSON.parse(line);
    const m = e.message ? `${e.message.role}:${String(e.message.content).slice(0, 18)}` : "-";
    console.log(
      `  id=${e.id} parent=${String(e.parentId).slice(0, 8)} type=${e.type.padEnd(11)} ${m}`,
    );
  }
  console.log(
    `\n当前活跃链上的历史（压缩后模型看到的）: ${session.getHistory().length} 条`,
  );
})();

// ============================================================
// 运行观察点：
//   node step4-session-file-layer.cjs
//
// 预期：
//   1. 第 7 句左右触发 compaction 日志："🗜 compaction: 7 条 → 1 条摘要"
//   2. 之后模型回复的"上下文N字"明显变小 —— 压缩生效
//   3. 文件 dump 里能看到 parentId 链：compaction 记录的 parent
//      指向最后一条旧消息 —— 新链从此分叉
//   4. ★ 再跑一次（node step4-session-file-layer.cjs）：
//      开头显示"重启恢复：从文件读回 N 条记录" —— 重启不丢！
//      这就是持久化的意义（第 3 步的内存版重启即失忆）
//
// 思考：
//   1. 为什么 appendFileSync 一行一条是"崩溃安全"的？（写一半断电，
//      最多最后一行残缺，前面的行都是完整 JSON，#load 会跳过空行）
//   2. compaction 为什么不删旧记录？（审计/回滚/分支还要用；
//      链结构让"排除出上下文"和"删除数据"解耦）
//   3. 绕过 append() 直接 fs.appendFileSync 写一行会发生什么？
//      （parentId 可能断链 → getActiveChain 回溯中断 → 历史丢失
//       —— 这就是工程红线存在的理由）
//   4. 真实系统二开为什么要加 Redis？（多实例/多进程共享会话 +
//      文件在 K8s 上不好共享 —— 见实验6的双写排障文档）
// ============================================================
