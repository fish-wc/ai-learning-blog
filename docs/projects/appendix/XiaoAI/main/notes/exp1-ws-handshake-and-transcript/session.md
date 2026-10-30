---
title: "session"
date: 2026-10-29
tags:

- OpenClaw
- session

categories:
- AI Agent
---
# 概念拆解

## 第一部分：parentId、compaction、appendMessage

先回顾会话`.jsonl`文件：它**不是普通数组，是链表/DAG**。
每一条消息记录：

```json
{"id":"A","parentId":null,"type":"session"}      //会话头
{"id":"B","parentId":"A","type":"message"}
{"id":"C","parentId":"B","type":"message"}
```

`id`是本条唯一编号；`parentId`指向**上一条消息的id**，把全部记录串成一条链条。

### 1. compaction 是什么？中文叫「会话压缩/整理」

> compaction：对话上下文压缩、折叠。

大模型有**上下文token上限**。对话很长，jsonl里面几十上百条消息，全部丢给模型会超token限制。
compaction就是系统做的长对话整理：

1. 把旧的一大段历史消息做摘要浓缩；
2. 生成一条**摘要新节点**；
3. 让新节点作为**新的链表起点**；
4. 旧完整消息链保留在文件里（可以回溯历史），但后续对话不再从旧链继续，而是从摘要节点往后继续写新消息。

举例子：
原始链：`session(A) → B → C → D → E → F`
对话太长触发compaction：

- 对B‑F做摘要，产出摘要节点 `G(id=G, parentId=D)`
- **后续新消息不再接F，而是接G**：`G → H → I`

系统靠`parentId`遍历整条链条，识别哪些是已经被折叠的旧历史，哪些是当前活跃对话分支。

**如果某一条记录缺失parentId：**

- 这条记录不知道自己接在哪条消息后面；链表直接断裂（切断链）。
- compaction逻辑遍历链条的时候，走到断裂点就无法完整追溯对话。
  - 无法判断哪些消息需要被摘要折叠；
  - 分不清哪条是当前活跃分支；
  - 回放历史会话、fork分叉会话（`/new`新开对话分支）全部错乱。

> 这就是“破坏compaction”。

### 2. appendMessage 是什么

`appendMessage()` 是 `SessionManager`（会话管理器）对外提供的**API函数/方法**，专门用来往会话jsonl追加一条消息记录。

它内部帮你自动做这些事：

1. 拿到当前会话**最新叶子节点id**（也就是链条的最后一条消息的id）；
2. 把这个id自动填到新记录的`parentId`字段；
3. 生成本条消息全新唯一`id`；
4. 做格式校验；
5. 安全append追加写入jsonl磁盘文件；
6. 同步更新内存、同步更新`sessions.json`元数据；
7. 处理会话懒创建：如果文件还不存在，则自动新建jsonl文件。

### 3. 为什么绝对不能裸写jsonl，必须调用appendMessage？

> 裸写 = 自己直接用`fs.appendFile`打开jsonl，手写一行json直接写入磁盘。

如果裸写，你很容易犯错误：

1. 忘记填写`parentId`，链条断裂 → compaction失效，会话回放、fork分支全部坏掉；
2. parentId填错，指到错误id，链表逻辑错乱；
3. id重复冲突；
4. 忘记更新内存状态、不更新`sessions.json`元数据，内存和磁盘不一致。

`appendMessage`把这些复杂逻辑封装起来，开发者不用手动处理id、parentId、文件创建、元数据同步，保证链表永远合法。

> 工程红线：**所有写会话记录的代码，必须调用SessionManager.appendMessage，禁止直接写磁盘jsonl**。

---

## 第二部分：sessions.json 和 *.jsonl 的分工，什么叫路由索引

> 题目原话：`sessions.json`：路由索引（sessionKey → sessionId）；`xxx.jsonl`：会话流水文件。

### 1. 两个文件分别存什么

| 文件                    | 类型                     | 存储内容                                                                                                                                 |
| ----------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `sessions.json`       | **索引元数据文件** | key‑value字典。key = `sessionKey`；value是一大堆会话运行时元信息，里面包含`sessionId`，以及会话模型、技能快照、状态、会话文件路径等 |
| `23dfc0d9…596.jsonl` | **流水日志文件**   | 全部对话原始记录：session头、用户消息、assistant消息、tool调用、model_change事件，一行一条json。                                         |

### 2. 什么是路由索引（sessionKey → sessionId）

#### sessionKey

是**网关路由用的别名、业务标识**。例子文档里是：`agent:dev:main`

- 语义：`agent`类型，`dev`开发环境，名字叫`main`主会话。
- 人类/上层客户端（TUI）、代码逻辑，习惯使用可读性好的sessionKey。

#### sessionId

是一串很长的随机哈希串：`23dfc0d9b60368d18fb934b897aea596`

- 这是会话文件的文件名，**磁盘上会话文件就用sessionId命名**。
- 机器底层读写jsonl文件，必须用这个sessionId找文件。

> 问题来了：上层传过来`sessionKey=agent:dev:main`，网关怎么找到磁盘上对应的`.jsonl`文件？
> 👉 **靠sessions.json做翻译映射（路由索引）**。

**路由查找流程：**

1. TUI发送`req:agent`请求，告诉网关：我要操作会话 `sessionKey: agent:dev:main`
2. Gateway读取`sessions.json`，以`agent:dev:main`作为key查字典；
3. 取出value里面的`sessionId: "23dfc0d9b60368d18fb934b897aea596"`；
4. 同时拿到`sessionFile`绝对路径：`/home/zzy/.xiaoaiclaw‑dev/agents/dev/sessions/23dfc0d9b60368d18fb934b897aea596.jsonl`；
5. 网关就可以打开这个jsonl流水文件，读取历史对话、调用`appendMessage`追加新消息。

> 索引的意义：就像书的目录。
>
> - sessionKey：章节名字（人好理解）
> - sessionId：页码（机器用来定位文件）
> - sessions.json：目录表，把章节名翻译成页码。
> - *.jsonl：书本正文，完整对话全部写在这里。

### 3. 拆解你这份真实的sessions.json样例

顶层key是`"agent:dev:main"`（就是sessionKey，路由键），它对应的value对象里面关键字段：

```json
"agent:dev:main": {
    "sessionId": "23dfc0d9b60368d18fb934b897aea596",
    "sessionFile": "/home/zzy/..../23dfc0d9b60368d18fb934b897aea596.jsonl",
    "updatedAt": 1793087248525,
    "model": "claude-opus-4-6",
    "status": "done",
    "startedAt": 1793087239206,
    "skillsSnapshot": {...},
    "systemPromptReport": {...}
}
```

1. **sessionId**：找到磁盘会话文件的id；
2. **sessionFile**：直接给出jsonl完整磁盘路径；
3. **updatedAt / startedAt / status**：会话状态，什么时候启动，最后更新时间，运行状态；
4. **model、contextTokens**：记录本次会话使用什么大模型、上下文token上限；
5. **skillsSnapshot**：快照，记录会话加载了哪些skill技能，每个技能的路径、描述。

> ⚠️为什么要存skillsSnapshot快照？
> 会话中途模型使用的技能集合是会变的。后续重放历史会话时，必须恢复**当时那一瞬间的技能集合**，而不是现在最新磁盘上的SKILL.md文件（磁盘文件可能被修改）。快照保存在`sessions.json`元数据里。

6. **systemPromptReport**：系统提示词的统计，注入了哪些工作区md文件、工具列表token占用统计。

### 4. 两个文件职责严格分开

1. **sessions.json（索引元数据）：不保存对话消息！**
   它只存：映射关系、会话配置快照、运行状态、模型、技能快照、时间戳。**没有用户说的一句话，没有AI回复内容**。对话消息全部存在jsonl。
2. **xxx.jsonl（流水文件）：存全部对话记录，但它不知道sessionKey！**
   jsonl第一行session头只有sessionId，没有sessionKey。它不知道上层业务别名。必须靠sessions.json才能把sessionKey映射到这个文件。

### 5. 懒创建联动（回顾实验文档知识点）

> TUI连上网关，sessions文件夹是空的，第一条消息产生之后，才会生成jsonl文件。

1. 刚建立WS连接，还没发消息：`sessions.json`里面可以先存在`agent:dev:main`这条索引记录（内存先初始化会话元信息）；
2. 但此时磁盘上**还没有对应的sessionId.jsonl文件**；
3. 当第一条用户消息过来，调用`appendMessage`，才真正在磁盘创建xxx.jsonl；
4. 填充jsonl的session头记录。

### 6. 完整消息流转串讲（把前面全部知识串起来）

1. TUI通过WebSocket发送`req:agent`，指定`sessionKey=agent:dev:main`；
2. Gateway读取`sessions.json`索引，由sessionKey查出sessionId和sessionFile路径；
3. 调用`SessionManager.appendMessage()`写入用户消息；
   - appendMessage自动获取上一条消息id作为parentId；
   - 如果磁盘文件不存在，则新建xxx.jsonl；
   - 追加一行json写入流水；
   - 更新`sessions.json`里的updatedAt时间戳；
4. 大模型生成assistant回复（可能携带toolCall）；
5. **再次调用appendMessage**写入assistant消息，parentId自动指向用户消息id，维持链表；
6. 如果对话太长触发compaction（会话压缩）：appendMessage写入一条摘要节点，parentId指向旧链节点，维持链表不断裂；
7. 当需要历史回放：sessions.json拿到sessionFile路径，读取jsonl，依靠每一行的id+parentId遍历完整消息链，还原对话。

### 7. 关键易错总结

1. 如果裸写jsonl、不使用`appendMessage`，丢失parentId → 链表断裂 → compaction会话压缩逻辑完全失效；
2. `sessions.json`是索引（目录）：sessionKey → sessionId映射，存储会话配置快照，**不存聊天消息**；
3. `sessionId.jsonl`是流水正文：存储全部对话消息，靠parentId构成链表/DAG；
4. sessionKey是人/上层业务用的别名；sessionId是机器用来定位磁盘文件的随机ID。

# 多会话场景：多个 sessionId 时 sessions.json 的结构

核心规则先记住：

1. **每一条独立对话，拥有自己唯一的 `sessionId`（随机哈希，作为`.jsonl`文件名）**；
2. `sessions.json` 的**顶层 key 就是 `sessionKey`**；
3. `sessionKey ↔ sessionId` 是映射关系：**一个 sessionKey 对应 1 个 sessionId；但可以存在多个不同 sessionKey，分别指向不同 sessionId**；
4. 磁盘上会生成多个会话流水文件：`xxx.jsonl`、`yyy.jsonl`、`zzz.jsonl`，一一对应不同 `sessionId`。

> 重要区分两个概念

- `sessionId`：底层磁盘标识，不可读，UUID/长哈希，**一个对话永远绑定一个sessionId**，文件名字就是它。
- `sessionKey`：业务路由别名，给网关、TUI使用，格式约定：`agent:<环境>:<会话别名>`。

## sessionKey 的命名惯例

格式：`agent:<env>:<name>`

- `agent:dev:main` → 开发环境，主会话（我们之前的例子）
- `agent:dev:chat‑todo` → 开发环境，todo任务会话
- `agent:dev:debug‑gateway` → 开发环境，调试网关会话
- `agent:prod:main` → 生产环境主会话

> 你可以理解：`sessionKey` 就是你给对话起的“逻辑名字”；系统通过这个名字，找到真实磁盘会话文件(sessionId)。

### 场景：同时3个独立对话

1. 会话A：主聊天，sessionKey=`agent:dev:main`，sessionId=`23dfc0d9b60368d18fb934b897aea596`，文件：`23dfc0d9b60368d18fb934b897aea596.jsonl`
2. 会话B：todo任务对话，sessionKey=`agent:dev:chat‑todo`，sessionId=`87ac210ef4411223344556677889900aa`，文件：`87ac210ef4411223344556677889900aa.jsonl`
3. 会话C：网关调试对话，sessionKey=`agent:dev:debug‑gateway`，sessionId=`bb9876543210fedcba1234567890abcde`，文件：`bb9876543210fedcba1234567890abcde.jsonl`

## sessions.json完整示例（3个会话）

`sessions.json` 是一个大JSON对象，**顶层每个key都是sessionKey，每个value是该会话的元数据对象**。

> 我做精简，保留关键字段，省略冗长的`skillsSnapshot`、`systemPromptReport`大段内容。

```
{
  "agent:dev:main": {
    "sessionId": "23dfc0d9b60368d18fb934b897aea596",
    "sessionFile": "/home/zzy/.xiaoaiclaw-dev/agents/dev/sessions/23dfc0d9b60368d18fb934b897aea596.jsonl",
    "updatedAt": 1793087248525,
    "startedAt": 1793087239206,
    "status": "done",
    "model": "claude-opus-4-6",
    "skillsSnapshot": { "...": "省略大段技能快照内容" },
    "systemPromptReport": { "...": "省略系统提示统计" }
  },
  "agent:dev:chat-todo": {
    "sessionId": "87ac210ef4411223344556677889900aa",
    "sessionFile": "/home/zzy/.xiaoaiclaw-dev/agents/dev/sessions/87ac210ef4411223344556677889900aa.jsonl",
    "updatedAt": 1793088012111,
    "startedAt": 1793088000000,
    "status": "active",
    "model": "claude-opus-4-6",
    "skillsSnapshot": { "...": "todo会话自己的技能快照" },
    "systemPromptReport": { "...": "todo会话系统提示统计" }
  },
  "agent:dev:debug-gateway": {
    "sessionId": "bb9876543210fedcba1234567890abcde",
    "sessionFile": "/home/zzy/.xiaoaiclaw-dev/agents/dev/sessions/bb9876543210fedcba1234567890abcde.jsonl",
    "updatedAt": 1793088100222,
    "startedAt": 1793088088000,
    "status": "active",
    "model": "claude‑sonnet‑4‑5",
    "skillsSnapshot": { "...": "调试会话技能快照" },
    "systemPromptReport": { "...": "调试会话系统提示统计" }
  }
}
```

### 磁盘目录样子

```
~/.xiaoaiclaw-dev/agents/dev/sessions/
├── sessions.json                          # 统一索引，保存上面3条映射
├── 23dfc0d9b60368d18fb934b897aea596.jsonl          # agent:dev:main对话流水
├── 87ac210ef4411223344556677889900aa.jsonl         # agent:dev:chat‑todo流水
└── bb9876543210fedcba1234567890abcde.jsonl        # agent:dev:debug‑gateway流水
```

## 网关是怎么切换不同对话（完整流程）

1. 用户在TUI中切换会话，TUI在`req:agent`请求带上对应的`sessionKey`，比如：`sessionKey: "agent:dev:chat-todo"`
2. Gateway读取`sessions.json`，拿`agent:dev:chat‑todo`作为key查询；
3. 取出对应的`sessionId`，得到`sessionFile`磁盘路径；
4. 打开这个`.jsonl`文件，读取id+parentId链表，恢复这个对话全部历史；
5. 用户发消息，调用`appendMessage()`追加写入**这个jsonl文件**，不会污染别的会话；
6. 更新该sessionKey对应的元数据`updatedAt`；>

> 切换另一个会话就换另一个sessionKey，走一模一样流程，操作另外一套jsonl。

## 特殊情况1：fork会话（会话分支，对应 `/new` 命令）

> `/new`：从当前某条消息分叉出新对话。

- 原有会话：`agent:dev:main` → `S1(sessionId=AAA)`
- fork之后生成新会话：`agent:dev:main‑fork‑01` → `S2(sessionId=BBB)`
- 磁盘生成新的`BBB.jsonl`；新会话的jsonl内部`parentId`会指向老会话里某条消息id；
- `sessions.json`新增一条顶层key：`agent:dev:main‑fork‑01`，映射到新sessionId BBB。

> ⚠️fork出来是**全新独立sessionId，全新jsonl文件**，不是修改旧文件。旧会话`AAA.jsonl`原封不动保留。

## 特殊情况2：懒创建在多会话下的表现

当你在TUI创建一个新会话别名`agent:dev:chat‑todo`，**还没有发送第一条消息的时候**：

1. `sessions.json`里就会新增这个sessionKey条目（内存落盘元数据）；
2. 但是磁盘**不会立刻生成 `xxx.jsonl` 文件**；
3. 直到发送第一条消息调用`appendMessage`，磁盘才创建对应的jsonl会话文件。

## 几个高频误区

1. ❌误区：一个`sessions.json`里面只有一个会话。
   ✅正确：**sessions.json是全部会话的总索引，可以放几十上百个会话映射**。每一个顶层key代表一个业务会话别名。
2. ❌误区：sessionKey和sessionId一一绑定不能变。
   ✅正确：理论上可以修改sessions.json，让旧sessionKey指向别的sessionId，但是业务上不建议这么做；业务上约定一个sessionKey生命周期绑定一个sessionId。
3. ❌误区：多个会话的聊天消息会混在同一个jsonl。
   ✅正确：**每个sessionId独立一个jsonl流水文件**，互相隔离。appendMessage只会操作当前sessionId对应的那一个文件。
4. ❌误区：sessionKey不能随便起名。
   ✅正确：推荐遵守`agent:<环境>:<名字>`规范；系统内部解析会依赖这个格式，不按规范会有部分功能异常。

## 自测小问题（检验是否懂）

1. 如果我有5个对话，磁盘会有多少个jsonl？sessions.json顶层会有多少key？

> 答案：5个jsonl，sessions.json顶层5个sessionKey。

2. TUI想要打开“todo对话”，它携带什么参数发给gateway？gateway拿什么去找jsonl文件？

> TUI携带`sessionKey=agent:dev:chat‑todo`；gateway查sessions.json得到sessionId，通过sessionId拼接得到jsonl路径。

3. fork分叉出新会话，旧会话的jsonl文件会被修改吗？

> 不会，fork生成全新sessionId、全新jsonl；旧会话文件保持原样。

# 两个问题深度解析

> 背景回顾：jsonl每一行外层`type`：`session`/`message`/`model_change`/`custom`；
> 在**`type:"message"`内部的message对象**，才有`role`字段（`user` / `assistant` / `toolResult`）。

## 问题1：为什么 toolResult 挂在 `user role`（message.role="user"）里面？为什么不单独设置顶层type=tool‑use？

> 注意区分两层：
>
> 1. jsonl文件**行的顶层type**：`type:"message"`，代表这一行整体是消息记录；
> 2. 行内部 `message:{role:xxx}`：角色，取值 `user / assistant / toolResult`。
>    toolResult**不是顶层type**，它是嵌套在`type:"message"`里面的一个role值。

### 根本原因1：对齐 Claude Messages API 的原生协议模型

Anthropic Claude API历史上只有两个顶层角色：`user`、`assistant`。

- assistant输出：`role:assistant`，content数组里面放`tool_use`块（发起工具调用）；
- 工具执行完，**把工具输出包装进 `role:user` 的消息，content里面放`tool_result`块回传给大模型**。

> API视角理解：**工具结果本质是「输入给大模型的外部数据」，相当于用户侧输入**，不是模型输出，所以归到user这一侧。
> 模型：我（assistant）喊你执行命令；系统/工具扮演user，把命令输出结果作为输入还给我。

完整工具循环：

1. user（人类）→提问
2. assistant → toolCall（stopReason=toolUse），停下，等待工具输出
3. **系统执行工具，生成toolResult，包装进role=user的message，塞回历史上下文**
4. assistant读取toolResult，继续思考、输出回答。

### 根本原因2：对话回合交替约束，保证消息时序奇偶交替

对话历史必须严格交替：`user → assistant → user → assistant`。

- assistant发起toolCall（assistant轮次结束，stopReason=toolUse）；
- 接下来必须是user侧消息，再轮到assistant；
- 如果新增一个独立顶层`type:toolUse`，会打破`user/assistant`交替回合，上层调用大模型时还需要做适配转换，增加大量适配胶水代码。

### 那为什么不新增一个独立顶层type（例如`type:"toolResult"`）？

如果新增顶层独立type会带来一堆负担：

1. **每次调用大模型，都要做转换**：把自定义`toolResult`类型，转成符合Claude/OpenAI API格式的user消息；
2. 多一层分支逻辑：回放会话、compaction压缩、fork分支、历史回放全部要写额外if‑else处理这个新type；
3. 跨模型兼容：OpenAI、Claude、Ollama各家工具返回的协议不一样，自定义顶层type会抬高迁移成本。

> 补充：OpenClaw内部message对象的role支持`toolResult`，但**依然封装在外层`type:"message"`行里面**，没有做成独立的顶层jsonl行type，这样既内部可以识别toolResult角色，又对外可以直接映射各大LLM厂商的消息格式。

> 通俗总结：
> toolResult是**给大模型的输入**，逻辑归属user一侧；
> 不单独做顶层type，是为了**尽量贴近底层大模型API原生格式，减少转换胶水，简化回放、压缩、fork分支逻辑**。

> 误区纠正：不是把工具结果当成人类用户输入；这里`role="user"`指的是**流向模型的输入方**，不等于人类真人。网关/工具代理扮演输入方。

示例结构（jsonl一行）

```json
{
  "type":"message",      // 顶层行类型，永远是message
  "id":"xxx",
  "parentId":"yyy",
  "message": {
    "role":"user",       // 内部角色：user，承载toolResult
    "content":[
      {"type":"toolResult","toolCallId":"call_8a2218713de14dba9eef3512","content":"命令输出......"}
    ]
  }
}
```

## 问题2：做compaction会话压缩之后，compaction摘要内容被标记为什么？

### 1. jsonl文件中，**压缩摘要的顶层type是 `compaction`**（独立的顶层行类型，不是message）。

> 它是独立一行追加写入jsonl，同样拥有`id`和`parentId`，接入parentId链表DAG。

典型`compaction`条目样例：

```json
{
  "type":"compaction",
  "id":"comp_001",
  "parentId":"a0db8f7b",  // parentId指向被压缩链的最后一条原始消息id，接入链表
  "timestamp":"2026‑10‑27T...",
  "summary":"【这里存放旧对话的摘要文本】",
  "firstKeptEntryId":"12ee1ec3", // 从这个id开始，后面的消息保留不压缩
  "tokensBefore":12000           // 压缩之前消耗token数量
}
```

关键点：

1. **旧的原始消息不会被删除！** compaction不会删掉jsonl里旧行，只是新增一行`type:compaction`节点；磁盘完整历史全部保留，支持回溯、fork分支。
2. parentId链表关系：

```
旧链：S → M1 → M2 → M3 → M4（触发压缩点）
新增compaction节点C，C.parentId=M4.id
新消息全部接在C节点后面：C → M5 → M6
```

- 磁盘上M1‑M4依然存在；
- **后续给大模型喂上下文的时候，从C（compaction摘要）开始，再接M5、M6；M1‑M4不再送入模型，但磁盘文件完整留存。**

3. 回放逻辑：读到`type:"compaction"`节点，读取`firstKeptEntryId`，就知道哪些消息保留给模型。

### compaction和普通message区别

| 字段                 | 普通对话消息                     | compaction压缩摘要条目                                   |
| -------------------- | -------------------------------- | -------------------------------------------------------- |
| 顶层type             | `"message"`                    | `"compaction"`（独立类型）                             |
| 是否有message.role   | 有user/assistant/toolResult      | 没有message字段，自有summary、firstKeptEntryId等元数据   |
| 是否参与parentId链表 | 是id+parentId                    | 是id+parentId，必须调用appendMessage写入，保证链表不断裂 |
| 是否删除旧记录       | 不会删除旧行；append‑only追加写 | **不会删除旧行，只新增compaction行**               |

### 补充两个相关type（拓展了解）

- `branch_summary`：当fork分支的时候，生成分支摘要节点，顶层type=`branch_summary`；
- `reset`：重置会话历史窗口，顶层type=`reset`。

### 工程要点（结合之前学的appendMessage红线）

> compaction节点也是调用`SessionManager.appendMessage()`写入jsonl，不是裸写！
> appendMessage自动：
>
> 1. 自动填充parentId指向压缩断点的最后一条消息；
> 2. 生成compaction节点唯一id；
> 3. 追加写入jsonl，维护DAG链表；
>    如果裸写compaction条目，parentId写错，会直接破坏会话回放、压缩、fork全部逻辑。

## 串联回顾，自测巩固

1. toolResult为什么放在`message.role=user`？

> ①贴合LLM API原生模型：工具输出属于模型输入，归user侧；②维持user‑assistant回合交替；③避免新增顶层type带来大量格式转换、分支代码。

2. compaction摘要在jsonl是什么type？会不会删除旧消息？

> 顶层type=`compaction`独立条目；**不会删除旧消息，只追加新行；旧消息完整留在磁盘；通过parentId接入链表，靠firstKeptEntryId标记保留消息的起点。**

3. compaction节点是否也需要appendMessage？

> 需要，所有写入jsonl的记录（message、compaction、model_change）都必须走appendMessage，维护parentId链。

# 核心一句话先讲清楚

OpenClaw **没有把 workspace 的 md 内容作为一条 `role: system` 消息塞进消息列表**（这是很多框架的做法）；
它的方案：

1. **System 侧内容（所有 workspace md + memory）全部先拼接成一段超大的前置系统提示文本**；
2. 这段文本**作为底层 API 的 `system` 参数**（Claude 原生支持顶层 system 参数），**不是作为 messages 数组里的一条 role=system 消息**；
3. `messages` 数组里面，就只有交替的 `user / assistant`（内部还包含你前面了解的 toolResult 挂在 user 里）；

```
// Claude API 原生格式示意（OpenClaw 最终组装后发给模型的结构）
{
  "system": "【这里是拼接好的 AGENTS.md + SOUL.md + IDENTITY.md + USER.md + TOOLS.md + memory 全部内容】",
  "messages": [
    {"role":"user", "content": "用户第一条提问"},
    {"role":"assistant", "content": "模型回答，附带tool_use"},
    {"role":"user", "content": "toolResult工具返回结果"},
    {"role":"assistant", "content": "继续回答"}
  ]
}
```

> 👉 这就是为什么你在会话 jsonl 的 message.role 只能看到 user / assistant，**看不到 system**。

> 补充：很多别的LLM框架（OpenAI风格）不支持顶层 `system` 字段，它们就会把系统提示包装成一条 `{"role":"system"}` 插到 messages 数组最前面，这就是你说的三角色模板。
> Claude 很早就把 system 单独抽成顶层参数，和 messages 数组解耦，OpenClaw 是贴合 Claude 原生设计走的。

---

## 完整拼接流程（结合你前面学的 compaction、jsonl 链表）

### 步骤 1：网关加载本会话快照里的系统素材（新建会话时快照固化）

从 `sessions.json` 当前会话的 `systemPromptReport` 拿到当时注入的全部 workspace 文件内容：
AGENTS.md、SOUL.md、IDENTITY.md、USER.md、TOOLS.md，加上读取 memory 今日/昨日md，拼接成一整块长文本 → 赋值给 API 的顶层 `system`。

> ⚠️ 重点：**一旦会话创建完成，system 内容就冻结在快照里**；后续改 workspace md，这个会话的顶层system不会变，只有新开session才会重新读取md。

### 步骤 2：遍历 jsonl 的消息链表，生成 messages 数组（会跳过被compaction折叠掉的旧消息）

SessionManager 顺着 `id ← parentId` 链条遍历，处理 compaction 节点：

- 如果遇到 `type:compaction`：**compaction.summary 会作为一条 user 消息插入消息流，替代被折叠的一大段历史**；
- compaction 的 `firstKeptEntryId` 标记：在它之前的原始消息，不再放进 messages 数组；
- 从 compaction 节点往后，正常依次收集 `type:message`，提取里面的 `message.role`；
- 收集规则严格维持交替：`user ↔ assistant`
  - 人类用户输入 → `role:user`
  - 模型输出（含tool_use） → `role:assistant`
  - toolResult 工具返回结果 → 封装为 `role:user` 的消息（前面解释过，符合Claude工具协议）

> compaction 节点本身**不是顶层system**，它是一段被压缩后的历史摘要，混入 messages 对话流，作为 user 侧的上下文给到模型。

举一个压缩后最终上送的结构示例：

```json
{
  "system": "【一大段：SOUL + IDENTITY + USER + AGENTS + TOOLS + memory】",
  "messages": [
    {"role":"user", "content":"[历史摘要：前面十几轮对话的浓缩总结（来自compaction.summary）]"},
    {"role":"assistant", "content":"摘要之后模型上一轮回答..."},
    {"role":"user", "content":"用户刚刚新问的问题"}
  ]
}
```

## 对比两种主流方案，帮你分清差异

### 方案A：OpenAI 风格（你印象里的三角色模板：system / user / assistant 都在messages数组）

```json
{
  "messages": [
    {"role":"system", "content":"系统提示词"},
    {"role":"user", "content":"用户问题"},
    {"role":"assistant", "content":"回答"}
  ]
}
```

缺点：system 消息也是消息列表一员，会参与消息轮转；部分工具、压缩逻辑要额外区分 system 消息不能被压缩、不能被fork。

### 方案B：Claude原生 + OpenClaw采用（顶层独立system + messages只有user/assistant）

```json
{
  "system": "系统提示词",
  "messages": [
    {"role":"user", "content":"用户问题"},
    {"role":"assistant", "content":"回答"}
  ]
}
```

优点：

1. 系统人格、规则、workspace 素材和**对话历史完全解耦**；
2. compaction、消息截断、fork、回放逻辑只需要处理 user/assistant 交替链，不用额外判断、跳过 system 消息；
3. 工具调用天然适配 Claude 原生工具规范（toolResult进user）；

## 几个高频疑问澄清

### Q1：那 OpenClaw 能不能兼容 OpenAI 这种“system 在 messages 里面”的模式？

网关做模型适配层的时候可以做转换：

- 对接 Claude：原样，system放顶层；
- 对接 OpenAI / Ollama：网关自动把顶层 system 抽出来，包装一条 `role:system` 插到 messages 数组头部再发给OpenAI接口。

> 这个转换是 Gateway 的适配器层做的，**会话内部存储（jsonl）永远不存 role=system 的消息**。

### Q2：compaction摘要为什么不放到顶层system里？

这是一个很关键的设计取舍：

- 顶层system：**Agent的人格、长期不变的规则、身份**（SOUL/IDENTITY这类，会话生命周期基本固定）；
- compaction摘要：**动态变化的对话历史摘要**，属于这一轮聊天上下文，会随着对话持续变长、反复压缩；
  如果把compaction塞进system，会污染Agent底层人格；而且每次压缩都要修改顶层system快照，快照机制就乱了。
  所以 compaction 摘要属于对话历史，进 messages，作为 user 消息。

### Q3：jsonl 里会不会出现 role=system 的message？

**不会**。
所有 system 来源：workspace md + memory，只在会话初始化时合成顶层system，写入`sessions.json`的`systemPromptReport`快照；不会持久化成一条 `type:message, role:system` 写到jsonl流水。

## 整条链路串起来复盘（和你之前知识点打通）

1. TUI发起 `req:agent`，指定sessionKey；
2. Gateway查`sessions.json`拿到会话快照：固化好的system文本；
3. 读取这个sessionId对应的jsonl，顺着parentId遍历链表，识别compaction节点；
4. 根据compaction裁剪历史，收集剩下的 user/assistant 消息，组装 messages；
5. 调用模型适配器：
   - 目标是Claude：`{system:快照文本, messages:[...]}`
   - 目标是OpenAI：适配器转成 `messages: [system消息, ...对话消息]`
6. 模型返回assistant内容（可能带tool_use）；
7. 网关执行工具，产出toolResult，调用appendMessage写入一条role=user的message；
8. 循环。

如果你想要，我可以给你画一个极简伪代码，模拟 Gateway 里面组装这个请求的函数。

# 一句话先把核心结论钉死

1. 在 **jsonl 持久存储（磁盘流水）**：
   `compaction` 是**和 `message` 同一顶层层级的独立条目**，顶层字段 `type="compaction"`，**它根本不是 type="message"，也没有 message.role**。

```json
// jsonl 一行：compaction，和 message 平级
{
  "type": "compaction",
  "id": "cmp_001",
  "parentId": "msg_99",
  "timestamp": "...",
  "summary": "这里是历史对话摘要文本",
  "firstKeptEntryId": "msg_50",
  "tokensBefore": 11400
}

// jsonl 一行：普通消息
{
  "type": "message",
  "id": "msg_99",
  "parentId": "msg_98",
  "message": { "role": "assistant", "content": "..." }
}
```

✅ 存储层：`compaction`、`message`、`session`、`model_change` 都是同级的节点类型，共享同一套 `id + parentId` DAG链表。

2. 在 **发给LLM的API请求（内存组装messages数组这一步）**：
   读到 `type="compaction"` 节点的时候，**把它的 `summary` 拿出来，包装成一条 `role="user"` 的消息塞进 messages 列表**。

> 👉 这里只是**运行时组装阶段的转换**，**不是把磁盘上的 compaction 节点改成了 message**。磁盘上那条 compaction 条目原样保留，永远不会被改写。

---

## 完整 compaction 流程拆解，分两大阶段：【写入阶段】、【加载回放/组装上下文阶段】

### 阶段1：触发压缩，写入 jsonl（appendMessage）

条件：遍历消息链统计 token，历史太长，超过阈值 → 触发 compaction。

1. 找到压缩断点：比如从 `msg_1 ~ msg_49` 这一大段要折叠；保留起点是 `firstKeptEntryId = msg_50`；断点最后一条是 `msg_49`
2. 对 `msg_1 ~ msg_49` 调用摘要模型，生成 summary
3. 调用 `appendMessage`，写入一条**顶层 type=compaction 的节点**：
   - `parentId = msg_49.id`（挂在旧链末尾）
   - `firstKeptEntryId = msg_50`（标记：从这个id往后的原始消息保留，不再折叠）
4. 后续所有新消息，`parentId` 都指向这个 `cmp_001`，不再接 `msg_49`
   链表变成：

```
msg1 → msg2 → ... → msg49 → cmp_001 → msg50 → msg51 → msg52 ...
```

> 磁盘上 msg1~msg49 依旧存在，**不会删除**，fork、回溯历史时还能读到完整原始链。

### 阶段2：读取jsonl、遍历链表、组装给大模型的 messages（最容易混淆的地方！）

SessionManager 遍历 DAG 的规则：

> 顺着链表走，一旦碰到 `type="compaction"` 节点：
>
> 1. 停止收集 compaction.firstKeptEntryId **之前**的所有原始消息（msg1~msg49 丢弃，不进messages）
> 2. 把 compaction.summary 包装一条 `{"role":"user", "content": summary文本}` 放入 messages
> 3. 继续往后收集：从 `firstKeptEntryId`（msg50）开始，往后正常收集 type=message（user/assistant/toolResult）

👉 组装出来给 Claude API 的 messages 示例：

```json
{
  "system": "workspace整套md快照",
  "messages": [
    // 由 compaction 节点运行时转换出来的摘要消息
    {"role":"user", "content":"【摘要：msg1到msg49对话浓缩】"},
    // 从 firstKeptEntryId 开始往后正常原始消息
    {"role":"assistant", "content":"msg50内容"},
    {"role":"user", "content":"msg51内容"},
    {"role":"assistant", "content":"msg52内容"}
  ]
}
```

⚠️ 重点区分两层，千万不要混在一起：

- 磁盘（持久层）：`compaction` ≠ message，独立type，平级节点
- 上送LLM（内存视图）：compaction 摘要被**临时转成一条 user 消息**塞进对话流，只是视图转换，不修改磁盘记录

## 为什么设计成这样，而不是直接存成一条 role=user 的message？

如果当初直接把摘要存成一条普通 `type:message, role:user`：

- 程序没法区分：**这一条是自动压缩摘要，还是用户真实输入的user消息**；
- 压缩逻辑、递归二次压缩、fork分支、统计token都会混乱；
- 没有地方放 `firstKeptEntryId`、`tokensBefore` 这些压缩元数据（这些是压缩逻辑专用标记，不是普通消息该带的字段）。

单独的顶层 `type:compaction` 节点，就是为了**带上压缩元信息，让程序能识别它是一个压缩锚点**；只是在喂给模型的时候，为了适配 Claude 只能 user/assistant 交替的消息流，才把摘要文本包装成 user 消息。

## 快速自测一句话辨析

问：jsonl里面compaction节点本身是message吗？
答：不是，type是compaction，和message同级。
问：那模型收到的messages数组里，摘要是什么role？
答：运行时转成 role=user 的消息。

## 顺带一个易错边界

一条 compaction **只会产生一条摘要user消息**；
compaction节点本身**永远不会进入messages数组，进入数组的只有它提取出来的summary文本包装后的user消息**。

如果你愿意，我给你写一段极简伪代码，模拟 SessionManager 遍历链表、处理compaction、产出messages的逻辑。
