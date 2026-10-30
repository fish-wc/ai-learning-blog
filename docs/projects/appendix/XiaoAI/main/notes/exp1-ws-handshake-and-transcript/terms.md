
# 关键词解释

## 1. TUI是什么

**TUI = Text‑based User Interface，文本用户界面**，是这个项目的**客户端交互程序**，跑在终端里，不是网页GUI。

- 作用：人在终端输入问题、看模型回复；它作为客户端，通过 WebSocket 去连接 Gateway，发送请求、接收流式事件消息。
- 文中场景：TUI 作为客户端发起 WS 连接，会带上 token、设备密钥签名，申请权限，踩过 `pairing required` 配对报错。

> 区分 CLI：CLI一般做简单查询；TUI是完整交互式聊天终端界面。

## 2. WebSocket（WS）是什么

WebSocket 是一种网络通信协议。
普通 HTTP：一问一答，收到响应就断开连接，短连接。
WebSocket：**建立一次TCP连接之后长久保持连接，双向实时通信**，客户端和服务端任何一方都可以随时发消息。

- 本项目里：全部消息走 WebSocket，载荷是**JSON文本帧**。
- 用途：实现流式输出、事件广播（比如设备配对通知、心跳tick事件）。

> 地址示例：`ws://127.0.0.1:19001`，ws:// 就是 WebSocket 的协议标识。

## 3. Gateway是什么，起到什么作用

Gateway：**网关服务，整个系统的服务端核心进程**。

### 角色

所有客户端（TUI / CLI）都不直接调用大模型，全部连接 Gateway。

### 主要作用

1. **连接接入与安全握手**：处理 WebSocket 连接，做三重校验：token校验、设备身份签名校验、设备配对+scope权限校验；拒绝非法设备，出现`pairing required`就是网关做的权限拦截。
2. **消息协议处理**：解析客户端 `req` 请求，返回 `res` 响应，主动推送 `event` 事件。
3. **路由转发**：把客户端的聊天请求，路由给对应的 Agent会话（通过 sessionKey → sessionId）。
4. **持久化会话**：调用 SessionManager，把对话追加写入 jsonl 会话文件。
5. 向上对接大模型，向下给客户端返回流式内容。

> 简单类比：Gateway 相当于保安前台 + 总调度。客户端全部先过它，校验身份权限，再分发任务。

## 4. scope

- 英文直译：**范围、作用域**
- 在本系统：**权限作用域，代表设备允许执行哪些操作的权限标签**。
  例子文档出现：
- `operator.read`：只读权限，只能查询状态，不能发起聊天、调用工具
- `operator.admin`：管理员完整权限，可以聊天、调用命令工具等。

> 业务逻辑：
> 设备配对记录里保存 `approvedScopes`（已批准scope集合）；客户端握手时会带上本次想要申请的scope。
> 如果本次请求的scope > 已批准scope，就触发 `scope‑upgrade`，要求重新配对审批，报`pairing required`错误。
> scope不是一次性授权，**权限升级必须人工审批，本地回环也不会自动放行权限升级**。

## 5. `req:connect`、`event: connect.challenge` 记号含义

这是项目自定义的**Wire protocol（线上通信协议）标记写法**，用来标记消息类别+消息方法/事件名。

### `req:xxx`

`req` = request，**客户端发往网关的请求消息**
格式：协议JSON中 `{"type":"req","method":"connect",...}`
`req:connect`：就是 `method="connect"` 的请求，也就是**WebSocket握手第一条必须发送的连接请求**，完成身份、设备、权限上报。

### `event: xxx.yyy`

`event` = 事件，**网关服务端主动推送给客户端的通知消息，客户端没有发起请求，服务端主动推送**
JSON字段：`{"type":"event","event":"connect.challenge",...}`

`event: connect.challenge`

- event类型，事件名叫`connect.challenge`
- 含义：网关在TCP建立之后，立刻主动推送挑战随机数nonce给客户端，作为“考题”，客户端要拿自己私钥对这个nonce签名应答，用来防重放攻击。

> 补充完整三类消息记号文档里出现：

1. `req:methodName`：客户端发起请求
2. `res:methodName`：网关返回对应请求的响应（一一对应，靠id匹配req和res）
3. `event:event.name`：服务端主动推送事件，不需要客户端请求。

举例子

- `event:presence`、`event:tick`：网关主动广播状态、心跳事件
- `req:agent`：客户端发起agent聊天请求；`res:agent`是网关给这个请求的应答。

---

### 小总结记忆

| 记号                        | 发送方        | 含义                                               |
| --------------------------- | ------------- | -------------------------------------------------- |
| `req:connect`             | TUI(客户端)   | 客户端发送连接握手请求，上报token、设备签名、scope |
| `event:connect.challenge` | Gateway(网关) | 网关主动下发防重放随机数nonce挑战                  |

> 协议强制约束：WebSocket连接建立后**第一条消息必须是 req:connect**，直接发别的请求会被网关拒绝。
