

# 先一句话总览这段 JS 代码

> 用 `ws` 库启动 WebSocket 服务；每当有客户端连上并发消息过来，就把收到的消息转发给 Anthropic Claude 大模型，拿到回答后通过 WebSocket 发回给客户端。
> 你熟悉 C++ / Python，我会大量对照帮你理解。

## 逐行拆解代码

```js
// toy-bot.js —— 我们的第一版 
const ws = require("ws"); 
const { Anthropic } = require("@anthropic-ai/sdk"); 

wss.on("connection", (socket) => {  
  socket.on("message", async (data) => {  
    const reply = await anthropic.messages.create({  
      model: "claude-opus-4-6",  
      messages: [{ role: "user", content: data.toString() }],  
    });  
    socket.send(reply.content[0].text);  
  }); 
});
```

1. `const ws = require("ws");`
   `require()` 是 Node.js 的**模块导入语法**，等价于 Python 的 `import ws` / C++ 的 `#include` + 链接库。

- `const`：定义**常量**，变量不能重新赋值（类似 C++ `const`，但注意：JS const 只是变量引用不可改，对象内部属性依旧可以改！这点和 C++ const 不一样）
- 这里加载第三方 `ws` WebSocket 库。

2. `const { Anthropic } = require("@anthropic-ai/sdk");`
   👉 **对象解构（JS 高频语法！）**
   等价于：

```js
const sdk = require("@anthropic-ai/sdk");
const Anthropic = sdk.Anthropic;
```

Python 近似：`from xxx import Anthropic`。C++ 没有原生解构。

3. `wss.on("connection", (socket) => { ... })`

- `wss` 是 WebSocket Server 实例（代码里省略了 `new ws.Server()` 那一行）
- `.on("事件名", 回调函数)`：Node.js 事件驱动模型，**注册事件监听器**

  - 当客户端连接上来 → 触发 `connection` 事件 → 执行回调，`socket` 代表这个客户端连接
- `(socket) => { ... }`：**箭头函数**，JS 的匿名函数简写

  ```js
  // 等价写法（传统 function）
  wss.on("connection", function(socket) { ... })
  ```

  类似 Python lambda，但功能更强；C++11 lambda `[](socket){}` 可以类比。

4. `socket.on("message", async (data) => { ... })`
   在单个连接 socket 上继续监听 `message` 事件：**客户端发消息过来的时候触发**

- `async (data) => {}`：标记这个回调函数是**异步函数**
  ✅ 只有 `async` 函数内部，才能写 `await`，这是 JS 异步核心！
  Python 对照：`async def func(): ...` + `await`，几乎一模一样；C++ 没有内置 await。

5. `const reply = await anthropic.messages.create({ ... })`

- `await`：等待这个异步 API 请求（网络调用）完成，拿到返回结果，不会阻塞事件循环（Node 异步非阻塞）
- `anthropic.messages.create({ ... })`：调用 SDK 接口，传参是一个 **JS 对象字面量 `{key: val}`**

  ```js
  {
    model: "claude-opus-4-6",
    messages: [{ role: "user", content: data.toString() }],
  }
  ```

  JS 对象 ≈ Python dict；和 C++ struct 不一样，是动态字典。- `messages` 的值是**数组 `[]`**，数组里面又包了一个对象

  - `data` 是 WebSocket 收到的二进制/Buffer，`.toString()` 转字符串

6. `socket.send(reply.content[0].text);`

- `reply.content` 是数组，`[0]` 取下标 0 的元素
- `.text` 访问这个对象的 text 属性
- `socket.send()`：把字符串通过 WebSocket 发回客户端

> ⚠️ 这段代码有明显坑（后面给你提一下）：没有异常捕获，客户端断开、API报错会直接崩服务；没有初始化 `wss` 和 `anthropic` 实例。
