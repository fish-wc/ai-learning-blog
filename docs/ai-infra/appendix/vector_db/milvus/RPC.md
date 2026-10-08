---
title: "Milvus 架构图"
date: 2026-09-24
tags: ["向量数据库", "Milvus","RPC（Remote Procedure Call）"]

---


# RPC（Remote Procedure Call）
**概述：Remote Procedure Call，远程过程调用**。是一种通信机制，**允许程序像调用本地函数一样调用远端机器上的服务函数，屏蔽底层网络通信细节**。Milvus集群内部节点之间通信采用gRPC（基于Protobuf二进制序列化，性能高）。Raft协议中节点间投票、日志复制消息也基于RPC实现。



## RPC 核心组成
1. **序列化/反序列化**：把内存里的对象（向量、参数）转成二进制字节流，方便网络传输；接收方收到字节流，还原成对象。Milvus内部用Protobuf。
2. **网络传输**：底层一般基于TCP。
3. **服务注册与发现（可选）**：客户端知道服务在哪台机器、哪个端口。
4. **Stub（桩）**：客户端侧的代理，伪装成本地函数，屏蔽网络细节。

## 和HTTP的简单区分
- **RPC**：专注于**函数调用**，Protobuf序列化，二进制协议，体积小、速度快，适合内部服务之间通信。Milvus集群内部节点之间全部用RPC通信。
- **HTTP/REST**：基于文本JSON，通用性强，适合对外暴露的API；开销更大。
> Milvus对外提供HTTP SDK接口；**集群内部Proxy、QueryNode、DataNode之间通信全部是gRPC（Google实现的RPC框架）**。

## 在Milvus / Woodpecker / Raft里的RPC场景
1. Milvus集群节点之间：Proxy → QueryNode，Proxy → DataNode，全部走gRPC。
2. Raft协议内部：节点之间发送投票请求、日志复制请求，本质都是RPC调用。
3. Woodpecker节点之间同步内存副本（QuorumBuffer），节点之间互相发消息也是RPC。

举个Milvus查询例子：
Proxy节点收到用户search请求，调用QueryNode的Search接口：
```
Proxy(客户端Stub) --gRPC--> QueryNode(服务端)
```
在代码层面，Proxy就像调用本地函数，实际请求通过网络发给另一台QueryNode机器执行检索，结果再通过网络返回Proxy。

### 通俗例子
代码在机器A（Proxy节点），想要调用机器B（QueryNode）的检索函数。如果不用RPC：要手动写socket、组装数据包、序列化、网络传输、接收解析、异常处理，非常麻烦。使用RPC：直接写一行代码 `query_node.search(vectors)`，看起来就像调用本机函数，底层网络收发全部由RPC框架完成。



