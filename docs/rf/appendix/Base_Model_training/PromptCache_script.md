---
title: "Prompt Cache是什么"
date: 2026-10-12
categories:
  - llm
tags:
  - Prompt Cache
---

# PPT 脚本

## 第 页，要解决的问题是什么

Prompt Cache是用来解决什么问题的？
Prompt Cache是什么？
Prompt Cache和KV Cache的关系是什么？

## 第 页，总结

1. 对于agent场景，通常情况下tool的描述信息，system prompt这类信息通常放在前缀，通常比较稳定。对于消息记录而言，逐步拼接起来的，新消息会利用到同一份旧消息的信息。
2. 针对跨请求之间，如果多次请求都有一样的前缀，那么这部分前缀的KV和保存起来，降低重复计算的次数。所以需要设计一种机制来存跨请求间的KV。
3. 该机制即Prompt Cache，即将多次请求之间的相同前缀存起来，如果下一次请求前缀匹配上了，那么这部分前缀可以不用重新计算。
4. 对于KV Cache而言，是一次请求之间，降低重复计算的数量。对于Prompt Cache而言，是跨请求之间降低重复计算的数量。
5. 为什么可以缓存前缀：因为掩码注意力机制，一个token只能看到前面token的内容，新token不会影响已有token的KV。

## 参考文献

[Prompt Cache到底是啥，它和KV Cache是什么关系？](https://www.bilibili.com/video/BV1CjH962EVb?spm_id_from=333.788.videopod.sections&vd_source=f520474b32080bfed1b47d9eea6866c7)