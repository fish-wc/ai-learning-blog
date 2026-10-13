---
title: "KV Cache是什么"
date: 2026-10-12
categories:
  - llm
tags:
  - KV Cache
math: true
---

# ppt制作脚本

## 第 页，解决的问题

KV Cache 要解决的问题是什么
KV Cache 这个技术是什么

## 第 页，总结

1. 对于最开始的整个transformer运算的时候，对于每个block而言。拿到一个token之后，需要生成新的token，需要把刚生成的token拼接起来，然后重新过一遍transformer的计算。需要重新计算所有token的Q，K，V，重新计算$QK^T$、$softmax(\frac{QK^T}{\sqrt{d_k}})V$。
2. 理论上，经过一份transformer网络的时候，旧token看不到新token，不会更新，所以旧token的Q,K,V，以及运算的中间变量都是不会变的。
3. 考虑到缓存中间变量的方式，对于新的token 有新的$Q_{new}$，再经过transformer网络的时候，不用用到之前的$Q_{old}$，之前运算的中间变量$Q_{old}K_{old}^T$、$softmax(\frac{Q_{old}K_{old}^T}{\sqrt{d_k}})V_{old}$都用不上，所以这些变量都不需要额外缓存。
4. 对于新的token 的$Q_{new}$只需要重新计算$QK^T$、$softmax(\frac{QK^T}{\sqrt{d_k}})V$即可，所以只需要缓存$K_{old}$和$V_{old}$，然后再拼接$K_{new}$和$V_{new}$即可进行增量计算。
5. 所以，只缓存每一层transformer block的K和V，每次只将新token 的$Q_new$往后传递增量计算即可。

## 参考文献

[到底什么是KV Cache，为啥不是QKV Cache，一个动画彻底搞懂！](https://www.bilibili.com/video/BV1fcaN6fENi?spm_id_from=333.788.videopod.sections&vd_source=f520474b32080bfed1b47d9eea6866c7)