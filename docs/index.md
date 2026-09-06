---
layout: home

hero:
  name: AI Learning Blog
  text: AI、大模型与智能体学习笔记
  tagline: 以求职为目标，以项目和可复现的实验为线索，持续记录理解与实践。
  actions:
    - theme: brand
      text: 查看学习路线
      link: /roadmap
    - theme: alt
      text: 浏览全部笔记
      link: /foundation/

features:
  - title: 系统学习
    details: 从基础能力出发，逐步覆盖大模型、RAG、Agent 与 AI Infra，形成可复用的知识框架。
  - title: 实验驱动
    details: 不只记录结论，也记录代码、指标、失败尝试和复盘，让每个观点都尽量可以被验证。
  - title: 面向求职
    details: 把学习沉淀为可展示的项目、清晰的技术文章和能够在面试中讲明白的工程经验。
---

<!-- markdownlint-disable MD033 MD041 -->
<div class="home-links">
  <a href="./llm/">大模型</a>
  <a href="./rag/">RAG</a>
  <a href="./agent/">智能体</a>
  <a href="./ai-infra/">AI Infra</a>
  <a href="./projects/">项目实践</a>
  <a href="./interview/">面试准备</a>
</div>

<style>
.home-links {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  justify-content: center;
  margin: 2rem auto;
  max-width: 760px;
}

.home-links a {
  border: 1px solid var(--vp-c-divider);
  border-radius: 999px;
  color: var(--vp-c-brand-1);
  padding: 0.5rem 1rem;
  text-decoration: none;
  transition: border-color 0.2s, background-color 0.2s;
}

.home-links a:hover {
  background: var(--vp-c-bg-soft);
  border-color: var(--vp-c-brand-1);
}
</style>
