# AI Learning Blog

这是一个面向 **人工智能、大模型、智能体与 AI Infra 求职准备** 的个人学习博客。

我会把学习过程中的知识整理、论文阅读、实验记录和项目复盘持续写成 Markdown，并通过 VitePress 自动发布到 GitHub Pages。博客的重点不是堆砌结论，而是记录「理解问题 → 动手验证 → 总结表达 → 持续迭代」的过程。

## 学习主线

```text
基础能力
   ↓
大模型原理与应用 ──┐
   ↓               │
RAG 检索增强       ├──→ 项目实践与作品集 ──→ 面试准备
   ↓               │
Agent / LangGraph ─┘
   ↓
AI Infra：训练、推理、服务化与可观测性
```

- **基础能力**：Python、Linux、Git、数据结构、机器学习和深度学习。
- **大模型**：Transformer、预训练、微调、推理和评测。
- **RAG**：数据处理、Embedding、检索、重排、生成和评测。
- **Agent**：工作流、工具调用、状态管理、记忆、可靠性和 LangGraph。
- **AI Infra**：GPU、并行计算、模型服务、推理优化、分布式训练和 MLOps。
- **项目实践**：把知识落成可运行、可测量、可解释的端到端项目。
- **论文阅读**：用论文理解前沿方法，并反哺工程实践。
- **面试准备**：基础知识、系统设计、项目表达和复盘。

详细安排见 [学习路线](docs/roadmap.md)。

## 目录结构

```text
.
├── .github/workflows/deploy.yml  # GitHub Pages 自动部署
├── .vitepress/
│   ├── config.mjs                # VitePress 配置
│   └── sidebar.mjs               # 根据 docs/ 自动生成侧边栏
├── docs/                         # 唯一的 Markdown 内容源
│   ├── foundation/               # 基础能力
│   ├── llm/                      # 大模型
│   ├── rag/                      # RAG
│   ├── agent/                    # Agent
│   ├── ai-infra/                 # AI Infra
│   ├── projects/                 # 项目与实验
│   ├── paper-reading/            # 论文阅读
│   ├── interview/                # 面试准备
│   ├── index.md                  # 博客首页
│   └── roadmap.md                # 学习路线
├── package.json
└── package-lock.json
```

## 本地运行

环境要求：Node.js 18+（推荐 Node.js 20）。

```bash
npm install
npm run docs:dev
```

开发服务器启动后，打开终端输出的本地地址即可预览。生产构建和预览：

```bash
npm run docs:build
npm run docs:preview
```

## 如何写新笔记

只需要在 `docs/` 下创建 Markdown 文件，不需要手动维护侧边栏：

```text
docs/llm/transformer.md
```

```markdown
# Transformer 学习笔记

## 为什么需要注意力机制

记录自己的理解、代码实验和结论。
```

侧边栏生成规则：

1. 目录会自动变成可折叠分组，文件名会自动转成页面标题。
2. 页面标题优先使用 frontmatter 的 `title`，其次使用第一个一级标题，最后使用文件名。
3. `index.md` 会成为对应目录的入口，不会在分组中重复显示。
4. 需要调整顺序时，在 frontmatter 中添加 `order`；不想出现在侧边栏时添加 `sidebar: false`。
5. 文件名建议使用小写 kebab-case，例如 `attention-mechanism.md`。

可选的 frontmatter：

```yaml
---
title: 自定义页面标题
order: 2
---
```

VitePress 会自动渲染 Markdown、代码高亮、表格、数学公式和自定义容器。数学公式由 `markdown-it-mathjax3` 统一处理，推荐使用标准 LaTeX 写法：

```markdown
行内公式：$J(\\theta)=\\mathbb{E}[R]$

块级公式：

$$
J(\\theta)=\\mathbb{E}_{\\tau\\sim p_\\theta}[R(\\tau)]
$$
```

不要使用 `[` 和 `]` 包裹公式（例如 `[ ... ]`），这种写法会被当作普通 Markdown 文本。frontmatter 中的 `math: true` 可以作为笔记标记，但不会单独启用渲染插件；插件已经在全站配置中启用。

## GitHub Pages 部署

仓库已提供 GitHub Actions 工作流：推送到 `main` 后会自动构建并部署到 GitHub Pages。

首次使用时，在 GitHub 仓库的 **Settings → Pages → Build and deployment** 中将来源设置为 **GitHub Actions**。当前仓库名为 `ai-learning-blog`，工作流使用 `/ai-learning-blog/` 作为站点路径；如果仓库改名，需同步修改 `.github/workflows/deploy.yml` 中的 `BASE_PATH`。使用自定义域名时通常改为 `/`。

## 写作原则

- 用自己的话解释，而不是只复制官方文档。
- 给出可复现的代码、环境、数据和实验结论。
- 记录失败尝试及原因，它们同样是工程经验。
- 每篇笔记尽量回答：问题是什么、怎么验证、结论是什么、下一步做什么。
- 以能在简历和面试中清楚讲明白为标准持续迭代。
