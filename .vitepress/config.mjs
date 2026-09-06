import { defineConfig } from 'vitepress'
import { getSidebar } from 'vitepress-plugin-auto-sidebar'

export default defineConfig({
  title: 'AI-Learning-Blog',
  description: 'AI Infra & Agent 求职学习笔记',

  // Markdown 文件放在 docs 目录
  srcDir: 'docs',

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: 'Agent开发', link: '/agent/langgraph-basic' },
      { text: 'AI Infra', link: '/ai-infra/' }
    ],

    sidebar: getSidebar({
      // 注意：Windows 下这里建议写 docs，不要写 /docs
      contentRoot: 'docs',

      // 只扫描这两个目录，因此根目录 docs/index.md 不会进入 sidebar
      contentDirs: [
        { path: 'agent', title: 'Agent开发' },
        { path: 'ai-infra', title: 'AI Infra' }
      ],

      // 目录是否可以折叠
      collapsible: true,

      // 默认是否折叠
      collapsed: false,

      // true = 优先使用 markdown frontmatter 中的 title
      // false = 使用文件名
      useFrontmatter: true
    })
  }
})
