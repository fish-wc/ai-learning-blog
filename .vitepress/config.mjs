import { defineConfig } from 'vitepress'
import { getSidebar } from './sidebar.mjs'

const configuredBase = process.env.BASE_PATH || '/'
const base = /^[A-Za-z]:[\\/]/.test(configuredBase) ? '/' : configuredBase
const normalizedBase =
  base === '/' ? '/' : `/${base.replace(/^\/+|\/+$/g, '')}/`

export default defineConfig({
  lang: 'zh-CN',
  title: 'AI Learning Blog',
  description: '面向人工智能、大模型与智能体求职的学习笔记',
  base: normalizedBase,
  srcDir: 'docs',
  lastUpdated: true,

  themeConfig: {
    siteTitle: 'AI Learning Blog',
    nav: [
      { text: '首页', link: '/' },
      { text: '学习路线', link: '/roadmap' },
      { text: '大模型', link: '/llm/' },
      { text: '智能体', link: '/agent/' },
      { text: '强化学习', link: '/rf/' },
      { text: 'GitHub', link: 'https://github.com/fish-wc/ai-learning-blog' }
    ],

    sidebar: getSidebar(),
    outline: {
      level: [2, 3],
      label: '本页目录'
    },
    search: {
      provider: 'local'
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/fish-wc/ai-learning-blog' }
    ],
    editLink: {
      pattern: 'https://github.com/fish-wc/ai-learning-blog/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },
    docFooter: {
      prev: '上一页',
      next: '下一页'
    },
    lastUpdated: {
      text: '最后更新于'
    },
    footer: {
      message: '持续学习，持续输出。',
      copyright: 'Copyright © 2026 AI Learning Blog'
    }
  },

  markdown: {
    lineNumbers: true
  }
})
