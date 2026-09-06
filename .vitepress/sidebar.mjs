import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const CONTENT_ROOT = fileURLToPath(new URL('../docs', import.meta.url))
const MARKDOWN_EXTENSION = '.md'
const HIDDEN_PREFIX = '_'

/**
 * Build the VitePress sidebar from the docs directory.
 *
 * A note only needs a Markdown file and an H1. Optional frontmatter:
 *
 * ---
 * title: A custom sidebar title
 * order: 1
 * sidebar: false
 * ---
 */
export function getSidebar() {
  return scanDirectory(CONTENT_ROOT, '')
    .filter((item) => item.link !== '/')
    .map(stripInternalFields)
}

function scanDirectory(directory, relativeDirectory) {
  if (!fs.existsSync(directory)) return []

  return fs
    .readdirSync(directory, { withFileTypes: true })
    .filter((entry) => !entry.name.startsWith(HIDDEN_PREFIX))
    .map((entry) => {
      const absolutePath = path.join(directory, entry.name)
      const relativePath = path.posix.join(relativeDirectory, entry.name)

      if (entry.isDirectory()) {
        const items = scanDirectory(absolutePath, relativePath)
        if (items.length === 0) return null

        const index = items.find((item) => item.isIndex)
        const directoryItems = items.filter((item) => !item.isIndex)
        const group = {
          text: index?.text ?? formatName(entry.name),
          link: index?.link,
          items: directoryItems,
          order: index?.order,
          sortKey: relativePath,
          collapsed: false
        }

        // A directory containing only index.md is a normal page, not an empty
        // collapsible group.
        return directoryItems.length === 0 && index
          ? { text: index.text, link: index.link, order: index.order, sortKey: relativePath }
          : group
      }

      if (!entry.isFile() || path.extname(entry.name).toLowerCase() !== MARKDOWN_EXTENSION) {
        return null
      }

      const metadata = readMetadata(absolutePath)
      if (metadata.sidebar === false) return null

      const name = path.basename(entry.name, MARKDOWN_EXTENSION)
      const isIndex = name.toLowerCase() === 'index'
      const link = makeLink(relativePath, relativeDirectory, isIndex)

      return {
        text: metadata.title ?? (isIndex ? formatName(relativeDirectory.split('/').pop() || '首页') : formatName(name)),
        link,
        order: metadata.order,
        sortKey: relativePath,
        isIndex
      }
    })
    .filter(Boolean)
    .sort(compareItems)
}

function readMetadata(filePath) {
  const source = fs.readFileSync(filePath, 'utf8')
  const frontmatterMatch = source.match(/^---[ \t]*\r?\n([\s\S]*?)\r?\n---[ \t]*(?:\r?\n|$)/)
  const frontmatter = frontmatterMatch?.[1] ?? ''
  const title = parseFrontmatterValue(frontmatter, 'title')
  const orderValue = parseFrontmatterValue(frontmatter, 'order')
  const heading = source.match(/^#\s+(.+?)\s*$/m)?.[1]?.replace(/\s+#+\s*$/, '').trim()
  const order = orderValue === undefined ? undefined : Number(orderValue)

  return {
    title: title || heading,
    order: Number.isFinite(order) ? order : undefined,
    sidebar: !/^sidebar:\s*(?:false|['"]false['"])\s*$/im.test(frontmatter)
  }
}

function parseFrontmatterValue(frontmatter, key) {
  const match = frontmatter.match(new RegExp(`^${key}:\\s*(.+?)\\s*$`, 'm'))
  if (!match) return undefined

  const value = match[1]
  if (value.length >= 2 && value.startsWith('"') && value.endsWith('"')) {
    try {
      return JSON.parse(value)
    } catch {
      return value.slice(1, -1)
    }
  }
  if (value.length >= 2 && value.startsWith("'") && value.endsWith("'")) {
    return value.slice(1, -1).replaceAll("''", "'")
  }
  return value
}

function makeLink(relativePath, relativeDirectory, isIndex) {
  if (isIndex) return relativeDirectory ? `/${relativeDirectory}/` : '/'
  return `/${relativePath.slice(0, -MARKDOWN_EXTENSION.length)}`
}

function compareItems(left, right) {
  const leftOrder = left.order ?? Number.POSITIVE_INFINITY
  const rightOrder = right.order ?? Number.POSITIVE_INFINITY
  if (leftOrder !== rightOrder) return leftOrder - rightOrder
  return (left.sortKey ?? left.text).localeCompare(right.sortKey ?? right.text, 'zh-CN', { numeric: true })
}

function stripInternalFields(item) {
  const { order, isIndex, sortKey, ...publicItem } = item
  if (publicItem.items?.length) {
    publicItem.items = publicItem.items.map(stripInternalFields)
  } else {
    delete publicItem.items
  }
  delete publicItem.order
  return publicItem
}

function formatName(name) {
  return name
    .replace(/[-_]+/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase())
}

export default getSidebar
