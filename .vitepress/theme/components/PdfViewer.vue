<script setup>
import { withBase } from 'vitepress'

const props = defineProps({
  file: { type: String, required: true },
  title: { type: String, default: 'PDF' }
})

const src = withBase(props.file)
</script>

<template>
  <iframe
    :src="src"
    :title="title"
    style="width: 100%; height: 85vh; border: 0"
    loading="lazy"
  />
</template>
