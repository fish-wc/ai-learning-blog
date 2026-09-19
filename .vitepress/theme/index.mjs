import DefaultTheme from 'vitepress/theme'
import './custom.css'
import PdfViewer from './components/PdfViewer.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('PdfViewer', PdfViewer)
  }
}
