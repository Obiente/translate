/// <reference types="vite/client" />

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}

interface ImportMetaEnv {
  readonly VITE_GOOGLE_TRANSLATE_API_KEY?: string
  readonly VITE_GOOGLE_CLOUD_PROJECT_ID?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}