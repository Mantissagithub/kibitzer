/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_KIBITZER_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare module "*.mdx" {
  import type { ComponentType } from "react";
  const Component: ComponentType;
  export default Component;
}
