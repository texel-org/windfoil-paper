import { defineConfig } from 'vite';
import { resolve } from 'node:path';

const root = import.meta.dirname;
const workspace = resolve(root, '../..');

export default defineConfig({
  root,
  server: {
    port: 5173,
    fs: { allow: [workspace] },
  },
  build: {
    target: 'esnext',
    outDir: resolve(root, 'dist'),
    emptyOutDir: true,
  },
});
