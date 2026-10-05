import { defineConfig } from 'vite';
import { resolve } from 'node:path';

const root = import.meta.dirname;
const workspace = resolve(root, '../..');

export default defineConfig({
  root,
  // Relative asset URLs, so the build also works under a subpath (GitHub Pages).
  base: './',
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
