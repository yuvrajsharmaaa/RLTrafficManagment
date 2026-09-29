import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { viteSingleFile } from 'vite-plugin-singlefile';

// server.py serves only the root index.html plus a few named files, so the
// build must be one self-contained HTML file (no separate asset chunks).
const backend = process.env.BACKEND_URL ?? 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react(), viteSingleFile()],
  server: {
    port: 5173,
    proxy: {
      '/api': backend,
      '/health': backend,
      '/frontend_data': backend,
      '/manifest.json': backend,
      '/icon.svg': backend,
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
