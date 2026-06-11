import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The backend port (RL Arena runs on 8001 because 8000 is taken by Hakuryu).
const API_PORT = '8001';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: true,
    // Proxy /api → backend so the browser talks SAME-ORIGIN (no CORS, no
    // IPv4/IPv6 localhost mismatch). Vite forwards server-side to 127.0.0.1.
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${API_PORT}`,
        changeOrigin: true,
      },
      // WebSocket proxy for the realtime frame stream (A2).
      '/ws': {
        target: `ws://127.0.0.1:${API_PORT}`,
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
