import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The backend runs on :8000 (GPU 1). In dev the browser talks to Vite on :5173
// and /api is proxied here; in single-process mode FastAPI serves the built
// dist/ and the API is same-origin, so no proxy is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    // Vite rejects any request whose Host header it doesn't recognise, which
    // blocks `tailscale serve` (it forwards the tailnet hostname through).
    // Both servers still bind to 127.0.0.1 only — the tailnet reaches them
    // through tailscaled, so nothing is exposed on the LAN.
    allowedHosts: ['.ts.net'],
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
