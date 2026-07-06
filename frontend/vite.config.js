import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  // Backend URL for dev proxy (if port 8000 is blocked on Windows, use 8001)
  const apiTarget =
    env.VITE_DEV_API_PROXY || "http://127.0.0.1:8000";

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/analyze": { target: apiTarget, changeOrigin: true, timeout: 3_600_000 },
        "/health": { target: apiTarget, changeOrigin: true, timeout: 30_000 },
        "/inference": { target: apiTarget, changeOrigin: true, timeout: 3_600_000 },
        "/history": { target: apiTarget, changeOrigin: true },
        "/auth": { target: apiTarget, changeOrigin: true },
      },
    },
  };
});
