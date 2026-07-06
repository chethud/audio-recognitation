# Deploy ALM-LITE — Render (backend) + Vercel (frontend)

## Prerequisites

- GitHub repo: https://github.com/chethud/audio-recognitation
- **Render Standard (2 GB RAM)** or higher — free tier will OOM on Whisper + CNN
- **Python 3.11** (Dockerfile uses 3.11-slim)
- **CNN checkpoints** (`outputs/*.pt`) — train on Render shell or copy from local machine

---

## Backend — Render

1. Go to [render.com](https://render.com) → **Sign in with GitHub**
2. **New +** → **Blueprint** (or **Web Service**)
3. Connect repo `chethud/audio-recognitation`
4. If using Blueprint: approve `render.yaml` (Docker, plan: standard)
5. If manual Web Service:
   - **Runtime:** Docker
   - **Dockerfile path:** `./Dockerfile`
   - **Plan:** Standard (2 GB+)
   - **Health check path:** `/health`
6. **Environment variables** (optional):
   - `HF_TOKEN` — Hugging Face token (faster model downloads)
7. **Deploy** — first build takes 15–25 min (PyTorch)
8. After deploy, open Shell and train CNN (once):
   ```bash
   python -m training.run_notebook_pipeline --quick --batch-size 2
   ```
9. Copy your API URL, e.g. `https://alm-lite-api.onrender.com`
10. Test: `https://YOUR-URL.onrender.com/health` → `"model_ready": true`

---

## Frontend — Vercel

1. Go to [vercel.com](https://vercel.com) → **Sign in with GitHub**
2. **Add New → Project** → import `chethud/audio-recognitation`
3. Settings:
   - **Root Directory:** `frontend`
   - **Framework Preset:** Vite
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. **Environment Variables** → Production:
   - `VITE_API_BASE` = `https://YOUR-RENDER-URL.onrender.com` (no trailing slash)
5. **Deploy**
6. Open your Vercel URL (e.g. `https://audio-recognitation.vercel.app`)

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Render crash / OOM | Upgrade to Standard 2 GB+; use `whisper-tiny` in `config.yaml` |
| `model_ready: false` | Wait 3–5 min; check logs; ensure RAM enough |
| No sounds | Run CNN training on Render or upload `outputs/*.pt` |
| Vercel analyze fails | Set `VITE_API_BASE` correctly; redeploy after changing env |
