# Deploy ALM-LITE — Render (backend) + Vercel (frontend)

## Prerequisites

- GitHub repo: https://github.com/chethud/audio-recognitation
- **Render Standard (2 GB RAM)** or higher — free tier will OOM on Whisper + CNN
- **Python 3.11** (Dockerfile uses 3.11-slim)
- **CNN checkpoints** (`outputs/*.pt`) — train on Render shell or copy from local machine
- **HF_TOKEN** — Hugging Face token with access to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) (required for WhisperX+PyAnnote speaker diarization)

**Local diarization setup (Windows):**

```powershell
$env:HF_TOKEN = "hf_your_token"
.\setup-diarization.ps1
```

Use **Python 3.11** (WhisperX does not support 3.14). The Docker image already uses 3.11.

---

## Backend — Render

### Option A — Docker (recommended)

1. Go to [render.com](https://render.com) → **Sign in with GitHub**
2. **New +** → **Blueprint** → select `chethud/audio-recognitation`
3. Approve `render.yaml` (Docker, plan: standard)
4. **Deploy**

### Option B — Manual Web Service

| Setting | Value |
|---------|--------|
| **Root Directory** | *(leave empty)* — **NOT** `frontend` |
| **Runtime** | **Docker** |
| **Dockerfile path** | `./Dockerfile` |
| **Plan** | Standard (2 GB+) |
| **Health check** | `/health` |

**If using Python runtime instead of Docker:**

| Setting | Value |
|---------|--------|
| **Root Directory** | *(empty)* |
| **Python version** | `3.11.9` (env: `PYTHON_VERSION=3.11.9`) |
| **Build** | `pip install --upgrade pip && pip install -r requirements.txt` |
| **Start** | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |

> **Do not use Python 3.14** — AST/HuggingFace may fail. Use **3.11**.

### After deploy

1. In Render → **Environment** → add secret **`HF_TOKEN`** (your Hugging Face token; accept pyannote model terms first).
2. Redeploy if the service was created before `HF_TOKEN` was set.
3. Optional — train CNN checkpoints on Render shell:

   ```bash
   python -m training.run_notebook_pipeline --quick --batch-size 2
   ```

4. Copy your API URL, e.g. `https://alm-lite-api.onrender.com`
5. Test: `https://YOUR-URL.onrender.com/health` → `"model_ready": true`

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
| `requirements.txt` not found | Set **Root Directory** to empty (repo root), not `frontend` |
| Python 3.14 on build | Set `PYTHON_VERSION=3.11.9` or use **Docker** runtime |
| Render crash / OOM | Upgrade to Standard 2 GB+; use `whisper-tiny` in `config.yaml` |
| `model_ready: false` | Wait 3–5 min; check logs; ensure RAM enough |
| No sounds | Run CNN training on Render or upload `outputs/*.pt` |
| Vercel analyze fails | Set `VITE_API_BASE` correctly; redeploy after changing env |
