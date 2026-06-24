# ALM-Lite — Full Project Guide

Complete documentation for the **Audio Language Model (ALM-Lite)** project: architecture, workflows, setup, execution commands, API usage, training, and troubleshooting.

Repository: [https://github.com/chethud/audio-recognitation](https://github.com/chethud/audio-recognitation)

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [What the system does](#2-what-the-system-does)
3. [Technology stack](#3-technology-stack)
4. [Project structure](#4-project-structure)
5. [Architecture & workflow](#5-architecture--workflow)
6. [Inference pipeline (step by step)](#6-inference-pipeline-step-by-step)
7. [Training workflow (CNN models)](#7-training-workflow-cnn-models)
8. [Database (SQLite)](#8-database-sqlite)
9. [Configuration](#9-configuration)
10. [Installation](#10-installation)
11. [Execution commands](#11-execution-commands)
12. [API endpoints](#12-api-endpoints)
13. [Frontend usage](#13-frontend-usage)
14. [Speed vs full analysis](#14-speed-vs-full-analysis)
15. [Troubleshooting](#15-troubleshooting)
16. [Project outputs](#16-project-outputs)

---

## 1. Project overview

**ALM-Lite** is a modular audio understanding system. You upload an audio file and receive:

| Output | Description |
|--------|-------------|
| **Transcript** | Speech converted to text (Whisper ASR) |
| **Sounds** | Environmental / background sound labels (SED) |
| **Emotion** | Estimated speaker emotion |
| **Answer** | AI-generated response to your question (LLM) |

The project includes:

- **FastAPI backend** — REST API for analysis and auth
- **React frontend** — glass-style UI to upload audio, play preview, and view results
- **Modular AI pipeline** — ASR, SED, emotion, and LLM in `src/`
- **SQLite database** — local storage for users and analysis history
- **Optional CNN training** — custom SED/emotion models on ESC-50 and RAVDESS

---

## 2. What the system does

### Inference (main use case)

1. User uploads audio (`.wav`, `.mp3`, `.flac`, etc.) via UI or API.
2. Backend decodes only the **first N seconds** (default: 12s) for speed.
3. Three models run **in parallel** (when full mode is on):
   - Whisper → transcript
   - AST → sound events
   - Wav2Vec2 → emotion
4. Results are merged into a **structured context**.
5. **Qwen LLM** generates an answer to the user’s question.
6. Response is returned to the UI; metadata is saved to SQLite in the background.

### Training (optional)

Separate mel-spectrogram CNNs can be trained on:

- **ESC-50** — sound event detection (50 classes)
- **RAVDESS** — emotion recognition (8 classes)

Checkpoints and metrics are saved under `outputs/`.

---

## 3. Technology stack

| Layer | Technologies |
|-------|----------------|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| AI / ML | PyTorch, Hugging Face Transformers, Librosa |
| ASR | `openai/whisper-tiny` |
| SED | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| Emotion | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` |
| LLM | `Qwen/Qwen2-0.5B-Instruct` |
| Database | SQLite (`data/alm.db`) |
| Frontend | React 18, Vite, Tailwind CSS, Axios |
| Training | scikit-learn, datasets, Jupyter notebook |

---

## 4. Project structure

```
audio-recognitation/
├── run.py                      # Start FastAPI server
├── config.yaml                 # Main configuration
├── requirements.txt            # Python dependencies
├── inference_worker_modular.py # CLI/subprocess inference worker
├── ALM_Lite_CNN_Training.ipynb # Notebook: download → train → merge
│
├── backend/
│   ├── main.py                 # FastAPI app & routes
│   ├── inference_service.py    # In-process model cache & analyze
│   ├── database.py             # SQLite schema & queries
│   ├── models.py               # Pydantic API models
│   └── api/server.py           # Alternate uvicorn entry
│
├── src/
│   ├── asr/                    # Whisper speech-to-text
│   ├── sed/                    # Environmental sound detection
│   ├── emotion/                # Emotion classification
│   ├── reasoning/              # LLM answer generation
│   ├── context_builder/        # Merge transcript + sounds + emotion
│   ├── pipeline/alm_lite.py    # Full pipeline orchestration
│   └── utils/audio.py          # Audio file loading
│
├── training/
│   ├── download_datasets.py
│   ├── train_sed.py
│   ├── train_emotion.py
│   ├── merge_cnn_models.py
│   └── data_utils.py
│
├── frontend/
│   ├── src/
│   │   ├── pages/Home.jsx      # Upload & results
│   │   ├── components/         # UploadAudio, ResultDisplay, GlassBackground
│   │   └── api/api.js          # Axios client
│   ├── .env.development        # API proxy URL for dev
│   └── package.json
│
├── data/
│   └── alm.db                  # SQLite (created at runtime, gitignored)
│
└── outputs/
    ├── sed_metrics.json
    ├── emotion_metrics.json
    └── *.pt                    # Trained checkpoints (gitignored)
```

---

## 5. Architecture & workflow

### High-level diagram

```
┌─────────────┐     POST /analyze      ┌──────────────────┐
│  React UI   │ ─────────────────────► │  FastAPI Backend │
│  (Vite)     │ ◄───────────────────── │  backend/main.py │
└─────────────┘     JSON response      └────────┬─────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │ inference_service.py   │
                                    │ (models cached in RAM) │
                                    └────────┬───────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    ▼                        ▼                        ▼
              ┌──────────┐           ┌──────────┐           ┌──────────┐
              │ Whisper  │           │   AST    │           │ Wav2Vec2 │
              │   ASR    │           │   SED    │           │ Emotion  │
              └────┬─────┘           └────┬─────┘           └────┬─────┘
                   │                      │                      │
                   └──────────────────────┼──────────────────────┘
                                          ▼
                               ┌─────────────────────┐
                               │  Context builder    │
                               └──────────┬──────────┘
                                          ▼
                               ┌─────────────────────┐
                               │   Qwen2 LLM         │
                               └──────────┬──────────┘
                                          ▼
                               ┌─────────────────────┐
                               │  SQLite (async)     │
                               │  analyze_logs       │
                               └─────────────────────┘
```

### Request flow (UI → API → models)

1. User selects audio file and optional question on **Home** page.
2. Frontend sends `multipart/form-data` to `POST /analyze`.
3. Backend writes file to a temp path.
4. `inference_service.analyze_file()` loads first 12s of audio.
5. `run_alm_lite()` runs ASR + SED + emotion in parallel, then LLM.
6. JSON returned: `transcript`, `sounds`, `emotion`, `answer`.
7. `ResultDisplay` shows all sections; audio player lets user preview upload.

---

## 6. Inference pipeline (step by step)

**File:** `src/pipeline/alm_lite.py`

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 1 | `load_audio_from_file` | File path | 16 kHz mono waveform (max 12s) |
| 2a | `transcribe_audio` | Waveform | Text transcript |
| 2b | `detect_sound_events` | Waveform | List of `{label, score}` |
| 2c | `predict_emotion_from_audio` | Waveform | Emotion label |
| 3 | `build_structured_context` | Transcript + sounds + emotion | Context string |
| 4 | `answer_question_from_context` | Context + question | Final answer |

Steps 2a–2c run **in parallel** using `ThreadPoolExecutor`.

**Startup warmup:** On API start, all models are loaded once into memory (`backend/inference_service.py` → `warmup()`). This makes later requests much faster.

---

## 7. Training workflow (CNN models)

### Option A — Jupyter notebook (recommended)

```bash
# Open from project root
jupyter notebook ALM_Lite_CNN_Training.ipynb
```

Notebook steps:

1. Download ESC-50 and RAVDESS
2. Train SED CNN → `outputs/sed_cnn.pt`
3. Train emotion CNN → `outputs/emotion_cnn.pt`
4. Merge → `outputs/alm_cnn_merged.pt`
5. Write metrics JSON files

### Option B — Command line

```bash
# 1. Download datasets
python -m training.download_datasets --all

# 2. Train sound event model (ESC-50)
python -m training.train_sed --epochs 25 --batch-size 16

# 3. Train emotion model (RAVDESS)
python -m training.train_emotion --epochs 30 --batch-size 32

# 4. Merge checkpoints + metrics
python -m training.merge_cnn_models
```

### Training outputs

| File | Description |
|------|-------------|
| `outputs/sed_cnn.pt` | SED checkpoint |
| `outputs/emotion_cnn.pt` | Emotion checkpoint |
| `outputs/alm_cnn_merged.pt` | Combined bundle |
| `outputs/sed_metrics.json` | ESC-50 validation metrics |
| `outputs/emotion_metrics.json` | RAVDESS validation metrics |

---

## 8. Database (SQLite)

**Path:** `data/alm.db` (auto-created on first API start)

| Table | Purpose |
|-------|---------|
| `users` | Registered accounts |
| `sessions` | Bearer auth tokens |
| `analyze_logs` | Full `/analyze` results |
| `inferences` | Legacy `/inference` runs |
| `dataset_samples` | Optional dataset metadata |

Audio BLOB storage is **optional** (`alm_lite.store_uploaded_audio` in `config.yaml`). Default is `false` for faster responses.

Retrieve stored audio (if saved):

```http
GET /analyze/history/{log_id}/audio
```

---

## 9. Configuration

**File:** `config.yaml`

### Current default (full analysis)

```yaml
alm_lite:
  fast_mode: false              # Full pipeline: ASR + SED + emotion + LLM
  store_uploaded_audio: false   # Skip BLOB save for speed
  asr:
    model_id: "openai/whisper-tiny"
    language: null              # Set "en" for faster English-only
  sed:
    enabled: true               # Sound detection ON
  emotion:
    enabled: true               # Emotion detection ON
  llm:
    enabled: true               # LLM answer ON
    max_new_tokens: 32

data:
  max_audio_length_sec: 12      # Only first 12 seconds analyzed
```

### Important flags

| Setting | `true` / enabled | `false` / disabled |
|---------|------------------|---------------------|
| `fast_mode` | ASR only + instant answer (fastest) | Full ASR + SED + emotion + LLM |
| `sed.enabled` | Environmental sounds detected | No sound labels |
| `emotion.enabled` | Emotion label returned | Always `neutral` |
| `llm.enabled` | AI-generated answer | Template answer from context |

**After changing `config.yaml`, restart the API server.**

---

## 10. Installation

### Prerequisites

- Python 3.10 or newer
- Node.js 18+ and npm
- Git
- ~4–8 GB free disk (for Hugging Face model cache)
- CPU works; GPU speeds up inference if available

### Step 1 — Clone repository

```bash
git clone https://github.com/chethud/audio-recognitation.git
cd audio-recognitation
```

### Step 2 — Python environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

First run downloads models from Hugging Face (several GB). Ensure internet access.

### Step 3 — Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 4 — Configure frontend API proxy (development)

Edit `frontend/.env.development` if the API port differs:

```env
VITE_DEV_API_PROXY=http://127.0.0.1:8001
```

Use `8000` if that port is free on your machine.

---

## 11. Execution commands

### Start FastAPI backend

**Recommended (project root):**

```bash
python run.py
```

**Custom port (if 8000 is blocked on Windows):**

```bash
python run.py --port 8001
```

**Alternative (uvicorn directly):**

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

**Wait for startup log:**

```
Model warmup complete [full (ASR + SED + emotion + LLM)].
All models loaded — ready for analysis.
```

First startup may take **3–5 minutes** (model download + load).

### Verify API health

```bash
python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8001/health').read().decode())"
```

Expected: `{"status": "ok", "model_ready": true}`

### Start React frontend

```bash
cd frontend
npm run dev
```

Open the URL shown (usually `http://localhost:5173`).

### Build frontend for production

```bash
cd frontend
npm run build
npm run preview
```

### Analyze audio via API (curl)

```bash
curl -X POST "http://127.0.0.1:8001/analyze" \
  -F "file=@your_audio.mp3" \
  -F "question=What can be inferred from the audio?"
```

### API documentation (Swagger)

Open in browser: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

---

## 12. API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server status & `model_ready` |
| `POST` | `/analyze` | **Main endpoint** — upload audio + question |
| `GET` | `/analyze/history` | List past analyses |
| `GET` | `/analyze/history/{id}` | Get one analysis record |
| `GET` | `/analyze/history/{id}/audio` | Download stored audio (if saved) |
| `POST` | `/inference` | Legacy inference (modular optional) |
| `GET` | `/history` | Legacy inference history |
| `POST` | `/auth/signup` | Create account |
| `POST` | `/auth/login` | Login → bearer token |
| `GET` | `/auth/me` | Current user |
| `POST` | `/auth/logout` | End session |

### `/analyze` response example

```json
{
  "transcript": "Hello, how are you today?",
  "sounds": ["Speech", "Music"],
  "emotion": "happy",
  "answer": "The speaker greets the listener in a positive tone...",
  "question": "What can be inferred from the audio?",
  "audio_filename": "sample.mp3",
  "log_id": null
}
```

---

## 13. Frontend usage

1. Start backend (`python run.py --port 8001`).
2. Start frontend (`cd frontend && npm run dev`).
3. Open the app in your browser.
4. **Upload** an audio file — preview player appears.
5. Edit the **question** if needed.
6. Click **Analyze** — wait for transcript, sounds, emotion, and answer.
7. Optional: **Sign up / Log in** for auth (stored in SQLite).

UI features:

- Glassmorphism design
- Audio preview before analysis
- Loading stages: transcribing → detecting sounds → generating answer

---

## 14. Speed vs full analysis

| Mode | Config | Features | Typical time (CPU) |
|------|--------|----------|---------------------|
| **Fast** | `fast_mode: true` | Transcript + template answer | ~10–20 s |
| **Full** | `fast_mode: false` | Transcript + sounds + emotion + LLM | ~30–90 s |

Speed tips (full mode):

- Keep `max_audio_length_sec: 12` (or lower)
- Set `asr.language: "en"` for English-only audio
- Keep `store_uploaded_audio: false`
- Do not use `--reload` on the API (reloads models on file changes)
- Leave server running between uploads (models stay in memory)

---

## 15. Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 8000 blocked (`WinError 10013`) | Use `python run.py --port 8001` and update `frontend/.env.development` |
| `model_ready: false` | Wait for warmup to finish; check API terminal for errors |
| Very slow on long MP3s | Only first 12s are analyzed; ensure latest `src/utils/audio.py` uses `duration=` in librosa |
| Empty sounds / emotion | Ensure `sed.enabled: true` and `emotion.enabled: true`; `fast_mode: false` |
| Frontend cannot reach API | Match `VITE_DEV_API_PROXY` to backend port; restart `npm run dev` |
| HuggingFace symlink warning | Harmless on Windows; suppressed via env in code |
| Out of memory | Close other apps; use `fast_mode: true` or disable emotion temporarily |

### Environment variables (optional)

```bash
# Faster Hugging Face downloads (optional)
set HF_TOKEN=your_huggingface_token

# Force subprocess inference (slower; not recommended)
set ALM_SUBPROCESS_INFERENCE=1
```

---

## 16. Project outputs

### Inference (runtime)

Returned in API/UI for each upload:

- **Transcript** — speech text
- **Sounds** — detected environmental labels
- **Emotion** — speaker emotion estimate
- **Answer** — LLM response to your question

### Training (offline)

| Metric (CNN) | Dataset | Typical accuracy |
|--------------|---------|------------------|
| SED CNN | ESC-50 | ~42.75% (see `outputs/sed_metrics.json`) |
| Emotion CNN | RAVDESS | ~59.72% (see `outputs/emotion_metrics.json`) |

ASR and LLM use pretrained models; quality depends on audio and language.

---

## Quick start (copy-paste)

```bash
# Terminal 1 — Backend
cd audio-recognitation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py --port 8001

# Terminal 2 — Frontend (after backend shows "ready for analysis")
cd audio-recognitation\frontend
npm install
npm run dev
```

Then open **http://localhost:5173**, upload audio, and click **Analyze**.

---

## License & credits

- Built as an academic / demo **Audio Language Model** project.
- Uses open models from OpenAI (Whisper), MIT (AST), Hugging Face community (emotion, Qwen).
- ESC-50 and RAVDESS datasets for CNN training experiments.

For the original training metrics and notebook details, see also `README.md` in the repository root.
