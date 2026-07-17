# Complete Audio Language Model (ALM-Lite) Project Guide

This master guide contains the comprehensive documentation for the **Audio Language Model (ALM-Lite)** project. It details the system capabilities, technology stack, project architecture, technical algorithms, configuration, installation, local execution commands, offline training workflows, and troubleshooting.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Key Capabilities & Output Channels](#2-key-capabilities--output-channels)
3. [Technology Stack](#3-technology-stack)
4. [Project Structure](#4-project-structure)
5. [System Architecture & Data Workflows](#5-system-architecture--data-workflows)
6. [Detailed Technical Algorithms](#6-detailed-technical-algorithms)
   * [6.1 Audio Preprocessing Pipeline](#61-audio-preprocessing-pipeline)
   * [6.2 Speech-to-Text (ASR)](#62-speech-to-text-asr)
   * [6.3 Speaker Diarization](#63-speaker-diarization)
   * [6.4 Sound Event Detection (SED)](#64-sound-event-detection-sed)
   * [6.5 Speech Emotion Recognition (SER)](#65-speech-emotion-recognition-ser)
   * [6.6 Reasoning & Question Answering Head](#66-reasoning--question-answering-head)
7. [Database (SQLite) Schema](#7-database-sqlite-schema)
8. [Configuration (config.yaml)](#8-configuration-configyaml)
9. [Installation & Setup](#9-installation--setup)
10. [Local Execution Commands](#10-local-execution-commands)
11. [Offline Training & Merging Pipelines](#11-offline-training--merging-pipelines)
12. [Troubleshooting Guide](#12-troubleshooting-guide)

---

## 1. Project Overview

**ALM-Lite** is a modular audio understanding application. It allows users to upload any audio file, transcribe the speech, detect background environmental sounds, identify the speakers' emotions, and receive an AI-generated answer to a specific question about the audio's contents.

The project features:
* **FastAPI Backend:** An asynchronous API handling audio processing, user database authentication, session logging, and ML inference.
* **React Frontend:** A modern, glassmorphic UI built with React 18, Vite, and Tailwind CSS.
* **Modular Pipeline Orchestration:** A modular Python backend coordinating parallel model execution.
* **Subprocess Isolation:** An inference wrapper running model inference in a child process on Windows to prevent native crashes from terminating the FastAPI server.
* **SQLite Storage:** In-house database handling accounts, authorization sessions, and request history.

---

## 2. Key Capabilities & Output Channels

When an audio file is uploaded to the backend via the UI or the `/analyze` REST endpoint, the system extracts:
1. **ASR Transcript:** Conversational speech transcribed into text.
2. **Diarization Turns:** Time-stamped segments indicating who spoke and what they said (e.g. `Speaker 1`, `Speaker 2`).
3. **Sound Events:** Environmental sounds detected in the audio (e.g. `car honking`, `clock ticking`, `water drops`).
4. **Speaker Emotion:** Estimated emotional tone of the speaker(s) (e.g., `happy`, `sad`, `neutral`).
5. **AI Answer:** A contextual response synthesized by the Qwen LLM answering the user's specific text question.

---

## 3. Technology Stack

| Layer | Technologies & Libraries |
| :--- | :--- |
| **Backend Framework** | Python 3.10+, FastAPI, Uvicorn |
| **Frontend Framework** | React 18, Vite, Tailwind CSS, Axios, Lucide Icons |
| **Audio Core** | Librosa, Soundfile, PyTorch, imageio-ffmpeg |
| **Speech-to-Text (ASR)** | Hugging Face Transformers (`openai/whisper-tiny` / `openai/whisper-base`), CTranslate2 |
| **Sound Event Detection** | `MIT/ast-finetuned-audioset-10-10-0.4593` (AST Transformer) / Custom PyTorch CNN |
| **Speech Emotion** | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` / Custom PyTorch CNN |
| **Reasoning Head (LLM)**| `Qwen/Qwen2-0.5B-Instruct` |
| **Database** | SQLite3 (`data/alm.db`) |
| **Training Pipeline** | scikit-learn, Jupyter Notebook, Matplotlib |

---

## 4. Project Structure

```
audio-recognitation/
├── run.py                      # Start FastAPI server
├── config.yaml                 # Configuration parameters
├── requirements.txt            # Python dependencies
├── inference_worker_modular.py # CLI/subprocess inference worker
├── ALM_Lite_CNN_Training.ipynb # Notebook: download → train → merge
├── PROJECT_TECHNICAL_DETAILS.md# Algorithms documentation
├── COMPLETE_PROJECT_GUIDE.md   # This master guide
│
├── backend/
│   ├── main.py                 # FastAPI endpoints & routing
│   ├── inference_service.py    # Backend model cache & analyze entry
│   ├── database.py             # SQLite schemas & db queries
│   └── models.py               # Pydantic request/response schemas
│
├── src/
│   ├── asr/                    # Whisper S2T & text cleanup modules
│   ├── sed/                    # AST Sound Event Detection modules
│   ├── emotion/                # Speech Emotion Recognition modules
│   ├── diarization/            # PyAnnote & Custom VAD Clustering
│   ├── reasoning/              # Qwen LLM reasoning implementation
│   ├── pipeline/alm_lite.py    # Main pipeline orchestration
│   └── utils/audio.py          # Audio loading utilities
│
├── training/
│   ├── download_datasets.py    # Dataset loaders (ESC-50, RAVDESS)
│   ├── train_sed.py            # Training script for SED CNN
│   ├── train_emotion.py        # Training script for Emotion CNN
│   ├── merge_cnn_models.py     # Checkpoint merging utility
│   └── data_utils.py           # PyTorch CNN model structure
│
├── frontend/
│   ├── src/
│   │   ├── pages/Home.jsx      # Upload page
│   │   ├── components/         # GlassBackground, ResultDisplay, AppHeader
│   │   └── api/api.js          # Axios API wrappers
│   ├── .env.development        # Development API proxy port
│   └── package.json
│
└── data/
    └── alm.db                  # Local database (generated at runtime, gitignored)
```

---

## 5. System Architecture & Data Workflows

The following diagrams illustrate how data flows through the application during an analysis request:

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
                               └─────────────────────┘
```

### The Request Lifecycle
1. **Upload & Form Data:** The frontend sends the audio file bytes and the user's question via a `multipart/form-data` request.
2. **Subprocess Isolation (Windows):** To protect the web server from native segment crashes, the backend executes `inference_worker_modular.py` in a child process, communicating parameters using JSON configurations.
3. **Parallel Feature Extraction:** The audio is processed by the ASR, SED, and SER components in parallel threads.
4. **LLM Context Compilation:** The extracted features are merged into a text context and parsed by the Qwen LLM.
5. **Database Log Action:** The FastAPI server logs the request metadata asynchronously into SQLite while returning the results to the React frontend.

---

## 6. Detailed Technical Algorithms

### 6.1 Audio Preprocessing Pipeline
Raw audio is resampled to a target of **16,000 Hz** and downmixed to **mono** (1-channel). The first 12 seconds are parsed by default, which ensures optimal CPU performance.
```python
waveform, sr = librosa.load(audio_path, sr=16000, mono=True, duration=12.0)
```

### 6.2 Speech-to-Text (ASR)
ASR uses Hugging Face Pipelines loaded with Whisper weights.
* **Deduplication:** A sliding window script (`_dedupe_whisper_chunks`) matches and removes repeating word tokens, preventing infinite Whisper hallucination loops.
* **Language Customizations:** The pipeline detects the spoken language automatically. For Kannada, the system uses the fine-tuned `vasista22/whisper-kannada-base` model.

### 6.3 Speaker Diarization
* **PyAnnote:** If a valid `HF_TOKEN` is found, the system utilizes PyAnnote to extract neural speaker segments.
* **Custom VAD + MFCC Clustering Fallback:**
  1. **Energy VAD:** Detects speech segments using Root Mean Square (RMS) frame values.
  2. **Feature Extractor:** Generates 20 MFCC features plus delta and delta-delta features (43 dimensions).
  3. **Agglomerative Clustering:** Groups vectors using Cosine Distance and Average Linkage.
  4. **Tuned Thresholds:** The thresholds are tuned for unequal conversations (e.g. interviews):
     ```python
     sil_accept = 0.08      # Min silhouette score to accept 2 speakers
     minority_accept = 0.08  # Min ratio of minority speaker talk-time
     minority_keep = 0.06    # Collapse limit for minor speaker
     ```

### 6.4 Sound Event Detection (SED)
* **Audio Spectrogram Transformer (AST):** Uses a Vision Transformer architecture pre-trained on AudioSet to classify 527 classes.
* **Custom PyTorch CNN:** A custom model trained on ESC-50 classes:
  - **Conv Blocks:** 3 Convolutional layers (filters: 32 $\rightarrow$ 64 $\rightarrow$ 128) with Batch Normalization, ReLU activations, and Max Pooling.
  - **Global Avg Pooling:** Formats outputs to a $(4, 8)$ grid.
  - **Linear Head:** Flatten layer, Dense layer (128*4*8 $\rightarrow$ 256) with 30% dropout, and output projection (50 logits).

### 6.5 Speech Emotion Recognition (SER)
* **Pretrained Wav2Vec2:** Extract features using `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` to classify 8 RAVDESS emotions.
* **Custom PyTorch Emotion CNN:** Shares the architecture of the SED CNN, but utilizes an 8-output classification head.

### 6.6 Reasoning & Question Answering Head
The reasoning system compiles transcripts, emotions, and background sounds into a single prompt:
```
[Language Instruction] Answer briefly from this audio context.
Context:
Speech (transcript): "I am fine, thank you."
Speaker emotion: happy
Non-speech (environmental sounds): rain, dog barking

Q: What is the speaker's mood and what is the background environment?
A:
```
This prompt is processed by `Qwen/Qwen2-0.5B-Instruct` with `temperature = 0.0` to generate a brief, deterministic answer.

---

## 7. Database (SQLite) Schema

Data is stored locally in `data/alm.db`. Tables include:

```sql
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    password_salt TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token TEXT UNIQUE NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analyze_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_filename TEXT NOT NULL,
    audio_mime TEXT,
    audio_data BLOB, -- Optional binary file save
    question TEXT NOT NULL,
    transcript TEXT,
    sounds_json TEXT,
    emotion TEXT,
    answer TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

---

## 8. Configuration (config.yaml)

The system behavior is defined in `config.yaml`:
```yaml
alm_lite:
  fast_mode: false              # If true, runs ASR only (fastest response)
  store_uploaded_audio: false   # Store file binary BLOBs in SQLite database
  asr:
    model_id: "openai/whisper-tiny"
    language: null              # Set "en" or "kn" to lock language
  sed:
    enabled: true
  emotion:
    enabled: true
  llm:
    enabled: true               # If false, returns rule-based templated answer
    max_new_tokens: 32

data:
  max_audio_length_sec: 12      # Maximum length of audio decoded
```

---

## 9. Installation & Setup

### Prerequisites
* Python 3.10 or newer
* Node.js 18+ and npm
* Git

### Step 1: Clone Repository
```bash
git clone https://github.com/chethud/audio-recognitation.git
cd audio-recognitation
```

### Step 2: Set up Python Virtual Environment
```bash
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate on macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Set up Frontend Node Modules
```bash
cd frontend
npm install
cd ..
```

---

## 10. Local Execution Commands

### 10.1 Running the Backend Server
Start the FastAPI server from the project root directory:
```bash
.venv\Scripts\python.exe run.py --host 127.0.0.1 --port 8002
```

### 10.2 Verifying Backend Status
Check that the server is alive and models have successfully loaded:
```bash
python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8002/health').read().decode())"
```
Expected output: `{"status":"ok","model_ready":true}`

### 10.3 Running the React Frontend
Start the Vite dev server:
```bash
cd frontend
npm run dev
```
Open your browser and navigate to the local address displayed in the console (default: `http://localhost:5173`).

---

## 11. Offline Training & Merging Pipelines

If you wish to retrain the custom CNN models rather than using the default pre-trained transformers:

### Option A: Jupyter Notebook (Recommended)
Open the Jupyter notebook inside the root directory:
```bash
jupyter notebook ALM_Lite_CNN_Training.ipynb
```
Run the cells sequentially to download datasets, train the SED and Emotion CNNs, merge their parameters, and write validation metric records.

### Option B: CLI Commands
```bash
# 1. Download ESC-50 and RAVDESS datasets
python -m training.download_datasets --all

# 2. Train SED CNN (ESC-50)
python -m training.train_sed --epochs 40 --batch-size 16

# 3. Train Emotion CNN (RAVDESS)
python -m training.train_emotion --epochs 60 --batch-size 32

# 4. Merge trained models into a single bundle
python -m training.merge_cnn_models
```

The merged weights will be outputted to `outputs/alm_cnn_merged.pt`, and your validation metrics will be saved in `outputs/sed_metrics.json` and `outputs/emotion_metrics.json`.

---

## 12. Troubleshooting Guide

* **Port Already in Use (`WinError 10013`):** 
  If port `8000` or `8002` is in use, start the backend with a custom port:
  ```bash
  python run.py --port 8003
  ```
  Then, edit `frontend/.env.development` to target your custom port:
  ```env
  VITE_DEV_API_PROXY=http://127.0.0.1:8003
  ```
  Finally, restart your Vite frontend.

* **Speaker Diarization not Separating Speaker Turns:**
  Ensure that `diarization.backend` is set to `vad` in `config.yaml` and that the thresholds in [vad_segment_pipeline.py](file:///d:/jn/audio-recognitation/src/diarization/vad_segment_pipeline.py) have been adjusted. If PyAnnote is preferred, ensure `HF_TOKEN` is correctly set in your environment variables.

* **Out of Memory (OOM) Errors:**
  If running out of RAM/VRAM on standard CPU systems, disable the LLM or emotion components temporarily in `config.yaml` by setting `llm.enabled: false` or `emotion.enabled: false`. Alternatively, enable `fast_mode: true`.
