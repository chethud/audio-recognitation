# Setup Guide: Running the Project on a New System

This guide provides step-by-step instructions on how to set up, configure, and run the **Audio Language Model (ALM-Lite)** project on a brand new system (Windows, macOS, or Linux).

---

## 1. System Requirements & Prerequisites

Before setting up the project, ensure the new system has the following installed:

1. **Python 3.11:** 
   * Use Python 3.11 only. Do **not** use Python 3.14 (WhisperX / several audio deps fail). Prefer `py -3.11` on Windows.
   * Make sure Python is added to the system PATH.
2. **Node.js (v18 or newer) & npm:** 
   * Required for running and building the Vite/React frontend.
3. **Git:** 
   * Required to clone the repository.
4. **FFmpeg:**
   * Crucial for audio file decoding and loading. 
   * *Windows:* The project automatically attempts to locate or download it via `imageio-ffmpeg` and add it to the path at runtime, but having a system-wide FFmpeg installation is highly recommended.
   * *macOS:* Install via Homebrew: `brew install ffmpeg`
   * *Linux:* Install via apt: `sudo apt install ffmpeg`
5. **Hardware:**
   * Minimum **8 GB RAM** (16 GB recommended) as loading Whisper, AST, and Qwen LLM models into memory simultaneously takes ~3 to 5 GB.
   * A dedicated Nvidia GPU (CUDA) is optional but will speed up inference significantly. CPU-only execution works out of the box.

---

## 2. Step-by-Step Setup Guide

Follow these commands in sequence to install the project:

### Step 1: Clone the Repository
Open a terminal/command prompt and clone the codebase:
```bash
git clone https://github.com/chethud/audio-recognitation.git
cd audio-recognitation
```

### Step 2: Set up the Python Virtual Environment
Creating a virtual environment ensures that the project dependencies do not conflict with other Python applications on the system.

**On Windows:**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Backend Dependencies
Install the required packages. Ensure the virtual environment is active (you should see `(.venv)` in your terminal prompt):
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*Note: Installing packages like PyTorch may take a few minutes depending on your internet connection.*

### Step 4: Install Frontend Dependencies
Navigate to the frontend folder and install the npm modules:
```bash
cd frontend
npm install
cd ..
```

---

## 3. Configuration Adjustments

Before starting the applications, configure the network settings:

1. **Verify Backend Port:** 
   Open `config.yaml` in the root folder. The default port configurations can remain, but make sure they do not conflict with local services on the new system.
2. **Update Frontend Environment Variables:**
   Open `frontend/.env.development`. Set the proxy target to match the backend port:
   ```env
   VITE_DEV_API_PROXY=http://127.0.0.1:8002
   ```
   *(If the backend is run on port 8000, change this to `http://127.0.0.1:8000`).*

---

## 4. Execution Workflow (First Run)

### Step 1: Start the Backend Server
Run the FastAPI application from the project root:
```bash
# Verify your virtual environment is active first!
python run.py --host 127.0.0.1 --port 8002
```

**What happens on the first run:**
1. The backend automatically creates a local SQLite database at `data/alm.db`.
2. The server begins downloading pretrained models from Hugging Face:
   * **Whisper ASR:** `openai/whisper-tiny` (~75 MB)
   * **AST Sound Detection:** `MIT/ast-finetuned-audioset-10-10-0.4593` (~340 MB)
   * **Wav2Vec2 Emotion:** `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (~1.2 GB)
   * **Qwen LLM:** `Qwen/Qwen2-0.5B-Instruct` (~950 MB)
3. **DO NOT interrupt this process.** Downloading can take 3 to 10 minutes depending on connection speeds.
4. When finished, you will see this log in the console:
   ```
   Model warmup complete [full (ASR + SED + emotion + LLM)].
   All models loaded — ready for analysis.
   INFO:     Uvicorn running on http://127.0.0.1:8002 (Press CTRL+C to quit)
   ```

### Step 2: Verify Backend Health
Open a separate terminal window and run:
* **Windows (PowerShell):**
  ```powershell
  Invoke-RestMethod -Uri "http://127.0.0.1:8002/health"
  ```
* **macOS / Linux / Git Bash:**
  ```bash
  curl http://127.0.0.1:8002/health
  ```
It should return: `{"status":"ok","model_ready":true}`.

### Step 3: Start the React Frontend
Navigate to the frontend folder and run the Vite server:
```bash
cd frontend
npm run dev
```
Navigate to `http://localhost:5173` in your browser.

---

## 5. Portability & Troubleshooting Guide

### Issue 1: "This site can't be reached" (Frontend cannot communicate with API)
* **Reason:** Vite development server is proxying to the wrong backend port.
* **Solution:** Check the terminal output of `run.py` to see what port the backend is running on (e.g. 8002). Open `frontend/.env.development` and ensure `VITE_DEV_API_PROXY` points to that exact port. Restart the Vite server (`npm run dev`).

### Issue 2: FFmpeg Not Found Error
* **Reason:** Librosa/Soundfile cannot load audio files without FFmpeg.
* **Solution:** Install FFmpeg on the system and add its `bin` directory to your system environment PATH variables (see Prerequisites section).

### Issue 3: CUDA / GPU Out of Memory
* **Reason:** The GPU does not have enough VRAM to hold all 4 models simultaneously.
* **Solution:** Run on CPU. PyTorch defaults to CPU if CUDA is not available or if forced. You can disable GPU loading by setting the environment variable `CUDA_VISIBLE_DEVICES=""` before launching `run.py`.

### Issue 4: Out of RAM (System Hangs on Startup)
* **Reason:** The computer is running out of physical memory (RAM) while warming up the models.
* **Solution:** Edit `config.yaml` to run in **Fast Mode** (ASR only, which loads only the small Whisper model) or disable specific pipeline steps:
  ```yaml
  alm_lite:
    fast_mode: true   # Skips loading AST, Wav2Vec2, and Qwen LLM
  ```
  After editing, restart `run.py`.
