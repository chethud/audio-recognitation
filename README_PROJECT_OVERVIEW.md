# Project Overview: ALM-Lite

## What this project is

ALM-Lite is an audio understanding project that combines multiple AI modules into one system:

- ASR (speech-to-text) for spoken words
- SED (sound event detection) for background/environment sounds
- Emotion recognition for speaker emotion
- LLM reasoning to generate a final intelligent response

The goal is to upload an audio clip and get a complete analysis: what was said, what sounds were present, the speaker emotion, and an AI-generated explanation.

## What we are doing now

Right now, we are building and improving the **training + inference pipeline** in a modular way:

1. Train custom CNN models:
   - SED model on ESC-50
   - Emotion model on RAVDESS
2. Save checkpoints and detailed metrics in `outputs/`
3. Merge trained model outputs for easier deployment
4. Run backend API + optional frontend for full workflow testing

Current focus is improving model quality, validating metrics, and making the end-to-end system stable for demo/report use.

## Current architecture

- `training/`  
  Dataset download, model training, metrics generation, model merging.

- `backend/`  
  FastAPI server, database integration, AI inference modules.

- `frontend/`  
  React UI for login/upload/result display.

- `src/`  
  Core modular pipeline (ASR, SED, emotion, context builder, reasoning).

- `outputs/`  
  Trained checkpoints and metrics JSON files.

## Main output files

- `outputs/sed_cnn.pt` - trained SED model
- `outputs/emotion_cnn.pt` - trained emotion model
- `outputs/alm_cnn_merged.pt` - merged model bundle
- `outputs/sed_metrics.json` - SED evaluation report
- `outputs/emotion_metrics.json` - emotion evaluation report
- `outputs/alm_cnn_merged_metrics.json` - combined metrics summary

## How to run the project

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Train models (optional if checkpoints already exist)

```bash
python -m training.download_datasets --all
python -m training.train_sed
python -m training.train_emotion
python -m training.merge_cnn_models
```

### 3) Start backend API

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4) Start frontend UI

```bash
cd frontend
npm install
npm run dev
```

## Short project status

- Core modules are implemented.
- CNN training pipeline is available.
- Metrics are being tracked in detail.
- Full-stack flow (API + UI) is available.
- We are currently improving accuracy and preparing cleaner final results/documentation.

## Next steps

- Train with more epochs and tuning for stronger accuracy
- Add more robust evaluation on real/noisy audio
- Improve deployment readiness and documentation
- Finalize report/demo narrative using latest metrics
