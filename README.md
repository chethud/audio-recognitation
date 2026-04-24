# ALM-Lite — Audio Language Model (modular)

Speech (ASR), environmental sounds (SED), speaker emotion, and LLM reasoning over a unified API and optional React UI.

---

## Model stack (inference — default)

These are the **pretrained** models used by the modular pipeline (`inference_worker_modular.py` / `POST /analyze`). Metrics are **task- and audio-dependent**; use your own clips to measure WER, label accuracy, or answer quality.

| Component | Model | Role |
|-----------|--------|------|
| ASR | `openai/whisper-small` | Speech-to-text (multilingual) |
| SED | `MIT/ast-finetuned-audioset-10-10-0.4593` | Environmental sound events (AudioSet labels) |
| Emotion | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | Speech emotion (English-oriented) |
| LLM | `Qwen/Qwen2-0.5B-Instruct` | Reasoning from built context |

Configuration: `config.yaml` under `alm_lite`.

### Module-wise accuracy (current run)

The table below uses values from your latest generated metrics files:
- `outputs/sed_metrics.json`
- `outputs/emotion_metrics.json`

| Module | Dataset | Accuracy | Balanced Accuracy | Source |
|--------|---------|----------|-------------------|--------|
| SED CNN (`MelSpecCNN`) | ESC-50 validation | **42.75%** (`0.4275`) | **42.75%** (`0.4275`) | `outputs/sed_metrics.json` |
| Emotion CNN (`MelSpecCNN`) | RAVDESS validation | **59.72%** (`0.5972`) | **60.26%** (`0.6026`) | `outputs/emotion_metrics.json` |

> Note: ASR and LLM in this project are pretrained inference modules. Their quality is typically reported with task-specific metrics (e.g., WER for ASR and human/benchmark scoring for LLM answers), not a single unified "accuracy" value.

### Database: SQLite + optional Supabase

- **SQLite** (default): `data/alm.db` — all runs are stored locally.
- **Supabase** (optional mirror for `POST /analyze`): copy `.env.example` to `.env`, set your project URL and API key, then `ALM_USE_SUPABASE=1`.

Example URL (your project): `https://ljxarxkxlgzkueedlrbi.supabase.co` — add keys from **Supabase Dashboard → Project Settings → API** (use **service_role** on the server for storage upload + `audio_logs` inserts, or configure **RLS** if you use the **anon** key).

In Supabase: create Storage bucket **`audio-files`**, run SQL from `supabase/schema.sql` for the **`audio_logs`** table.

---

## Jupyter notebook (recommended): download → train both → merge

Open **`ALM_Lite_CNN_Training.ipynb`** from the project root. It:

1. Downloads **ESC-50** (SED) and **RAVDESS** (emotion).
2. Trains **two separate** CNNs and writes **full accuracy details** to JSON.
3. **Merges** weights into `outputs/alm_cnn_merged.pt` and copies metrics into `outputs/alm_cnn_merged_metrics.json`.

**Accuracy details (per model, validation set):**

- Overall **accuracy** and **balanced accuracy**
- **Macro / micro / weighted** precision, recall, F1
- **Per-class** precision, recall, F1, support (sklearn `classification_report`)
- **Confusion matrix** (rows = true class, columns = predicted)

Files:

| File | Contents |
|------|----------|
| `outputs/sed_metrics.json` | All of the above for ESC-50 (50 classes) |
| `outputs/emotion_metrics.json` | All of the above for RAVDESS (8 emotions) |
| `outputs/alm_cnn_merged_metrics.json` | Both reports + paths to checkpoints |

**Separate checkpoints:** `outputs/sed_cnn.pt`, `outputs/emotion_cnn.pt`  
**Merged bundle:** `outputs/alm_cnn_merged.pt` (Python dict with keys `sed`, `emotion`, `meta`).

Programmatic merge (same as notebook last step):

```bash
python -m training.merge_cnn_models
```

---

## CNN training results (custom checkpoints)

Custom **mel-spectrogram CNNs** are trained offline. Checkpoints are saved under `outputs/` (see `config.yaml` → `alm_cnn`). The notebook uses `training/train_cnn_pipeline.py` for richer metrics than the CLI scripts alone.

### Sound event CNN (ESC-50 subset)

ESC-50 is used as a **manageable AudioSet-style** environmental benchmark (50 classes). Not the full AudioSet corpus.

| Metric | Value | Notes |
|--------|--------|--------|
| Dataset | ESC-50 (Hugging Face `ashraq/esc50`) | 50 classes, ~2k clips |
| Validation split | 20% | Stratified, seed 42 |
| Model | `MelSpecCNN` | Log-mel → conv blocks → classifier (`training/data_utils.py`) |
| Checkpoint | `outputs/sed_cnn.pt` | Best validation accuracy saved during training |

**Reported run (smoke test — 1 epoch, batch size 8, CPU/GPU auto):**

| Epochs | Train loss (approx.) | Best validation accuracy |
|--------|----------------------|---------------------------|
| 1 | ~3.82 | **~10.3%** |

> **Note:** One epoch is only a sanity check. For a project report, train longer (e.g. 20–50 epochs), tune learning rate, and prefer GPU. Random baseline for 50-way uniform classification is ~2%; the smoke run already exceeds that but is **not** representative of a converged model.

**Reproduce training**

```bash
python -m training.download_datasets --sed
python -m training.train_sed --epochs 25 --batch-size 16
```

---

### Emotion CNN (RAVDESS Speech)

| Metric | Value | Notes |
|--------|--------|--------|
| Dataset | RAVDESS Speech (`Audio_Speech_Actors_01-24`) | 8 emotion classes (speech modality `03-*` filenames) |
| Validation split | 20% | Stratified where possible |
| Model | `MelSpecCNN` | Same backbone as SED, 8-way head |
| Checkpoint | `outputs/emotion_cnn.pt` | Best validation accuracy saved |

**After you train**, replace the line below with your best run:

| Epochs | Train loss (approx.) | Best validation accuracy |
|--------|----------------------|---------------------------|
| *(run training)* | — | — |

**Reproduce training**

```bash
python -m training.download_datasets --ravdess
python -m training.train_emotion --epochs 30 --batch-size 32
```

---

## Quick commands

| Task | Command |
|------|---------|
| Full workflow + metrics + merge | Open `ALM_Lite_CNN_Training.ipynb` |
| Download datasets | `python -m training.download_datasets --all` |
| Train SED CNN | `python -m training.train_sed` |
| Train emotion CNN | `python -m training.train_emotion` |
| Merge `.pt` + metrics JSON | `python -m training.merge_cnn_models` |
| API server | `uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000` |
| React UI | `cd frontend && npm run dev` |

---

## Updating this README with your final numbers

1. Run full training (many epochs) for SED and emotion.
2. Copy the **best validation accuracy** (and optional final train loss) from the terminal log into the tables above.
3. For the pretrained inference stack, add optional rows (e.g. sample WER on LibriSpeech, or qualitative examples) if your report requires them.

---

## Project explanation (up to model training)

Use this section for presentations, viva, or report writing.

1. **Problem statement**
   Build an audio understanding system that can detect speech content, environmental sounds, and speaker emotion, then provide meaningful reasoning.

2. **System design**
   The project has a modular ALM-Lite architecture:
   - ASR module converts speech to text.
   - SED module detects environmental events.
   - Emotion module predicts speaker emotion.
   - LLM module reasons over the combined context.

3. **Training objective**
   In the training phase, we train custom CNNs for:
   - Sound Event Detection (SED) on ESC-50.
   - Emotion Recognition on RAVDESS speech.

4. **Dataset preparation**
   - ESC-50 is loaded through Hugging Face as an AudioSet-style subset.
   - RAVDESS speech is downloaded from Zenodo and extracted locally.
   - Commands:
     - `python -m training.download_datasets --sed`
     - `python -m training.download_datasets --ravdess`
     - `python -m training.download_datasets --all`

5. **Feature extraction**
   Audio is transformed into log-mel spectrograms.
   Fixed-size mel inputs are created using padding/cropping so CNN input dimensions remain consistent.

6. **Model used for training**
   - `MelSpecCNN` backbone for both tasks.
   - Different classifier heads:
     - 50 classes for ESC-50 SED.
     - 8 classes for RAVDESS emotion.

7. **Training strategy**
   - Train/validation split: 80/20 (stratified when possible).
   - Optimizer: AdamW.
   - Loss: CrossEntropyLoss.
   - Best checkpoint is saved based on validation accuracy.

8. **Evaluation metrics**
   Saved in `outputs/sed_metrics.json` and `outputs/emotion_metrics.json`:
   - Accuracy and balanced accuracy
   - Precision/Recall/F1 (macro, micro, weighted)
   - Per-class report
   - Confusion matrix

9. **Output artifacts**
   - `outputs/sed_cnn.pt`
   - `outputs/emotion_cnn.pt`
   - `outputs/alm_cnn_merged.pt` (merged model bundle)
   - `outputs/alm_cnn_merged_metrics.json`

10. **Current observed results (not final ceiling)**
    - SED (ESC-50): best validation accuracy around `0.4275` (from notebook run).
    - Emotion (RAVDESS): best validation accuracy around `0.5972` (from notebook run).
    These improve further with more tuning and longer training.

---

## Possible questions and sample answers

### Q1) Why did you choose a modular pipeline instead of one end-to-end model?
**Answer:** A modular design is easier to debug, train, and improve independently. We can replace ASR, SED, or emotion models without retraining the full system.

### Q2) Why use ESC-50 instead of full AudioSet for training?
**Answer:** ESC-50 is a manageable benchmark with clean labels and quick experimentation. Full AudioSet is much larger and requires heavier compute/storage.

### Q3) Why do you convert audio to mel spectrograms?
**Answer:** Mel spectrograms provide a compact time-frequency representation that CNNs learn effectively for acoustic patterns.

### Q4) What is the role of `MelSpecCNN` in your project?
**Answer:** It is the shared backbone used to learn acoustic features for both SED and emotion tasks, with separate output heads based on class count.

### Q5) Why use an 80/20 train-validation split?
**Answer:** It gives enough training samples while preserving a reliable holdout set to estimate generalization.

### Q6) Why do you track macro/micro/weighted F1, not only accuracy?
**Answer:** Accuracy alone can hide class imbalance effects. F1 metrics provide better insight into per-class behavior and overall robustness.

### Q7) How do you decide the final saved model?
**Answer:** We save the checkpoint with the best validation accuracy during training.

### Q8) What are key limitations of current results?
**Answer:** Results are dataset-dependent and may drop in real-world noisy conditions. More diverse training data and augmentation can improve robustness.

### Q9) How can this be improved further?
**Answer:** Use longer training, stronger augmentation, hyperparameter tuning, larger backbones, and domain-specific data.

### Q10) What is produced at the end of training?
**Answer:** Separate `.pt` checkpoints for SED and emotion, detailed JSON metric reports, and one merged checkpoint for deployment integration.

---

## Algorithms used in this project

This section lists the core algorithms/methods used across training and inference.

### 1) Audio preprocessing algorithms
- **Resampling** to a fixed sample rate (`16000 Hz` in config) for consistent input.
- **Mono conversion** by channel averaging when input is multi-channel.
- **Log-Mel Spectrogram extraction** (`librosa.feature.melspectrogram` + `power_to_db`).
- **Fixed-length time normalization** using random crop (train), center crop (validation), or zero/min padding.
- **Per-sample normalization** using mean/std scaling.

### 2) CNN training algorithms (custom models)
- **Convolutional Neural Network (CNN)**: `MelSpecCNN` with:
  - Conv2D + BatchNorm + ReLU blocks
  - MaxPooling
  - Adaptive Average Pooling
  - Fully connected classification head + Dropout
- **Optimization algorithm**: `AdamW`
- **Loss function**: `CrossEntropyLoss`
- **Model selection strategy**: Save checkpoint with best validation accuracy.
- **Data split algorithm**: Stratified train/validation split (`train_test_split` with `stratify` where possible).

### 3) Task-specific modeling algorithms
- **SED (Sound Event Detection):**
  - Trained on ESC-50 (AudioSet-style subset), 50-way multiclass classification.
- **Emotion Recognition:**
  - Trained on RAVDESS speech, 8-way multiclass classification.
  - Emotion label is parsed from RAVDESS filename pattern (`03-...` schema).

### 4) Inference pipeline algorithms (ALM-Lite modular)
- **ASR algorithm**: Transformer Whisper (`openai/whisper-small`) for speech-to-text.
- **SED inference algorithm**: Audio Spectrogram Transformer (AST) classifier (`MIT/ast-finetuned-audioset-10-10-0.4593`).
- **Emotion inference algorithm**: Wav2Vec2-based speech emotion classifier (`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`).
- **Context construction algorithm**: Rule-based fusion of transcript + detected sounds + emotion.
- **Reasoning algorithm**: Causal LLM generation (`Qwen/Qwen2-0.5B-Instruct`) using deterministic decoding (`do_sample=False`) with repetition controls (`repetition_penalty`, `no_repeat_ngram_size`).

### 5) Evaluation algorithms/metrics
- **Accuracy**
- **Balanced Accuracy**
- **Precision / Recall / F1** (macro, micro, weighted)
- **Per-class classification report**
- **Confusion matrix**

### 6) Security/auth algorithms (web app)
- **Password hashing**: PBKDF2-HMAC-SHA256 with salt (`310000` iterations).
- **Session management**: Bearer token sessions stored in SQLite with expiry.
"# audio-recognitation" 
