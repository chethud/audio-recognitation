# Audio Language Model (ALM-Lite) - Technical Specifications & Algorithms

This document provides a comprehensive technical breakdown of the algorithms, models, database schemas, and workflows used in the **ALM-Lite** (Audio Language Model) project.

---

## 1. High-Level Architecture

The project is built as a split-architecture system:
* **Frontend:** Built with React 18, Vite, and Tailwind CSS. It communicates with the backend via Axios and displays a glassmorphic dashboard showing a file upload preview, a real-time progress bar, and tabulated results.
* **Backend:** Built with FastAPI, Uvicorn, and SQLite. The backend supports user authentication (sessions + bearer tokens) and processes incoming audio files.
* **ML Inference Worker:** Subprocess-isolated on Windows using `inference_worker_modular.py` to prevent native model errors (e.g., PyTorch segment violations) from crashing the FastAPI server.

```
┌─────────────────────────────────┐
│          React UI (Vite)        │
└────────────────┬────────────────┘
                 │ POST /analyze
                 ▼
┌─────────────────────────────────┐
│        FastAPI Backend          │
│       (backend/main.py)         │
└────────────────┬────────────────┘
                 │ Subprocess Spawn (isolated)
                 ▼
┌─────────────────────────────────┐
│    inference_worker_modular.py  │
└────────────────┬────────────────┘
                 ├─► Speech-to-Text (Whisper ASR)
                 ├─► Speaker Diarization (PyAnnote / MFCC Clustering)
                 ├─► Sound Event Detection (AST / Custom CNN)
                 ├─► Emotion Recognition (Wav2Vec2 / Custom CNN)
                 ▼
┌─────────────────────────────────┐
│        Context Builder          │
└────────────────┬────────────────┘
                 │ Compiles: Transcript + Sounds + Emotion
                 ▼
┌─────────────────────────────────┐
│         Qwen2 LLM Head          │
│    (Generates final answer)     │
└────────────────┬────────────────┘
                 │ Response JSON
                 ▼
┌─────────────────────────────────┐
│         FastAPI Backend          │
└────────────────┬────────────────┘
                 │ Save history (SQLite async)
                 ▼
┌─────────────────────────────────┐
│         SQLite database         │
└─────────────────────────────────┘
```

---

## 2. Audio Processing Pipeline

Before feeding audio into any of the ML models, the raw file must be processed:
* **Decoding & Resampling:** Raw audio uploads (`.mp3`, `.wav`, `.m4a`, etc.) are decoded using `librosa.load` and resampled to a strict target rate of **16,000 Hz**.
* **Mono Conversion:** Multi-channel files are averaged down to 1-channel (mono) to ensure feature consistency across models.
* **Duration Limiting:** The backend decodes only the first $N$ seconds (configured in `config.yaml` as `data.max_audio_length_sec`, default 12s) to speed up CPU inference.

---

## 3. Speech-to-Text (ASR) Algorithms

Speech-to-Text converts voice signals into text strings.

### 3.1 Models Used
* **Default:** `openai/whisper-tiny` or `openai/whisper-base` via Hugging Face.
* **Kannada Fine-Tuned (Automatic Override):** If the spoken language is detected as or configured as Kannada (`kn`), the model swaps to `vasista22/whisper-kannada-base` (which has specialized decoder prompt scripts to force native Kannada script output instead of romanized characters).
* **CTranslate2 (Optional):** Supports faster-whisper execution via CTranslate2 serialization to improve performance under CPU constraints.

### 3.2 Post-Processing & Text Cleanup
* **Repeated Chunk Deduplication:** Whisper occasionally suffers from hallucination loops (especially on short or noisy clips). The system uses word-level sliding window comparisons (`_dedupe_whisper_chunks`) to detect and collapse duplicate overlapping sentences.
* **Script Mismatch Filtering:** If the user specifies English only, but Whisper outputs Indic/CJK scripts, the system filters out the script hallucinations.
* **Indic Bias Mitigation:** Forces the Whisper tokenizer language tokens while stripping hardcoded prompt tokens that could cause identical short transcripts across unrelated uploads.

---

## 4. Speaker Diarization (Who Spoke When)

Speaker diarization splits the transcript into segments annotated with speaker IDs (e.g., `Speaker 1`, `Speaker 2`).

### 4.1 PyAnnote Backend (Primary)
If a Hugging Face token is provided and the `pyannote/speaker-diarization-3.1` model is loaded, diarization runs using PyAnnote's neural voice activity detection and clustering.

### 4.2 Custom VAD + MFCC Clustering Backend (Fallback)
If PyAnnote is unavailable, a custom pipeline is executed:
1. **Energy VAD:** Detects speech windows using Root Mean Square (RMS) frame energy.
   $$\text{RMS} = \sqrt{\frac{1}{M}\sum_{i=1}^M y_i^2}$$
   Windows are merged or split using speech duration thresholds (`vad_min_speech_sec = 0.3` and `vad_min_silence_sec = 0.35`).
2. **MFCC Feature Extraction:** Generates 20 Mel-Frequency Cepstral Coefficients (MFCCs) plus their delta and delta-delta coefficients (43 features total) for sliding windows within the speech segments using `librosa.feature.mfcc`.
3. **Clustering:** Normalizes the feature vectors and clusters them using **Agglomerative Hierarchical Clustering** with Cosine Distance and Average Linkage.
4. **Speaker Ratio Validation:** Uses the **Silhouette Score** ($s$) to determine if the multi-speaker split is statistically strong:
   $$s = \frac{b - a}{\max(a, b)}$$
   * *Threshold Tuning:* Lowered to `sil_accept = 0.08`, `minority_accept = 0.08`, and `minority_keep = 0.06` to prevent collapsing dialogues with a dominant speaker (e.g. interviews) into a single speaker.
5. **Transcribe-Per-Segment:** Whisper transcribes the exact audio crops belonging to each cluster and associates them with the clustered speaker labels.

---

## 5. Sound Event Detection (SED)

Sound Event Detection extracts non-speech labels to identify background events.

### 5.1 Pretrained Audio Spectrogram Transformer (AST)
* **Model:** `MIT/ast-finetuned-audioset-10-10-0.4593`.
* **Algorithm:** Processes audio Mel Spectrograms using a Vision Transformer (ViT) architecture pre-trained on AudioSet's 527 environmental classes.

### 5.2 Custom PyTorch MelSpecCNN Architecture (Offline Training)
The project contains a custom Convolutional Neural Network trained on the **ESC-50** dataset (50 classes):

* **Input:** $[B, 1, 64, 128]$ representing log-mel spectrogram features.
* **Feature Extractor Blocks (3 Layers):**
  - **Conv Layer 1:** 32 filters ($3\times3$), Batch Normalization, ReLU, MaxPool2d ($2\times2$).
  - **Conv Layer 2:** 64 filters ($3\times3$), Batch Normalization, ReLU, MaxPool2d ($2\times2$).
  - **Conv Layer 3:** 128 filters ($3\times3$), Batch Normalization, ReLU.
  - **Global Pooling:** AdaptiveAvgPool2d to compress feature maps down to a fixed $(4, 8)$ grid.
* **Classification Head:**
  - Flattening ($128 \times 4 \times 8 = 4096$ units).
  - Linear Layer (4096 $\rightarrow$ 256), ReLU, Dropout (0.3).
  - Linear Layer (256 $\rightarrow$ 50 classes).

```
[Input log-mel] ────► [Conv1 + BN + ReLU + MaxPool] ────► [Conv2 + BN + ReLU + MaxPool]
                                                                  │
[Linear Head] ◄──── [Flatten] ◄──── [AdaptiveAvgPool] ◄──── [Conv3 + BN + ReLU]
```

---

## 6. Speech Emotion Recognition (SER)

SER evaluates the emotional state of the speakers.

### 6.1 Pretrained Model
* **Model:** `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`.
* **Algorithm:** Fine-tuned Wav2Vec2 transformer that outputs probabilities for 8 primary emotions (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised).

### 6.2 Custom PyTorch Emotion CNN (Offline Training)
Trained on the **RAVDESS** dataset (8 classes), it shares the same `MelSpecCNN` structural block layout but is initialized with a classification head outputting 8 logits.

---

## 7. Context Builder & LLM Reasoning Head

Once speech, environmental sounds, and speaker emotions are compiled, they must be formatted for reasoning.

### 7.1 Context Builder Template
The modular pipeline formats outputs into a structural prompt text:
```
Speech (transcript): "I arrived here in Chicago in 1985..."
Speaker-separated transcript:
  Speaker 1: I do it. You?
  Speaker 2: Entering the city through...
Speaker emotion (estimated): neutral
Non-speech (environmental sounds): church bells, airplane, siren
```

### 7.2 Reasoning Head (Qwen2)
* **Model:** `Qwen/Qwen2-0.5B-Instruct` (causal language model).
* **Generation Settings:**
  - Temperature: `0.0` (deterministic outputs).
  - Repetition Penalty: `1.1`.
  - Max New Tokens: `32`.
  - Prompt:
    ```
    [Language Instruction] Answer briefly from this audio context.
    Context:
    {context}

    Q: {question}
    A:
    ```

---

## 8. Database (SQLite) Schema

Data persistence is managed via an asynchronous SQLite engine using Write-Ahead Logging (`WAL`) mode to prevent database locks on simultaneous requests.

### 8.1 Schema Tables
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

## 9. Model Training & Merging Workflow

You can retrain the custom SED and SER networks using the `training` modules:

1. **Feature Alignment Constraints:** Resamples all training audio to **16,000 Hz** (via `librosa.resample`) during dataset loading (`ESC50MelDataset` in `train_sed.py` and `Esc50FileMelDataset` in `esc50_files.py`). This prevents frequency stretching and time dilation artifacts during inference.
2. **Loss Function:** PyTorch `CrossEntropyLoss` with the Adam optimizer:
   - Learning Rate (SED): `1e-3` (with CosineAnnealingLR scheduler).
   - Learning Rate (SER): `5e-4`.
3. **Merging Checkpoints:** `merge_cnn_models.py` bundles the state dictionaries of both trained models into a single file `outputs/alm_cnn_merged.pt` to simplify backend loading:
   ```python
   torch.save({
       "sed": {
           "model": sed_model.state_dict(),
           "class_names": sed_classes,
           "n_mels": n_mels,
           "time_frames": time_frames
       },
       "emotion": {
           "model": emo_model.state_dict(),
           "class_names": emo_classes,
           "n_mels": n_mels,
           "time_frames": time_frames
       }
   }, "outputs/alm_cnn_merged.pt")
   ```
