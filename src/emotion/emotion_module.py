"""
Speech emotion from audio using local CNN (preferred) or HuggingFace wav2vec2.
Falls back only when models fail or audio is too short — never skips analysis when enabled.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional, Union

import numpy as np
import torch

from src.env_setup import configure_ml_env

configure_ml_env()

logger = logging.getLogger(__name__)

_pipeline_lock = threading.Lock()
_pipeline_cache: dict[str, object] = {}


def _get_emotion_pipeline(model_id: str, device: torch.device):
    global _pipeline_cache
    key = f"{model_id}|{device.type}"
    if key in _pipeline_cache:
        return _pipeline_cache[key]
    with _pipeline_lock:
        if key in _pipeline_cache:
            return _pipeline_cache[key]
        from transformers import pipeline

        dev = 0 if device.type == "cuda" else -1
        p = pipeline("audio-classification", model=model_id, device=dev)
        _pipeline_cache[key] = p
        return p


def _prefer_cnn(backend: str) -> bool:
    from src.cnn.loader import should_use_cnn

    b = (backend or "auto").lower()
    if b == "cnn":
        return True
    if b in ("hf", "wav2vec2", "huggingface"):
        return False
    return should_use_cnn()


def _cnn_available() -> bool:
    from src.cnn.loader import _load_emo_bundle

    return _load_emo_bundle() is not None


def _predict_hf(
    wav: np.ndarray,
    *,
    sample_rate: int,
    model_id: str,
    device: torch.device,
) -> Optional[str]:
    try:
        pipe = _get_emotion_pipeline(model_id, device)
        out = pipe({"raw": wav, "sampling_rate": sample_rate}, top_k=1)
        if out and isinstance(out, list) and out[0].get("label"):
            return str(out[0]["label"]).lower().strip()
    except Exception as exc:
        logger.warning("HF emotion inference failed: %s", exc)
    return None


def predict_emotion_from_audio(
    audio: Union[np.ndarray, torch.Tensor],
    *,
    sample_rate: int = 16000,
    model_id: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    device: Optional[Union[str, torch.device]] = None,
    min_samples: int = 4000,
    enabled: bool = True,
    backend: str = "auto",
) -> str:
    """
    Return a single emotion label (e.g. angry, happy, sad, neutral, calm, …).

    When ``enabled``, always runs a model (CNN and/or HF). Returns ``neutral``
    only if audio is too short or every backend fails.
    """
    if not enabled:
        return "neutral"

    if isinstance(audio, torch.Tensor):
        wav = audio.detach().float().cpu().numpy().reshape(-1)
    else:
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)

    if wav.size < min_samples:
        logger.info("Emotion skipped: audio too short (%d samples)", wav.size)
        return "neutral"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    # 1) CNN when preferred / available
    if _prefer_cnn(backend) and _cnn_available():
        from src.cnn.inference import predict_emotion_cnn

        label = predict_emotion_cnn(
            wav,
            sample_rate=sample_rate,
            min_samples=min_samples,
        )
        if label:
            logger.info("Emotion (CNN): %s", label)
            return label

    # 2) HuggingFace wav2vec2 classifier
    if (backend or "auto").lower() != "cnn" or not _cnn_available():
        label = _predict_hf(wav, sample_rate=sample_rate, model_id=model_id, device=device)
        if label:
            logger.info("Emotion (HF): %s", label)
            return label

    # 3) Last resort: try the other backend once
    if not _prefer_cnn(backend) and _cnn_available():
        from src.cnn.inference import predict_emotion_cnn

        label = predict_emotion_cnn(
            wav,
            sample_rate=sample_rate,
            min_samples=min_samples,
        )
        if label:
            logger.info("Emotion (CNN fallback): %s", label)
            return label

    if _prefer_cnn(backend):
        label = _predict_hf(wav, sample_rate=sample_rate, model_id=model_id, device=device)
        if label:
            logger.info("Emotion (HF fallback): %s", label)
            return label

    logger.warning("Emotion analysis failed; returning neutral")
    return "neutral"


def predict_emotions_per_speaker(
    audio: Union[np.ndarray, torch.Tensor],
    speaker_turns: list,
    *,
    sample_rate: int = 16000,
    model_id: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    device: Optional[Union[str, torch.device]] = None,
    enabled: bool = True,
    backend: str = "auto",
    min_sec: float = 0.35,
    max_sec_per_speaker: float = 12.0,
) -> dict[str, str]:
    """Run emotion on each speaker's concatenated speech slices."""
    if not enabled:
        return {}

    if isinstance(audio, torch.Tensor):
        wav = audio.detach().float().cpu().numpy().reshape(-1)
    else:
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)

    buckets: dict[str, list[np.ndarray]] = {}
    for turn in speaker_turns or []:
        if isinstance(turn, dict):
            sp = str(turn.get("speaker") or turn.get("speaker_id") or "").strip()
            start = float(turn.get("start") or turn.get("start_sec") or 0.0)
            end = float(turn.get("end") or turn.get("end_sec") or 0.0)
        else:
            sp = str(getattr(turn, "speaker", "") or "").strip()
            start = float(getattr(turn, "start_sec", 0.0) or 0.0)
            end = float(getattr(turn, "end_sec", 0.0) or 0.0)
        if not sp or end <= start:
            continue
        i0 = max(0, int(start * sample_rate))
        i1 = min(len(wav), int(end * sample_rate))
        if i1 - i0 < int(min_sec * sample_rate):
            continue
        buckets.setdefault(sp, []).append(wav[i0:i1])

    max_samples = int(max_sec_per_speaker * sample_rate)
    out: dict[str, str] = {}
    for sp, chunks in buckets.items():
        piece = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if piece.size > max_samples:
            piece = piece[:max_samples]
        out[sp] = predict_emotion_from_audio(
            piece,
            sample_rate=sample_rate,
            model_id=model_id,
            device=device,
            enabled=True,
            backend=backend,
        )
    return out
