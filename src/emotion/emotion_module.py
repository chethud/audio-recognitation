"""
Speech emotion from audio using a pretrained HuggingFace model (e.g. wav2vec2 + classifier).
Falls back to "neutral" if inference fails or audio is too short.
"""
from __future__ import annotations

import threading
from typing import Optional, Union

import numpy as np
import torch

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


def predict_emotion_from_audio(
    audio: Union[np.ndarray, torch.Tensor],
    *,
    sample_rate: int = 16000,
    model_id: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    device: Optional[Union[str, torch.device]] = None,
    min_samples: int = 4000,
    enabled: bool = True,
) -> str:
    """
    Return a single emotion label string (model-dependent, e.g. angry, happy, neutral).

    Args:
        audio: 1D waveform.
        sample_rate: Input sample rate.
        model_id: HuggingFace model id for audio emotion classification.
        device: torch device; None = auto.
        min_samples: Below this length, return neutral without running the model.
        enabled: If False, skip model and return neutral.
    """
    if not enabled:
        return "neutral"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(audio, torch.Tensor):
        wav = audio.detach().float().cpu().numpy().reshape(-1)
    else:
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)

    if wav.size < min_samples:
        return "neutral"

    try:
        pipe = _get_emotion_pipeline(model_id, device)
        out = pipe({"raw": wav, "sampling_rate": sample_rate}, top_k=1)
        if out and isinstance(out, list) and out[0].get("label"):
            return str(out[0]["label"]).lower().strip()
    except Exception:
        pass

    return "neutral"
