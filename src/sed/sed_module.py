"""Environmental sound events via AST (audio-classification pipeline)."""
from __future__ import annotations

import threading
from typing import List, Optional, Union

import numpy as np
import torch

from src.env_setup import configure_ml_env

configure_ml_env()

_lock = threading.Lock()
_pipe_cache: dict[str, object] = {}


def _get_sed_pipe(model_id: str, device: Union[str, torch.device, int]):
    if isinstance(device, str):
        dev = 0 if device == "cuda" else -1
    elif isinstance(device, torch.device):
        dev = 0 if device.type == "cuda" else -1
    elif device == 0:
        dev = 0
    else:
        dev = -1
    key = f"{model_id}|{dev}"
    with _lock:
        if key in _pipe_cache:
            return _pipe_cache[key]
        from transformers import pipeline

        p = pipeline("audio-classification", model=model_id, device=dev)
        _pipe_cache[key] = p
        return p


def detect_sound_events(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    device: Optional[Union[str, torch.device]] = None,
    top_k: int = 10,
    threshold: float = 0.2,
) -> List[dict]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        pass  # pipeline device index below
    elif isinstance(device, torch.device):
        device = "cuda" if device.type == "cuda" else "cpu"

    if isinstance(audio, torch.Tensor):
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        wav = audio.cpu().float().numpy()
    else:
        wav = np.asarray(audio, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.squeeze(0)

    if sample_rate != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    pipe = _get_sed_pipe(model_id, device)
    try:
        result = pipe({"array": wav, "sampling_rate": sample_rate}, top_k=top_k)
    except TypeError:
        result = pipe({"raw": wav, "sampling_rate": sample_rate}, top_k=top_k)

    out: List[dict] = []
    for item in result:
        score = float(item["score"])
        if score >= threshold:
            out.append({"label": str(item["label"]), "score": round(score, 4)})
    return out
