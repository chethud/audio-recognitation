"""Whisper-based speech-to-text via Hugging Face."""
from __future__ import annotations

import os

# Before any `transformers` import (notebooks / scripts that skip the worker).
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import threading
from typing import Optional, Union

import numpy as np
import torch

_lock = threading.Lock()
_pipe_cache: dict[str, object] = {}


def _get_asr_pipe(model_id: str, device: Union[str, torch.device]):
    if isinstance(device, torch.device):
        dev = 0 if device.type == "cuda" else -1
    else:
        dev = 0 if device == "cuda" else -1
    key = f"{model_id}|{dev}"
    with _lock:
        if key in _pipe_cache:
            return _pipe_cache[key]
        from transformers import pipeline
        p = pipeline("automatic-speech-recognition", model=model_id, device=dev)
        _pipe_cache[key] = p
        return p


def transcribe_audio(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    model_id: str = "openai/whisper-small",
    device: Optional[Union[str, torch.device]] = None,
    language: Optional[str] = None,
) -> str:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(audio, torch.Tensor):
        wav = audio.detach().float().cpu().numpy().reshape(-1)
    else:
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)

    pipe = _get_asr_pipe(model_id, device)
    kwargs = {}
    if language:
        kwargs["generate_kwargs"] = {"language": language}
    try:
        out = pipe({"array": wav, "sampling_rate": sample_rate}, **kwargs)
        text = out.get("text", "") if isinstance(out, dict) else str(out)
        return (text or "").strip()
    except Exception as e:
        return f"[ASR error: {e}]"
