"""Environmental sound events via AST (audio-classification pipeline)."""
from __future__ import annotations

import threading
from typing import List, Optional, Union

import numpy as np
import torch

import logging

from src.env_setup import configure_ml_env

configure_ml_env()

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pipe_cache: dict[str, object] = {}
_hf_sed_unavailable = False


def _hf_fallback_allowed(backend: str) -> bool:
    """Use HuggingFace AST only when explicitly requested and CNN is not in use."""
    from src.cnn.loader import cnn_checkpoints_exist

    b = backend.lower()
    if b == "cnn":
        return False
    if b in ("hf", "ast", "huggingface"):
        return True
    # auto: prefer CNN checkpoints; avoid broken HF load on Python 3.14+
    return not cnn_checkpoints_exist()


def _get_sed_pipe(model_id: str, device: Union[str, torch.device, int]):
    global _hf_sed_unavailable
    if _hf_sed_unavailable:
        raise RuntimeError("HuggingFace SED (AST) is unavailable on this system")

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

        try:
            p = pipeline("audio-classification", model=model_id, device=dev)
        except Exception as exc:
            _hf_sed_unavailable = True
            raise RuntimeError(
                f"Could not load HuggingFace SED model {model_id}: {exc}"
            ) from exc
        _pipe_cache[key] = p
        return p


def _to_mono_wav(
    audio: Union[torch.Tensor, np.ndarray], sample_rate: int
) -> tuple[np.ndarray, int]:
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
    return wav, sample_rate


def _classify_chunk(
    pipe,
    wav: np.ndarray,
    sample_rate: int,
    top_k: int,
) -> List[dict]:
    try:
        result = pipe({"array": wav, "sampling_rate": sample_rate}, top_k=top_k)
    except TypeError:
        result = pipe({"raw": wav, "sampling_rate": sample_rate}, top_k=top_k)

    out: List[dict] = []
    for item in result:
        out.append({"label": str(item["label"]), "score": float(item["score"])})
    return out


def _use_cnn_backend(backend: str) -> bool:
    from src.cnn.loader import should_use_cnn

    b = backend.lower()
    if b == "cnn":
        return True
    if b in ("hf", "ast", "huggingface"):
        return False
    return should_use_cnn()


def detect_sound_events(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    device: Optional[Union[str, torch.device]] = None,
    top_k: int = 10,
    threshold: float = 0.2,
    backend: str = "auto",
) -> List[dict]:
    if _use_cnn_backend(backend):
        from src.cnn.inference import predict_sed_cnn

        events = predict_sed_cnn(
            audio,
            sample_rate=sample_rate,
            top_k=top_k,
            threshold=threshold,
        )
        if events or backend == "cnn":
            return events

    if not _hf_fallback_allowed(backend):
        return []

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        wav, sample_rate = _to_mono_wav(audio, sample_rate)
        pipe = _get_sed_pipe(model_id, device)
        merged = _classify_chunk(pipe, wav, sample_rate, top_k)
    except Exception as exc:
        logger.warning("SED skipped (HF model unavailable): %s", exc)
        return []

    return [
        {"label": e["label"], "score": round(e["score"], 4)}
        for e in merged
        if e["score"] >= threshold
    ]


def detect_sound_events_segmented(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    device: Optional[Union[str, torch.device]] = None,
    top_k: int = 8,
    threshold: float = 0.12,
    segment_sec: float = 2.0,
    max_windows: int = 2,
    backend: str = "auto",
) -> List[dict]:
    """
    Fast multi-sound scan: full clip plus at most one extra window (not every 2s).
    """
    if _use_cnn_backend(backend):
        from src.cnn.inference import predict_sed_cnn

        events = predict_sed_cnn(
            audio,
            sample_rate=sample_rate,
            top_k=top_k,
            threshold=threshold,
            segment_sec=segment_sec,
            max_windows=max_windows,
        )
        if events or backend == "cnn":
            return events

    if not _hf_fallback_allowed(backend):
        return []

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        wav, sample_rate = _to_mono_wav(audio, sample_rate)
        pipe = _get_sed_pipe(model_id, device)
    except Exception as exc:
        logger.warning("SED skipped (HF model unavailable): %s", exc)
        return []

    best: dict[str, float] = {}

    def _merge(chunk: np.ndarray) -> None:
        if len(chunk) < sample_rate // 4:
            return
        for item in _classify_chunk(pipe, chunk, sample_rate, top_k):
            label = item["label"]
            best[label] = max(best.get(label, 0.0), item["score"])

    _merge(wav)

    if max_windows > 1 and len(wav) > int(3 * sample_rate):
        mid = len(wav) // 2
        half = max(int(segment_sec * sample_rate), sample_rate)
        _merge(wav[max(0, mid - half // 2) : min(len(wav), mid + half // 2)])

    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return [
        {"label": label, "score": round(score, 4)}
        for label, score in ranked
        if score >= threshold
    ]
