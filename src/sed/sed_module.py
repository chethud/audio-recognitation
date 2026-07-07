"""Environmental sound events via CNN (ESC-50) and/or AST (AudioSet)."""
from __future__ import annotations

import logging
import threading
from typing import List, Optional, Union

import numpy as np
import torch

from src.env_setup import configure_ml_env
from src.sed.labels import merge_sound_events, should_skip_sound_label
from src.sed.windows import sliding_audio_windows

configure_ml_env()

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pipe_cache: dict[str, object] = {}
_hf_sed_unavailable = False


def _backend_uses_cnn(backend: str) -> bool:
    from src.cnn.loader import should_use_cnn

    b = backend.lower()
    if b in ("cnn", "hybrid", "both"):
        return True
    if b in ("hf", "ast", "huggingface"):
        return False
    return should_use_cnn()


def _backend_uses_ast(backend: str) -> bool:
    from src.cnn.loader import cnn_checkpoints_exist

    b = backend.lower()
    if b in ("hf", "ast", "huggingface"):
        return True
    if b == "cnn":
        return False
    if b in ("hybrid", "both"):
        return True
    # auto: AST when no CNN checkpoints, or hybrid when checkpoints exist
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
        label = str(item["label"])
        if should_skip_sound_label(label):
            continue
        out.append({"label": label, "score": float(item["score"])})
    return out


def _run_cnn_sed(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int,
    top_k: int,
    threshold: float,
    segment_sec: float,
    max_windows: int,
) -> List[dict]:
    from src.cnn.inference import predict_sed_cnn

    return predict_sed_cnn(
        audio,
        sample_rate=sample_rate,
        top_k=top_k,
        threshold=threshold,
        segment_sec=segment_sec,
        max_windows=max_windows,
    )


def _run_ast_sed(
    wav: np.ndarray,
    sample_rate: int,
    *,
    model_id: str,
    device: Union[str, torch.device],
    top_k: int,
    threshold: float,
    segment_sec: float,
    max_windows: int,
) -> List[dict]:
    pipe = _get_sed_pipe(model_id, device)
    best: dict[str, float] = {}

    for chunk in sliding_audio_windows(
        wav,
        sample_rate,
        segment_sec=segment_sec,
        max_windows=max_windows,
    ):
        if len(chunk) < sample_rate // 4:
            continue
        for item in _classify_chunk(pipe, chunk, sample_rate, top_k):
            label = item["label"]
            best[label] = max(best.get(label, 0.0), item["score"])

    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return [
        {"label": label, "score": round(score, 4)}
        for label, score in ranked
        if score >= threshold
    ]


def detect_sound_events(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    device: Optional[Union[str, torch.device]] = None,
    top_k: int = 10,
    threshold: float = 0.2,
    segment_sec: float = 2.5,
    max_windows: int = 12,
    backend: str = "auto",
    max_results: int = 12,
) -> List[dict]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    wav, sample_rate = _to_mono_wav(audio, sample_rate)
    cnn_events: List[dict] = []
    ast_events: List[dict] = []

    if _backend_uses_cnn(backend):
        cnn_events = _run_cnn_sed(
            wav,
            sample_rate=sample_rate,
            top_k=top_k,
            threshold=threshold,
            segment_sec=segment_sec,
            max_windows=max_windows,
        )

    if _backend_uses_ast(backend):
        try:
            ast_events = _run_ast_sed(
                wav,
                sample_rate,
                model_id=model_id,
                device=device,
                top_k=top_k,
                threshold=threshold,
                segment_sec=segment_sec,
                max_windows=max_windows,
            )
        except Exception as exc:
            logger.warning("AST SED skipped: %s", exc)

    if cnn_events or ast_events:
        return merge_sound_events(
            cnn_events,
            ast_events,
            max_results=max_results,
            min_score=0.0,
        )

    if backend.lower() == "cnn":
        return cnn_events
    return ast_events


def detect_sound_events_segmented(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    device: Optional[Union[str, torch.device]] = None,
    top_k: int = 10,
    threshold: float = 0.05,
    segment_sec: float = 2.5,
    max_windows: int = 12,
    backend: str = "auto",
    max_results: int = 12,
) -> List[dict]:
    """
    Scan the full clip with evenly spaced windows (CNN + AST in hybrid mode).
    """
    return detect_sound_events(
        audio,
        sample_rate=sample_rate,
        model_id=model_id,
        device=device,
        top_k=top_k,
        threshold=threshold,
        segment_sec=segment_sec,
        max_windows=max_windows,
        backend=backend,
        max_results=max_results,
    )
