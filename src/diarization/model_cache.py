"""Singleton cache for WhisperX and PyAnnote diarization models."""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_whisperx_model: Any = None
_whisperx_device: str = "cpu"
_whisperx_compute_type: str = "int8"
_whisperx_model_size: str = ""
_align_models: dict[str, tuple[Any, Any]] = {}
_diarize_pipeline: Any = None
_hf_token: str | None = None


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_compute_type(device: str) -> str:
    if device == "cuda":
        return "float16"
    return "int8"


def hf_auth_token() -> str | None:
    """Read Hugging Face token for gated PyAnnote models."""
    global _hf_token
    if _hf_token is not None:
        return _hf_token or None
    _hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("PYANNOTE_TOKEN")
        or ""
    )
    return _hf_token or None


def whisper_size_from_model_id(model_id: str) -> str:
    """Map HuggingFace Whisper id to WhisperX size name."""
    mid = (model_id or "").lower()
    for size in ("large-v3", "large-v2", "large", "medium", "small", "base", "tiny"):
        if size in mid:
            return size
    return "base"


def get_whisperx_model(model_size: str = "base") -> tuple[Any, str, str]:
    """Load (or return cached) WhisperX ASR model."""
    global _whisperx_model, _whisperx_device, _whisperx_compute_type, _whisperx_model_size

    device = _resolve_device()
    compute_type = _resolve_compute_type(device)

    with _lock:
        if (
            _whisperx_model is not None
            and _whisperx_model_size == model_size
            and _whisperx_device == device
        ):
            return _whisperx_model, device, compute_type

        import whisperx

        logger.info(
            "Loading WhisperX model '%s' on %s (compute_type=%s)",
            model_size,
            device,
            compute_type,
        )
        model = whisperx.load_model(
            model_size,
            device,
            compute_type=compute_type,
        )
        _whisperx_model = model
        _whisperx_device = device
        _whisperx_compute_type = compute_type
        _whisperx_model_size = model_size
        logger.info("WhisperX ASR model ready (%s)", model_size)
        return model, device, compute_type


def get_align_model(language_code: str) -> tuple[Any, Any, str]:
    """Load (or return cached) wav2vec2 alignment model for a language."""
    lang = (language_code or "en").split("-")[0].lower()
    device = _resolve_device()

    with _lock:
        if lang in _align_models:
            model_a, metadata = _align_models[lang]
            return model_a, metadata, device

        import whisperx

        logger.info("Loading WhisperX align model for language '%s'", lang)
        model_a, metadata = whisperx.load_align_model(
            language_code=lang,
            device=device,
        )
        _align_models[lang] = (model_a, metadata)
        logger.info("WhisperX align model ready for '%s'", lang)
        return model_a, metadata, device


def get_diarization_pipeline() -> tuple[Any, str]:
    """Load (or return cached) PyAnnote diarization pipeline via WhisperX."""
    global _diarize_pipeline

    device = _resolve_device()
    token = hf_auth_token()

    with _lock:
        if _diarize_pipeline is not None:
            return _diarize_pipeline, device

        if not token:
            raise RuntimeError(
                "HF_TOKEN (or HUGGINGFACE_TOKEN) is required for PyAnnote diarization. "
                "Accept pyannote/speaker-diarization-3.1 on Hugging Face and set the token."
            )

        from whisperx.diarize import DiarizationPipeline

        logger.info("Loading PyAnnote diarization pipeline on %s", device)
        _diarize_pipeline = DiarizationPipeline(
            use_auth_token=token,
            device=torch.device(device),
        )
        logger.info("PyAnnote diarization pipeline ready")
        return _diarize_pipeline, device


def warmup_whisperx_models(model_size: str = "base") -> bool:
    """Pre-load WhisperX ASR model. PyAnnote loads on first diarize call."""
    try:
        get_whisperx_model(model_size)
        return True
    except Exception as exc:
        logger.warning("WhisperX warmup failed: %s", exc)
        return False


def release_gpu_memory() -> None:
    """Free CUDA cache after long diarization runs."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
