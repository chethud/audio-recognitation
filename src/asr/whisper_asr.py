"""Whisper-based speech-to-text via Hugging Face."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Union

from src.env_setup import configure_ml_env

configure_ml_env()

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import threading

import numpy as np
import torch

_lock = threading.Lock()
_pipe_cache: dict[str, object] = {}


@dataclass
class TranscriptionResult:
    """English transcript for display + original speech language metadata."""

    transcript: str
    transcript_original: str
    language: str


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

        p = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=dev,
            torch_dtype=torch.float16 if dev == 0 else torch.float32,
        )
        _pipe_cache[key] = p
        return p


def _prepare_wav(
    audio: Union[torch.Tensor, np.ndarray],
    sample_rate: int,
    max_duration_sec: Optional[float],
) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        wav = audio.detach().float().cpu().numpy().reshape(-1)
    else:
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)

    if max_duration_sec and max_duration_sec > 0:
        cap = int(max_duration_sec * sample_rate)
        if len(wav) > cap:
            wav = wav[:cap]
    return wav


def _whisper_language_code(pipe, lang_id: int) -> str:
    tokenizer = pipe.tokenizer
    try:
        from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

        token = tokenizer.convert_ids_to_tokens([lang_id])[0]
        if token.startswith("<|") and token.endswith("|>"):
            name = token[2:-2]
            if name == "english":
                return "en"
            return TO_LANGUAGE_CODE.get(name, name)
    except Exception:
        pass
    return "en"


def _detect_language(pipe, wav: np.ndarray, sample_rate: int) -> str:
    try:
        model = pipe.model
        fe = getattr(pipe, "feature_extractor", None) or getattr(pipe, "processor", None)
        if fe is None:
            return "en"
        if hasattr(fe, "feature_extractor"):
            fe = fe.feature_extractor

        inputs = fe([wav], sampling_rate=sample_rate, return_tensors="pt", padding=True)
        input_features = inputs.input_features
        if hasattr(model, "device"):
            input_features = input_features.to(model.device)
        dtype = getattr(model, "dtype", torch.float32)
        input_features = input_features.to(dtype=dtype)

        with torch.no_grad():
            lang_ids = model.detect_language(input_features)
            lang_id = int(lang_ids[0, 0].item())
        return _whisper_language_code(pipe, lang_id)
    except Exception:
        return "en"


def _run_whisper(
    pipe,
    wav: np.ndarray,
    sample_rate: int,
    *,
    task: str,
    language: Optional[str] = None,
) -> str:
    duration_s = len(wav) / max(sample_rate, 1)
    pipe_kwargs: dict = {}
    if duration_s > 30:
        pipe_kwargs = {"chunk_length_s": 30, "batch_size": 4, "stride_length_s": 5}

    gen_kwargs: dict = {
        "task": task,
        "condition_on_prev_tokens": False,
        "num_beams": 1,
    }
    if language:
        gen_kwargs["language"] = language

    out = pipe(
        {"array": wav, "sampling_rate": sample_rate},
        generate_kwargs=gen_kwargs,
        **pipe_kwargs,
    )
    text = out.get("text", "") if isinstance(out, dict) else str(out)
    return (text or "").strip()


def transcribe_bilingual(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    model_id: str = "openai/whisper-tiny",
    device: Optional[Union[str, torch.device]] = None,
    max_duration_sec: Optional[float] = None,
) -> TranscriptionResult:
    """
    Auto-detect spoken language, return English transcript + original-language text.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav = _prepare_wav(audio, sample_rate, max_duration_sec)
    pipe = _get_asr_pipe(model_id, device)

    try:
        language = _detect_language(pipe, wav, sample_rate)
        transcript_original = _run_whisper(
            pipe,
            wav,
            sample_rate,
            task="transcribe",
            language=language if language != "en" else None,
        )

        if language == "en":
            transcript_en = transcript_original
        else:
            transcript_en = _run_whisper(
                pipe,
                wav,
                sample_rate,
                task="translate",
            )

        if not transcript_en and transcript_original:
            transcript_en = transcript_original
        if not transcript_original and transcript_en:
            transcript_original = transcript_en
            language = "en"

        return TranscriptionResult(
            transcript=transcript_en,
            transcript_original=transcript_original,
            language=language or "en",
        )
    except Exception as e:
        err = f"[ASR error: {e}]"
        return TranscriptionResult(
            transcript=err,
            transcript_original=err,
            language="en",
        )


def transcribe_audio(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    model_id: str = "openai/whisper-tiny",
    device: Optional[Union[str, torch.device]] = None,
    language: Optional[str] = None,
    max_duration_sec: Optional[float] = None,
) -> str:
    """Legacy single-language transcription (English when ``language='en'``)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav = _prepare_wav(audio, sample_rate, max_duration_sec)
    pipe = _get_asr_pipe(model_id, device)

    try:
        task = "translate" if language == "en" else "transcribe"
        return _run_whisper(
            pipe,
            wav,
            sample_rate,
            task=task,
            language=language,
        )
    except Exception as e:
        return f"[ASR error: {e}]"
