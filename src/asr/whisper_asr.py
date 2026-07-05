"""Whisper-based speech-to-text via Hugging Face."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Union

from src.asr.whisper_languages import (
    language_label,
    token_to_language_code,
    whisper_language_code,
    whisper_language_name,
)
from src.asr.text_cleanup import clean_asr_text, is_likely_english_text
from src.env_setup import configure_ml_env

configure_ml_env()

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import threading

import numpy as np
import torch

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pipe_cache: dict[str, object] = {}


@dataclass
class TranscriptionResult:
    """English transcript + original speech with multi-language metadata."""

    transcript: str
    transcript_original: str
    language: str
    language_name: str
    languages: list[str]
    language_names: list[str]


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


def _feature_extractor(pipe):
    fe = getattr(pipe, "feature_extractor", None) or getattr(pipe, "processor", None)
    if fe is None:
        return None
    if hasattr(fe, "feature_extractor"):
        return fe.feature_extractor
    return fe


def _detect_language(pipe, wav: np.ndarray, sample_rate: int) -> str:
    """Auto-detect spoken language using Whisper (all supported languages)."""
    try:
        model = pipe.model
        fe = _feature_extractor(pipe)
        if fe is None:
            return "en"

        # Use up to 30s for reliable detection on short clips.
        detect_samples = min(len(wav), sample_rate * 30)
        detect_wav = wav[:detect_samples]

        inputs = fe([detect_wav], sampling_rate=sample_rate, return_tensors="pt", padding=True)
        input_features = inputs.input_features
        if hasattr(model, "device"):
            input_features = input_features.to(model.device)
        dtype = getattr(model, "dtype", torch.float32)
        input_features = input_features.to(dtype=dtype)

        # Whisper language detection expects ~30s mel frames (3000); pad short clips.
        target_frames = 3000
        cur_frames = input_features.shape[-1]
        if cur_frames < target_frames:
            input_features = torch.nn.functional.pad(
                input_features, (0, target_frames - cur_frames)
            )
        elif cur_frames > target_frames:
            input_features = input_features[..., :target_frames]

        with torch.no_grad():
            lang_ids = model.detect_language(input_features)
            lang_id = int(lang_ids[0, 0].item())

        tokenizer = pipe.tokenizer
        token = tokenizer.convert_ids_to_tokens([lang_id])[0]
        code = token_to_language_code(token)
        logger.info("Detected language: %s (%s)", language_label(code), code)
        return whisper_language_code(code)
    except Exception as exc:
        logger.warning("Language detection failed, falling back to auto: %s", exc)
        return "en"


def _run_whisper(
    pipe,
    wav: np.ndarray,
    sample_rate: int,
    *,
    task: str,
    language_code: Optional[str] = None,
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
    # Do not pass no_speech_threshold / logprob_threshold here — on short clips
    # transformers can compare None to float and crash transcription.

    whisper_lang = whisper_language_name(language_code)
    if whisper_lang and whisper_lang != "english":
        gen_kwargs["language"] = whisper_lang
    elif language_code == "en":
        gen_kwargs["language"] = "english"

    out = pipe(
        {"array": wav, "sampling_rate": sample_rate},
        generate_kwargs=gen_kwargs,
        **pipe_kwargs,
    )
    text = out.get("text", "") if isinstance(out, dict) else str(out)
    return (text or "").strip()



def _transcribe_chunk(
    pipe,
    chunk: np.ndarray,
    sample_rate: int,
    language: str,
) -> tuple[str, str]:
    """Transcribe one chunk; translate to English when not English."""
    original = _run_whisper(
        pipe, chunk, sample_rate, task="transcribe", language_code=language
    )
    if not original.strip():
        return "", ""

    if language == "en":
        return original.strip(), original.strip()

    english = _run_whisper(
        pipe, chunk, sample_rate, task="translate", language_code=language
    )
    return (english or original).strip(), original.strip()


def _finalize_transcripts(en_parts: list[str], orig_parts: list[str]) -> tuple[str, str]:
    transcript_en = clean_asr_text(" ".join(en_parts))
    transcript_original = clean_asr_text(" ".join(orig_parts))
    return transcript_en, transcript_original


def _split_time_chunks(
    wav: np.ndarray,
    sample_rate: int,
    segment_sec: float,
    max_segments: int,
) -> list[np.ndarray]:
    """Split audio into overlapping windows for full-length transcription."""
    if max_segments <= 1:
        return [wav]

    seg_samples = max(int(segment_sec * sample_rate), sample_rate)
    if len(wav) <= seg_samples:
        return [wav]

    overlap = min(int(0.5 * sample_rate), seg_samples // 4)
    step = max(seg_samples - overlap, sample_rate // 2)
    chunks: list[np.ndarray] = []
    for start in range(0, len(wav), step):
        chunk = wav[start : start + seg_samples]
        if len(chunk) < sample_rate // 2:
            break
        chunks.append(chunk)
        if len(chunks) >= max_segments:
            break
    return chunks or [wav]


def _probe_languages(
    pipe, wav: np.ndarray, sample_rate: int, probe_sec: float = 2.0
) -> tuple[str, str]:
    """Compare language at start vs end (2 lightweight detect passes)."""
    probe = min(int(probe_sec * sample_rate), max(len(wav) // 3, sample_rate))
    if len(wav) <= probe:
        lang = _detect_language(pipe, wav, sample_rate)
        return lang, lang
    lang_start = _detect_language(pipe, wav[:probe], sample_rate)
    lang_end = _detect_language(pipe, wav[-probe:], sample_rate)
    return lang_start, lang_end


def transcribe_bilingual(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    model_id: str = "openai/whisper-tiny",
    device: Optional[Union[str, torch.device]] = None,
    max_duration_sec: Optional[float] = None,
    segment_sec: float = 4.0,
    max_segments: int = 2,
) -> TranscriptionResult:
    """
    Auto-detect language; transcribe full clip in time windows when needed.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav = _prepare_wav(audio, sample_rate, max_duration_sec)
    pipe = _get_asr_pipe(model_id, device)

    try:
        lang_start, lang_end = _probe_languages(pipe, wav, sample_rate)

        # Whole clip may be English even when language probe is wrong.
        if lang_start == lang_end and lang_start != "en":
            auto_en = _run_whisper(
                pipe, wav, sample_rate, task="transcribe", language_code=None
            )
            if is_likely_english_text(auto_en):
                lang_start = lang_end = "en"

        multi = lang_start != lang_end and max_segments > 1

        en_parts: list[str] = []
        orig_parts: list[str] = []
        langs_ordered: list[str] = []

        if multi:
            mid = len(wav) // 2
            chunks = [(lang_start, wav[:mid]), (lang_end, wav[mid:])]
            logger.info("Mixed languages detected: %s + %s", lang_start, lang_end)
        else:
            lang = lang_start
            time_chunks = _split_time_chunks(wav, sample_rate, segment_sec, max_segments)
            chunks = [(lang, c) for c in time_chunks]
            if len(time_chunks) > 1:
                logger.info(
                    "Transcribing %d segments (%.1fs total)",
                    len(time_chunks),
                    len(wav) / sample_rate,
                )

        for lang, chunk in chunks:
            english, original = _transcribe_chunk(pipe, chunk, sample_rate, lang)
            if not original:
                continue

            # English mis-detected as another language — keep speech in English.
            if lang != "en" and is_likely_english_text(original):
                lang = "en"
                english = original

            langs_ordered.append(lang)
            if lang == "en":
                orig_parts.append(original)
                en_parts.append(original)
            else:
                orig_parts.append(f"[{language_label(lang)}] {original}")
                en_parts.append(english)

        if not en_parts:
            return TranscriptionResult(
                transcript="",
                transcript_original="",
                language="en",
                language_name="English",
                languages=[],
                language_names=[],
            )

        languages = list(dict.fromkeys(langs_ordered))
        language_names = [language_label(code) for code in languages]
        transcript_en, transcript_original = _finalize_transcripts(en_parts, orig_parts)

        if len(languages) == 1:
            primary = languages[0]
            primary_name = language_names[0]
        else:
            primary = "multi"
            primary_name = ", ".join(language_names)

        return TranscriptionResult(
            transcript=transcript_en,
            transcript_original=transcript_original,
            language=primary,
            language_name=primary_name,
            languages=languages,
            language_names=language_names,
        )
    except Exception as e:
        err = f"[ASR error: {e}]"
        return TranscriptionResult(
            transcript=err,
            transcript_original=err,
            language="en",
            language_name="English",
            languages=["en"],
            language_names=["English"],
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
    """Legacy single-language transcription."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav = _prepare_wav(audio, sample_rate, max_duration_sec)
    pipe = _get_asr_pipe(model_id, device)

    try:
        lang = whisper_language_code(language) if language else _detect_language(pipe, wav, sample_rate)
        task = "translate" if lang == "en" else "transcribe"
        return _run_whisper(pipe, wav, sample_rate, task=task, language_code=lang)
    except Exception as e:
        return f"[ASR error: {e}]"
