"""Stable Kannada ASR via faster-whisper (CTranslate2).

HuggingFace Transformers Whisper fine-tunes often hallucinate loops or
crash (0xC0000005) on Windows CPU. CTranslate2 is more stable and supports
built-in VAD filtering, which greatly reduces silence hallucinations.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from src.asr.text_cleanup import clean_asr_text

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_CT2_DIR = _ROOT / ".cache" / "ct2"
_PREFERRED = (
    # Prefer base for accuracy; tiny strongly hallucinates news/COVID boilerplate.
    _CT2_DIR / "whisper-kannada-base",
    _CT2_DIR / "whisper-kannada-tiny",
)

_lock = threading.Lock()
_model_cache: dict[str, object] = {}


def _resolve_ct2_model_dir() -> Path | None:
    for path in _PREFERRED:
        if (path / "model.bin").is_file():
            return path
    return None


def kannada_faster_available() -> bool:
    """
    Opt-in via ALM_ENABLE_CT2=1.

    Default OFF on this stack: failed CT2 mkl_malloc attempts leave Windows
    memory fragmented and the following HF Whisper stage often dies with
    0xC0000005 (exit 3221225477).
    """
    import os

    flag = (os.environ.get("ALM_ENABLE_CT2") or "").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return False
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False
    return _resolve_ct2_model_dir() is not None


def _get_model():
    path = _resolve_ct2_model_dir()
    if path is None:
        raise FileNotFoundError("No CT2 Kannada Whisper model under .cache/ct2")
    key = str(path)
    with _lock:
        if key in _model_cache:
            return _model_cache[key], path
        from faster_whisper import WhisperModel

        # Prefer int8; fall back to float32 if int8 runtime fails.
        last_err: Exception | None = None
        model = None
        for ctype in ("int8", "float32"):
            try:
                model = WhisperModel(
                    key,
                    device="cpu",
                    compute_type=ctype,
                    cpu_threads=1,
                    num_workers=1,
                )
                logger.info(
                    "Loaded faster-whisper Kannada model from %s (%s)",
                    path.name,
                    ctype,
                )
                break
            except Exception as exc:
                last_err = exc
                logger.warning("CT2 load compute_type=%s failed: %s", ctype, last_err)
        if model is None:
            raise RuntimeError(f"Could not load CT2 Kannada model: {last_err}")
        _model_cache[key] = model
        return model, path


def transcribe_kannada_faster(
    wav: np.ndarray,
    sample_rate: int = 16000,
    *,
    language: Optional[str] = "kn",
) -> list[tuple[float, float, str]]:
    """
    Transcribe full audio to timed (start, end, text) segments.
    Uses VAD + no previous-text conditioning to limit Whisper loops.
    Long clips are processed in ~90s pieces to avoid Windows malloc OOM.
    """
    if wav.ndim > 1:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    else:
        wav = np.asarray(wav, dtype=np.float32)

    if sample_rate != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    model, path = _get_model()
    duration = len(wav) / float(sample_rate)
    print(
        f"[alm-worker] faster_whisper_kn model={path.name} dur={duration:.1f}s",
        flush=True,
    )

    # Chunk long audio so CT2+VAD does not exhaust MKL allocators on Windows.
    piece_sec = 90.0
    piece = int(piece_sec * sample_rate)
    pieces: list[tuple[float, np.ndarray]] = []
    if len(wav) <= piece:
        pieces.append((0.0, wav))
    else:
        hop = int(85.0 * sample_rate)  # slight overlap
        pos = 0
        while pos < len(wav):
            end = min(len(wav), pos + piece)
            if end - pos < int(0.5 * sample_rate):
                break
            pieces.append((pos / float(sample_rate), wav[pos:end]))
            if end >= len(wav):
                break
            pos += hop

    out: list[tuple[float, float, str]] = []
    prev_text = ""
    for piece_i, (offset, chunk) in enumerate(pieces):
        print(
            f"[alm-worker] faster_whisper_kn piece={piece_i} "
            f"offset={offset:.1f}s len={len(chunk)/sample_rate:.1f}s",
            flush=True,
        )
        try:
            segments, _info = model.transcribe(
                chunk,
                language="kn",
                task="transcribe",
                beam_size=5,
                best_of=5,
                patience=1.0,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 400,
                    "speech_pad_ms": 200,
                },
                compression_ratio_threshold=2.2,
                log_prob_threshold=-0.8,
                no_speech_threshold=0.55,
                without_timestamps=False,
                word_timestamps=False,
            )
        except Exception as exc:
            logger.warning("faster-whisper piece %d failed: %s", piece_i, exc)
            continue

        for seg in segments:
            text = clean_asr_text((seg.text or "").strip())
            text = text.replace("\ufffd", "").replace("�", "").strip()
            if not text or len(text.strip(" .,?!।।")) < 2:
                continue
            start = float(seg.start) + offset
            end = float(seg.end) + offset
            if prev_text and _token_overlap(prev_text, text) >= 0.75:
                print(
                    f"[alm-worker] faster_whisper_kn skip_dup "
                    f"{start:.1f}-{end:.1f}s",
                    flush=True,
                )
                continue
            out.append((start, end, text))
            prev_text = text
            print(
                f"[alm-worker] faster_whisper_kn seg "
                f"{start:.1f}-{end:.1f}s chars={len(text)}",
                flush=True,
            )
    return out


def _token_overlap(a: str, b: str) -> float:
    wa = set(a.split())
    wb = set(b.split())
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    return inter / float(min(len(wa), len(wb)))
