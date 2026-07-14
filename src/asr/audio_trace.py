"""Per-request audio identity logging (no transcript caching)."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_waveform(wav: np.ndarray) -> str:
    arr = np.asarray(wav, dtype=np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def request_trace_fields() -> dict[str, str]:
    """Fields the API/worker inject via env (safe defaults)."""
    return {
        "upload_name": (os.environ.get("ALM_UPLOAD_NAME") or "").strip(),
        "temp_name": (os.environ.get("ALM_TEMP_NAME") or "").strip(),
        "audio_sha256": (os.environ.get("ALM_AUDIO_SHA256") or "").strip(),
        "audio_bytes": (os.environ.get("ALM_AUDIO_BYTES") or "").strip(),
        "language": (os.environ.get("ALM_ASR_LANGUAGE") or "").strip(),
    }


def log_audio_identity(
    *,
    stage: str,
    upload_name: str = "",
    temp_path: str | Path | None = None,
    file_bytes: int | None = None,
    file_sha256: str = "",
    language: str = "",
    wav: np.ndarray | None = None,
    sample_rate: int | None = None,
    whisper_out: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Emit a structured identity line so each upload can be verified end-to-end.
    Returns the dict for inclusion in worker JSON / API response.
    """
    trace = request_trace_fields()
    name = upload_name or trace["upload_name"]
    sha = file_sha256 or trace["audio_sha256"]
    nbytes = file_bytes
    if nbytes is None and trace["audio_bytes"].isdigit():
        nbytes = int(trace["audio_bytes"])
    temp = str(temp_path) if temp_path else trace["temp_name"]
    lang = language or trace["language"] or "auto"

    wav_sha = ""
    wav_samples = 0
    dur = 0.0
    if wav is not None:
        wav_sha = sha256_waveform(wav)
        wav_samples = int(np.asarray(wav).size)
        if sample_rate and sample_rate > 0:
            dur = wav_samples / float(sample_rate)

    preview = (whisper_out or "").replace("\n", " ").strip()
    if len(preview) > 160:
        preview = preview[:160] + "…"

    payload: dict[str, Any] = {
        "stage": stage,
        "upload_name": name,
        "temp_path": temp,
        "audio_bytes": nbytes,
        "audio_sha256": sha,
        "wav_sha256": wav_sha,
        "wav_samples": wav_samples,
        "wav_duration_sec": round(dur, 3) if dur else None,
        "language": lang,
        "whisper_preview": preview,
    }
    if extra:
        payload.update(extra)

    line = (
        f"[alm-audio] stage={stage} upload={name!r} temp={Path(temp).name if temp else ''} "
        f"bytes={nbytes} sha256={sha[:16] + '…' if sha else ''} "
        f"wav_sha={wav_sha[:16] + '…' if wav_sha else ''} "
        f"lang={lang} whisper={preview[:80]!r}"
    )
    print(line, flush=True)
    logger.info(
        "audio_trace stage=%s upload=%s bytes=%s sha256=%s wav_sha=%s lang=%s whisper_len=%d",
        stage,
        name,
        nbytes,
        sha,
        wav_sha,
        lang,
        len(whisper_out or ""),
    )
    return payload
