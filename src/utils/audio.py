"""Load mono waveform from file for inference."""
from __future__ import annotations

import io
import logging
import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


def _resample_if_needed(y: np.ndarray, file_sr: int, target_sr: int) -> np.ndarray:
    if file_sr == target_sr:
        return y.astype(np.float32, copy=False)
    return librosa.resample(y.astype(np.float32), orig_sr=file_sr, target_sr=target_sr)


def _load_with_soundfile(path: Path, sr: int, max_sec: float) -> torch.Tensor | None:
    """Fast partial read for WAV/FLAC/OGG (seeks without decoding the whole file)."""
    try:
        info = sf.info(str(path))
    except Exception:
        return None

    max_frames = info.frames
    if max_sec and max_sec > 0:
        max_frames = min(max_frames, int(max_sec * info.samplerate))

    y, file_sr = sf.read(str(path), frames=max_frames, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = _resample_if_needed(y, file_sr, sr)
    return torch.from_numpy(y).unsqueeze(0)


def _load_with_ffmpeg(path: Path, sr: int, max_sec: float) -> torch.Tensor | None:
    """Extract only the first N seconds via ffmpeg (fast for long MP3/M4A/video)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-t",
        str(max_sec),
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "wav",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True, timeout=90)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug("ffmpeg trim failed for %s: %s", path.name, e)
        return None

    if not proc.stdout:
        return None

    try:
        y, file_sr = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=False)
    except Exception as e:
        logger.debug("ffmpeg wav decode failed for %s: %s", path.name, e)
        return None

    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = _resample_if_needed(y, file_sr, sr)
    return torch.from_numpy(y).unsqueeze(0)


def load_audio_from_file(
    path: str | Path, sr: int = 16000, max_sec: float | int = 30
) -> torch.Tensor:
    """
    Load audio as float tensor shape (1, samples) at sample rate `sr`.
    If max_sec > 0, only the first max_sec seconds are decoded (fast for long uploads).
    If max_sec == 0, load the full file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    limit = float(max_sec) if max_sec and float(max_sec) > 0 else 0.0

    if limit > 0:
        # Prefer ffmpeg for compressed/long media; soundfile for native PCM formats.
        suffix = path.suffix.lower()
        if suffix in {".wav", ".flac", ".ogg"}:
            tensor = _load_with_soundfile(path, sr, limit)
            if tensor is not None:
                return tensor
        else:
            tensor = _load_with_ffmpeg(path, sr, limit)
            if tensor is not None:
                return tensor

    load_kw: dict = {"sr": sr, "mono": True}
    if limit > 0:
        load_kw["duration"] = limit

    y, _file_sr = librosa.load(str(path), **load_kw)
    x = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)
    return x
