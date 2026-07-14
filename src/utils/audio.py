"""Load mono waveform from file for inference."""
from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

_FFMPEG_FORMATS = {
    ".mp3",
    ".mp4",
    ".m4a",
    ".mkv",
    ".webm",
    ".avi",
    ".mov",
    ".aac",
    ".wma",
    ".mpeg",
    ".mpg",
}

_cached_ffmpeg: str | None | bool = False  # False = not resolved yet


def _find_ffmpeg() -> str | None:
    """System ffmpeg on PATH, else bundled binary from imageio-ffmpeg."""
    global _cached_ffmpeg
    if _cached_ffmpeg is not False:
        return _cached_ffmpeg  # type: ignore[return-value]

    exe = shutil.which("ffmpeg")
    if exe:
        _cached_ffmpeg = exe
        return exe

    env_exe = os.environ.get("IMAGEIO_FFMPEG_EXE", "").strip()
    if env_exe and Path(env_exe).is_file():
        _cached_ffmpeg = env_exe
        return env_exe

    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).is_file():
            # Ensure later shutil.which / child tools can find it
            bin_dir = str(Path(bundled).resolve().parent)
            path = os.environ.get("PATH", "")
            if bin_dir.lower() not in path.lower():
                os.environ["PATH"] = bin_dir + os.pathsep + path
            os.environ.setdefault("IMAGEIO_FFMPEG_EXE", bundled)
            _cached_ffmpeg = bundled
            logger.info("Using bundled ffmpeg: %s", bundled)
            return bundled
    except Exception as exc:
        logger.warning("imageio_ffmpeg unavailable: %s", exc)

    _cached_ffmpeg = None
    return None


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
    """Decode via ffmpeg (partial when max_sec > 0, full file when max_sec == 0)."""
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return None

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "wav",
        "pipe:1",
    ]
    if max_sec and max_sec > 0:
        cmd[6:6] = ["-t", str(max_sec)]

    timeout = 90
    if not max_sec or max_sec <= 0:
        # Scale timeout for long videos (up to 15 min decode).
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            timeout = int(min(900, max(120, size_mb * 12)))
        except OSError:
            timeout = 300

    try:
        proc = subprocess.run(cmd, capture_output=True, check=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace").strip()
        logger.warning("ffmpeg decode failed for %s: %s", path.name, err or e)
        return None
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("ffmpeg decode failed for %s: %s", path.name, e)
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
    suffix = path.suffix.lower()

    if suffix in {".wav", ".flac", ".ogg"}:
        tensor = _load_with_soundfile(path, sr, limit)
        if tensor is not None:
            return tensor

    tensor = _load_with_ffmpeg(path, sr, limit)
    if tensor is not None:
        return tensor

    ffmpeg = _find_ffmpeg()
    if suffix in _FFMPEG_FORMATS:
        if not ffmpeg:
            raise RuntimeError(
                "Could not decode this audio/video file: ffmpeg not found. "
                "In the project venv run: pip install imageio-ffmpeg"
            )
        raise RuntimeError(
            f"Could not decode '{path.name}' with ffmpeg ({ffmpeg}). "
            "Try converting to WAV/MP3, or re-export the file."
        )

    try:
        load_kw: dict = {"sr": sr, "mono": True}
        if limit > 0:
            load_kw["duration"] = limit
        y, _file_sr = librosa.load(str(path), **load_kw)
        return torch.from_numpy(y.astype(np.float32)).unsqueeze(0)
    except Exception as exc:
        raise RuntimeError(
            f"Could not decode '{path.name}'. Tried soundfile/ffmpeg/librosa. ({exc})"
        ) from exc
