"""Load mono waveform from file for inference."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch


def load_audio_from_file(path: str | Path, sr: int = 16000, max_sec: float | int = 30) -> torch.Tensor:
    """
    Load audio as float tensor shape (1, samples) at sample rate `sr`.
    If max_sec > 0, only the first max_sec seconds are decoded (fast for long uploads).
    If max_sec == 0, load the full file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    load_kw: dict = {"sr": sr, "mono": True}
    if max_sec and float(max_sec) > 0:
        load_kw["duration"] = float(max_sec)

    y, _file_sr = librosa.load(str(path), **load_kw)
    x = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)
    return x
