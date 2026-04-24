"""Load mono waveform from file for inference."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch


def load_audio_from_file(path: str | Path, sr: int = 16000, max_sec: float | int = 30) -> torch.Tensor:
    """
    Load audio as float tensor shape (1, samples) at sample rate `sr`.
    If max_sec > 0, trim to first max_sec seconds; if max_sec == 0, load full file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    y, file_sr = librosa.load(str(path), sr=sr, mono=True)
    if max_sec and float(max_sec) > 0:
        max_len = int(float(max_sec) * sr)
        if len(y) > max_len:
            y = y[:max_len]
    x = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)
    return x
