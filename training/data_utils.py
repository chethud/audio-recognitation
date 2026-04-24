"""Mel-spectrogram transforms and padding for CNN training."""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def waveform_to_mel(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Return log-mel [n_mels, T] float32."""
    import librosa

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,
        fmax=sample_rate // 2,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def pad_or_crop_time(mel: np.ndarray, target_frames: int, center: bool = False) -> np.ndarray:
    """mel shape [n_mels, T] -> fixed T. Use center=True for deterministic validation metrics."""
    _, t = mel.shape
    if t >= target_frames:
        if center:
            start = max(0, (t - target_frames) // 2)
        else:
            start = random.randint(0, t - target_frames) if t > target_frames else 0
        return mel[:, start : start + target_frames]
    pad = target_frames - t
    return np.pad(mel, ((0, 0), (0, pad)), mode="constant", constant_values=mel.min())


def normalize_mel(mel: torch.Tensor) -> torch.Tensor:
    m = mel.mean()
    s = mel.std().clamp(min=1e-6)
    return (mel - m) / s


class MelSpecCNN(nn.Module):
    """Small CNN for [B, 1, n_mels, T] log-mel inputs."""

    def __init__(self, num_classes: int, n_mels: int = 64, time_frames: int = 128):
        super().__init__()
        self.time_frames = time_frames
        self.n_mels = n_mels
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
