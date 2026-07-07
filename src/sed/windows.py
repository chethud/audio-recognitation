"""Sliding windows for sound-event scanning across long clips."""
from __future__ import annotations

import numpy as np


def sliding_audio_windows(
    wav: np.ndarray,
    sample_rate: int,
    *,
    segment_sec: float = 2.5,
    max_windows: int = 12,
) -> list[np.ndarray]:
    """
    Return fixed-length chunks spread across the full waveform.
    Uses evenly spaced start positions so background events are not missed
    when only the start/end of a long file is scanned.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return []

    seg_samples = max(int(segment_sec * sample_rate), sample_rate)
    min_chunk = sample_rate // 4

    if wav.size <= seg_samples:
        return [wav]

    if max_windows <= 0:
        duration = wav.size / sample_rate
        max_windows = min(32, max(8, int(duration / max(segment_sec * 0.5, 0.5)) + 1))

    max_start = max(0, wav.size - seg_samples)
    if max_windows == 1:
        starts = [0]
    else:
        starts = [
            int(round(i * max_start / (max_windows - 1))) for i in range(max_windows)
        ]

    windows: list[np.ndarray] = []
    seen: set[int] = set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        chunk = wav[start : start + seg_samples]
        if chunk.size < min_chunk:
            continue
        windows.append(chunk)

    return windows if windows else [wav]
