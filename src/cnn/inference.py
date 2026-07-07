"""CNN inference for SED (ESC-50) and emotion (RAVDESS)."""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch

from src.cnn.loader import _load_emo_bundle, _load_sed_bundle, should_use_cnn
from src.sed.windows import sliding_audio_windows
from training.data_utils import normalize_mel, pad_or_crop_time, waveform_to_mel


def _prepare_wav(audio: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        return audio.detach().float().cpu().numpy().reshape(-1)
    wav = np.asarray(audio, dtype=np.float32)
    return wav.reshape(-1) if wav.ndim == 1 else wav.squeeze()


def _mel_tensor(
    wav: np.ndarray,
    sample_rate: int,
    n_mels: int,
    time_frames: int,
    device: torch.device,
) -> torch.Tensor:
    mel = waveform_to_mel(wav, sample_rate, n_mels=n_mels)
    mel = pad_or_crop_time(mel, time_frames, center=True)
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
    x = normalize_mel(x)
    return x.to(device)


def _top_k_from_logits(
    logits: torch.Tensor,
    class_names: list[str],
    top_k: int,
    threshold: float,
) -> List[dict]:
    probs = torch.softmax(logits, dim=-1)[0]
    k = min(top_k, len(class_names))
    values, indices = torch.topk(probs, k)
    out: List[dict] = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        if score < threshold:
            continue
        out.append({"label": str(class_names[idx]), "score": round(float(score), 4)})
    # Quick-trained CNN may score below threshold — still return top hits.
    if not out:
        for score, idx in zip(values.tolist(), indices.tolist()):
            if score < 0.03:
                break
            out.append({"label": str(class_names[idx]), "score": round(float(score), 4)})
    return out


def _sed_time_windows(
    wav: np.ndarray,
    sample_rate: int,
    segment_sec: float,
    max_windows: int,
) -> list[np.ndarray]:
    return sliding_audio_windows(
        wav,
        sample_rate,
        segment_sec=segment_sec,
        max_windows=max_windows,
    )


def predict_sed_cnn(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    top_k: int = 5,
    threshold: float = 0.15,
    segment_sec: float = 3.0,
    max_windows: int = 2,
) -> List[dict]:
    bundle = _load_sed_bundle()
    if bundle is None:
        return []

    wav = _prepare_wav(audio)
    model = bundle["model"]
    device = bundle["device"]
    n_mels = bundle["n_mels"]
    time_frames = bundle["time_frames"]
    class_names = bundle["class_names"]

    best: dict[str, float] = {}

    for chunk in _sed_time_windows(wav, sample_rate, segment_sec, max_windows):
        if len(chunk) < sample_rate // 4:
            continue
        with torch.no_grad():
            x = _mel_tensor(chunk, sample_rate, n_mels, time_frames, device)
            logits = model(x)
        for item in _top_k_from_logits(logits, class_names, top_k, threshold):
            best[item["label"]] = max(best.get(item["label"], 0.0), item["score"])

    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return [
        {"label": label, "score": round(score, 4)}
        for label, score in ranked[: max(top_k * 2, top_k)]
    ]


def predict_emotion_cnn(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    min_samples: int = 4000,
) -> str:
    bundle = _load_emo_bundle()
    if bundle is None:
        return "neutral"

    wav = _prepare_wav(audio)
    if len(wav) < min_samples:
        return "neutral"

    model = bundle["model"]
    device = bundle["device"]
    with torch.no_grad():
        x = _mel_tensor(wav, sample_rate, bundle["n_mels"], bundle["time_frames"], device)
        logits = model(x)
        idx = int(logits.argmax(dim=-1).item())
    return str(bundle["class_names"][idx]).lower().strip()


def cnn_models_available() -> bool:
    return should_use_cnn() and _load_sed_bundle() is not None
