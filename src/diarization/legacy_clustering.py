"""Legacy Wav2Vec2 embedding + agglomerative clustering diarization."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


def _speech_windows(
    wav: np.ndarray,
    sample_rate: int,
    *,
    window_sec: float = 1.2,
    hop_sec: float = 0.6,
    min_rms: float = 0.02,
) -> list[tuple[float, float, np.ndarray]]:
    win = max(int(window_sec * sample_rate), sample_rate // 2)
    hop = max(int(hop_sec * sample_rate), sample_rate // 4)
    out: list[tuple[float, float, np.ndarray]] = []

    for start in range(0, max(len(wav) - win // 2, 1), hop):
        end = min(start + win, len(wav))
        chunk = wav[start:end]
        if len(chunk) < sample_rate // 3:
            continue
        rms = float(np.sqrt(np.mean(chunk**2)))
        if rms < min_rms:
            continue
        out.append((start / sample_rate, end / sample_rate, chunk))
    return out


def _embed_chunks_batched(
    chunks: list[np.ndarray],
    sample_rate: int,
    model,
    processor,
    device: torch.device,
    batch_size: int = 16,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        inputs = processor(
            batch,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            hidden = model(input_values).last_hidden_state
        for j in range(hidden.shape[0]):
            out.append(hidden[j].mean(dim=0).cpu().numpy())
    return out


def _cluster_embeddings(
    embeddings: np.ndarray,
    *,
    max_speakers: int = 6,
    distance_threshold: float = 0.55,
) -> np.ndarray:
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)
    if len(set(labels)) > max_speakers:
        clustering = AgglomerativeClustering(
            n_clusters=max_speakers,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)
    return labels


def _merge_labeled_windows(
    windows: list[tuple[float, float, np.ndarray]],
    labels: np.ndarray,
    *,
    min_segment_sec: float = 0.4,
    gap_merge_sec: float = 0.35,
) -> list[dict[str, Any]]:
    if not windows:
        return []

    labeled: list[dict[str, Any]] = []
    for (start, end, _), label in zip(windows, labels):
        labeled.append({"cluster": int(label), "start_sec": start, "end_sec": end})

    labeled.sort(key=lambda x: x["start_sec"])
    merged: list[dict[str, Any]] = []
    for seg in labeled:
        if not merged:
            merged.append(dict(seg))
            continue
        prev = merged[-1]
        gap = seg["start_sec"] - prev["end_sec"]
        if seg["cluster"] == prev["cluster"] and gap <= gap_merge_sec:
            prev["end_sec"] = max(prev["end_sec"], seg["end_sec"])
        else:
            merged.append(dict(seg))

    return [s for s in merged if (s["end_sec"] - s["start_sec"]) >= min_segment_sec]


def _assign_person_labels(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order: dict[int, int] = {}
    out: list[dict[str, Any]] = []
    for seg in segments:
        cluster = seg["cluster"]
        if cluster not in order:
            order[cluster] = len(order) + 1
        out.append(
            {
                "speaker": f"Speaker {order[cluster]}",
                "start_sec": round(seg["start_sec"], 2),
                "end_sec": round(seg["end_sec"], 2),
            }
        )
    return out


def cluster_speaker_segments(
    wav: np.ndarray,
    sample_rate: int,
    *,
    model,
    processor,
    device: torch.device,
    max_speakers: int = 6,
    window_sec: float = 1.2,
    hop_sec: float = 0.6,
    min_segment_sec: float = 0.4,
    distance_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    windows = _speech_windows(wav, sample_rate, window_sec=window_sec, hop_sec=hop_sec)
    if len(windows) < 2:
        return []

    chunk_audio = [w[2] for w in windows]
    embeddings = _embed_chunks_batched(
        chunk_audio, sample_rate, model, processor, device
    )
    if len(embeddings) < 2:
        return []

    labels = _cluster_embeddings(
        np.vstack(embeddings),
        max_speakers=max_speakers,
        distance_threshold=distance_threshold,
    )
    merged = _merge_labeled_windows(windows, labels, min_segment_sec=min_segment_sec)
    return _assign_person_labels(merged)


def _speaker_for_time(segments: list[dict[str, Any]], t_sec: float) -> str:
    for seg in segments:
        if seg["start_sec"] <= t_sec <= seg["end_sec"]:
            return seg["speaker"]
    best = min(
        segments,
        key=lambda s: min(abs(t_sec - s["start_sec"]), abs(t_sec - s["end_sec"])),
    )
    return best["speaker"]


def align_phrases_to_segments(
    phrases: list[tuple[str, tuple[float, float]]],
    segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, str]:
    if not phrases or not segments:
        return [], "", ""

    turns: list[dict[str, Any]] = []
    for text, (start, end) in phrases:
        mid = (start + end) / 2.0
        speaker = _speaker_for_time(segments, mid)
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] = f"{turns[-1]['text']} {text}".strip()
            turns[-1]["text_original"] = turns[-1]["text"]
            turns[-1]["end_sec"] = round(end, 2)
        else:
            turns.append(
                {
                    "speaker": speaker,
                    "start_sec": round(start, 2),
                    "end_sec": round(end, 2),
                    "text": text,
                    "text_original": text,
                    "alignment": "legacy",
                }
            )

    en_lines = [f"{t['speaker']}: {t['text']}" for t in turns]
    return turns, "\n".join(en_lines), "\n".join(en_lines)
