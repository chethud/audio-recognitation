"""Speaker diarization via Wav2Vec2 embeddings + clustering."""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_embed_model = None
_embed_processor = None
_embed_device: Optional[torch.device] = None

_EMBED_MODEL_ID = "facebook/wav2vec2-base"


def _get_embedding_model():
    global _embed_model, _embed_processor, _embed_device
    with _lock:
        if _embed_model is not None:
            return _embed_model, _embed_processor, _embed_device
        try:
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            processor = Wav2Vec2FeatureExtractor.from_pretrained(_EMBED_MODEL_ID)
            model = Wav2Vec2Model.from_pretrained(_EMBED_MODEL_ID)
            model.to(device)
            model.eval()
            _embed_model = model
            _embed_processor = processor
            _embed_device = device
            logger.info("Loaded speaker embedding model (%s)", _EMBED_MODEL_ID)
            return model, processor, device
        except Exception as exc:
            logger.warning("Speaker diarization unavailable: %s", exc)
            return None, None, None


def warmup_diarization() -> bool:
    model, _, _ = _get_embedding_model()
    return model is not None


def _prepare_wav(audio: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        wav = audio.detach().float().cpu().numpy().reshape(-1)
    else:
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = np.max(np.abs(wav)) if len(wav) else 0.0
    if peak > 0:
        wav = wav / peak
    return wav.astype(np.float32)


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


def _speaker_for_time(segments: list[dict[str, Any]], t_sec: float) -> str:
    for seg in segments:
        if seg["start_sec"] <= t_sec <= seg["end_sec"]:
            return seg["speaker"]
    best = min(
        segments,
        key=lambda s: min(
            abs(t_sec - s["start_sec"]),
            abs(t_sec - s["end_sec"]),
        ),
    )
    return best["speaker"]


def _align_transcript_to_speakers(
    phrases: list[tuple[str, tuple[float, float]]],
    segments: list[dict[str, Any]],
    *,
    language: str,
) -> tuple[list[dict[str, Any]], str, str]:
    from src.asr.whisper_languages import language_label

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
                }
            )

    en_lines: list[str] = []
    orig_lines: list[str] = []
    for t in turns:
        en_lines.append(f"{t['speaker']}: {t['text']}")
        if language == "en":
            orig_lines.append(f"{t['speaker']}: {t['text_original']}")
        else:
            orig_lines.append(
                f"{t['speaker']}: [{language_label(language)}] {t['text_original']}"
            )

    return turns, "\n".join(en_lines), "\n".join(orig_lines)

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


def _cluster_embeddings(
    embeddings: np.ndarray,
    *,
    max_speakers: int = 6,
    distance_threshold: float = 0.55,
) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

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
    n_found = len(set(labels))

    if n_found > max_speakers:
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
                "speaker": f"Person {order[cluster]}",
                "start_sec": round(seg["start_sec"], 2),
                "end_sec": round(seg["end_sec"], 2),
            }
        )
    return out


def diarize_speakers(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    max_speakers: int = 6,
    window_sec: float = 1.2,
    hop_sec: float = 0.6,
    min_segment_sec: float = 0.4,
    distance_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    """
    Return time segments labeled Person 1, Person 2, …
    Empty list when diarization is unavailable or no speech found.
    """
    model, processor, device = _get_embedding_model()
    if model is None or processor is None or device is None:
        return []

    wav = _prepare_wav(audio)
    windows = _speech_windows(
        wav, sample_rate, window_sec=window_sec, hop_sec=hop_sec
    )
    if len(windows) < 2:
        return []

    embeddings: list[np.ndarray] = []
    kept: list[tuple[float, float, np.ndarray]] = []
    chunk_audio = [w[2] for w in windows]
    try:
        embeddings = _embed_chunks_batched(
            chunk_audio, sample_rate, model, processor, device
        )
        kept = list(windows)
    except Exception as exc:
        logger.debug("Batched embedding failed, falling back: %s", exc)
        for start, end, chunk in windows:
            try:
                single = _embed_chunks_batched(
                    [chunk], sample_rate, model, processor, device, batch_size=1
                )
                if single:
                    embeddings.append(single[0])
                    kept.append((start, end, chunk))
            except Exception:
                pass

    if len(embeddings) < 2:
        return []

    labels = _cluster_embeddings(
        np.vstack(embeddings),
        max_speakers=max_speakers,
        distance_threshold=distance_threshold,
    )
    merged = _merge_labeled_windows(
        kept, labels, min_segment_sec=min_segment_sec
    )
    segments = _assign_person_labels(merged)

    n_speakers = len({s["speaker"] for s in segments})
    logger.info("Diarization: %d speaker(s), %d segments", n_speakers, len(segments))
    return segments


def run_diarized_transcription(
    audio: Union[torch.Tensor, np.ndarray],
    pipe,
    *,
    sample_rate: int = 16000,
    language: str = "en",
    max_speakers: int = 6,
    window_sec: float = 1.2,
    hop_sec: float = 0.6,
    min_segment_sec: float = 0.4,
    distance_threshold: float = 0.55,
    max_diarization_sec: float = 0,
) -> tuple[list[dict[str, Any]], str, str]:
    """
    Fast path: diarize once, transcribe once with timestamps, then align.
    Returns (speaker_turns, transcript_en, transcript_original).
    """
    from src.asr.whisper_asr import _run_whisper_timestamped

    wav = _prepare_wav(audio)
    dia_wav = wav
    if max_diarization_sec and max_diarization_sec > 0:
        cap = int(max_diarization_sec * sample_rate)
        if len(wav) > cap:
            dia_wav = wav[:cap]
            logger.info(
                "Diarization uses first %.0fs of %.0fs clip for speed",
                max_diarization_sec,
                len(wav) / sample_rate,
            )

    segments = diarize_speakers(
        dia_wav,
        sample_rate=sample_rate,
        max_speakers=max_speakers,
        window_sec=window_sec,
        hop_sec=hop_sec,
        min_segment_sec=min_segment_sec,
        distance_threshold=distance_threshold,
    )

    if not segments:
        return [], "", ""

    phrases = _run_whisper_timestamped(
        pipe, wav, sample_rate, language_code=language
    )
    if not phrases:
        return [], "", ""

    return _align_transcript_to_speakers(phrases, segments, language=language)
