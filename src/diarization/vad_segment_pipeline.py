"""VAD → speaker labeling → Whisper-per-segment diarization pipeline."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from src.diarization.types import DiarizationSegment, SpeakerTurn

logger = logging.getLogger(__name__)


def _prepare_wav(audio: Union[torch.Tensor, np.ndarray], sample_rate: int) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        wav = audio.detach().float().cpu().numpy().reshape(-1)
    else:
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak > 0:
        wav = wav / peak
    return wav.astype(np.float32)


def energy_vad(
    wav: np.ndarray,
    sample_rate: int,
    *,
    min_speech_sec: float = 0.3,
    min_silence_sec: float = 0.35,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> list[tuple[float, float]]:
    """Simple energy VAD: return (start_sec, end_sec) speech regions."""
    if len(wav) == 0:
        return []

    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    if len(wav) < frame:
        return [(0.0, len(wav) / float(sample_rate))]

    # RMS per frame
    n_frames = 1 + max(0, (len(wav) - frame) // hop)
    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        chunk = wav[start : start + frame]
        rms[i] = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))

    # Adaptive threshold: above noise floor
    noise = float(np.percentile(rms, 20))
    thr = max(noise * 2.5, float(np.percentile(rms, 40)) * 0.6, 1e-4)
    speech = rms >= thr

    # Close short silence gaps inside speech
    silence_frames = max(1, int(min_silence_sec * 1000.0 / hop_ms))
    speech_frames = max(1, int(min_speech_sec * 1000.0 / hop_ms))

    # Fill small holes
    filled = speech.copy()
    i = 0
    while i < n_frames:
        if filled[i]:
            i += 1
            continue
        j = i
        while j < n_frames and not filled[j]:
            j += 1
        if j - i <= silence_frames and i > 0 and j < n_frames:
            filled[i:j] = True
        i = j

    regions: list[tuple[float, float]] = []
    i = 0
    while i < n_frames:
        if not filled[i]:
            i += 1
            continue
        j = i
        while j < n_frames and filled[j]:
            j += 1
        if j - i >= speech_frames:
            start = (i * hop) / float(sample_rate)
            end = min(len(wav), j * hop + frame) / float(sample_rate)
            if end > start:
                regions.append((start, end))
        i = j

    if not regions:
        # Whole clip as one region if any energy
        if float(np.max(rms)) > 1e-5:
            return [(0.0, len(wav) / float(sample_rate))]
    return regions


def _mfcc_embedding(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    import librosa

    if len(wav) < sample_rate // 10:
        wav = np.pad(wav, (0, sample_rate // 10 - len(wav)))
    mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=20)
    feats = [mfcc.mean(axis=1), mfcc.std(axis=1)]
    # Pitch stats — the strongest voice-identity cue (e.g. male vs female median f0).
    try:
        f0 = librosa.yin(
            np.asarray(wav, dtype=np.float32),
            fmin=65,
            fmax=400,
            sr=sample_rate,
            frame_length=1024,
        )
        voiced = f0[(f0 > 66.0) & (f0 < 399.0)]
        if voiced.size >= 3:
            logf0 = np.log(voiced)
            feats.append(
                np.array(
                    [logf0.mean(), logf0.std(), voiced.size / max(f0.size, 1)],
                    dtype=np.float32,
                )
            )
        else:
            feats.append(np.zeros(3, dtype=np.float32))
    except Exception:
        feats.append(np.zeros(3, dtype=np.float32))
    return np.concatenate(feats).astype(np.float32)


def _normalize_embeddings(X: np.ndarray) -> np.ndarray:
    """
    Z-score each feature across regions, then L2-normalize.

    Raw MFCC coefficient 0 (energy) is ~2 orders of magnitude larger than the
    voice-identity features, so plain L2 cosine distance reduced to a loudness
    comparison and two-speaker interviews clustered as one speaker.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return X
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    Z = (X - mu) / sd
    # Pitch dims (last 3) are the strongest speaker cue — give them extra weight.
    if Z.shape[1] >= 43:
        Z[:, -3:] *= 3.0
    norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    return (Z / norms).astype(np.float32)


def _speech_mask_from_regions(
    n_samples: int,
    sample_rate: int,
    regions: list[tuple[float, float]],
) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    for start, end in regions:
        s = max(0, int(start * sample_rate))
        e = min(n_samples, int(end * sample_rate))
        if e > s:
            mask[s:e] = True
    return mask


def _window_embeddings(
    wav: np.ndarray,
    sample_rate: int,
    regions: list[tuple[float, float]],
    *,
    window_sec: float = 1.5,
    hop_sec: float = 0.75,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Sliding MFCC embeddings over speech regions (not one embedding per VAD blob)."""
    mask = _speech_mask_from_regions(len(wav), sample_rate, regions)
    win = max(int(window_sec * sample_rate), sample_rate // 4)
    hop = max(int(hop_sec * sample_rate), sample_rate // 8)
    embeds: list[np.ndarray] = []
    spans: list[tuple[float, float]] = []

    for start, end in regions:
        s0 = max(0, int(start * sample_rate))
        e0 = min(len(wav), int(end * sample_rate))
        if e0 - s0 < win:
            # Short region — still one window
            crop = wav[s0:e0]
            if crop.size < sample_rate // 10:
                continue
            # Require mostly speech
            if mask[s0:e0].mean() < 0.3:
                continue
            embeds.append(_mfcc_embedding(crop, sample_rate))
            spans.append((s0 / sample_rate, e0 / sample_rate))
            continue

        pos = s0
        while pos + win <= e0:
            crop = wav[pos : pos + win]
            if mask[pos : pos + win].mean() >= 0.5:
                embeds.append(_mfcc_embedding(crop, sample_rate))
                spans.append((pos / sample_rate, (pos + win) / sample_rate))
            pos += hop
        # Tail window
        if e0 - s0 >= win and (e0 - win) > s0:
            pos = e0 - win
            crop = wav[pos:e0]
            if mask[pos:e0].mean() >= 0.5:
                embeds.append(_mfcc_embedding(crop, sample_rate))
                spans.append((pos / sample_rate, e0 / sample_rate))

    if not embeds:
        return np.zeros((0, 43), dtype=np.float32), []
    return _normalize_embeddings(np.stack(embeds, axis=0)), spans


def _is_kannada_language(language: str | None) -> bool:
    if not language:
        return False
    return language.strip().lower() in {"kn", "kannada"}


def _cluster_labels(
    X: np.ndarray,
    *,
    max_speakers: int,
    distance_threshold: float,
    force_min_speakers: int = 1,
    strict_single: bool = False,
) -> list[int]:
    """
    Cluster speech embeddings into speakers.

    Defaults to ONE speaker. Only returns 2+ when clustering quality is strong
    (silhouette + balanced talk share). Never invents alternating speakers.

    ``strict_single`` (Kannada): much higher bar before accepting 2 speakers —
    Kannada monologues often look multi-speaker to weak MFCC clustering.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    n = X.shape[0]
    if n == 0:
        return []
    if n == 1 or max_speakers < 2:
        return [0] * n

    # Kannada: prefer single-speaker unless separation is very clear.
    sil_accept = 0.28 if strict_single else 0.08
    minority_accept = 0.32 if strict_single else 0.08
    switch_accept = 0.35 if strict_single else 0.15
    sil_keep_multi = 0.22 if strict_single else 0.06
    minority_keep = 0.30 if strict_single else 0.06

    def _fit(k: int) -> list[int]:
        kwargs = {"n_clusters": k, "linkage": "average"}
        try:
            model = AgglomerativeClustering(metric="cosine", **kwargs)
        except TypeError:
            model = AgglomerativeClustering(**kwargs)
        return model.fit_predict(X).tolist()

    def _minority_ratio(labels: list[int]) -> float:
        from collections import Counter

        c = Counter(labels)
        if not c:
            return 0.0
        return min(c.values()) / max(sum(c.values()), 1)

    def _mean_switch_distance(labels: list[int]) -> float:
        dists = []
        for i in range(1, n):
            if labels[i] != labels[i - 1]:
                dists.append(float(1.0 - np.clip(np.dot(X[i], X[i - 1]), -1.0, 1.0)))
        return float(np.mean(dists)) if dists else 0.0

    # Natural clustering by distance — often yields 1 speaker for monologues.
    # Kannada: use a looser threshold so pauses don't become extra speakers.
    dist_thr = float(distance_threshold)
    if strict_single:
        dist_thr = max(dist_thr, 0.55)
    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_thr,
            metric="cosine",
            linkage="average",
        )
        labels = model.fit_predict(X).tolist()
    except TypeError:
        labels = [0] * n

    if len(set(labels)) > max_speakers:
        labels = _fit(max_speakers)

    # Optional: test a 2-speaker fit only when we have enough regions.
    # Kannada needs more regions before we even try.
    min_regions_for_2 = 6 if strict_single else 4
    if n >= min_regions_for_2 and max_speakers >= 2 and force_min_speakers <= 2:
        labels2 = _fit(2)
        if len(set(labels2)) >= 2:
            try:
                sil = float(silhouette_score(X, labels2, metric="cosine"))
            except Exception:
                sil = -1.0
            minority = _minority_ratio(labels2)
            switch_dist = _mean_switch_distance(labels2)
            if sil >= sil_accept and minority >= minority_accept and switch_dist >= switch_accept:
                logger.info(
                    "MFCC clustering: 2 speakers (sil=%.3f minority=%.2f switch=%.3f strict=%s)",
                    sil,
                    minority,
                    switch_dist,
                    strict_single,
                )
                return labels2
            logger.info(
                "MFCC clustering: rejecting 2-speaker split "
                "(sil=%.3f minority=%.2f switch=%.3f strict=%s) → 1 speaker",
                sil,
                minority,
                switch_dist,
                strict_single,
            )

    if len(set(labels)) < 2:
        logger.info("MFCC clustering: 1 speaker (%d regions)", n)
        return [0] * n

    # Cap weak multi-way distance clustering back to 1 if poorly separated.
    if len(set(labels)) >= 2:
        try:
            sil = float(silhouette_score(X, labels, metric="cosine"))
        except Exception:
            sil = -1.0
        if sil < sil_keep_multi or _minority_ratio(labels) < minority_keep:
            logger.info(
                "MFCC clustering: collapsing weak multi-cluster (sil=%.3f) → 1 speaker",
                sil,
            )
            return [0] * n

    return labels


def _labels_to_segments(
    spans: list[tuple[float, float]],
    labels: list[int],
) -> list[DiarizationSegment]:
    order: dict[int, int] = {}
    segs: list[DiarizationSegment] = []
    for (start, end), lab in zip(spans, labels):
        lab_i = int(lab)
        if lab_i not in order:
            order[lab_i] = len(order) + 1
        segs.append(
            DiarizationSegment(
                speaker_id=f"Speaker {order[lab_i]}",
                start_sec=float(start),
                end_sec=float(end),
            )
        )
    return segs


def _label_speakers_mfcc(
    wav: np.ndarray,
    sample_rate: int,
    regions: list[tuple[float, float]],
    *,
    max_speakers: int = 6,
    distance_threshold: float = 0.35,
    window_sec: float = 1.5,
    hop_sec: float = 0.75,
    strict_single: bool = False,
) -> list[DiarizationSegment]:
    """
    Speaker labels from VAD regions (preferred) or sliding windows.

    Region-level clustering works much better for turn-taking dialogue:
    sliding windows often drown the second speaker into a single blip.
    """
    if not regions:
        return []

    # Prefer one embedding per VAD speech region when we have enough turns.
    # Kannada: ignore tiny VAD blips (<400ms) that spuriously change speakers.
    min_reg = 0.4 if strict_single else 0.2
    if len(regions) >= 3:
        embeds = []
        kept_regions: list[tuple[float, float]] = []
        for start, end in regions:
            if end - start < min_reg:
                continue
            s = int(start * sample_rate)
            e = int(end * sample_rate)
            embeds.append(_mfcc_embedding(wav[s:e], sample_rate))
            kept_regions.append((start, end))
        if len(kept_regions) >= 3:
            X = _normalize_embeddings(np.stack(embeds, axis=0))
            labels = _cluster_labels(
                X,
                max_speakers=max_speakers,
                distance_threshold=distance_threshold,
                force_min_speakers=1,
                strict_single=strict_single,
            )
            segs = _labels_to_segments(
                [(a, b) for a, b in kept_regions],
                labels,
            )
            n_spk = len({s.speaker_id for s in segs})
            logger.info(
                "MFCC region clustering: %d regions → %d speakers (strict=%s)",
                len(kept_regions),
                n_spk,
                strict_single,
            )
            return segs

    X, spans = _window_embeddings(
        wav,
        sample_rate,
        regions,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )
    if len(spans) == 0:
        return []

    labels = _cluster_labels(
        X,
        max_speakers=max_speakers,
        distance_threshold=distance_threshold,
        force_min_speakers=1,
        strict_single=strict_single,
    )
    return _labels_to_segments(spans, labels)


def _label_speakers_pyannote(
    wav: np.ndarray,
    sample_rate: int,
    *,
    max_speakers: int = 6,
) -> list[DiarizationSegment] | None:
    """Try PyAnnote; return None if unavailable."""
    from src.diarization.model_cache import hf_auth_token

    token = hf_auth_token()
    if not token:
        return None

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        logger.info("pyannote.audio not available: %s", exc)
        return None

    path = None
    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
        if pipe is None:
            return None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe.to(device)

        fd, path = tempfile.mkstemp(suffix=".wav")
        import os

        import soundfile as sf

        os.close(fd)
        sf.write(path, wav, sample_rate)
        # min_speakers=1 — never force two speakers on a monologue.
        diarization = pipe(
            path,
            min_speakers=1,
            max_speakers=max_speakers,
        )

        order: dict[str, int] = {}
        segs: list[DiarizationSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            sp = str(speaker)
            if sp not in order:
                order[sp] = len(order) + 1
            segs.append(
                DiarizationSegment(
                    speaker_id=f"Speaker {order[sp]}",
                    start_sec=float(turn.start),
                    end_sec=float(turn.end),
                )
            )
        segs.sort(key=lambda s: (s.start_sec, s.end_sec))
        return segs if segs else None
    except Exception as exc:
        logger.warning("PyAnnote labeling failed, using MFCC: %s", exc)
        return None
    finally:
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass


def _merge_same_speaker_segments(
    segs: list[DiarizationSegment],
    *,
    gap_merge_sec: float = 0.6,
) -> list[DiarizationSegment]:
    if not segs:
        return []
    ordered = sorted(segs, key=lambda s: (s.start_sec, s.end_sec))
    out = [ordered[0]]
    for seg in ordered[1:]:
        prev = out[-1]
        if (
            seg.speaker_id == prev.speaker_id
            and seg.start_sec - prev.end_sec <= gap_merge_sec
        ):
            out[-1] = DiarizationSegment(
                speaker_id=prev.speaker_id,
                start_sec=prev.start_sec,
                end_sec=max(prev.end_sec, seg.end_sec),
            )
        else:
            out.append(seg)
    return out


def _resolve_overlapping_segments(
    segs: list[DiarizationSegment],
) -> list[DiarizationSegment]:
    """
    Convert possibly-overlapping window labels into a non-overlapping timeline.
    At each conflict, keep the earlier segment and push the next start forward.
    """
    if not segs:
        return []
    ordered = sorted(segs, key=lambda s: (s.start_sec, s.end_sec))
    out: list[DiarizationSegment] = []
    for seg in ordered:
        if not out:
            out.append(seg)
            continue
        prev = out[-1]
        if seg.start_sec < prev.end_sec:
            if seg.speaker_id == prev.speaker_id:
                out[-1] = DiarizationSegment(
                    speaker_id=prev.speaker_id,
                    start_sec=prev.start_sec,
                    end_sec=max(prev.end_sec, seg.end_sec),
                )
            else:
                # Split at midpoint of overlap
                mid = 0.5 * (prev.end_sec + seg.start_sec)
                if mid <= prev.start_sec + 0.05:
                    mid = prev.start_sec + 0.05
                out[-1] = DiarizationSegment(
                    speaker_id=prev.speaker_id,
                    start_sec=prev.start_sec,
                    end_sec=mid,
                )
                if seg.end_sec > mid + 0.15:
                    out.append(
                        DiarizationSegment(
                            speaker_id=seg.speaker_id,
                            start_sec=mid,
                            end_sec=seg.end_sec,
                        )
                    )
        else:
            out.append(seg)
    return [s for s in out if s.end_sec - s.start_sec >= 0.2]


def _phrase_midpoint(start: float, end: float) -> float:
    return 0.5 * (float(start) + float(end))


def _assign_phrases_to_speakers(
    phrases: list[tuple[str, tuple[float, float]]],
    segs: list[DiarizationSegment],
) -> list[SpeakerTurn]:
    """Attach timestamped Whisper phrases to speaker segments by time overlap."""
    if not phrases or not segs:
        return []

    buckets: dict[int, list[tuple[float, float, str]]] = {i: [] for i in range(len(segs))}
    for text, (p0, p1) in phrases:
        text = (text or "").strip()
        if not text or text in {".", ",", "?", "!", "…"}:
            continue
        overlaps: list[tuple[int, float, float, float]] = []
        for i, seg in enumerate(segs):
            ov0 = max(float(p0), seg.start_sec)
            ov1 = min(float(p1), seg.end_sec)
            ov = ov1 - ov0
            if ov > 0.05:
                overlaps.append((i, ov0, ov1, ov))
        if not overlaps:
            mid = _phrase_midpoint(p0, p1)
            best_i = min(
                range(len(segs)),
                key=lambda i: abs(
                    _phrase_midpoint(segs[i].start_sec, segs[i].end_sec) - mid
                ),
            )
            buckets[best_i].append((float(p0), float(p1), text))
            continue

        overlaps.sort(key=lambda x: x[1])
        words = text.split()
        total_ov = sum(o[3] for o in overlaps) or 1.0
        if len(overlaps) == 1 or len(words) <= 2:
            i, ov0, ov1, _ = max(overlaps, key=lambda x: x[3])
            buckets[i].append((ov0, ov1, text))
            continue

        cursor = 0
        for idx, (i, ov0, ov1, ov) in enumerate(overlaps):
            if idx == len(overlaps) - 1:
                chunk = words[cursor:]
            else:
                take = max(1, int(round(len(words) * (ov / total_ov))))
                chunk = words[cursor : cursor + take]
                cursor += take
            if chunk:
                buckets[i].append((ov0, ov1, " ".join(chunk)))

    turns: list[SpeakerTurn] = []
    for i, seg in enumerate(segs):
        items = sorted(buckets[i], key=lambda x: x[0])
        if not items:
            continue
        text = " ".join(t for _, _, t in items).strip()
        if not text:
            continue
        turns.append(
            SpeakerTurn(
                speaker=seg.speaker_id,
                start_sec=float(items[0][0]),
                end_sec=float(items[-1][1]),
                text=text,
                text_original=text,
                confidence=0.0,
                alignment="vad",
            )
        )
    return turns


def _filter_script_mismatch(text: str, language: str | None) -> str:
    """Drop clear wrong-script hallucinations; keep Kannada romanization (Latin)."""
    import re

    from src.asr.text_cleanup import (
        clean_asr_text,
        infer_language_from_text,
        is_low_quality_indic,
        _SCRIPT_RANGES,
    )

    text = clean_asr_text(text or "")
    if not text or not language:
        return text
    lang = language.strip().lower()
    if lang in ("en", "english"):
        return text

    low = text.lower()
    if "commission" in low or "subscribe" in low or "thank you for watching" in low:
        return ""
    if len(re.findall(r"[.。]", text)) >= 3 and len(text) < 80:
        return ""

    # Drop Arabic / CJK / Hangul when expecting Indic
    if re.search(r"[\u0600-\u06FF\u0750-\u077F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]", text):
        target_pat = dict(_SCRIPT_RANGES).get(lang)
        if not target_pat or not re.search(target_pat, text):
            return ""

    inferred = infer_language_from_text(text)
    if inferred is None:
        return text
    if inferred == lang:
        if is_low_quality_indic(text):
            return ""
        return text
    if lang in {"kn", "ta", "te", "ml", "hi", "bn", "gu", "pa", "or"}:
        target_pat = dict(_SCRIPT_RANGES).get(lang)
        if target_pat and re.search(target_pat, text) and not is_low_quality_indic(text):
            return text
        logger.info(
            "Dropping script-mismatch ASR (%s inferred, %s forced): %s",
            inferred,
            lang,
            text[:80],
        )
        return ""
    return text


def _pack_speech_windows(
    segs: list[DiarizationSegment],
    *,
    max_sec: float = 18.0,
    gap_join_sec: float = 0.6,
) -> list[tuple[float, float]]:
    """
    Pack consecutive same-speaker speech into ~max_sec windows for ASR.
    Better Kannada quality than tiny crops or one giant clip.
    """
    if not segs:
        return []
    ordered = sorted(segs, key=lambda s: (s.start_sec, s.end_sec))
    windows: list[tuple[float, float]] = []
    cur_start = float(ordered[0].start_sec)
    cur_end = float(ordered[0].end_sec)
    for seg in ordered[1:]:
        s, e = float(seg.start_sec), float(seg.end_sec)
        gap = s - cur_end
        if gap <= gap_join_sec and (e - cur_start) <= max_sec:
            cur_end = max(cur_end, e)
            continue
        if cur_end > cur_start:
            windows.append((cur_start, cur_end))
        # If a single region is longer than max_sec, split it.
        if e - s > max_sec:
            t = s
            while t < e:
                nt = min(e, t + max_sec)
                windows.append((t, nt))
                t = nt
            cur_start = e
            cur_end = e
        else:
            cur_start, cur_end = s, e
    if cur_end > cur_start:
        windows.append((cur_start, cur_end))
    return windows


def _token_jaccard(a: str, b: str) -> float:
    wa = set((a or "").split())
    wb = set((b or "").split())
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    union = len(wa | wb)
    return inter / float(union) if union else 0.0


def _transcribe_segments(
    wav: np.ndarray,
    sample_rate: int,
    segs: list[DiarizationSegment],
    *,
    language: str | None,
    asr_model_id: str,
) -> list[SpeakerTurn]:
    """
    Full-clip Whisper + assign phrases to speaker segments (best for Kannada),
    with per-segment fallback.
    """
    from src.asr.whisper_asr import _get_asr_pipe, _run_whisper, _run_whisper_timestamped

    segs = _resolve_overlapping_segments(segs)
    segs = _merge_same_speaker_segments(segs, gap_merge_sec=1.0)
    if not segs:
        return []

    n_spk = len({s.speaker_id for s in segs})
    print(
        f"[alm-worker] whisper_align lang={language or 'auto'} "
        f"segs={len(segs)} speakers={n_spk} model={asr_model_id}",
        flush=True,
    )

    from src.asr.whisper_asr import _is_kannada_finetune, run_whisper_speech_windows

    # Kannada: prefer faster-whisper CT2 (stable + VAD) over HF tiny hallucinations.
    lang_l = (language or "").strip().lower()
    if lang_l in {"kn", "kannada"}:
        try:
            from src.asr.kannada_faster import (
                kannada_faster_available,
                transcribe_kannada_faster,
            )

            if kannada_faster_available():
                print("[alm-worker] kannada_faster_whisper", flush=True)
                scored = transcribe_kannada_faster(
                    wav, sample_rate, language=language
                )
                turns_fw: list[SpeakerTurn] = []
                speaker = segs[0].speaker_id
                # Map timed ASR segments onto the majority speaker label.
                for start, end, text in scored:
                    text = _filter_script_mismatch(text, language)
                    if not text:
                        continue
                    # If multi-speaker diarization, assign by max overlap.
                    sp = speaker
                    if n_spk >= 2:
                        best = None
                        best_ov = 0.0
                        for seg in segs:
                            ov = max(
                                0.0,
                                min(end, seg.end_sec) - max(start, seg.start_sec),
                            )
                            if ov > best_ov:
                                best_ov = ov
                                best = seg.speaker_id
                        if best:
                            sp = best
                    turns_fw.append(
                        SpeakerTurn(
                            speaker=sp,
                            start_sec=start,
                            end_sec=end,
                            text=text,
                            text_original=text,
                            confidence=0.0,
                            alignment="vad",
                        )
                    )
                if turns_fw:
                    return _merge_asr_turns(turns_fw)
                logger.warning("faster-whisper Kannada returned empty; falling back")
        except Exception as exc:
            logger.warning("faster-whisper Kannada failed, falling back: %s", exc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = _get_asr_pipe(asr_model_id, device)

    # Single speaker → pack speech into ~12s windows and ASR each (full coverage).
    if n_spk <= 1:
        print("[alm-worker] whisper_speech_windows single_speaker", flush=True)
        windows = _pack_speech_windows(segs, max_sec=12.0, gap_join_sec=0.45)
        if not windows:
            windows = [(float(segs[0].start_sec), float(segs[-1].end_sec))]
        scored = run_whisper_speech_windows(
            pipe,
            wav,
            sample_rate,
            windows,
            language_code=language,
        )
        turns: list[SpeakerTurn] = []
        for start, end, text in scored:
            text = _filter_script_mismatch(text, language)
            if not text:
                continue
            # Drop near-duplicate consecutive windows (exact hallucination loops only).
            # Threshold was 0.7 which collapsed distinct window content into one block.
            if turns and _token_jaccard(turns[-1].text, text) >= 0.92:
                continue
            turns.append(
                SpeakerTurn(
                    speaker=segs[0].speaker_id,
                    start_sec=start,
                    end_sec=end,
                    text=text,
                    text_original=text,
                    confidence=0.0,
                    alignment="vad",
                )
            )
        if turns:
            return _merge_asr_turns(turns)
        # Fallback: longform on full clip
        print("[alm-worker] whisper_full_clip single_speaker_fallback", flush=True)
        try:
            text = _run_whisper(
                pipe,
                wav,
                sample_rate,
                task="transcribe",
                language_code=language,
            )
        except Exception as exc:
            logger.warning("Full-clip Whisper failed: %s", exc)
            text = ""
        text = _filter_script_mismatch(text, language)
        if not text:
            return []
        return [
            SpeakerTurn(
                speaker=segs[0].speaker_id,
                start_sec=float(segs[0].start_sec),
                end_sec=float(segs[-1].end_sec),
                text=text,
                text_original=text,
                confidence=0.0,
                alignment="vad",
            )
        ]

    # Multi-speaker: Kannada fine-tunes prefer per-segment (no reliable timestamps).
    # Other models support reliable timestamps, so we use single-pass timestamped ASR.
    use_per_segment = _is_kannada_finetune(asr_model_id)

    phrases: list[tuple[str, tuple[float, float]]] = []
    if not use_per_segment:
        try:
            phrases = _run_whisper_timestamped(
                pipe, wav, sample_rate, language_code=language
            )
        except Exception as exc:
            logger.warning("Timestamped Whisper failed: %s", exc)

    cleaned: list[tuple[str, tuple[float, float]]] = []
    for text, ts in phrases:
        text = _filter_script_mismatch(text, language)
        if text and len(text.strip(" .,?!")) >= 2:
            cleaned.append((text, ts))

    turns = _assign_phrases_to_speakers(cleaned, segs) if cleaned else []
    turn_speakers = {t.speaker for t in turns}
    if turns and (len(turn_speakers) >= 2 or n_spk < 2) and not _is_kannada_finetune(asr_model_id):
        return _merge_asr_turns(turns)

    print(
        f"[alm-worker] whisper_per_segment segs={len(segs)}",
        flush=True,
    )
    turns = []
    for seg in segs:
        if seg.end_sec - seg.start_sec < 0.35:
            continue
        s = max(0, int(seg.start_sec * sample_rate))
        e = min(len(wav), int(seg.end_sec * sample_rate))
        try:
            text = _run_whisper(
                pipe,
                wav[s:e],
                sample_rate,
                task="transcribe",
                language_code=language,
            )
        except Exception as exc:
            logger.warning(
                "Whisper failed on %.2f-%.2f: %s", seg.start_sec, seg.end_sec, exc
            )
            continue
        text = _filter_script_mismatch(text, language)
        if not text or len(text.strip(" .,?!")) < 2:
            continue
        turns.append(
            SpeakerTurn(
                speaker=seg.speaker_id,
                start_sec=float(seg.start_sec),
                end_sec=float(seg.end_sec),
                text=text,
                text_original=text,
                confidence=0.0,
                alignment="vad",
            )
        )
    return _merge_asr_turns(turns)


def _merge_asr_turns(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    if len(turns) < 2:
        return turns
    merged: list[SpeakerTurn] = [turns[0]]
    for t in turns[1:]:
        prev = merged[-1]
        if t.speaker == prev.speaker and t.start_sec - prev.end_sec <= 0.5:
            merged[-1] = SpeakerTurn(
                speaker=prev.speaker,
                start_sec=prev.start_sec,
                end_sec=t.end_sec,
                text=f"{prev.text} {t.text}".strip(),
                text_original=f"{prev.text_original} {t.text_original}".strip(),
                confidence=prev.confidence,
                alignment=prev.alignment,
            )
        else:
            merged.append(t)
    return merged


def _collapse_segments_to_single_speaker(
    segs: list[DiarizationSegment],
) -> list[DiarizationSegment]:
    """Relabel every segment as Speaker 1 and merge adjacent gaps."""
    if not segs:
        return []
    out = [
        DiarizationSegment(
            speaker_id="Speaker 1",
            start_sec=float(s.start_sec),
            end_sec=float(s.end_sec),
        )
        for s in segs
    ]
    return _merge_same_speaker_segments(out, gap_merge_sec=0.6)


def _looks_like_kannada_monologue(segs: list[DiarizationSegment]) -> bool:
    """
    Heuristic: balanced? No — ping-pong of short alternating segments =
    phantom second speaker on a monologue.
    """
    if len(segs) < 4:
        return False
    speakers = {s.speaker_id for s in segs}
    if len(speakers) != 2:
        return False
    switches = sum(
        1
        for i in range(1, len(segs))
        if segs[i].speaker_id != segs[i - 1].speaker_id
    )
    switch_ratio = switches / max(len(segs) - 1, 1)
    durs = [max(0.0, s.end_sec - s.start_sec) for s in segs]
    avg_dur = sum(durs) / max(len(durs), 1)
    # Majority talk share
    from collections import defaultdict

    share: dict[str, float] = defaultdict(float)
    for s in segs:
        share[s.speaker_id] += max(0.0, s.end_sec - s.start_sec)
    total = sum(share.values()) or 1.0
    minority = min(share.values()) / total
    # Short alternating turns OR one speaker dominant → treat as monologue
    if switch_ratio >= 0.55 and avg_dur <= 6.0:
        return True
    if minority < 0.30:
        return True
    return False


def run_vad_segment_pipeline(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    language: str | None = "en",
    max_speakers: int = 6,
    max_diarization_sec: float = 0,
    min_speech_sec: float = 0.3,
    min_silence_sec: float = 0.35,
    distance_threshold: float = 0.35,
    gap_merge_sec: float = 0.45,
    window_sec: float = 1.5,
    hop_sec: float = 0.75,
    asr_model_id: str = "openai/whisper-base",
) -> tuple[list[dict[str, Any]], str, str]:
    """
    VAD → speaker labels → Whisper each crop → chronological turns.

    Returns (speaker_turns_api_dicts, transcript, transcript_original).
    Empty turns means caller should fall back to text diarization.
    """
    wav = _prepare_wav(audio, sample_rate)
    if max_diarization_sec and max_diarization_sec > 0:
        cap = int(max_diarization_sec * sample_rate)
        if len(wav) > cap:
            wav = wav[:cap]

    if len(wav) < int(0.2 * sample_rate):
        return [], "", ""

    kn = _is_kannada_language(language)

    print("[alm-worker] vad_energy", flush=True)
    regions = energy_vad(
        wav,
        sample_rate,
        min_speech_sec=min_speech_sec if not kn else max(min_speech_sec, 0.4),
        min_silence_sec=min_silence_sec if not kn else max(min_silence_sec, 0.45),
    )
    if not regions:
        regions = [(0.0, len(wav) / float(sample_rate))]
        logger.info("VAD found no regions — using full clip")

    print("[alm-worker] vad_speaker_label", flush=True)
    segs = _label_speakers_pyannote(wav, sample_rate, max_speakers=max_speakers)
    n_py = len({s.speaker_id for s in segs}) if segs else 0

    # Kannada: trust a confident 1-speaker PyAnnote result — do NOT re-cluster
    # with MFCC (main cause of fake Speaker 1/2 splits on monologues).
    if kn and segs is not None and n_py == 1:
        logger.info("Kannada: keeping PyAnnote 1-speaker result (skip MFCC)")
        segs = _collapse_segments_to_single_speaker(segs)
    elif segs is None or n_py < 2:
        if segs is not None and n_py < 2 and not kn:
            logger.info("PyAnnote returned %d speaker(s); trying windowed MFCC", n_py)
        elif kn:
            logger.info("Kannada: MFCC speaker labeling with strict single-speaker prior")
        segs = _label_speakers_mfcc(
            wav,
            sample_rate,
            regions,
            max_speakers=max_speakers,
            distance_threshold=distance_threshold,
            window_sec=window_sec,
            hop_sec=hop_sec,
            strict_single=kn,
        )

    # Drop micro-segments that flip speakers for Kannada
    if kn and segs:
        segs = [s for s in segs if (s.end_sec - s.start_sec) >= 0.4] or segs

    merge_gap = 0.6 if kn else gap_merge_sec
    segs = _merge_same_speaker_segments(segs, gap_merge_sec=merge_gap)
    if not segs:
        return [], "", ""

    # Kannada monologue safety net before ASR locks bad labels.
    if kn and len({s.speaker_id for s in segs}) >= 2 and _looks_like_kannada_monologue(segs):
        logger.info("Kannada: collapsing monologue-like multi-speaker labels → Speakers 1")
        segs = _collapse_segments_to_single_speaker(segs)

    n_spk = len({s.speaker_id for s in segs})
    print(
        f"[alm-worker] vad_whisper_crops n={len(segs)} speakers={n_spk} lang={language or 'auto'}",
        flush=True,
    )
    turns = _transcribe_segments(
        wav,
        sample_rate,
        segs,
        language=language,
        asr_model_id=asr_model_id,
    )
    if not turns:
        return [], "", ""

    # If ASR ends with 1 speaker after merges, collapse any residual dual labels.
    if kn and len({t.speaker for t in turns}) == 1:
        only = next(iter({t.speaker for t in turns}))
        for i, t in enumerate(turns):
            if t.speaker != only:
                turns[i] = SpeakerTurn(
                    speaker=only,
                    start_sec=t.start_sec,
                    end_sec=t.end_sec,
                    text=t.text,
                    text_original=t.text_original,
                    confidence=t.confidence,
                    alignment=t.alignment,
                )

    api = [t.to_api_dict() for t in turns]
    transcript = " ".join(t.text for t in turns).strip()
    transcript_original = " ".join(t.text_original for t in turns).strip()
    logger.info(
        "VAD pipeline: %d speakers, %d turns",
        len({t.speaker for t in turns}),
        len(turns),
    )
    return api, transcript, transcript_original
