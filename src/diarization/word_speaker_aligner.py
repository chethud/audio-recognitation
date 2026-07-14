"""Word-level speaker assignment, smoothing, and turn merging."""
from __future__ import annotations

import logging
import re
from typing import Sequence

from src.diarization.types import DiarizationSegment, SpeakerTurn, WordToken

logger = logging.getLogger(__name__)


def _overlap_sec(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers_by_overlap(
    words: list[WordToken],
    diar_segments: Sequence[DiarizationSegment],
    *,
    min_confidence: float = 0.35,
) -> list[WordToken]:
    """
    Assign each word to the diarization segment with maximum temporal overlap.
    Confidence = overlap / word duration.
    """
    if not words or not diar_segments:
        return words

    prev_speaker: str | None = None
    for word in words:
        best_sp: str | None = None
        best_overlap = 0.0
        word_dur = max(word.duration_sec, 1e-3)

        for seg in diar_segments:
            ov = _overlap_sec(word.start_sec, word.end_sec, seg.start_sec, seg.end_sec)
            if ov > best_overlap:
                best_overlap = ov
                best_sp = seg.speaker_id

        if best_sp is None:
            # No overlap — nearest segment by midpoint
            mid = (word.start_sec + word.end_sec) / 2.0
            nearest = min(
                diar_segments,
                key=lambda s: min(abs(mid - s.start_sec), abs(mid - s.end_sec)),
            )
            best_sp = nearest.speaker_id
            best_overlap = word_dur * 0.25

        confidence = best_overlap / word_dur
        if confidence < min_confidence and prev_speaker is not None:
            word.speaker_id = prev_speaker
            word.confidence = confidence
        else:
            word.speaker_id = best_sp
            word.confidence = confidence
            prev_speaker = best_sp

    return words


def apply_whisperx_word_speakers(
    words: list[WordToken],
    aligned_segments: list[dict],
) -> list[WordToken]:
    """Import speaker labels already assigned by whisperx.assign_word_speakers."""
    out: list[WordToken] = []
    for seg in aligned_segments:
        for w in seg.get("words") or []:
            text = (w.get("word") or "").strip()
            if not text:
                continue
            start = float(w.get("start") or seg.get("start") or 0.0)
            end = float(w.get("end") or start)
            sp = w.get("speaker")
            out.append(
                WordToken(
                    word=text,
                    start_sec=start,
                    end_sec=end,
                    score=float(w.get("score") or 0.0),
                    speaker_id=str(sp) if sp else None,
                    confidence=float(w.get("score") or 0.8) if sp else 0.0,
                )
            )
    return out if out else words


def remove_short_speaker_switches(
    words: list[WordToken],
    *,
    min_switch_sec: float = 0.3,
) -> list[WordToken]:
    """Relabel isolated speaker islands shorter than min_switch_sec."""
    if len(words) < 2:
        return words

    labels = [w.speaker_id for w in words]
    i = 0
    while i < len(words):
        j = i + 1
        while j < len(words) and words[j].speaker_id == words[i].speaker_id:
            j += 1
        run_start = words[i].start_sec
        run_end = words[j - 1].end_sec
        duration = run_end - run_start

        if duration < min_switch_sec and i > 0 and j < len(words):
            prev_sp = words[i - 1].speaker_id
            next_sp = words[j].speaker_id if j < len(words) else None
            if prev_sp and prev_sp == next_sp:
                for k in range(i, j):
                    words[k].speaker_id = prev_sp
                    labels[k] = prev_sp

        i = j

    changed = sum(1 for a, b in zip(labels, [w.speaker_id for w in words]) if a != b)
    if changed:
        logger.info("Smoothed %d words in short speaker switches (<%.0fms)", changed, min_switch_sec * 1000)
    return words


def _speaker_id_to_person(speaker_id: str, mapping: dict[str, str]) -> str:
    if speaker_id in mapping:
        return mapping[speaker_id]
    idx = len(mapping) + 1
    mapping[speaker_id] = f"Speaker {idx}"
    return mapping[speaker_id]


def merge_words_to_turns(
    words: list[WordToken],
    *,
    gap_merge_sec: float = 0.6,
) -> list[SpeakerTurn]:
    """
    Merge consecutive words from the same speaker into dialogue blocks.
    Adjacent same-speaker blocks separated by a short gap are also merged.
    """
    if not words:
        return []

    mapping: dict[str, str] = {}
    turns: list[SpeakerTurn] = []

    for word in words:
        if not word.word.strip() or not word.speaker_id:
            continue
        person = _speaker_id_to_person(word.speaker_id, mapping)

        if turns and turns[-1].speaker == person:
            gap = word.start_sec - turns[-1].end_sec
            if gap <= gap_merge_sec:
                turns[-1].text = f"{turns[-1].text} {word.word}".strip()
                turns[-1].text_original = turns[-1].text
                turns[-1].end_sec = word.end_sec
                turns[-1].confidence = (turns[-1].confidence + word.confidence) / 2.0
                continue

        turns.append(
            SpeakerTurn(
                speaker=person,
                start_sec=word.start_sec,
                end_sec=word.end_sec,
                text=word.word,
                text_original=word.word,
                confidence=word.confidence,
            )
        )

    return turns


def format_timestamp(sec: float) -> str:
    """Format seconds as HH:MM:SS for logging."""
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def turns_to_transcript(turns: Sequence[SpeakerTurn]) -> str:
    """Build formatted multi-speaker transcript string."""
    lines: list[str] = []
    for t in turns:
        if not t.text.strip():
            continue
        lines.append(f"{t.speaker}: {t.text}")
    return "\n\n".join(lines)


def extract_words_from_aligned(aligned: dict) -> list[WordToken]:
    """Flatten WhisperX aligned segments into WordToken list."""
    words: list[WordToken] = []
    for seg in aligned.get("segments") or []:
        seg_words = seg.get("words")
        if seg_words:
            for w in seg_words:
                text = (w.get("word") or "").strip()
                if not text:
                    continue
                start = w.get("start")
                end = w.get("end")
                if start is None or end is None:
                    continue
                words.append(
                    WordToken(
                        word=text,
                        start_sec=float(start),
                        end_sec=float(end),
                        score=float(w.get("score") or 0.0),
                        speaker_id=str(w["speaker"]) if w.get("speaker") else None,
                    )
                )
        else:
            text = (seg.get("text") or "").strip()
            if text and seg.get("start") is not None and seg.get("end") is not None:
                for token in re.findall(r"\S+", text):
                    words.append(
                        WordToken(
                            word=token,
                            start_sec=float(seg["start"]),
                            end_sec=float(seg["end"]),
                            score=0.5,
                        )
                    )
    return words
