"""Normalize speaker turns — only expose multi-speaker UI when 2+ real speakers."""
from __future__ import annotations

from typing import Any


def _turn_duration(turn: dict[str, Any]) -> float:
    start = turn.get("start_sec")
    end = turn.get("end_sec")
    if start is not None and end is not None and end > start:
        return float(end - start)
    return max(len((turn.get("text") or "").split()), 1) / 4.0


def _merge_consecutive_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for turn in turns:
        if merged and merged[-1]["speaker"] == turn["speaker"]:
            merged[-1]["text"] = f"{merged[-1]['text']} {turn['text']}".strip()
            merged[-1]["text_original"] = (
                f"{merged[-1]['text_original']} {turn.get('text_original', turn['text'])}"
            ).strip()
            if turn.get("end_sec") is not None:
                merged[-1]["end_sec"] = turn["end_sec"]
        else:
            merged.append(dict(turn))
    return merged


def _collapse_weak_second_speaker(
    turns: list[dict[str, Any]],
    *,
    min_minority_ratio: float = 0.15,
) -> list[dict[str, Any]]:
    """Merge a spurious second cluster (common with one narrator / one language)."""
    speakers = {t["speaker"] for t in turns}
    if len(speakers) < 2:
        return turns

    durations: dict[str, float] = {}
    for turn in turns:
        sp = str(turn.get("speaker", ""))
        durations[sp] = durations.get(sp, 0.0) + _turn_duration(turn)

    total = sum(durations.values())
    if total <= 0:
        return turns

    ordered = sorted(durations.items(), key=lambda x: x[1], reverse=True)
    majority_sp = ordered[0][0]
    minority_ratio = ordered[-1][1] / total
    if minority_ratio >= min_minority_ratio:
        return turns

    relabeled = []
    for turn in turns:
        item = dict(turn)
        item["speaker"] = majority_sp
        relabeled.append(item)
    return _merge_consecutive_turns(relabeled)


def _plain_transcript_from_turns(turns: list[dict[str, Any]]) -> str:
    return " ".join((t.get("text") or "").strip() for t in turns).strip()


def _plain_original_from_turns(turns: list[dict[str, Any]]) -> str:
    return " ".join(
        (t.get("text_original") or t.get("text") or "").strip() for t in turns
    ).strip()


def format_multispeaker_transcript(turns: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"{t['speaker']}: {t['text']}" for t in turns if (t.get("text") or "").strip()
    )


def normalize_speaker_turns(
    turns: list[dict[str, Any]],
    transcript: str,
    transcript_original: str,
) -> tuple[list[dict[str, Any]], str, str, int]:
    """
    Return speaker turns only when 2+ distinct speakers remain after cleanup.
    Single-speaker audio keeps a plain transcript (no fake Person 1 / Person 2 split).
    """
    if not turns:
        return [], transcript, transcript_original, 0

    turns = _merge_consecutive_turns(turns)
    turns = _collapse_weak_second_speaker(turns)
    speakers = {t.get("speaker") for t in turns if t.get("speaker")}

    if len(speakers) < 2:
        plain = _plain_transcript_from_turns(turns) or transcript
        plain_orig = _plain_original_from_turns(turns) or transcript_original or plain
        return [], plain, plain_orig, 0

    formatted = format_multispeaker_transcript(turns)
    return turns, formatted or transcript, transcript_original, len(speakers)
