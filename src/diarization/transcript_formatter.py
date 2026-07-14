"""Strict conversation transcript formatting for diarized turns."""
from __future__ import annotations

from typing import Any, Optional


def format_timestamp(sec: Optional[float]) -> str:
    """Format seconds as MM:SS.mmm (or --:--.--- when unknown)."""
    if sec is None:
        return "--:--.---"
    try:
        s = max(0.0, float(sec))
    except (TypeError, ValueError):
        return "--:--.---"
    minutes = int(s // 60)
    rem = s - minutes * 60
    return f"{minutes:02d}:{rem:06.3f}"


def detected_speakers(turns: list[dict[str, Any]]) -> list[str]:
    """Unique speaker IDs in first-appearance order."""
    out: list[str] = []
    seen: set[str] = set()
    for turn in turns:
        sp = str(turn.get("speaker") or "").strip()
        if not sp or sp in seen:
            continue
        seen.add(sp)
        out.append(sp)
    return out


def format_conversation_transcript(turns: list[dict[str, Any]]) -> str:
    """
    Exact block layout:

    [00:00.000 - 00:05.120]
    Speaker 1
    Hello everyone.
    """
    blocks: list[str] = []
    for turn in turns:
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        speaker = str(turn.get("speaker") or "Speaker 1").strip()
        start = format_timestamp(turn.get("start_sec"))
        end = format_timestamp(turn.get("end_sec"))
        blocks.append(f"[{start} - {end}]\n{speaker}\n{text}")
    return "\n\n".join(blocks)


def format_speaker_emotion_block(
    speakers: list[str],
    emotion: str,
) -> str:
    """Per-speaker emotion lines (clip-level emotion until per-speaker exists)."""
    label = (emotion or "neutral").strip() or "neutral"
    if not speakers:
        return f"Speaker 1:\n{label}"
    return "\n\n".join(f"{sp}:\n{label}" for sp in speakers)
