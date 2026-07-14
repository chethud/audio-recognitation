"""Text-based dialogue splitting when voice diarization finds one speaker."""
from __future__ import annotations

import re
from typing import Any

# Addressing "Mrs. Roy" -> Speaker 1 (neighbor A); "Mrs. Gupta" -> Speaker 2 (neighbor B)
_ADDR_ROY = re.compile(r"\bmrs\.?\s*roy\b", re.I)
_ADDR_GUPTA = re.compile(r"\bmrs\.?\s*gupt?a\b", re.I)
_DIALOGUE_START = re.compile(
    r"let'?s start with the conversation|start with the conversation|"
    r"without any delay|let us start with the conversation|"
    r"start the conversation",
    re.I,
)
_INTERJECTION = re.compile(
    r"\s+(?=(?:Oh|Yeah|Hmm|OK|Okay|Yes|No|Mom|Dear)\b)",
    re.I,
)
_ABBREV_PERIOD = re.compile(
    r"\b(Mrs|Mr|Ms|Dr|Prof|Sr|Jr|vs|etc|i\.e|e\.g)\.",
    re.I,
)
_SHORT_FOLLOWUP = re.compile(
    r"^(?:how are you|how are you doing|how'?s it going|and you)\??\.?$",
    re.I,
)
_SELF_GUPTA = re.compile(r"\b(?:i am|i'?m)\s+mrs\.?\s*gupt?a\b", re.I)
_SELF_ROY = re.compile(r"\b(?:i am|i'?m)\s+mrs\.?\s*roy\b", re.I)


def _mask_abbrev_periods(text: str) -> str:
    return _ABBREV_PERIOD.sub(lambda m: f"{m.group(1)}\u2024", text)


def _unmask_abbrev_periods(text: str) -> str:
    return text.replace("\u2024", ".")


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []

    masked = _mask_abbrev_periods(text)
    parts = re.split(r"(?<=[.!?])\s+", masked)
    out: list[str] = []
    for part in parts:
        part = _unmask_abbrev_periods(part.strip())
        if not part:
            continue
        for sub in _INTERJECTION.split(part):
            sub = _unmask_abbrev_periods(sub.strip(" ,"))
            if sub:
                out.append(sub if sub[-1] in ".!?" else sub + ".")
    return out


def _has_strong_dialogue_cues(text: str) -> bool:
    """True only for scripted two-person dialogue (neighbor lesson pattern)."""
    roy = bool(_ADDR_ROY.search(text))
    gupta = bool(_ADDR_GUPTA.search(text))
    return roy and gupta


def _guess_speaker(
    sentence: str,
    *,
    last: str | None,
    in_dialogue: bool,
    dialogue_active: bool,
) -> str:
    if not in_dialogue:
        return "Speaker 1"

    if _SELF_GUPTA.search(sentence):
        return "Speaker 1"
    if _SELF_ROY.search(sentence):
        return "Speaker 2"
    if _ADDR_ROY.search(sentence) and not _ADDR_GUPTA.search(sentence):
        return "Speaker 1"
    if _ADDR_GUPTA.search(sentence) and not _ADDR_ROY.search(sentence):
        return "Speaker 2"

    if dialogue_active and last and _SHORT_FOLLOWUP.match(sentence.strip()):
        return last

    if dialogue_active and last:
        if last == "Speaker 1":
            return "Speaker 2"
        if last == "Speaker 2":
            return "Speaker 1"

    return "Speaker 1"


def _merge_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for turn in turns:
        if merged and merged[-1]["speaker"] == turn["speaker"]:
            merged[-1]["text"] = f"{merged[-1]['text']} {turn['text']}".strip()
        else:
            merged.append(dict(turn))
    return merged


def split_dialogue_speakers(transcript: str) -> tuple[list[dict[str, Any]], str, int]:
    """
    Split into Speaker 1 / Speaker 2 only for clear two-character scripted dialogue.
    Single-speaker monologue returns no speaker turns (plain transcript).
    """
    if not _has_strong_dialogue_cues(transcript):
        return [], transcript, 0

    sentences = _split_sentences(transcript)
    if len(sentences) < 2:
        return [], transcript, 0

    dialogue_idx = 0
    for i, sent in enumerate(sentences):
        if _DIALOGUE_START.search(sent):
            dialogue_idx = min(i + 1, len(sentences) - 1)
            break
        if re.search(r"\bgood morning mrs\.?\s", sent, re.I):
            dialogue_idx = i
            break

    turns: list[dict[str, Any]] = []
    last: str | None = None
    dialogue_active = False

    for i, sent in enumerate(sentences):
        in_dialogue = i >= dialogue_idx
        if in_dialogue and (
            _ADDR_ROY.search(sent)
            or _ADDR_GUPTA.search(sent)
            or _SELF_ROY.search(sent)
            or _SELF_GUPTA.search(sent)
        ):
            dialogue_active = True

        speaker = _guess_speaker(
            sent,
            last=last,
            in_dialogue=in_dialogue,
            dialogue_active=dialogue_active,
        )
        last = speaker
        turns.append(
            {
                "speaker": speaker,
                "text": sent,
                "text_original": sent,
                "start_sec": None,
                "end_sec": None,
            }
        )

    turns = _merge_turns(turns)
    speakers = {t["speaker"] for t in turns}

    if len(speakers) < 2:
        return [], transcript, 0

    from src.diarization.transcript_formatter import format_conversation_transcript

    formatted = format_conversation_transcript(turns)
    return turns, formatted, len(speakers)
