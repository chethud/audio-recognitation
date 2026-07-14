"""Normalize speaker turns for API / UI consumption."""
from __future__ import annotations

from typing import Any


def _turn_duration(turn: dict[str, Any]) -> float:
    start = turn.get("start_sec")
    end = turn.get("end_sec")
    if start is not None and end is not None and end > start:
        return float(end - start)
    return max(len((turn.get("text") or "").split()), 1) / 4.0


def _merge_consecutive_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge only consecutive SAME-speaker turns (never different speakers)."""
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
    min_minority_ratio: float = 0.05,
    min_minority_turns: int = 2,
) -> list[dict[str, Any]]:
    """Merge a spurious second cluster (common with one narrator / one language).

    Only collapses when the minority speaker is truly negligible — a small share
    of talk-time AND only a turn or two — so genuine short interjections
    (e.g. one person speaking much less) stay as a separate speaker.
    """
    speakers = {t["speaker"] for t in turns}
    if len(speakers) < 2:
        return turns

    durations: dict[str, float] = {}
    counts: dict[str, int] = {}
    for turn in turns:
        sp = str(turn.get("speaker", ""))
        durations[sp] = durations.get(sp, 0.0) + _turn_duration(turn)
        counts[sp] = counts.get(sp, 0) + 1

    total = sum(durations.values())
    if total <= 0:
        return turns

    ordered = sorted(durations.items(), key=lambda x: x[1], reverse=True)
    majority_sp = ordered[0][0]
    minority_sp = ordered[-1][0]
    minority_ratio = ordered[-1][1] / total
    minority_turns = counts.get(minority_sp, 0)
    if minority_ratio >= min_minority_ratio or minority_turns > min_minority_turns:
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
    """Plain speaker-prefixed transcript (not the timed UI layout)."""
    return "\n\n".join(
        f"{t['speaker']}: {t['text']}" for t in turns if (t.get("text") or "").strip()
    )


def normalize_speaker_turns(
    turns: list[dict[str, Any]],
    transcript: str,
    transcript_original: str,
    *,
    skip_weak_collapse: bool = False,
    language: str | None = None,
) -> tuple[list[dict[str, Any]], str, str, int]:
    """
    Keep speaker turns whenever present (including a single speaker).
    Same-speaker consecutive turns may merge; different speakers never merge.

    For Kannada (``language=kn``): aggressively collapse phantom second
    speakers and, when only one remains, return a continuous transcript
    without Speaker 1/2 timed blocks (UI shows plain Kannada text).
    """
    if not turns:
        return [], transcript, transcript_original, 0

    lang = (language or "").strip().lower()
    is_kn = lang in {"kn", "kannada"}

    turns = _merge_consecutive_turns(turns)

    # Always allow weak-second collapse for VAD — MFCC can invent a phantom Speaker 2.
    # Keep whisperx alignment strictly (high quality word-level diarization).
    if not skip_weak_collapse and any(t.get("alignment") == "whisperx" for t in turns):
        skip_weak_collapse = True

    if not skip_weak_collapse:
        if is_kn:
            # Stricter Kannada collapse: unbalanced OR short ping-pong → 1 speaker.
            turns = _collapse_weak_second_speaker(
                turns,
                min_minority_ratio=0.32,
                min_minority_turns=2,
            )
            turns = _collapse_alternating_fake_dialogue(
                turns,
                min_switch_ratio=0.55,
                max_avg_dur=8.0,
            )
            # Drop sub-400ms flips before merge.
            turns = [
                t
                for t in turns
                if _turn_duration(t) >= 0.4 or len(turns) <= 2
            ] or turns
            turns = _merge_consecutive_turns(turns)
        else:
            turns = _collapse_weak_second_speaker(
                turns,
                min_minority_ratio=0.22,
                min_minority_turns=3,
            )
            turns = _collapse_alternating_fake_dialogue(turns)

    speakers = {t.get("speaker") for t in turns if t.get("speaker")}
    if not speakers:
        plain = _plain_transcript_from_turns(turns) or transcript
        plain_orig = _plain_original_from_turns(turns) or transcript_original or plain
        return [], plain, plain_orig, 0

    # Single speaker → continuous Kannada transcript (no Person 1/2 UI blocks).
    if len(speakers) == 1:
        plain = _plain_transcript_from_turns(turns) or transcript
        plain_orig = _plain_original_from_turns(turns) or transcript_original or plain
        if is_kn:
            # Empty speaker_turns → frontend shows one continuous block.
            return [], plain, plain_orig, 1
        return turns, plain, plain_orig, 1

    formatted = format_multispeaker_transcript(turns)
    plain_fallback = _plain_transcript_from_turns(turns) or transcript
    plain_orig = _plain_original_from_turns(turns) or transcript_original or plain_fallback
    return turns, formatted or plain_fallback, plain_orig, len(speakers)


def _collapse_alternating_fake_dialogue(
    turns: list[dict[str, Any]],
    *,
    min_switch_ratio: float = 0.7,
    max_avg_dur: float = 5.0,
) -> list[dict[str, Any]]:
    """
    Detect monologue falsely split into short alternating Speaker 1/2 turns.
    If nearly every turn switches speaker and turns are short → collapse to one.
    """
    if len(turns) < 4:
        return turns
    speakers = {t.get("speaker") for t in turns if t.get("speaker")}
    if len(speakers) != 2:
        return turns

    switches = sum(
        1
        for i in range(1, len(turns))
        if turns[i].get("speaker") != turns[i - 1].get("speaker")
    )
    switch_ratio = switches / max(len(turns) - 1, 1)
    avg_dur = sum(_turn_duration(t) for t in turns) / len(turns)
    # Ping-pong of short turns is typical of false VAD alternation.
    if switch_ratio >= min_switch_ratio and avg_dur <= max_avg_dur:
        majority = max(
            speakers,
            key=lambda sp: sum(
                _turn_duration(t) for t in turns if t.get("speaker") == sp
            ),
        )
        try:
            import logging

            logging.getLogger(__name__).info(
                "Collapsing alternating fake dialogue "
                "(switch_ratio=%.2f avg_dur=%.1fs) → %s",
                switch_ratio,
                avg_dur,
                majority,
            )
        except Exception:
            pass
        relabeled = []
        for turn in turns:
            item = dict(turn)
            item["speaker"] = majority
            relabeled.append(item)
        return _merge_consecutive_turns(relabeled)
    return turns



def _strip_speaker_prefixes(text: str) -> str:
    """Remove 'Speaker N:' / 'Person N:' line prefixes for text dialogue split."""
    import re

    lines: list[str] = []
    for line in re.split(r"[\r\n]+", text or ""):
        line = re.sub(
            r"^(?:Speaker|Person)\s+\d+:\s*", "", line.strip(), flags=re.I
        )
        if line:
            lines.append(line)
    if lines:
        return " ".join(lines)
    return re.sub(
        r"(?:Speaker|Person)\s+\d+:\s*", "", text or "", flags=re.I
    ).strip()


def try_dialogue_speaker_split(
    transcript: str,
    transcript_original: str,
) -> tuple[list[dict[str, Any]], str, str, int]:
    """
    Text-based Speaker 1 / Speaker 2 split for scripted English dialogue
    (e.g. Mrs. Gupta / Mrs. Roy). Used when voice diarization collapses.
    """
    from src.diarization.dialogue_splitter import split_dialogue_speakers

    plain = _strip_speaker_prefixes(transcript) or transcript
    turns, formatted, n = split_dialogue_speakers(plain)
    if n < 2 or not turns:
        return [], transcript, transcript_original, 0

    return normalize_speaker_turns(
        turns,
        formatted,
        transcript_original or plain,
        skip_weak_collapse=True,
    )
