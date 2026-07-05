"""Clean common ASR artifacts and hallucinations from Whisper output."""
from __future__ import annotations

import re


def clean_asr_text(text: str) -> str:
    """Remove repetitive hallucination tokens and normalize whitespace."""
    if not text:
        return text

    cleaned = text.strip()

    # Whisper often emits A-A-A-A-... on barking, music, or silence.
    cleaned = re.sub(
        r"(?:\b[A-Za-z]-){12,}[A-Za-z]?\b",
        "[vocalization]",
        cleaned,
    )
    cleaned = re.sub(
        r"\b([A-Za-z])(?:-\1){8,}(?:-\1)*\b",
        r"[vocalization]",
        cleaned,
    )
    # Collapse repeated words (the the the)
    cleaned = re.sub(r"\b(\w+)(?:\s+\1){3,}\b", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_meaningful_speech(text: str) -> bool:
    """True if transcript contains real words, not just noise markers."""
    t = (text or "").strip()
    if not t or t == "[No speech detected]":
        return False
    without_markers = t.replace("[vocalization]", "").strip()
    if len(without_markers) < 3:
        return False
    return bool(re.search(r"[A-Za-z]{2,}", without_markers))


def is_likely_english_text(text: str) -> bool:
    """Heuristic: Latin script English vs mis-detected other languages."""
    t = (text or "").strip()
    if not t:
        return False
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 4:
        return False
    latin = sum(1 for c in letters if ord(c) < 128)
    return (latin / len(letters)) >= 0.85
