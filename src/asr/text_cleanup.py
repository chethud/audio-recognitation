"""Clean common ASR artifacts and hallucinations from Whisper output."""
from __future__ import annotations

import re

# Letters from major Indic scripts + Latin (for Kannada, Hindi, Tamil, …).
_UNICODE_WORD = r"[\w\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]+"
_SENTENCE_END = r"[.!?।॥]\s*"

_SCRIPT_RANGES: list[tuple[str, str]] = [
    ("kn", r"[\u0C80-\u0CFF]"),  # Kannada
    ("te", r"[\u0C01-\u0C7F]"),  # Telugu
    ("ta", r"[\u0B80-\u0BFF]"),  # Tamil
    ("ml", r"[\u0D00-\u0D7F]"),  # Malayalam
    ("hi", r"[\u0900-\u097F]"),  # Hindi / Devanagari
    ("bn", r"[\u0980-\u09FF]"),  # Bengali
    ("gu", r"[\u0A80-\u0AFF]"),  # Gujarati
    ("pa", r"[\u0A00-\u0A7F]"),  # Gurmukhi
    ("or", r"[\u0B00-\u0B7F]"),  # Odia
]

INDIC_LANGUAGE_CODES = frozenset(code for code, _ in _SCRIPT_RANGES)


def strip_language_tag_prefix(text: str) -> str:
    """Remove leading '[Kannada] ' style tag from stored originals."""
    return re.sub(r"^\[[^\]]+\]\s*", "", (text or "").strip()).strip()


def infer_language_from_text(text: str) -> str | None:
    """Guess ISO code from Unicode script in transcript."""
    t = text or ""
    best_code: str | None = None
    best_count = 0
    for code, pattern in _SCRIPT_RANGES:
        count = len(re.findall(pattern, t))
        if count > best_count:
            best_count = count
            best_code = code
    if best_count >= 2:
        return best_code
    return None


def contains_indic_script(text: str) -> bool:
    return infer_language_from_text(text) is not None


def _normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def collapse_repeated_phrases(text: str) -> str:
    """
    Remove Whisper stutter loops — same line or sentence repeated many times.
    Works for Kannada and other non-Latin scripts.
    """
    cleaned = _normalize_phrase(text)
    if not cleaned:
        return cleaned

    # Entire string is one short phrase copied N times in a row.
    cleaned = _collapse_whole_string_loops(cleaned)

    # Drop consecutive duplicate sentences/clauses.
    parts = re.split(f"({_SENTENCE_END})", cleaned)
    rebuilt: list[str] = []
    prev_key: str | None = None
    buf = ""
    for part in parts:
        buf += part
        if re.search(_SENTENCE_END + r"$", buf):
            sentence = buf.strip()
            key = _normalize_phrase(sentence)
            if key and key != prev_key:
                rebuilt.append(sentence)
                prev_key = key
            buf = ""
    if buf.strip():
        tail = buf.strip()
        key = _normalize_phrase(tail)
        if key != prev_key:
            rebuilt.append(tail)

    if rebuilt:
        cleaned = " ".join(rebuilt)
    else:
        cleaned = _normalize_phrase(cleaned)

    # Repeated single words (any script): word word word word
    cleaned = re.sub(
        rf"({_UNICODE_WORD})(?:\s+\1){{3,}}",
        r"\1",
        cleaned,
        flags=re.UNICODE,
    )

    # Repeated multi-word phrase in a row (4+ chars, 2+ repeats).
    cleaned = re.sub(
        r"(.{4,120}?)(?:\s*\1){2,}",
        r"\1",
        cleaned,
        flags=re.DOTALL,
    )

    return _normalize_phrase(cleaned)


def _collapse_whole_string_loops(text: str, *, min_unit: int = 6) -> str:
    """If text == unit * N, return unit once."""
    t = text.strip()
    n = len(t)
    if n < min_unit * 2:
        return t

    for size in range(min(n // 2, 240), min_unit - 1, -1):
        unit = t[:size]
        if len(unit.strip()) < min_unit:
            continue
        if n % size == 0 and t == unit * (n // size):
            return unit.strip()

        reps = 0
        pos = 0
        while pos + size <= n and t[pos : pos + size].strip() == unit.strip():
            reps += 1
            pos += size
        if reps >= 3:
            remainder = t[pos:].strip()
            return f"{unit.strip()} {remainder}".strip() if remainder else unit.strip()

    return t


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
    # Collapse repeated Latin words (the the the)
    cleaned = re.sub(r"\b(\w+)(?:\s+\1){3,}\b", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = collapse_repeated_phrases(cleaned)
    return cleaned


def is_meaningful_speech(text: str) -> bool:
    """True if transcript contains real words, not just noise markers."""
    t = (text or "").strip()
    if not t or t == "[No speech detected]":
        return False
    without_markers = t.replace("[vocalization]", "").strip()
    if len(without_markers) < 3:
        return False
    if re.search(r"[A-Za-z\u0900-\u0D7F]{2,}", without_markers):
        return True
    return len(without_markers) >= 4


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
