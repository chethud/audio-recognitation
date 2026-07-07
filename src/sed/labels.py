"""Normalize SED labels from CNN (ESC-50) and AST (AudioSet) for display."""
from __future__ import annotations

import re

_ESC50_DISPLAY: dict[str, str] = {
    "dog": "dog barking",
    "car_horn": "car honking",
    "clock_alarm": "alarm or phone ringing",
    "clock_tick": "clock ticking",
    "siren": "siren",
    "engine": "car engine",
    "helicopter": "helicopter",
    "airplane": "airplane",
    "train": "train",
    "rain": "rain",
    "thunderstorm": "thunderstorm",
    "wind": "wind",
    "crackling_fire": "fire crackling",
    "glass_breaking": "glass breaking",
    "door_wood_knock": "door knocking",
    "footsteps": "footsteps",
    "vacuum_cleaner": "vacuum cleaner",
    "washing_machine": "washing machine",
    "keyboard_typing": "keyboard typing",
    "mouse_click": "mouse click",
    "fireworks": "fireworks",
    "church_bells": "church bells",
    "crow": "crow",
    "chirping_birds": "birds chirping",
    "crickets": "crickets",
}

_SKIP_SUBSTRINGS = (
    "speech",
    "conversation",
    "narration",
    "monologue",
    "silence",
    "music",
    "sine wave",
    "white noise",
    "pink noise",
)


def should_skip_sound_label(label: str) -> bool:
    lower = label.lower()
    return any(s in lower for s in _SKIP_SUBSTRINGS)


def normalize_sound_label(label: str) -> dict[str, str]:
    """Return merge key and user-facing display name."""
    raw = (label or "").strip()
    if not raw:
        return {"key": "", "display": ""}

    esc_key = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    if esc_key in _ESC50_DISPLAY:
        return {"key": esc_key, "display": _ESC50_DISPLAY[esc_key]}

    lower = raw.lower()
    if "bark" in lower or re.search(r"\bdog\b", lower):
        return {"key": "dog_bark", "display": "dog barking"}
    if "horn" in lower or "honk" in lower:
        return {"key": "car_horn", "display": "car honking"}
    if "telephone" in lower or "phone" in lower or "ringtone" in lower:
        return {"key": "phone_ring", "display": "phone ringing"}
    if "alarm" in lower or "bell ringing" in lower:
        return {"key": "alarm_ring", "display": "alarm ringing"}
    if "siren" in lower:
        return {"key": "siren", "display": "siren"}
    if "engine" in lower or "motor" in lower:
        return {"key": "engine", "display": "vehicle engine"}
    if "rain" in lower:
        return {"key": "rain", "display": "rain"}
    if "thunder" in lower:
        return {"key": "thunder", "display": "thunder"}
    if "wind" in lower:
        return {"key": "wind", "display": "wind"}
    if "footstep" in lower or "walking" in lower:
        return {"key": "footsteps", "display": "footsteps"}
    if "knock" in lower or "door" in lower:
        return {"key": "door_knock", "display": "door knocking"}
    if "bird" in lower or "chirp" in lower:
        return {"key": "birds", "display": "birds chirping"}
    if "baby" in lower or "cry" in lower:
        return {"key": "crying", "display": "crying"}
    if "laugh" in lower:
        return {"key": "laughing", "display": "laughing"}
    if "applause" in lower or "clap" in lower:
        return {"key": "clapping", "display": "clapping"}

    key = re.sub(r"[^a-z0-9]+", "_", lower).strip("_")[:80]
    display = re.sub(r"\s+", " ", raw.replace("_", " ")).strip()
    return {"key": key or lower[:80], "display": display}


def merge_sound_events(
    *event_lists: list[dict],
    max_results: int = 12,
    min_score: float = 0.0,
) -> list[dict]:
    """Merge CNN + AST hits, dedupe by semantic key, keep best score."""
    best: dict[str, float] = {}
    display: dict[str, str] = {}

    for events in event_lists:
        for item in events or []:
            raw = str(item.get("label", "")).strip()
            if not raw or should_skip_sound_label(raw):
                continue
            score = float(item.get("score", 0.0))
            if score < min_score:
                continue
            norm = normalize_sound_label(raw)
            key = norm["key"]
            if not key:
                continue
            if score >= best.get(key, 0.0):
                best[key] = score
                display[key] = norm["display"]

    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:max_results]
    return [{"label": display[k], "score": round(s, 4)} for k, s in ranked]
