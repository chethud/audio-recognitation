"""Whisper language code ↔ name helpers (all supported languages)."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _whisper_maps() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """name→iso, iso→name, iso→display label."""
    from transformers.models.whisper.tokenization_whisper import LANGUAGES

    name_to_iso = dict(LANGUAGES)
    iso_to_name = {iso: name for name, iso in LANGUAGES.items()}
    labels = {iso: name.replace("_", " ").title() for name, iso in LANGUAGES.items()}
    labels["en"] = "English"
    return name_to_iso, iso_to_name, labels


def whisper_language_name(code_or_name: str | None) -> str | None:
    """Convert ISO code or Whisper name to Whisper pipeline language name."""
    if not code_or_name:
        return None
    key = code_or_name.strip().lower()
    name_to_iso, iso_to_name, _ = _whisper_maps()
    if key in name_to_iso:
        return key
    if key in iso_to_name:
        return iso_to_name[key]
    return None


def whisper_language_code(code_or_name: str | None) -> str:
    """Normalize to ISO 639-1 (or Whisper code) for API responses."""
    if not code_or_name:
        return "en"
    key = code_or_name.strip().lower()
    name_to_iso, iso_to_name, _ = _whisper_maps()
    if key in name_to_iso:
        return name_to_iso[key]
    if key in iso_to_name:
        return key
    return key


def language_label(code_or_name: str | None) -> str:
    """Human-readable label for UI (e.g. kn → Kannada)."""
    iso = whisper_language_code(code_or_name)
    _, _, labels = _whisper_maps()
    return labels.get(iso, (code_or_name or "English").upper())


def token_to_language_code(token: str) -> str:
    """Map Whisper language token like <|kannada|> to ISO code."""
    if not token.startswith("<|") or not token.endswith("|>"):
        return "en"
    name = token[2:-2].lower()
    if name == "english":
        return "en"
    name_to_iso, _, _ = _whisper_maps()
    return name_to_iso.get(name, name)
