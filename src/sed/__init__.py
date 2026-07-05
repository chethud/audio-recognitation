"""Sound event detection (AST / AudioSet)."""
from .sed_module import detect_sound_events, detect_sound_events_segmented

__all__ = ["detect_sound_events", "detect_sound_events_segmented"]
