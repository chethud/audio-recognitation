"""
Builds structured context from speech transcript and sound event labels
for LLM-based reasoning over the audio scene.
"""
from typing import List, Optional


def build_structured_context(
    transcript: str,
    sound_events: List[dict],
    *,
    emotion: Optional[str] = None,
    include_scores: bool = False,
    template: Optional[str] = None,
) -> str:
    """
    Combine transcript and SED labels into a single context string for the LLM.

    Args:
        transcript: ASR output (speech-to-text).
        sound_events: List of {"label": str, "score": float} from SED.
        include_scores: If True, append confidence scores for events.
        template: Custom format string with placeholders {transcript}, {events}.
                  Default uses "Speech: ... | Non-speech: ...".

    Returns:
        Single string describing the audio scene.
    """
    speech_part = (transcript or "").strip()
    if not speech_part:
        speech_part = "[No speech detected]"

    if sound_events:
        if include_scores:
            event_str = ", ".join(
                f"{e['label']} ({e.get('score', 0):.2f})" for e in sound_events
            )
        else:
            event_str = ", ".join(e["label"] for e in sound_events)
        non_speech_part = event_str
    else:
        non_speech_part = "[No environmental sounds detected]"

    emo = (emotion or "").strip() or "unknown"
    if template is not None:
        return template.format(transcript=speech_part, events=non_speech_part, emotion=emo)

    return (
        f"Speech (transcript): {speech_part}\n"
        f"Speaker emotion (estimated): {emo}\n"
        f"Non-speech (environmental sounds): {non_speech_part}"
    )
