"""ASR (Whisper)."""
from .whisper_asr import transcribe_audio, transcribe_bilingual, TranscriptionResult
from .whisper_languages import language_label, whisper_language_code

__all__ = [
    "transcribe_audio",
    "transcribe_bilingual",
    "TranscriptionResult",
    "language_label",
    "whisper_language_code",
]
