"""ASR (Whisper)."""
from .whisper_asr import transcribe_audio, transcribe_bilingual, TranscriptionResult

__all__ = ["transcribe_audio", "transcribe_bilingual", "TranscriptionResult"]
