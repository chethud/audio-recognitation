"""Shared types for the WhisperX + PyAnnote diarization pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DiarizationSegment:
    """A contiguous time span attributed to one speaker cluster."""

    speaker_id: str
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class WordToken:
    """Single word with timestamps and optional speaker assignment."""

    word: str
    start_sec: float
    end_sec: float
    score: float = 0.0
    speaker_id: str | None = None
    confidence: float = 0.0

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class SpeakerTurn:
    """Merged dialogue block for API / frontend consumption."""

    speaker: str
    start_sec: float
    end_sec: float
    text: str
    text_original: str
    confidence: float = 0.0
    alignment: str = "whisperx"

    def to_api_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "text": self.text,
            "text_original": self.text_original,
            "confidence": round(self.confidence, 3),
            "alignment": self.alignment,
        }


@dataclass
class DiarizationResult:
    """Full diarization output before API normalization."""

    turns: list[SpeakerTurn] = field(default_factory=list)
    transcript: str = ""
    transcript_original: str = ""
    num_speakers: int = 0
    processing_sec: float = 0.0
    word_count: int = 0
