"""
Pydantic models for API request/response.
"""
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class InferenceRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class InferenceResponse(BaseModel):
    id: int
    answer: str
    question: str
    audio_filename: str


class InferenceResponseModular(BaseModel):
    """ALM-Lite modular pipeline: includes transcript, sound_events, context."""
    id: int
    answer: str
    question: str
    audio_filename: str
    transcript: str = ""
    sound_events: list = []  # [{"label": str, "score": float}]
    emotion: str = ""
    context: str = ""


class AnalyzeResponse(BaseModel):
    """Production-style ALM-Lite output (matches POST /analyze contract)."""
    transcript: str = ""
    transcript_original: str = ""
    language: str = "en"
    sounds: List[str] = []
    emotion: str = ""
    answer: str = ""
    question: str = ""
    audio_filename: str = ""
    log_id: Optional[str] = None  # SQLite analyze_logs.id


class AnalyzeLogItem(BaseModel):
    """One row from SQLite ``analyze_logs``."""
    id: int
    audio_filename: str
    question: str
    transcript: str = ""
    sounds: List[str] = []
    emotion: str = ""
    answer: str = ""
    created_at: str


class HistoryItem(BaseModel):
    id: int
    audio_filename: str
    question: str
    answer: str
    created_at: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model_ready: bool = False


class UserPublic(BaseModel):
    id: int
    email: str


class SignupRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        s = v.strip().lower()
        if "@" not in s or "." not in s.split("@", 1)[-1]:
            raise ValueError("Invalid email address")
        return s


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic
