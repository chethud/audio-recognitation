"""Speech emotion estimation from raw audio (CNN + HuggingFace)."""
from src.emotion.emotion_module import (
    predict_emotion_from_audio,
    predict_emotions_per_speaker,
)

__all__ = ["predict_emotion_from_audio", "predict_emotions_per_speaker"]
