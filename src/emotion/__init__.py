"""Speech emotion estimation from raw audio (pretrained HF models)."""
from src.emotion.emotion_module import predict_emotion_from_audio

__all__ = ["predict_emotion_from_audio"]
