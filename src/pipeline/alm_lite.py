"""
Modular ALM-Lite pipeline: joint speech and non-speech understanding with reasoning.
Flow: Audio → ASR (transcript) + SED (events) → Structured Context → LLM → Answer.
"""
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from src.asr import transcribe_audio
from src.context_builder import build_structured_context
from src.emotion import predict_emotion_from_audio
from src.reasoning import answer_question_from_context
from src.sed import detect_sound_events


def run_alm_lite(
    audio: Union[torch.Tensor, np.ndarray],
    question: str,
    *,
    sample_rate: int = 16000,
    asr_model_id: str = "openai/whisper-small",
    sed_model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    llm_model_id: str = "Qwen/Qwen2-0.5B-Instruct",
    max_new_tokens: int = 256,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 4,
    device: Optional[Union[str, torch.device]] = None,
    asr_language: Optional[str] = None,
    sed_top_k: int = 10,
    sed_threshold: float = 0.2,
    include_sed_scores: bool = False,
    emotion_model_id: Optional[str] = None,
    emotion_enabled: bool = True,
) -> Dict[str, Any]:
    """
    Run the full ALM-Lite modular pipeline on one audio and one question.

    Returns:
        dict with keys: transcript, sound_events, emotion, context, answer
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) ASR: speech-to-text
    transcript = transcribe_audio(
        audio,
        sample_rate=sample_rate,
        model_id=asr_model_id,
        device=device,
        language=asr_language,
    )

    # 2) SED: environmental sound events
    sound_events = detect_sound_events(
        audio,
        sample_rate=sample_rate,
        model_id=sed_model_id,
        device=device,
        top_k=sed_top_k,
        threshold=sed_threshold,
    )

    emo_id = emotion_model_id or "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    emotion_label = predict_emotion_from_audio(
        audio,
        sample_rate=sample_rate,
        model_id=emo_id,
        device=device,
        enabled=emotion_enabled,
    )

    # 3) Structured context
    context = build_structured_context(
        transcript,
        sound_events,
        emotion=emotion_label,
        include_scores=include_sed_scores,
    )

    # 4) LLM reasoning
    answer = answer_question_from_context(
        context=context,
        question=question,
        model_id=llm_model_id,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        device=torch.device(device) if isinstance(device, str) else device,
    )

    return {
        "transcript": transcript,
        "sound_events": sound_events,
        "emotion": emotion_label,
        "context": context,
        "answer": answer,
    }
