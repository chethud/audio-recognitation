"""
Modular ALM-Lite pipeline: joint speech and non-speech understanding with reasoning.
Flow: Audio → ASR (+ optional SED/emotion in parallel) → Context → LLM or fast answer.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from src.asr import transcribe_audio
from src.context_builder import build_structured_context
from src.emotion import predict_emotion_from_audio
from src.reasoning import answer_from_context_fast, answer_question_from_context
from src.sed import detect_sound_events


def run_alm_lite(
    audio: Union[torch.Tensor, np.ndarray],
    question: str,
    *,
    sample_rate: int = 16000,
    asr_model_id: str = "openai/whisper-tiny",
    sed_model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    llm_model_id: str = "Qwen/Qwen2-0.5B-Instruct",
    max_new_tokens: int = 48,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3,
    device: Optional[Union[str, torch.device]] = None,
    asr_language: Optional[str] = None,
    sed_top_k: int = 3,
    sed_threshold: float = 0.3,
    include_sed_scores: bool = False,
    emotion_model_id: Optional[str] = None,
    emotion_enabled: bool = False,
    sed_enabled: bool = False,
    llm_enabled: bool = False,
    fast_mode: bool = True,
) -> Dict[str, Any]:
    """
    Run the full ALM-Lite modular pipeline on one audio and one question.

    Returns:
        dict with keys: transcript, sound_events, emotion, context, answer
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if fast_mode:
        sed_enabled = False
        llm_enabled = False
        emotion_enabled = False

    emo_id = emotion_model_id or "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

    if not sed_enabled and not emotion_enabled:
        transcript = transcribe_audio(
            audio,
            sample_rate=sample_rate,
            model_id=asr_model_id,
            device=device,
            language=asr_language,
        )
        sound_events: list = []
        emotion_label = "neutral"
    else:
        with ThreadPoolExecutor(max_workers=3) as pool:
            f_asr = pool.submit(
                transcribe_audio,
                audio,
                sample_rate=sample_rate,
                model_id=asr_model_id,
                device=device,
                language=asr_language,
            )
            f_sed = (
                pool.submit(
                    detect_sound_events,
                    audio,
                    sample_rate=sample_rate,
                    model_id=sed_model_id,
                    device=device,
                    top_k=sed_top_k,
                    threshold=sed_threshold,
                )
                if sed_enabled
                else None
            )
            f_emo = (
                pool.submit(
                    predict_emotion_from_audio,
                    audio,
                    sample_rate=sample_rate,
                    model_id=emo_id,
                    device=device,
                    enabled=True,
                )
                if emotion_enabled
                else None
            )

            transcript = f_asr.result()
            sound_events = f_sed.result() if f_sed else []
            emotion_label = f_emo.result() if f_emo else "neutral"

    context = build_structured_context(
        transcript,
        sound_events,
        emotion=emotion_label,
        include_scores=include_sed_scores,
    )

    if fast_mode or not llm_enabled:
        answer = answer_from_context_fast(context, question)
    else:
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
