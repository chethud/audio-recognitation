"""
Modular ALM-Lite pipeline: ASR + SED + emotion (parallel) → context → LLM answer.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from src.asr import transcribe_bilingual
from src.context_builder import build_structured_context
from src.emotion import predict_emotion_from_audio
from src.reasoning import answer_from_context_fast, answer_question_from_context
from src.sed import detect_sound_events_segmented


def run_alm_lite(
    audio: Union[torch.Tensor, np.ndarray],
    question: str,
    *,
    sample_rate: int = 16000,
    asr_model_id: str = "openai/whisper-tiny",
    sed_model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    llm_model_id: str = "Qwen/Qwen2-0.5B-Instruct",
    max_new_tokens: int = 32,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 2,
    device: Optional[Union[str, torch.device]] = None,
    asr_language: Optional[str] = None,
    asr_segment_sec: float = 2.5,
    sed_top_k: int = 8,
    sed_threshold: float = 0.12,
    sed_segment_sec: float = 2.0,
    include_sed_scores: bool = False,
    emotion_model_id: Optional[str] = None,
    emotion_enabled: bool = True,
    sed_enabled: bool = True,
    llm_enabled: bool = True,
    fast_mode: bool = False,
    max_duration_sec: Optional[float] = None,
) -> Dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if fast_mode:
        llm_enabled = False
        emotion_enabled = False

    emo_id = emotion_model_id or "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_asr = pool.submit(
            transcribe_bilingual,
            audio,
            sample_rate=sample_rate,
            model_id=asr_model_id,
            device=device,
            max_duration_sec=max_duration_sec,
            segment_sec=asr_segment_sec,
        )
        f_sed = (
            pool.submit(
                detect_sound_events_segmented,
                audio,
                sample_rate=sample_rate,
                model_id=sed_model_id,
                device=device,
                top_k=sed_top_k,
                threshold=sed_threshold,
                segment_sec=sed_segment_sec,
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

        asr = f_asr.result()
        sound_events = f_sed.result() if f_sed else []
        emotion_label = f_emo.result() if f_emo else "neutral"

    transcript = asr.transcript
    transcript_original = asr.transcript_original
    language = asr.language
    language_name = asr.language_name
    languages = asr.languages
    language_names = asr.language_names

    context = build_structured_context(
        transcript,
        sound_events,
        emotion=emotion_label,
        include_scores=include_sed_scores,
    )

    response_lang = language
    if language == "multi" and languages:
        response_lang = languages[0]

    if fast_mode or not llm_enabled:
        answer = answer_from_context_fast(
            context,
            question,
            language=response_lang,
            transcript_original=transcript_original,
            languages=languages,
        )
    else:
        answer = answer_question_from_context(
            context=context,
            question=question,
            model_id=llm_model_id,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            device=torch.device(device) if isinstance(device, str) else device,
            response_language=response_lang,
            languages=languages,
        )

    return {
        "transcript": transcript,
        "transcript_original": transcript_original,
        "language": language,
        "language_name": language_name,
        "languages": languages,
        "language_names": language_names,
        "sound_events": sound_events,
        "emotion": emotion_label,
        "context": context,
        "answer": answer,
    }
