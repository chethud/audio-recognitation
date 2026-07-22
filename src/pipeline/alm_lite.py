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
from src.emotion import predict_emotion_from_audio, predict_emotions_per_speaker
from src.reasoning import answer_from_context_fast, answer_question_from_context
from src.sed import detect_sound_events_segmented


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _audio_snapshot(audio: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Independent copy so ASR / SED / emotion do not share mutable buffers."""
    if isinstance(audio, torch.Tensor):
        return audio.detach().float().cpu().numpy().reshape(-1).copy()
    return np.asarray(audio, dtype=np.float32).reshape(-1).copy()


def _run_asr_sed_emo(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int,
    asr_model_id: str,
    device: torch.device,
    max_duration_sec: Optional[float],
    asr_segment_sec: float,
    asr_max_segments: int,
    asr_language: Optional[str],
    diarization_enabled: bool,
    diarization_cfg: dict,
    sed_enabled: bool,
    sed_model_id: str,
    sed_top_k: int,
    sed_threshold: float,
    sed_segment_sec: float,
    sed_max_windows: int,
    sed_max_results: int,
    sed_backend: str,
    emotion_enabled: bool,
    emo_id: str,
    emotion_backend: str,
):
    wav = _audio_snapshot(audio)

    asr = transcribe_bilingual(
        wav,
        sample_rate=sample_rate,
        model_id=asr_model_id,
        device=device,
        max_duration_sec=max_duration_sec,
        segment_sec=asr_segment_sec,
        max_segments=asr_max_segments,
        language=asr_language,
        diarization_enabled=diarization_enabled,
        diarization_max_speakers=int(diarization_cfg.get("max_speakers", 6)),
        diarization_window_sec=float(diarization_cfg.get("window_sec", 1.2)),
        diarization_hop_sec=float(diarization_cfg.get("hop_sec", 0.6)),
        diarization_min_segment_sec=float(diarization_cfg.get("min_segment_sec", 0.4)),
        diarization_distance_threshold=float(
            diarization_cfg.get("distance_threshold", 0.72)
        ),
        diarization_max_sec=float(diarization_cfg.get("max_audio_sec", 0)),
    )

    # Run SED + emotion in parallel after ASR (Whisper must finish first on CPU).
    sound_events: list = []
    emotion_label = "neutral"
    if sed_enabled or emotion_enabled:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_sed = (
                pool.submit(
                    detect_sound_events_segmented,
                    wav,
                    sample_rate=sample_rate,
                    model_id=sed_model_id,
                    device=device,
                    top_k=sed_top_k,
                    threshold=sed_threshold,
                    segment_sec=sed_segment_sec,
                    max_windows=sed_max_windows,
                    max_results=sed_max_results,
                    backend=sed_backend,
                )
                if sed_enabled
                else None
            )
            f_emo = (
                pool.submit(
                    predict_emotion_from_audio,
                    wav,
                    sample_rate=sample_rate,
                    model_id=emo_id,
                    device=device,
                    enabled=True,
                    backend=emotion_backend,
                )
                if emotion_enabled
                else None
            )
            sound_events = f_sed.result() if f_sed else []
            emotion_label = f_emo.result() if f_emo else "neutral"

    return asr, sound_events, emotion_label


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
    asr_segment_sec: float = 4.0,
    asr_max_segments: int = 2,
    diarization_enabled: bool = False,
    diarization_cfg: Optional[dict] = None,
    sed_top_k: int = 5,
    sed_threshold: float = 0.15,
    sed_segment_sec: float = 3.0,
    sed_max_windows: int = 12,
    sed_max_results: int = 12,
    include_sed_scores: bool = False,
    emotion_model_id: Optional[str] = None,
    emotion_enabled: bool = True,
    sed_enabled: bool = True,
    llm_enabled: bool = True,
    fast_mode: bool = False,
    max_duration_sec: Optional[float] = None,
    sed_backend: str = "auto",
    emotion_backend: str = "auto",
    parallel: bool = False,
) -> Dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dev = _resolve_device(device)
    # Run ASR + SED + emotion concurrently on GPU, or on CPU when explicitly enabled.
    run_parallel = dev.type == "cuda" or parallel

    if fast_mode:
        llm_enabled = False
        # Prefer lightweight CNN for SED/emotion. Do NOT force emotion off —
        # that made the UI always show "neutral".
        import sys

        from src.cnn.loader import should_use_cnn

        cnn_ok = should_use_cnn()
        if sys.platform == "win32":
            # HF wav2vec2 + Whisper often segfaults on Windows; CNN is safe.
            if emotion_enabled:
                if cnn_ok and emotion_backend.lower() in ("auto", "cnn"):
                    emotion_backend = "cnn"
                elif emotion_backend.lower() in ("hf", "wav2vec2", "huggingface"):
                    # Keep HF only when explicitly requested.
                    pass
                elif cnn_ok:
                    emotion_backend = "cnn"
                else:
                    # No CNN weights: still try HF rather than fake "neutral".
                    emotion_backend = "hf"
            if sed_enabled:
                configured_backend = str(sed_backend).lower()
                if should_use_cnn() and configured_backend in ("auto", "cnn"):
                    sed_backend = "cnn"
                elif configured_backend in ("hybrid", "both"):
                    sed_backend = "hybrid"
                elif should_use_cnn():
                    sed_backend = "cnn"
                else:
                    sed_enabled = False
        else:
            if emotion_enabled and emotion_backend.lower() in ("auto", "cnn") and cnn_ok:
                emotion_backend = "cnn"
            elif emotion_enabled and emotion_backend.lower() in ("auto", "cnn") and not cnn_ok:
                emotion_backend = "hf"

    emo_id = emotion_model_id or "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    dia_cfg = diarization_cfg or {}

    common = dict(
        audio=audio,
        sample_rate=sample_rate,
        asr_model_id=asr_model_id,
        device=dev,
        max_duration_sec=max_duration_sec,
        asr_segment_sec=asr_segment_sec,
        asr_max_segments=asr_max_segments,
        asr_language=asr_language,
        diarization_enabled=diarization_enabled,
        diarization_cfg=dia_cfg,
        sed_enabled=sed_enabled,
        sed_model_id=sed_model_id,
        sed_top_k=sed_top_k,
        sed_threshold=sed_threshold,
        sed_segment_sec=sed_segment_sec,
        sed_max_windows=sed_max_windows,
        sed_max_results=sed_max_results,
        sed_backend=sed_backend,
        emotion_enabled=emotion_enabled,
        emo_id=emo_id,
        emotion_backend=emotion_backend,
    )

    if run_parallel:
        wav = _audio_snapshot(audio)
        with ThreadPoolExecutor(max_workers=3) as pool:
            f_asr = pool.submit(
                transcribe_bilingual,
                wav,
                sample_rate=sample_rate,
                model_id=asr_model_id,
                device=dev,
                max_duration_sec=max_duration_sec,
                segment_sec=asr_segment_sec,
                max_segments=asr_max_segments,
                language=asr_language,
                diarization_enabled=diarization_enabled,
                diarization_max_speakers=int(dia_cfg.get("max_speakers", 6)),
                diarization_window_sec=float(dia_cfg.get("window_sec", 1.2)),
                diarization_hop_sec=float(dia_cfg.get("hop_sec", 0.6)),
                diarization_min_segment_sec=float(dia_cfg.get("min_segment_sec", 0.4)),
                diarization_distance_threshold=float(
                    dia_cfg.get("distance_threshold", 0.72)
                ),
                diarization_max_sec=float(dia_cfg.get("max_audio_sec", 0)),
            )
            f_sed = (
                pool.submit(
                    detect_sound_events_segmented,
                    wav,
                    sample_rate=sample_rate,
                    model_id=sed_model_id,
                    device=dev,
                    top_k=sed_top_k,
                    threshold=sed_threshold,
                    segment_sec=sed_segment_sec,
                    max_windows=sed_max_windows,
                    max_results=sed_max_results,
                    backend=sed_backend,
                )
                if sed_enabled
                else None
            )
            f_emo = (
                pool.submit(
                    predict_emotion_from_audio,
                    wav,
                    sample_rate=sample_rate,
                    model_id=emo_id,
                    device=dev,
                    enabled=True,
                    backend=emotion_backend,
                )
                if emotion_enabled
                else None
            )
            asr = f_asr.result()
            sound_events = f_sed.result() if f_sed else []
            emotion_label = f_emo.result() if f_emo else "neutral"
    else:
        # Sequential on CPU — avoids Whisper + CNN torch conflicts on Windows.
        asr, sound_events, emotion_label = _run_asr_sed_emo(**common)

    transcript = asr.transcript
    transcript_original = asr.transcript_original
    language = asr.language
    language_name = asr.language_name
    languages = asr.languages
    language_names = asr.language_names
    speaker_turns = asr.speaker_turns
    num_speakers = asr.num_speakers

    # Speech-heavy clips rarely hit ESC-50 classes confidently — if ASR found
    # speech, surface that so "Detected Sounds" is not empty.
    from src.asr.text_cleanup import is_meaningful_speech

    if is_meaningful_speech(transcript) or is_meaningful_speech(transcript_original):
        has_speech_label = any(
            isinstance(e, dict) and "speech" in str(e.get("label", "")).lower()
            for e in sound_events
        )
        if not has_speech_label:
            sound_events = [
                {"label": "Speech", "score": 0.95},
                *list(sound_events or []),
            ]

    context = build_structured_context(
        transcript,
        sound_events,
        emotion=emotion_label,
        include_scores=include_sed_scores,
        speaker_turns=speaker_turns,
    )

    response_lang = language
    if language == "multi" and languages:
        response_lang = languages[0]

    sound_labels = [
        e.get("label", "")
        for e in sound_events
        if isinstance(e, dict) and e.get("label")
    ]

    if fast_mode or not llm_enabled:
        answer = answer_from_context_fast(
            context,
            question,
            language=response_lang,
            transcript_original=transcript_original,
            languages=languages,
            transcript=transcript,
            emotion=emotion_label,
            sound_labels=sound_labels,
            speaker_turns=speaker_turns,
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

    from src.diarization.transcript_formatter import (
        detected_speakers,
        format_conversation_transcript,
    )

    speakers = detected_speakers(speaker_turns) if speaker_turns else []
    formatted = (
        format_conversation_transcript(speaker_turns)
        if speaker_turns
        else (transcript or "")
    )

    speaker_emotions: dict[str, str] = {}
    if emotion_enabled and speaker_turns:
        try:
            speaker_emotions = predict_emotions_per_speaker(
                audio,
                speaker_turns,
                sample_rate=sample_rate,
                model_id=emo_id,
                device=dev,
                enabled=True,
                backend=emotion_backend,
            )
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("Per-speaker emotion failed: %s", exc)
    if not speaker_emotions and emotion_label:
        for sp in (speakers or ["Speaker 1"]):
            speaker_emotions[sp] = emotion_label

    return {
        "transcript": transcript,
        "transcript_original": transcript_original,
        "language": language,
        "language_name": language_name,
        "languages": languages,
        "language_names": language_names,
        "speaker_turns": speaker_turns,
        "num_speakers": num_speakers,
        "detected_speakers": speakers,
        "formatted_transcript": formatted,
        "sound_events": sound_events,
        "emotion": emotion_label,
        "speaker_emotions": speaker_emotions,
        "context": context,
        "answer": answer,
        "summary": answer,
    }
