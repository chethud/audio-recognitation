"""
In-process ALM-Lite inference with model warmup and reuse (much faster than per-request subprocess).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from src.env_setup import configure_ml_env

configure_ml_env()

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE / "config.yaml"

_lock = __import__("threading").Lock()
_warmed = False


def _alm_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("alm_lite", {})


def _is_fast(cfg: dict[str, Any]) -> bool:
    return bool(_alm_cfg(cfg).get("fast_mode", True))


def _load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sed_uses_cnn(cfg: dict[str, Any]) -> bool:
    from src.cnn.loader import should_use_cnn

    sed_cfg = _alm_cfg(cfg).get("sed", {})
    backend = str(sed_cfg.get("backend", "auto")).lower()
    if backend in ("hf", "ast", "huggingface"):
        return False
    if backend in ("cnn", "hybrid", "both"):
        return should_use_cnn(cfg)
    return should_use_cnn(cfg)


def _needs_ast_sed(cfg: dict[str, Any]) -> bool:
    sed_cfg = _alm_cfg(cfg).get("sed", {})
    if not sed_cfg.get("enabled", True):
        return False
    backend = str(sed_cfg.get("backend", "auto")).lower()
    if backend == "cnn":
        return False
    if backend in ("hybrid", "both", "ast", "hf", "huggingface"):
        return True
    from src.cnn.loader import should_use_cnn

    return not should_use_cnn(cfg)


def _emotion_uses_cnn(cfg: dict[str, Any]) -> bool:
    from src.cnn.loader import should_use_cnn

    emo_cfg = _alm_cfg(cfg).get("emotion", {})
    backend = str(emo_cfg.get("backend", "auto")).lower()
    if backend in ("hf", "wav2vec2", "huggingface"):
        return False
    if backend == "cnn":
        return should_use_cnn(cfg)
    return should_use_cnn(cfg)


def warmup() -> None:
    """Load models once at API startup."""
    global _warmed
    with _lock:
        if _warmed:
            return
        cfg = _load_config()
        alm = _alm_cfg(cfg)
        fast = _is_fast(cfg)
        asr_cfg = alm.get("asr", {})
        sed_cfg = alm.get("sed", {})
        emo_cfg = alm.get("emotion", {})
        dia_cfg = alm.get("diarization", {})
        llm_cfg = alm.get("llm", {})

        import torch

        torch.set_grad_enabled(False)
        if not torch.cuda.is_available():
            threads = min(4, max(1, (os.cpu_count() or 4) // 2))
            torch.set_num_threads(threads)

        from src.asr.whisper_asr import _get_asr_pipe

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn_sed = _sed_uses_cnn(cfg)
        cnn_emo = _emotion_uses_cnn(cfg) and emo_cfg.get("enabled", True)
        cnn_ready = False
        if cnn_sed or cnn_emo:
            from src.cnn import warmup_cnn

            cnn_ready = warmup_cnn(cfg)

        sed_backend = str(sed_cfg.get("backend", "auto")).lower()
        skip_hf_sed = sed_backend == "cnn" and not cnn_ready

        if fast:
            if cnn_ready and cnn_sed and cnn_emo:
                mode = "fast (ASR + hybrid SED + CNN emotion)"
            elif cnn_ready and cnn_sed:
                mode = "fast (ASR + hybrid SED)"
            elif skip_hf_sed or not sed_cfg.get("enabled", True):
                mode = "fast (ASR only)"
            else:
                mode = "fast (ASR + SED)"
        else:
            mode = "full (ASR + SED + emotion + LLM)"
        logger.info("Warming up models [%s] on %s …", mode, device)

        _get_asr_pipe(asr_cfg.get("model_id", "openai/whisper-tiny"), device)

        if _needs_ast_sed(cfg) and not skip_hf_sed:
            try:
                from src.sed.sed_module import _get_sed_pipe

                _get_sed_pipe(
                    sed_cfg.get("model_id", "MIT/ast-finetuned-audioset-10-10-0.4593"),
                    device,
                )
            except Exception as exc:
                logger.warning("HF SED warmup skipped: %s", exc)
        elif skip_hf_sed:
            logger.info(
                "SED backend is cnn but checkpoints are missing — skipping HF AST download."
            )
        emo_on = emo_cfg.get("enabled", True)
        if emo_on and not fast and not (cnn_ready and cnn_emo):
            from src.emotion.emotion_module import _get_emotion_pipeline

            _get_emotion_pipeline(
                emo_cfg.get(
                    "model_id",
                    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                ),
                device,
            )
        elif emo_on and fast and cnn_ready and cnn_emo:
            from src.cnn.loader import _load_emo_bundle

            _load_emo_bundle(cfg)
        # Diarization (Wav2Vec2) loads on first analyze — keeps /health fast at startup.
        if not fast and llm_cfg.get("enabled", True):
            from src.reasoning.llm_reasoning import _get_llm

            _get_llm(llm_cfg.get("model_id", "Qwen/Qwen2-0.5B-Instruct"), device)

        _warmed = True
        logger.info("Model warmup complete [%s].", mode)


def is_ready() -> bool:
    return _warmed


def analyze_file(audio_path: str, question: str) -> dict[str, Any]:
    """Run pipeline on an audio file path. Returns worker-style result dict."""
    from src.pipeline import run_alm_lite
    from src.utils import load_audio_from_file

    cfg = _load_config()
    data_cfg = cfg.get("data", {})
    alm = _alm_cfg(cfg)
    fast = _is_fast(cfg)
    asr_cfg = alm.get("asr", {})
    sed_cfg = alm.get("sed", {})
    llm_cfg = alm.get("llm", {})
    emo_cfg = alm.get("emotion", {})
    dia_cfg = alm.get("diarization", {})
    max_sec = data_cfg.get("max_audio_length_sec", 10)

    # Decode only the first N seconds (do not hold model lock while loading file).
    audio = load_audio_from_file(
        audio_path,
        sr=data_cfg.get("sample_rate", 16000),
        max_sec=max_sec,
    )
    if audio.dim() == 2:
        audio = audio.squeeze(0)

    with _lock:
        try:
            result = run_alm_lite(
                audio.numpy(),
                question,
                sample_rate=data_cfg.get("sample_rate", 16000),
                asr_model_id=asr_cfg.get("model_id", "openai/whisper-tiny"),
                asr_language=asr_cfg.get("language"),
                sed_model_id=sed_cfg.get(
                    "model_id", "MIT/ast-finetuned-audioset-10-10-0.4593"
                ),
                sed_top_k=sed_cfg.get("top_k", 10),
                sed_threshold=sed_cfg.get("threshold", 0.04),
                sed_segment_sec=sed_cfg.get("segment_sec", 2.5),
                sed_max_windows=sed_cfg.get("max_windows", 12),
                sed_max_results=sed_cfg.get("max_results", 12),
                asr_segment_sec=asr_cfg.get("segment_sec", 4.0),
                asr_max_segments=asr_cfg.get("max_segments", 2),
                diarization_enabled=bool(dia_cfg.get("enabled", False)),
                diarization_cfg=dia_cfg,
                llm_model_id=llm_cfg.get("model_id", "Qwen/Qwen2-0.5B-Instruct"),
                max_new_tokens=llm_cfg.get("max_new_tokens", 32),
                repetition_penalty=llm_cfg.get("repetition_penalty", 1.1),
                no_repeat_ngram_size=llm_cfg.get("no_repeat_ngram_size", 2),
                emotion_model_id=emo_cfg.get("model_id"),
                emotion_enabled=emo_cfg.get("enabled", True),
                sed_enabled=sed_cfg.get("enabled", True),
                llm_enabled=llm_cfg.get("enabled", True),
                fast_mode=fast,
                max_duration_sec=max_sec if max_sec and max_sec > 0 else None,
                sed_backend=sed_cfg.get("backend", "auto"),
                emotion_backend=emo_cfg.get("backend", "auto"),
            )
            return {
                "ok": True,
                "answer": result["answer"],
                "transcript": result["transcript"],
                "transcript_original": result.get("transcript_original", ""),
                "language": result.get("language", "en"),
                "language_name": result.get("language_name", "English"),
                "languages": result.get("languages", []),
                "language_names": result.get("language_names", []),
                "speaker_turns": result.get("speaker_turns", []),
                "num_speakers": result.get("num_speakers", 0),
                "sound_events": result["sound_events"],
                "emotion": result.get("emotion", "neutral"),
                "context": result["context"],
            }
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {str(e)}"}
