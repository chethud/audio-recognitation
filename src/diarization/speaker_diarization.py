"""
Speaker diarization entry points.

Primary backend: WhisperX word alignment + PyAnnote speaker diarization.
Legacy fallback: Wav2Vec2 embeddings + clustering (when WhisperX unavailable).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import yaml

from src.diarization.types import DiarizationResult

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def _load_diarization_cfg() -> dict[str, Any]:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f).get("alm_lite", {}).get("diarization", {}) or {}
    except Exception:
        return {}


def _prepare_wav(audio: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        wav = audio.detach().float().cpu().numpy().reshape(-1)
    else:
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = np.max(np.abs(wav)) if len(wav) else 0.0
    if peak > 0:
        wav = wav / peak
    return wav.astype(np.float32)


def _build_whisperx_pipeline(cfg: dict[str, Any]):
    from src.diarization.model_cache import whisper_size_from_model_id
    from src.diarization.whisperx_pipeline import WhisperXDiarizationPipeline

    alm = {}
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            alm = yaml.safe_load(f).get("alm_lite", {}) or {}
    except Exception:
        pass

    asr_model = (alm.get("asr") or {}).get("model_id", "openai/whisper-base")
    size = cfg.get("whisper_model") or whisper_size_from_model_id(asr_model)

    return WhisperXDiarizationPipeline(
        whisper_model_size=size,
        min_switch_sec=float(cfg.get("min_switch_sec", 0.3)),
        min_confidence=float(cfg.get("min_confidence", 0.35)),
        gap_merge_sec=float(cfg.get("gap_merge_sec", 0.6)),
        batch_size=int(cfg.get("batch_size", 8)),
        align_batch_size=int(cfg.get("align_batch_size", 16)),
    )


def _result_to_api(
    result: DiarizationResult,
) -> tuple[list[dict[str, Any]], str, str]:
    turns = [t.to_api_dict() for t in result.turns]
    return turns, result.transcript, result.transcript_original


def warmup_diarization() -> bool:
    """Pre-load WhisperX ASR model (PyAnnote loads on first diarize)."""
    cfg = _load_diarization_cfg()
    backend = str(cfg.get("backend", "whisperx")).lower()

    if backend == "whisperx":
        try:
            from src.diarization.model_cache import warmup_whisperx_models, whisper_size_from_model_id

            alm = {}
            try:
                with open(_CONFIG_PATH, encoding="utf-8") as f:
                    alm = yaml.safe_load(f).get("alm_lite", {}) or {}
            except Exception:
                pass
            asr_model = (alm.get("asr") or {}).get("model_id", "openai/whisper-base")
            size = cfg.get("whisper_model") or whisper_size_from_model_id(asr_model)
            return warmup_whisperx_models(size)
        except Exception as exc:
            logger.warning("WhisperX diarization warmup failed: %s", exc)
            return False

    return _legacy_warmup()


def _legacy_warmup() -> bool:
    try:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        return True
    except Exception:
        return False


def diarize_speakers(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    max_speakers: int = 6,
    window_sec: float = 1.2,
    hop_sec: float = 0.6,
    min_segment_sec: float = 0.4,
    distance_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    """
    Return time segments labeled Person 1, Person 2, …
    Uses PyAnnote when WhisperX backend is active; legacy clustering otherwise.
    """
    cfg = _load_diarization_cfg()
    backend = str(cfg.get("backend", "whisperx")).lower()

    if backend == "whisperx":
        try:
            from src.diarization.whisperx_pipeline import WhisperXDiarizationPipeline

            pipeline = _build_whisperx_pipeline(cfg)
            wav = _prepare_wav(audio)
            import whisperx

            if sample_rate != 16000:
                import librosa

                wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=16000)

            import tempfile
            import soundfile as sf

            fd, path = tempfile.mkstemp(suffix=".wav")
            import os

            os.close(fd)
            sf.write(path, wav, 16000)
            try:
                audio_16k = whisperx.load_audio(path)
                segments = pipeline._run_pyannote(audio_16k, max_speakers=max_speakers)
            finally:
                Path(path).unlink(missing_ok=True)

            mapping: dict[str, str] = {}
            out: list[dict[str, Any]] = []
            for seg in segments:
                if seg.speaker_id not in mapping:
                    mapping[seg.speaker_id] = f"Speaker {len(mapping) + 1}"
                out.append(
                    {
                        "speaker": mapping[seg.speaker_id],
                        "start_sec": round(seg.start_sec, 2),
                        "end_sec": round(seg.end_sec, 2),
                    }
                )
            n_speakers = len(mapping)
            logger.info("Diarization: %d speaker(s), %d segments", n_speakers, len(out))
            return out
        except Exception as exc:
            logger.warning("WhisperX diarize_speakers failed, using legacy: %s", exc)

    return _legacy_diarize_speakers(
        audio,
        sample_rate=sample_rate,
        max_speakers=max_speakers,
        window_sec=window_sec,
        hop_sec=hop_sec,
        min_segment_sec=min_segment_sec,
        distance_threshold=distance_threshold,
    )


def run_diarized_transcription(
    audio: Union[torch.Tensor, np.ndarray],
    pipe,
    *,
    sample_rate: int = 16000,
    language: str | None = "en",
    max_speakers: int = 6,
    window_sec: float = 1.2,
    hop_sec: float = 0.6,
    min_segment_sec: float = 0.4,
    distance_threshold: float = 0.55,
    max_diarization_sec: float = 0,
) -> tuple[list[dict[str, Any]], str, str]:
    """
    Word-level speaker diarization + transcription.

    Uses WhisperX + PyAnnote when configured (default). The ``pipe`` argument is
    kept for backward compatibility with the ASR module but is not used by the
    WhisperX path.

    Returns (speaker_turns, transcript_en, transcript_original).
    """
    del pipe  # WhisperX path is self-contained; legacy fallback uses whisper_asr import.

    cfg = _load_diarization_cfg()
    backend = str(cfg.get("backend", "text")).lower()

    # Safe local mode: no voice embeddings / WhisperX — ASR + text dialogue only.
    if backend in ("text", "off", "none", "disabled"):
        logger.info("Diarization backend=%s — skipping voice diarization", backend)
        return [], "", ""

    if backend in ("vad", "segment", "segmented"):
        try:
            from src.diarization.vad_segment_pipeline import run_vad_segment_pipeline

            alm = {}
            try:
                with open(_CONFIG_PATH, encoding="utf-8") as f:
                    alm = yaml.safe_load(f).get("alm_lite", {}) or {}
            except Exception:
                pass
            asr_model = (alm.get("asr") or {}).get("model_id", "openai/whisper-base")
            lang_models = (alm.get("asr") or {}).get("language_models") or {}
            from src.asr.whisper_asr import resolve_asr_model_id

            asr_model = resolve_asr_model_id(
                asr_model, language, language_models=lang_models
            )
            turns, tr_en, tr_orig = run_vad_segment_pipeline(
                audio,
                sample_rate=sample_rate,
                language=language,
                max_speakers=max_speakers,
                max_diarization_sec=max_diarization_sec,
                min_speech_sec=float(cfg.get("vad_min_speech_sec", 0.3)),
                min_silence_sec=float(cfg.get("vad_min_silence_sec", 0.35)),
                distance_threshold=float(cfg.get("distance_threshold", 0.35)),
                gap_merge_sec=float(cfg.get("gap_merge_sec", 0.45)),
                window_sec=float(cfg.get("window_sec", 1.5)),
                hop_sec=float(cfg.get("hop_sec", 0.75)),
                asr_model_id=asr_model,
            )
            if turns:
                return turns, tr_en, tr_orig
            logger.info("VAD pipeline returned no turns — falling back to text diarization")
            return [], "", ""
        except Exception as exc:
            logger.error(
                "VAD segment pipeline failed (%s); falling back to text diarization",
                exc,
                exc_info=True,
            )
            return [], "", ""

    if backend == "whisperx":
        from src.diarization.whisperx_pipeline import is_whisperx_available

        if is_whisperx_available():
            try:
                pipeline = _build_whisperx_pipeline(cfg)
                result = pipeline.run(
                    audio,
                    sample_rate=sample_rate,
                    language=language,
                    max_speakers=max_speakers,
                    max_diarization_sec=max_diarization_sec,
                )
                return _result_to_api(result)
            except Exception as exc:
                logger.error(
                    "WhisperX+PyAnnote diarization failed (%s); falling back to legacy",
                    exc,
                    exc_info=True,
                )
        else:
            logger.warning(
                "WhisperX backend configured but unavailable "
                "(requires Python 3.9–3.13, pip install whisperx, and HF_TOKEN). "
                "Using legacy Wav2Vec2 diarization."
            )

    return _legacy_run_diarized_transcription(
        audio,
        sample_rate=sample_rate,
        language=language,
        max_speakers=max_speakers,
        window_sec=window_sec,
        hop_sec=hop_sec,
        min_segment_sec=min_segment_sec,
        distance_threshold=distance_threshold,
        max_diarization_sec=max_diarization_sec,
    )


# ---------------------------------------------------------------------------
# Legacy Wav2Vec2 + clustering fallback (unchanged behaviour)
# ---------------------------------------------------------------------------

_legacy_lock = __import__("threading").Lock()
_embed_model = None
_embed_processor = None
_embed_device: Optional[torch.device] = None
_EMBED_MODEL_ID = "facebook/wav2vec2-base"


def _get_embedding_model():
    global _embed_model, _embed_processor, _embed_device
    with _legacy_lock:
        if _embed_model is not None:
            return _embed_model, _embed_processor, _embed_device
        # Wav2Vec2 embedding diarization hard-crashes on Windows (0xC0000005)
        # after clustering in this project stack. Use WhisperX when available,
        # otherwise fall through to text dialogue split / plain transcript.
        logger.warning(
            "Skipping Wav2Vec2 legacy diarization (known Windows native crash). "
            "Using WhisperX if configured, else text dialogue fallback."
        )
        return None, None, None


def _legacy_diarize_speakers(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    max_speakers: int = 6,
    window_sec: float = 1.2,
    hop_sec: float = 0.6,
    min_segment_sec: float = 0.4,
    distance_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    from src.diarization.legacy_clustering import cluster_speaker_segments

    model, processor, device = _get_embedding_model()
    if model is None:
        return []

    wav = _prepare_wav(audio)
    segments = cluster_speaker_segments(
        wav,
        sample_rate,
        model=model,
        processor=processor,
        device=device,
        max_speakers=max_speakers,
        window_sec=window_sec,
        hop_sec=hop_sec,
        min_segment_sec=min_segment_sec,
        distance_threshold=distance_threshold,
    )
    n_speakers = len({s["speaker"] for s in segments})
    logger.info("Legacy diarization: %d speaker(s), %d segments", n_speakers, len(segments))
    return segments


def _legacy_run_diarized_transcription(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    language: str | None = "en",
    max_speakers: int = 6,
    window_sec: float = 1.2,
    hop_sec: float = 0.6,
    min_segment_sec: float = 0.4,
    distance_threshold: float = 0.55,
    max_diarization_sec: float = 0,
) -> tuple[list[dict[str, Any]], str, str]:
    from src.asr.whisper_asr import _run_whisper_timestamped
    from src.diarization.legacy_clustering import align_phrases_to_segments

    wav = _prepare_wav(audio)
    dia_wav = wav
    if max_diarization_sec and max_diarization_sec > 0:
        cap = int(max_diarization_sec * sample_rate)
        if len(wav) > cap:
            dia_wav = wav[:cap]

    segments = _legacy_diarize_speakers(
        dia_wav,
        sample_rate=sample_rate,
        max_speakers=max_speakers,
        window_sec=window_sec,
        hop_sec=hop_sec,
        min_segment_sec=min_segment_sec,
        distance_threshold=distance_threshold,
    )
    if not segments:
        return [], "", ""

    from src.asr.whisper_asr import _get_asr_pipe

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = _get_asr_pipe("openai/whisper-base", device)
    phrases = _run_whisper_timestamped(
        pipe, wav, sample_rate, language_code=language or None
    )
    if not phrases:
        return [], "", ""

    return align_phrases_to_segments(phrases, segments)
