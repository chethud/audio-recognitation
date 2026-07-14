"""Production diarization pipeline: WhisperX word alignment + PyAnnote speakers."""
from __future__ import annotations

import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import soundfile as sf
import torch

from src.diarization.model_cache import (
    get_align_model,
    get_diarization_pipeline,
    get_whisperx_model,
    hf_auth_token,
    release_gpu_memory,
    whisper_size_from_model_id,
)
from src.diarization.types import DiarizationResult, DiarizationSegment, SpeakerTurn, WordToken
from src.diarization.word_speaker_aligner import (
    assign_speakers_by_overlap,
    extract_words_from_aligned,
    format_timestamp,
    merge_words_to_turns,
    remove_short_speaker_switches,
    turns_to_transcript,
)

logger = logging.getLogger(__name__)


class WhisperXDiarizationPipeline:
    """
    WhisperX (Whisper ASR + forced alignment) + PyAnnote speaker diarization.

    Pipeline:
      1. Transcribe audio with WhisperX (Whisper backend).
      2. Generate word-level timestamps via WhisperX align.
      3. Run PyAnnote diarization in parallel where possible.
      4. Assign each word to a speaker by timestamp overlap.
      5. Smooth false switches, merge consecutive same-speaker words.
    """

    def __init__(
        self,
        *,
        whisper_model_size: str = "base",
        min_switch_sec: float = 0.3,
        min_confidence: float = 0.35,
        gap_merge_sec: float = 0.6,
        batch_size: int = 8,
        align_batch_size: int = 16,
    ) -> None:
        self.whisper_model_size = whisper_model_size
        self.min_switch_sec = min_switch_sec
        self.min_confidence = min_confidence
        self.gap_merge_sec = gap_merge_sec
        self.batch_size = batch_size
        self.align_batch_size = align_batch_size

    def _write_temp_wav(self, wav: np.ndarray, sample_rate: int) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="alm_dia_")
        import os

        os.close(fd)
        sf.write(path, wav, sample_rate, subtype="PCM_16")
        return path

    def _load_audio_whisperx(
        self, wav: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, str | None]:
        """WhisperX expects 16 kHz mono float audio."""
        import whisperx

        if sample_rate != 16000:
            import librosa

            wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        peak = np.max(np.abs(wav)) if len(wav) else 0.0
        if peak > 0:
            wav = wav / peak

        tmp_path = self._write_temp_wav(wav.astype(np.float32), sample_rate)
        try:
            audio = whisperx.load_audio(tmp_path)
            return audio, tmp_path
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def _transcribe_and_align(
        self,
        audio: np.ndarray,
        *,
        language: str | None,
    ) -> dict[str, Any]:
        import whisperx

        t0 = time.perf_counter()
        model, device, _ = get_whisperx_model(self.whisper_model_size)

        transcribe_kwargs: dict[str, Any] = {"batch_size": self.batch_size}
        if language:
            transcribe_kwargs["language"] = language.split("-")[0].lower()

        result = model.transcribe(audio, **transcribe_kwargs)
        lang_code = (result.get("language") or language or "en").split("-")[0].lower()
        transcribe_sec = time.perf_counter() - t0
        logger.info(
            "WhisperX transcribe done in %.2fs (language=%s, segments=%d)",
            transcribe_sec,
            lang_code,
            len(result.get("segments") or []),
        )

        t1 = time.perf_counter()
        align_model, metadata, device = get_align_model(lang_code)
        aligned = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        align_sec = time.perf_counter() - t1
        word_count = sum(len(s.get("words") or []) for s in aligned.get("segments") or [])
        logger.info(
            "WhisperX align done in %.2fs (%d words)",
            align_sec,
            word_count,
        )
        aligned["language"] = lang_code
        return aligned

    def _run_pyannote(self, audio: np.ndarray, *, max_speakers: int) -> list[DiarizationSegment]:
        t0 = time.perf_counter()
        pipeline, _ = get_diarization_pipeline()

        diarize_kwargs: dict[str, Any] = {"max_speakers": max_speakers}
        diarize_kwargs["min_speakers"] = 1

        diarize_output = pipeline(audio, **diarize_kwargs)
        segments = self._parse_diarization_output(diarize_output)
        elapsed = time.perf_counter() - t0
        speakers = {s.speaker_id for s in segments}
        logger.info(
            "PyAnnote diarization done in %.2fs — %d speaker(s), %d segment(s)",
            elapsed,
            len(speakers),
            len(segments),
        )
        for seg in segments[:8]:
            logger.debug(
                "  %s %s→%s (%.2fs)",
                seg.speaker_id,
                format_timestamp(seg.start_sec),
                format_timestamp(seg.end_sec),
                seg.duration_sec,
            )
        if len(segments) > 8:
            logger.debug("  … %d more segments", len(segments) - 8)
        return segments

    @staticmethod
    def _parse_diarization_output(diarize_output: Any) -> list[DiarizationSegment]:
        segments: list[DiarizationSegment] = []

        if hasattr(diarize_output, "iterrows"):
            for _, row in diarize_output.iterrows():
                segments.append(
                    DiarizationSegment(
                        speaker_id=str(row.get("speaker", row.name)),
                        start_sec=float(row["start"]),
                        end_sec=float(row["end"]),
                    )
                )
            return segments

        if isinstance(diarize_output, dict):
            for sp, spans in diarize_output.items():
                for span in spans:
                    segments.append(
                        DiarizationSegment(
                            speaker_id=str(sp),
                            start_sec=float(span[0]),
                            end_sec=float(span[1]),
                        )
                    )
            return sorted(segments, key=lambda s: s.start_sec)

        for item in diarize_output or []:
            if isinstance(item, dict):
                segments.append(
                    DiarizationSegment(
                        speaker_id=str(item.get("speaker", "SPEAKER_00")),
                        start_sec=float(item["start"]),
                        end_sec=float(item["end"]),
                    )
                )
            elif hasattr(item, "start"):
                segments.append(
                    DiarizationSegment(
                        speaker_id=str(getattr(item, "speaker", "SPEAKER_00")),
                        start_sec=float(item.start),
                        end_sec=float(item.end),
                    )
                )
        return sorted(segments, key=lambda s: s.start_sec)

    def _assign_word_speakers(
        self,
        aligned: dict[str, Any],
        diar_segments: list[DiarizationSegment],
    ) -> list[WordToken]:
        """Assign speakers to words using WhisperX assign_word_speakers + overlap fallback."""
        from whisperx.diarize import assign_word_speakers

        try:
            import pandas as pd

            rows = [
                {"start": s.start_sec, "end": s.end_sec, "speaker": s.speaker_id}
                for s in diar_segments
            ]
            diarize_df = pd.DataFrame(rows)
            aligned = assign_word_speakers(diarize_df, aligned)
        except Exception as exc:
            logger.warning("whisperx.assign_word_speakers failed, using overlap: %s", exc)

        words = extract_words_from_aligned(aligned)
        if not words:
            return []

        if not any(w.speaker_id for w in words):
            words = assign_speakers_by_overlap(
                words,
                diar_segments,
                min_confidence=self.min_confidence,
            )

        words = remove_short_speaker_switches(words, min_switch_sec=self.min_switch_sec)
        return words

    def run(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        *,
        sample_rate: int = 16000,
        language: str | None = "en",
        max_speakers: int = 6,
        max_diarization_sec: float = 0,
    ) -> DiarizationResult:
        """Execute full diarization + word alignment pipeline."""
        t_total = time.perf_counter()

        if isinstance(audio, torch.Tensor):
            wav = audio.detach().float().cpu().numpy().reshape(-1)
        else:
            wav = np.asarray(audio, dtype=np.float32).reshape(-1)

        if max_diarization_sec and max_diarization_sec > 0:
            cap = int(max_diarization_sec * sample_rate)
            if len(wav) > cap:
                logger.info(
                    "Capping diarization input to %.0fs of %.0fs",
                    max_diarization_sec,
                    len(wav) / sample_rate,
                )
                wav = wav[:cap]

        duration_sec = len(wav) / max(sample_rate, 1)
        logger.info(
            "Starting WhisperX+PyAnnote diarization (%.1fs audio, max_speakers=%d)",
            duration_sec,
            max_speakers,
        )

        tmp_path: str | None = None
        try:
            audio_16k, tmp_path = self._load_audio_whisperx(wav, sample_rate)

            aligned: dict[str, Any] = {}
            diar_segments: list[DiarizationSegment] = []

            with ThreadPoolExecutor(max_workers=2) as pool:
                f_transcribe = pool.submit(
                    self._transcribe_and_align,
                    audio_16k,
                    language=language,
                )
                f_diarize = pool.submit(
                    self._run_pyannote,
                    audio_16k,
                    max_speakers=max_speakers,
                )
                for fut in as_completed([f_transcribe, f_diarize]):
                    exc = fut.exception()
                    if exc:
                        raise exc
                aligned = f_transcribe.result()
                diar_segments = f_diarize.result()

            words = self._assign_word_speakers(aligned, diar_segments)
            turns = merge_words_to_turns(words, gap_merge_sec=self.gap_merge_sec)
            speakers = {t.speaker for t in turns}
            num_speakers = len(speakers)

            plain_parts = []
            for seg in aligned.get("segments") or []:
                t = (seg.get("text") or "").strip()
                if t:
                    plain_parts.append(t)
            plain_transcript = " ".join(plain_parts).strip()

            if num_speakers < 2:
                logger.info(
                    "Single speaker detected after alignment (%d turns) — plain transcript",
                    len(turns),
                )
                return DiarizationResult(
                    turns=[],
                    transcript=plain_transcript,
                    transcript_original=plain_transcript,
                    num_speakers=0,
                    processing_sec=time.perf_counter() - t_total,
                    word_count=len(words),
                )

            for turn in turns:
                logger.info(
                    "Speaker block: %s %s→%s conf=%.2f | %s",
                    turn.speaker,
                    format_timestamp(turn.start_sec),
                    format_timestamp(turn.end_sec),
                    turn.confidence,
                    (turn.text[:80] + "…") if len(turn.text) > 80 else turn.text,
                )

            formatted = turns_to_transcript(turns)
            elapsed = time.perf_counter() - t_total
            logger.info(
                "Diarization complete in %.2fs — %d speakers, %d turns, %d words",
                elapsed,
                num_speakers,
                len(turns),
                len(words),
            )

            return DiarizationResult(
                turns=turns,
                transcript=formatted,
                transcript_original=plain_transcript,
                num_speakers=num_speakers,
                processing_sec=elapsed,
                word_count=len(words),
            )
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
            release_gpu_memory()


def is_whisperx_available() -> bool:
    """Check if WhisperX and HF token are available."""
    try:
        import whisperx  # noqa: F401

        return hf_auth_token() is not None
    except ImportError:
        return False
