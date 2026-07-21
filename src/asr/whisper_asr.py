"""Whisper-based speech-to-text via Hugging Face."""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union

from src.asr.whisper_languages import (
    language_label,
    token_to_language_code,
    whisper_language_code,
    whisper_language_name,
)
from src.asr.text_cleanup import (
    clean_asr_text,
    clean_asr_text_preserve_content,
    contains_indic_script,
    infer_language_from_text,
    is_likely_english_text,
    is_low_quality_indic,
    INDIC_LANGUAGE_CODES,
    PRIMARY_LANGUAGE_CODES,
)
from src.env_setup import configure_ml_env

configure_ml_env()

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import threading

import numpy as np
import torch

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pipe_cache: dict[str, object] = {}

# Fine-tuned models that emit native Indic script (openai/whisper-* often romanizes Kannada).
_DEFAULT_LANGUAGE_MODELS = {
    "kn": "vasista22/whisper-kannada-base",
}


def resolve_asr_model_id(
    default_model_id: str,
    language: Optional[str] = None,
    *,
    language_models: Optional[dict] = None,
) -> str:
    """Pick a language-specific ASR checkpoint when configured (e.g. Kannada → Kannada script)."""
    if not language:
        return default_model_id
    code = whisper_language_code(language)
    overrides = language_models or {}
    if code in overrides and overrides[code]:
        return str(overrides[code])
    return _DEFAULT_LANGUAGE_MODELS.get(code, default_model_id)


def _is_kannada_finetune(model_id: str) -> bool:
    mid = (model_id or "").lower()
    return "kannada" in mid


def _pipe_is_kannada_finetune(pipe) -> bool:
    cfg = getattr(pipe, "model", None)
    cfg = getattr(cfg, "config", None) if cfg is not None else None
    name = str(
        getattr(cfg, "_name_or_path", None)
        or getattr(cfg, "name_or_path", None)
        or ""
    ).lower()
    return "kannada" in name


def _get_asr_pipe(model_id: str, device: Union[str, torch.device]):
    if isinstance(device, torch.device):
        dev = 0 if device.type == "cuda" else -1
    else:
        dev = 0 if device == "cuda" else -1
    key = f"{model_id}|{dev}"
    with _lock:
        if key in _pipe_cache:
            return _pipe_cache[key]
        from transformers import GenerationConfig, pipeline

        logger.info("Loading Whisper ASR model %s ...", model_id)
        print(f"[alm-worker] loading_whisper:{model_id}", file=sys.stderr, flush=True)
        pipe_init: dict = {
            "task": "automatic-speech-recognition",
            "model": model_id,
            "device": dev,
            "torch_dtype": torch.float16 if dev == 0 else torch.float32,
        }
        # Kannada fine-tunes: avoid pipeline long-form chunking (crashes / empties).
        if _is_kannada_finetune(model_id):
            pipe_init["model_kwargs"] = {"attn_implementation": "eager"}
        else:
            pipe_init["chunk_length_s"] = 30
        p = pipeline(**pipe_init)
        # Fine-tuned Kannada models: refresh generation_config + force kn decoder ids.
        # Passing generate_kwargs language=… breaks these older checkpoints.
        if _is_kannada_finetune(model_id):
            try:
                p.model.generation_config = GenerationConfig.from_pretrained(model_id)
            except Exception as exc:
                logger.debug("Kannada generation_config refresh skipped: %s", exc)
            try:
                forced = p.tokenizer.get_decoder_prompt_ids(language="kn", task="transcribe")
                p.model.config.forced_decoder_ids = forced
                if getattr(p.model, "generation_config", None) is not None:
                    p.model.generation_config.forced_decoder_ids = forced
                    # Avoid stale language conflict with generate()
                    for attr in ("language", "task"):
                        if hasattr(p.model.generation_config, attr):
                            setattr(p.model.generation_config, attr, None)
                    # Older fine-tunes miss attrs newer transformers expect.
                    gc = p.model.generation_config
                    if not hasattr(gc, "no_timestamps_token_id"):
                        try:
                            ts_id = p.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
                            setattr(gc, "no_timestamps_token_id", ts_id)
                        except Exception:
                            setattr(gc, "no_timestamps_token_id", None)
                    if getattr(gc, "return_timestamps", None) is None:
                        try:
                            gc.return_timestamps = False
                        except Exception:
                            pass
            except Exception as exc:
                logger.warning("Could not set Kannada forced_decoder_ids: %s", exc)
        _pipe_cache[key] = p
        print("[alm-worker] whisper_ready", file=sys.stderr, flush=True)
        return p


@dataclass
class TranscriptionResult:
    """English transcript + original speech with multi-language metadata."""

    transcript: str
    transcript_original: str
    language: str
    language_name: str
    languages: list[str]
    language_names: list[str]
    speaker_turns: list[dict] = field(default_factory=list)
    num_speakers: int = 0


def _prepare_wav(
    audio: Union[torch.Tensor, np.ndarray],
    sample_rate: int,
    max_duration_sec: Optional[float],
) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        wav = audio.detach().float().cpu().numpy().reshape(-1)
    else:
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)

    if max_duration_sec and max_duration_sec > 0:
        cap = int(max_duration_sec * sample_rate)
        if len(wav) > cap:
            wav = wav[:cap]
    return wav


def _feature_extractor(pipe):
    fe = getattr(pipe, "feature_extractor", None) or getattr(pipe, "processor", None)
    if fe is None:
        return None
    if hasattr(fe, "feature_extractor"):
        return fe.feature_extractor
    return fe


def _detect_language(pipe, wav: np.ndarray, sample_rate: int) -> str:
    """Auto-detect spoken language using Whisper (all supported languages)."""
    try:
        model = pipe.model
        fe = _feature_extractor(pipe)
        if fe is None:
            return "en"

        # Use up to 8s for language detection on long clips (faster).
        detect_samples = min(len(wav), sample_rate * 8)
        detect_wav = wav[:detect_samples]

        inputs = fe([detect_wav], sampling_rate=sample_rate, return_tensors="pt", padding=True)
        input_features = inputs.input_features
        if hasattr(model, "device"):
            input_features = input_features.to(model.device)
        dtype = getattr(model, "dtype", torch.float32)
        input_features = input_features.to(dtype=dtype)

        # Whisper language detection expects ~30s mel frames (3000); pad short clips.
        target_frames = 3000
        cur_frames = input_features.shape[-1]
        if cur_frames < target_frames:
            input_features = torch.nn.functional.pad(
                input_features, (0, target_frames - cur_frames)
            )
        elif cur_frames > target_frames:
            input_features = input_features[..., :target_frames]

        with torch.no_grad():
            lang_ids = model.detect_language(input_features)
            if getattr(lang_ids, "dim", lambda: 0)() == 1:
                lang_id = int(lang_ids[0].item())
            else:
                lang_id = int(lang_ids[0, 0].item())

        tokenizer = pipe.tokenizer
        token = tokenizer.convert_ids_to_tokens([lang_id])[0]
        code = token_to_language_code(token)
        logger.info("Detected language: %s (%s)", language_label(code), code)
        return whisper_language_code(code)
    except Exception as exc:
        logger.warning("Language detection failed, falling back to auto: %s", exc)
        return ""


def _dedupe_whisper_chunks(
    parsed: list[tuple[str, tuple[float, float]]],
) -> list[tuple[str, tuple[float, float]]]:
    """Drop overlapping Whisper chunks that repeat the same line (common on Indic audio)."""
    import re

    if not parsed:
        return parsed

    out: list[tuple[str, tuple[float, float]]] = []
    for text, (start, end) in parsed:
        key = re.sub(r"\s+", " ", (text or "").strip())
        if not key:
            continue
        if out:
            prev_text, (prev_start, prev_end) = out[-1]
            prev_key = re.sub(r"\s+", " ", prev_text.strip())
            if key == prev_key:
                out[-1] = (prev_text, (prev_start, max(prev_end, end)))
                continue
            if prev_key and key in prev_key and end <= prev_end + 0.5:
                continue
            if prev_key and prev_key in key and start <= prev_end + 0.5:
                out[-1] = (text, (prev_start, max(prev_end, end)))
                continue
        out.append((text, (start, end)))
    return out


def _run_whisper(
    pipe,
    wav: np.ndarray,
    sample_rate: int,
    *,
    task: str,
    language_code: Optional[str] = None,
) -> str:
    duration_s = len(wav) / max(sample_rate, 1)
    kannada_ft = _pipe_is_kannada_finetune(pipe)

    # Kannada fine-tunes: HF pipeline long-form chunking often collapses to one
    # short hallucinated sentence — use explicit windows instead.
    if kannada_ft and duration_s > 28:
        return _run_whisper_longform(
            pipe,
            wav,
            sample_rate,
            task=task,
            language_code=language_code,
            chunk_sec=18.0,
            hop_sec=16.0,
        )

    pipe_kwargs: dict = {}
    if duration_s > 30 and not kannada_ft:
        pipe_kwargs = {"chunk_length_s": 30, "batch_size": 4, "stride_length_s": 5}

    # Dynamically limit max new tokens for short segments to speed up CPU inference
    # and prevent long-running hallucination loops on short segments.
    max_tokens = 224
    if duration_s < 28:
        max_tokens = max(32, min(224, int(duration_s * 12)))

    gen_kwargs: dict = {
        "condition_on_prev_tokens": False,
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": max_tokens,
    }
    # Stock Whisper supports these; Kannada fine-tunes often lack no_timestamps_token_id
    # and crash if timestamp thresholds are passed.
    if not kannada_ft:
        gen_kwargs["compression_ratio_threshold"] = 2.2
        gen_kwargs["logprob_threshold"] = -0.8
        gen_kwargs["no_speech_threshold"] = 0.55
        gen_kwargs["num_beams"] = 1
        gen_kwargs["no_repeat_ngram_size"] = 3
    else:
        gen_kwargs["num_beams"] = 1
    # Fine-tuned Kannada checkpoints already bake language into forced_decoder_ids.
    # Passing language=/task= here raises "generation config is outdated".
    if not kannada_ft:
        gen_kwargs["task"] = task
        whisper_lang = whisper_language_name(language_code)
        if whisper_lang:
            gen_kwargs["language"] = whisper_lang
        elif language_code:
            gen_kwargs["language"] = language_code.strip().lower()
    else:
        whisper_lang = "kannada"

    call_kwargs = dict(pipe_kwargs)
    if kannada_ft:
        call_kwargs["return_timestamps"] = False
    try:
        out = pipe(
            {"array": wav, "sampling_rate": sample_rate},
            generate_kwargs=gen_kwargs,
            **call_kwargs,
        )
    except Exception as exc:
        # Surface the real ASR error — never fall back to a previous/sample transcript.
        logger.error("Whisper inference failed: %s", exc)
        raise
    text = out.get("text", "") if isinstance(out, dict) else str(out)
    if kannada_ft or (language_code or "").lower() in {"kn", "kannada"}:
        text = clean_asr_text_preserve_content((text or "").strip())
    else:
        text = clean_asr_text((text or "").strip())
    if kannada_ft or (language_code or "").lower() in {"kn", "kannada"}:
        try:
            from src.asr.audio_trace import log_audio_identity

            log_audio_identity(
                stage="whisper_out",
                language=language_code or "kn",
                wav=wav,
                sample_rate=sample_rate,
                whisper_out=text,
                extra={"model": "kannada_ft" if kannada_ft else "generic", "dur_sec": round(duration_s, 2)},
            )
        except Exception:
            pass
    if kannada_ft:
        return text

    # Drop Arabic/CJK junk even on first pass when language is Indic.
    if whisper_lang and whisper_lang != "english" and text:
        import re

        if re.search(
            r"[\u0600-\u06FF\u0750-\u077F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]",
            text,
        ) and not contains_indic_script(text):
            text = ""
    # Prefer readable romanization over junk native-script hallucinations.
    if (
        whisper_lang
        and whisper_lang != "english"
        and (not text or not contains_indic_script(text))
    ):
        if text and not contains_indic_script(text):
            return text
        retry_kwargs = dict(gen_kwargs)
        _apply_indic_script_bias(pipe, retry_kwargs, whisper_lang, strong=True)
        try:
            out2 = pipe(
                {"array": wav, "sampling_rate": sample_rate},
                generate_kwargs=retry_kwargs,
                **pipe_kwargs,
            )
            text2 = out2.get("text", "") if isinstance(out2, dict) else str(out2)
            text2 = clean_asr_text((text2 or "").strip())
            if (
                text2
                and contains_indic_script(text2)
                and not is_low_quality_indic(text2)
            ):
                return text2
        except Exception:
            pass
    return text


def _run_whisper_longform(
    pipe,
    wav: np.ndarray,
    sample_rate: int,
    *,
    task: str,
    language_code: Optional[str] = None,
    chunk_sec: float = 18.0,
    hop_sec: float = 16.0,
) -> str:
    """Transcribe long audio in fixed windows and concatenate (best for Kannada FT)."""
    n = len(wav)
    if n < int(0.4 * sample_rate):
        return ""
    chunk = max(int(chunk_sec * sample_rate), 1)
    hop = max(int(hop_sec * sample_rate), 1)
    rebuilt: list[str] = []
    pos = 0
    idx = 0
    while pos < n:
        end = min(n, pos + chunk)
        if end - pos < int(0.4 * sample_rate):
            break
        # Skip near-silent windows (saves time, avoids hallucinations).
        seg = wav[pos:end]
        if float(np.sqrt(np.mean(np.square(seg)) + 1e-12)) < 0.008:
            idx += 1
            if end >= n:
                break
            pos += hop
            continue
        try:
            text = _run_whisper(
                pipe,
                seg,
                sample_rate,
                task=task,
                language_code=language_code,
            )
        except Exception as exc:
            logger.error("Longform Whisper chunk %d failed: %s", idx, exc)
            if (language_code or "").lower() in {"kn", "kannada"} or _pipe_is_kannada_finetune(pipe):
                raise
            text = ""
        text = (text or "").strip()
        print(
            f"[alm-worker] whisper_longform chunk={idx} "
            f"t={pos / sample_rate:.1f}-{end / sample_rate:.1f}s "
            f"chars={len(text)}",
            flush=True,
        )
        if text and len(text.strip(" .,?!।।")) >= 2:
            if not rebuilt:
                rebuilt.append(text)
            else:
                suffix = _overlap_suffix(rebuilt[-1], text)
                if suffix:
                    rebuilt.append(suffix)
        idx += 1
        if end >= n:
            break
        pos += hop

    return clean_asr_text_preserve_content(" ".join(rebuilt).strip()) if (
        (language_code or "").lower() in {"kn", "kannada"} or _pipe_is_kannada_finetune(pipe)
    ) else clean_asr_text(" ".join(rebuilt).strip())


def _overlap_suffix(prev: str, cur: str) -> str:
    """Return the non-overlapping tail of cur relative to prev (word-level)."""
    if not cur:
        return ""
    if not prev:
        return cur
    prev_w = prev.split()
    cur_w = cur.split()
    if not cur_w:
        return ""
    max_k = min(len(prev_w), len(cur_w), 16)
    for k in range(max_k, 0, -1):
        if prev_w[-k:] == cur_w[:k]:
            return " ".join(cur_w[k:]).strip()
    return cur


def run_whisper_speech_windows(
    pipe,
    wav: np.ndarray,
    sample_rate: int,
    windows: list[tuple[float, float]],
    *,
    language_code: Optional[str] = None,
) -> list[tuple[float, float, str]]:
    """
    Transcribe each (start_sec, end_sec) speech window.
    Returns list of (start, end, text) with non-empty text.
    """
    out: list[tuple[float, float, str]] = []
    for i, (start_sec, end_sec) in enumerate(windows):
        if end_sec - start_sec < 0.35:
            continue
        s = max(0, int(start_sec * sample_rate))
        e = min(len(wav), int(end_sec * sample_rate))
        if e - s < int(0.35 * sample_rate):
            continue
        try:
            text = _run_whisper(
                pipe,
                wav[s:e],
                sample_rate,
                task="transcribe",
                language_code=language_code,
            )
        except Exception as exc:
            logger.error(
                "Speech-window Whisper failed %.2f-%.2f: %s", start_sec, end_sec, exc
            )
            # Kannada: do not silently skip and invent continuity from other windows /
            # prior results — bubble the failure so the API returns the real error.
            if (language_code or "").strip().lower() in {"kn", "kannada"}:
                raise
            continue
        text = (text or "").strip()
        print(
            f"[alm-worker] whisper_window {i} "
            f"t={start_sec:.1f}-{end_sec:.1f}s chars={len(text)}",
            flush=True,
        )
        if text and len(text.strip(" .,?!।।")) >= 2:
            out.append((float(start_sec), float(end_sec), text))
    return out

_INDIC_PROMPTS: dict[str, str] = {}  # Intentionally empty — hardcoded prompts were regurgitated.


def _apply_indic_script_bias(pipe, gen_kwargs: dict, whisper_lang: str, *, strong: bool = False) -> None:
    """Force Whisper language tokens only. Never inject hardcoded script prompts."""
    try:
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is None:
            return
        if hasattr(tokenizer, "get_decoder_prompt_ids"):
            gen_kwargs["forced_decoder_ids"] = tokenizer.get_decoder_prompt_ids(
                language=whisper_lang, task=gen_kwargs.get("task", "transcribe")
            )
        # Never attach prompt_ids — hardcoded Kannada/Hindi lines caused identical
        # short transcripts across unrelated uploads.
        gen_kwargs.pop("prompt_ids", None)
    except Exception as exc:
        logger.debug("Indic script bias skipped: %s", exc)
        gen_kwargs.pop("prompt_ids", None)


def _run_whisper_timestamped(
    pipe,
    wav: np.ndarray,
    sample_rate: int,
    *,
    language_code: Optional[str] = None,
) -> list[tuple[str, tuple[float, float]]]:
    """Single-pass transcription with (start_sec, end_sec) per phrase."""
    # Fine-tuned Kannada checkpoints often lack timestamp generation config.
    if _pipe_is_kannada_finetune(pipe):
        text = _run_whisper(
            pipe, wav, sample_rate, task="transcribe", language_code=language_code or "kn"
        )
        duration_s = len(wav) / max(sample_rate, 1)
        if text:
            return [(text, (0.0, duration_s))]
        return []

    duration_s = len(wav) / max(sample_rate, 1)
    pipe_kwargs: dict = {}
    if duration_s > 30:
        pipe_kwargs = {"chunk_length_s": 30, "batch_size": 8, "stride_length_s": 12}

    gen_kwargs: dict = {
        "task": "transcribe",
        "condition_on_prev_tokens": False,
        "num_beams": 1,
        "no_repeat_ngram_size": 3,
    }
    whisper_lang = whisper_language_name(language_code)
    if whisper_lang:
        gen_kwargs["language"] = whisper_lang
    elif language_code:
        gen_kwargs["language"] = language_code.strip().lower()
    if whisper_lang and whisper_lang != "english":
        gen_kwargs.setdefault("temperature", 0.0)

    out = pipe(
        {"array": wav, "sampling_rate": sample_rate},
        return_timestamps=True,
        generate_kwargs=gen_kwargs,
        **pipe_kwargs,
    )
    chunks = out.get("chunks") if isinstance(out, dict) else None
    if not chunks:
        text = out.get("text", "") if isinstance(out, dict) else str(out)
        if text and text.strip():
            return [(text.strip(), (0.0, duration_s))]
        return []

    parsed: list[tuple[str, tuple[float, float]]] = []
    for item in chunks:
        text = (item.get("text") or "").strip()
        ts = item.get("timestamp")
        if not text or not ts or ts[0] is None or ts[1] is None:
            continue
        parsed.append((text, (float(ts[0]), float(ts[1]))))
    return _dedupe_whisper_chunks(parsed)



def _normalize_lang_code(language: str | None) -> str:
    code = (language or "").strip().lower()
    return code if code and code != "auto" else ""


def _transcribe_chunk(
    pipe,
    chunk: np.ndarray,
    sample_rate: int,
    language: str,
) -> tuple[str, str, str]:
    """
    Transcribe one chunk; translate to English when not English.
    Returns (english_or_display, original, resolved_language_code).
    """
    lang = _normalize_lang_code(language)

    original = _run_whisper(
        pipe,
        chunk,
        sample_rate,
        task="transcribe",
        language_code=lang or None,
    )
    if not original.strip():
        return "", "", lang

    inferred = infer_language_from_text(original)
    if inferred:
        lang = inferred
    elif not lang:
        lang = "en" if is_likely_english_text(original) else ""

    if not lang or lang == "en":
        if not lang and not is_likely_english_text(original):
            # Last resort: let Whisper auto-translate without forcing English.
            english = _run_whisper(
                pipe, chunk, sample_rate, task="translate", language_code=None
            )
            return (english or original).strip(), original.strip(), lang or "en"
        return original.strip(), original.strip(), "en"

    # Kannada fine-tune models are trained only for transcription, not translation.
    # Calling task="translate" on them uses the same forced_decoder_ids and returns
    # Kannada script again — identical to the transcription, just wasting compute.
    # Skip translation and return the native-script text in both fields.
    if _pipe_is_kannada_finetune(pipe) or lang == "kn":
        return original.strip(), original.strip(), lang

    english = _run_whisper(
        pipe, chunk, sample_rate, task="translate", language_code=lang
    )
    return (english or original).strip(), original.strip(), lang


def _finalize_transcripts(en_parts: list[str], orig_parts: list[str]) -> tuple[str, str]:
    transcript_en = clean_asr_text(" ".join(en_parts))
    transcript_original = clean_asr_text(" ".join(orig_parts))
    return transcript_en, transcript_original


def _split_time_chunks(
    wav: np.ndarray,
    sample_rate: int,
    segment_sec: float,
    max_segments: int,
) -> list[np.ndarray]:
    """Split audio into overlapping windows for full-length transcription."""
    if max_segments <= 1:
        return [wav]

    seg_samples = max(int(segment_sec * sample_rate), sample_rate)
    if len(wav) <= seg_samples:
        return [wav]

    overlap = min(int(0.5 * sample_rate), seg_samples // 4)
    step = max(seg_samples - overlap, sample_rate // 2)
    chunks: list[np.ndarray] = []
    for start in range(0, len(wav), step):
        chunk = wav[start : start + seg_samples]
        if len(chunk) < sample_rate // 2:
            break
        chunks.append(chunk)
        if len(chunks) >= max_segments:
            break
    return chunks or [wav]


def _majority_language_vote(
    votes: list[str],
    preferred: frozenset[str] | None = None,
) -> str:
    from collections import Counter

    pool = [v for v in votes if v]
    if not pool:
        return ""
    preferred = preferred or PRIMARY_LANGUAGE_CODES
    filtered = [v for v in pool if v in preferred]
    return Counter(filtered or pool).most_common(1)[0][0]


def _resolve_spoken_language(
    pipe,
    wav: np.ndarray,
    sample_rate: int,
    *,
    preferred: frozenset[str] | None = None,
) -> str:
    """
    Robust language pick for English + Kannada/Hindi/Tamil/Telugu/Malayalam.
    Script from a short auto-transcribe overrides unreliable Whisper votes.
    """
    preferred = preferred or PRIMARY_LANGUAGE_CODES
    n = len(wav)
    votes: list[str] = []

    if n <= sample_rate * 3:
        votes.append(_detect_language(pipe, wav, sample_rate))
    else:
        win = sample_rate * 8
        for frac in (0.15, 0.35, 0.5, 0.65, 0.85):
            center = int(frac * n)
            start = max(0, center - win // 2)
            end = min(n, start + win)
            chunk = wav[start:end]
            if len(chunk) >= sample_rate * 2:
                code = _detect_language(pipe, chunk, sample_rate)
                if code:
                    votes.append(whisper_language_code(code))

    whisper_guess = _majority_language_vote(votes, preferred)

    sample_len = min(n, sample_rate * 25)
    mid = n // 2
    probe = wav[max(0, mid - sample_len // 2) : max(0, mid - sample_len // 2) + sample_len]
    if len(probe) < sample_rate * 2:
        probe = wav[: min(n, sample_rate * 20)]

    auto_text = _run_whisper(
        pipe, probe, sample_rate, task="transcribe", language_code=None
    )
    script_lang = infer_language_from_text(auto_text)
    if script_lang:
        if whisper_guess and script_lang != whisper_guess:
            logger.info(
                "Language script=%s overrides Whisper=%s",
                language_label(script_lang),
                language_label(whisper_guess),
            )
        return script_lang

    if is_likely_english_text(auto_text) and not contains_indic_script(auto_text):
        return "en"

    return whisper_guess


def _probe_languages(
    pipe, wav: np.ndarray, sample_rate: int, probe_sec: float = 2.0
) -> tuple[str, str]:
    """Return (lang, lang) using multi-point detection + script verification."""
    del probe_sec  # kept for API compatibility
    lang = _resolve_spoken_language(pipe, wav, sample_rate)
    return lang, lang


def transcribe_bilingual(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    model_id: str = "openai/whisper-tiny",
    device: Optional[Union[str, torch.device]] = None,
    max_duration_sec: Optional[float] = None,
    segment_sec: float = 4.0,
    max_segments: int = 2,
    language: Optional[str] = None,
    diarization_enabled: bool = False,
    diarization_max_speakers: int = 6,
    diarization_window_sec: float = 1.2,
    diarization_hop_sec: float = 0.6,
    diarization_min_segment_sec: float = 0.4,
    diarization_distance_threshold: float = 0.72,
    diarization_max_sec: float = 0,
) -> TranscriptionResult:
    """
    Auto-detect language; transcribe full clip in time windows when needed.
    When diarization_enabled, label speech by Speaker 1, Speaker 2, …
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav = _prepare_wav(audio, sample_rate, max_duration_sec)

    # When the caller specifies the language (e.g. user picked English or Kannada),
    # skip all auto-detection passes — this is a large speedup on long clips.
    forced_lang = _normalize_lang_code(language)
    if forced_lang:
        forced_lang = whisper_language_code(forced_lang)

    # Kannada + CT2: transcribe WITHOUT loading HuggingFace Whisper first.
    # Loading both on a full ~4min clip OOMs Windows (mkl_malloc) then HF fallback
    # also breaks on outdated GenerationConfig → empty transcript.
    if forced_lang == "kn":
        try:
            from src.asr.kannada_faster import (
                kannada_faster_available,
                transcribe_kannada_faster,
            )

            if kannada_faster_available():
                print("[alm-worker] kannada_faster_primary", flush=True)
                scored = transcribe_kannada_faster(
                    wav, sample_rate, language="kn"
                )
                turns = [
                    {
                        "speaker": "Speaker 1",
                        "start_sec": float(s),
                        "end_sec": float(e),
                        "text": t,
                        "text_original": t,
                        "confidence": 0.0,
                        "alignment": "vad",
                    }
                    for s, e, t in scored
                    if (t or "").strip()
                ]
                if turns:
                    from src.diarization.speaker_utils import normalize_speaker_turns

                    turns, transcript_en, transcript_original, num_speakers = (
                        normalize_speaker_turns(
                            turns,
                            " ".join(t["text"] for t in turns),
                            " ".join(t["text_original"] for t in turns),
                            language="kn",
                        )
                    )
                    if (transcript_en or transcript_original) and num_speakers >= 1:
                        return TranscriptionResult(
                            transcript=clean_asr_text(transcript_en),
                            transcript_original=clean_asr_text(
                                transcript_original or transcript_en
                            ),
                            language="kn",
                            language_name=language_label("kn"),
                            languages=["kn"],
                            language_names=[language_label("kn")],
                            speaker_turns=turns,
                            num_speakers=max(num_speakers, 1),
                        )
                logger.warning("Kannada CT2 returned no text; falling back to HF")
        except Exception as exc:
            logger.warning("Kannada CT2 primary path failed: %s", exc)

    # Kannada (and other language-specific checkpoints) → native script.
    try:
        import yaml
        from src.config_path import resolve_config_path

        cfg_path = resolve_config_path(Path(__file__).resolve().parents[2])
        with open(cfg_path, encoding="utf-8") as f:
            asr_cfg = (yaml.safe_load(f) or {}).get("alm_lite", {}).get("asr", {}) or {}
        lang_models = asr_cfg.get("language_models") or {}
    except Exception:
        lang_models = {}
    model_id = resolve_asr_model_id(
        model_id, forced_lang or language, language_models=lang_models
    )
    pipe = _get_asr_pipe(model_id, device)

    try:
        if forced_lang:
            lang_start = lang_end = forced_lang
            logger.info("Using user-selected language: %s (%s)", language_label(forced_lang), forced_lang)
        else:
            lang_start, lang_end = _probe_languages(pipe, wav, sample_rate)
        duration_s = len(wav) / max(sample_rate, 1)
        lang = lang_start

        # Re-check English only when auto-detected (skip when the user forced a language).
        if (
            not forced_lang
            and lang
            and lang not in INDIC_LANGUAGE_CODES
            and lang != "en"
            and duration_s < 90
        ):
            auto_en = _run_whisper(
                pipe, wav, sample_rate, task="transcribe", language_code=None
            )
            if is_likely_english_text(auto_en) and not contains_indic_script(auto_en):
                lang_start = lang_end = lang = "en"

        if diarization_enabled:
            from src.diarization import run_diarized_transcription

            turns, transcript_en, transcript_original = run_diarized_transcription(
                wav,
                pipe,
                sample_rate=sample_rate,
                language=lang or None,
                max_speakers=diarization_max_speakers,
                window_sec=diarization_window_sec,
                hop_sec=diarization_hop_sec,
                min_segment_sec=diarization_min_segment_sec,
                distance_threshold=diarization_distance_threshold,
                max_diarization_sec=diarization_max_sec,
            )
            from src.diarization.speaker_utils import normalize_speaker_turns, try_dialogue_speaker_split

            turns, transcript_en, transcript_original, num_speakers = (
                normalize_speaker_turns(
                    turns,
                    transcript_en,
                    transcript_original,
                    language=lang or forced_lang or language,
                )
            )
            # Only accept VAD/voice diarization when 2+ speakers were found.
            # A 1-speaker VAD result must not block text dialogue fallback.
            if turns and num_speakers >= 2:
                languages = [lang] if lang_start == lang_end else [lang_start, lang_end]
                languages = list(dict.fromkeys(languages))
                language_names = [language_label(code) for code in languages]
                primary = languages[0] if len(languages) == 1 else "multi"
                primary_name = (
                    language_names[0]
                    if len(language_names) == 1
                    else ", ".join(language_names)
                )
                return TranscriptionResult(
                    transcript=clean_asr_text(transcript_en),
                    transcript_original=clean_asr_text(transcript_original),
                    language=primary,
                    language_name=primary_name,
                    languages=languages,
                    language_names=language_names,
                    speaker_turns=turns,
                    num_speakers=num_speakers,
                )
            # Kannada 1-speaker: turns may be [] (continuous transcript) with num_speakers=1.
            single_speaker_fallback = None
            if num_speakers >= 1 and (transcript_en or transcript_original):
                # English keeps timed Speaker 1 turns; Kannada may return empty turns.
                if turns or whisper_language_code(lang or lang_start or "en") == "kn":
                    single_speaker_fallback = (
                        turns,
                        transcript_en,
                        transcript_original,
                        num_speakers,
                    )
            if transcript_en and whisper_language_code(lang or lang_start or "en") in (
                "en",
                "multi",
            ):
                dlg_turns, dlg_en, dlg_orig, num_speakers = try_dialogue_speaker_split(
                    transcript_en, transcript_original or transcript_en
                )
                if dlg_turns and num_speakers >= 2:
                    logger.info(
                        "Dialogue split into %d speakers (%d turns) [text fallback]",
                        num_speakers,
                        len(dlg_turns),
                    )
                    languages = [lang] if lang_start == lang_end else [lang_start, lang_end]
                    languages = list(dict.fromkeys(languages))
                    language_names = [language_label(code) for code in languages]
                    primary = languages[0] if len(languages) == 1 else "multi"
                    primary_name = (
                        language_names[0]
                        if len(language_names) == 1
                        else ", ".join(language_names)
                    )
                    return TranscriptionResult(
                        transcript=clean_asr_text(dlg_en),
                        transcript_original=clean_asr_text(dlg_orig),
                        language=primary,
                        language_name=primary_name,
                        languages=languages,
                        language_names=language_names,
                        speaker_turns=dlg_turns,
                        num_speakers=num_speakers,
                    )
            if single_speaker_fallback is not None:
                fb_turns, fb_en, fb_orig, fb_n = single_speaker_fallback
                logger.info(
                    "Keeping single-speaker result (turns=%d, n=%d); no 2-speaker split",
                    len(fb_turns),
                    fb_n,
                )
                languages = [lang] if lang_start == lang_end else [lang_start, lang_end]
                languages = list(dict.fromkeys(languages))
                language_names = [language_label(code) for code in languages]
                primary = languages[0] if len(languages) == 1 else "multi"
                primary_name = (
                    language_names[0]
                    if len(language_names) == 1
                    else ", ".join(language_names)
                )
                return TranscriptionResult(
                    transcript=clean_asr_text(fb_en),
                    transcript_original=clean_asr_text(fb_orig),
                    language=primary,
                    language_name=primary_name,
                    languages=languages,
                    language_names=language_names,
                    speaker_turns=fb_turns,
                    num_speakers=fb_n,
                )
            if transcript_en:
                logger.info(
                    "Diarization collapsed to single speaker; using plain transcript"
                )
                languages = [lang] if lang_start == lang_end else [lang_start, lang_end]
                languages = list(dict.fromkeys(languages))
                language_names = [language_label(code) for code in languages]
                primary = languages[0] if len(languages) == 1 else "multi"
                primary_name = (
                    language_names[0]
                    if len(language_names) == 1
                    else ", ".join(language_names)
                )
                return TranscriptionResult(
                    transcript=clean_asr_text(transcript_en),
                    transcript_original=clean_asr_text(transcript_original),
                    language=primary,
                    language_name=primary_name,
                    languages=languages,
                    language_names=language_names,
                    speaker_turns=[],
                    num_speakers=0,
                )
            logger.info("Diarization found no separable speakers; using standard ASR")

        multi = False  # avoid false Hindi+Telugu splits from noisy 2s probes

        en_parts: list[str] = []
        orig_parts: list[str] = []
        langs_ordered: list[str] = []

        if multi:
            mid = len(wav) // 2
            chunks = [(lang_start, wav[:mid]), (lang_end, wav[mid:])]
            logger.info("Mixed languages detected: %s + %s", lang_start, lang_end)
        else:
            lang = lang_start
            time_chunks = _split_time_chunks(wav, sample_rate, segment_sec, max_segments)
            chunks = [(lang, c) for c in time_chunks]
            if len(time_chunks) > 1:
                logger.info(
                    "Transcribing %d segments (%.1fs total)",
                    len(time_chunks),
                    len(wav) / sample_rate,
                )

        for lang, chunk in chunks:
            english, original, resolved_lang = _transcribe_chunk(
                pipe, chunk, sample_rate, lang
            )
            if not original:
                continue

            if resolved_lang:
                lang = resolved_lang
            else:
                inferred = infer_language_from_text(original)
                if inferred:
                    lang = inferred

            if (
                lang != "en"
                and is_likely_english_text(original)
                and not contains_indic_script(original)
            ):
                lang = "en"
                english = original

            langs_ordered.append(lang or "en")
            if lang == "en":
                orig_parts.append(original)
                en_parts.append(original)
            else:
                orig_parts.append(original)
                en_parts.append(english)

        if not en_parts:
            return TranscriptionResult(
                transcript="",
                transcript_original="",
                language="en",
                language_name="English",
                languages=[],
                language_names=[],
                speaker_turns=[],
                num_speakers=0,
            )

        languages = list(dict.fromkeys(langs_ordered))
        language_names = [language_label(code) for code in languages]
        transcript_en, transcript_original = _finalize_transcripts(en_parts, orig_parts)

        if len(languages) == 1:
            primary = languages[0]
            primary_name = language_names[0]
        else:
            primary = "multi"
            primary_name = ", ".join(language_names)

        speaker_turns: list[dict] = []
        num_speakers = 0
        if diarization_enabled and transcript_en and primary == "en":
            from src.diarization.speaker_utils import try_dialogue_speaker_split

            dlg_turns, dlg_en, dlg_orig, num_speakers = try_dialogue_speaker_split(
                transcript_en, transcript_original or transcript_en
            )
            if dlg_turns and num_speakers >= 2:
                speaker_turns = dlg_turns
                transcript_en = clean_asr_text(dlg_en)
                transcript_original = clean_asr_text(dlg_orig)
                logger.info(
                    "Dialogue split into %d speakers (%d turns)",
                    num_speakers,
                    len(dlg_turns),
                )

        return TranscriptionResult(
            transcript=transcript_en,
            transcript_original=transcript_original,
            language=primary,
            language_name=primary_name,
            languages=languages,
            language_names=language_names,
            speaker_turns=speaker_turns,
            num_speakers=num_speakers,
        )
    except Exception as e:
        err = f"[ASR error: {e}]"
        return TranscriptionResult(
            transcript=err,
            transcript_original=err,
            language="en",
            language_name="English",
            languages=["en"],
            language_names=["English"],
            speaker_turns=[],
            num_speakers=0,
        )


def transcribe_audio(
    audio: Union[torch.Tensor, np.ndarray],
    *,
    sample_rate: int = 16000,
    model_id: str = "openai/whisper-tiny",
    device: Optional[Union[str, torch.device]] = None,
    language: Optional[str] = None,
    max_duration_sec: Optional[float] = None,
) -> str:
    """Legacy single-language transcription."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav = _prepare_wav(audio, sample_rate, max_duration_sec)
    pipe = _get_asr_pipe(model_id, device)

    try:
        lang = whisper_language_code(language) if language else _detect_language(pipe, wav, sample_rate)
        task = "translate" if lang == "en" else "transcribe"
        return _run_whisper(pipe, wav, sample_rate, task=task, language_code=lang)
    except Exception as e:
        return f"[ASR error: {e}]"
