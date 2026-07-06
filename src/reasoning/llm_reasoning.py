"""LLM answers from structured audio context (Qwen / similar)."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Optional

import torch

from src.asr.text_cleanup import is_meaningful_speech
from src.asr.whisper_languages import language_label
from src.env_setup import configure_ml_env

configure_ml_env()

_lock = threading.Lock()
_llm_cache: dict[str, tuple] = {}


@contextmanager
def _quiet_load():
    """Hide tqdm / weight-loading progress bars during from_pretrained."""
    prev = None
    try:
        from transformers.utils import logging as tf_logging

        prev = tf_logging.get_verbosity()
        tf_logging.set_verbosity_error()
        yield
    finally:
        if prev is not None:
            try:
                from transformers.utils import logging as tf_logging

                tf_logging.set_verbosity(prev)
            except Exception:
                pass


def _get_llm(model_id: str, device: torch.device):
    key = f"{model_id}|{device.type}"
    with _lock:
        if key in _llm_cache:
            return _llm_cache[key]
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with _quiet_load():
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        model = model.to(device)
        model.eval()
        _llm_cache[key] = (tokenizer, model)
        return tokenizer, model


def _format_sound_summary(labels: list[str]) -> str:
    if not labels:
        return "No distinct environmental sounds detected."
    # Drop generic "Speech" — shown separately in transcript line.
    filtered = [l for l in labels if l.lower() not in {"speech", "conversation"}]
    if not filtered:
        return "General speech-like audio."
    return ", ".join(filtered)


def answer_from_context_fast(
    context: str,
    question: str,
    *,
    language: str = "en",
    transcript_original: str = "",
    languages: list[str] | None = None,
    transcript: str = "",
    emotion: str = "neutral",
    sound_labels: list[str] | None = None,
    speaker_turns: list[dict] | None = None,
) -> str:
    """Build a readable answer without loading the LLM (fast_mode)."""
    lang = (language or "en").lower()
    langs = languages or []
    sounds = sound_labels or []
    speech = (transcript or "").strip()
    emo = (emotion or "neutral").strip().capitalize()
    turns = speaker_turns or []

    has_speech = is_meaningful_speech(speech)
    has_sounds = bool(sounds)

    if not has_speech and not has_sounds:
        return "No clear speech or identifiable sounds were detected in this clip."

    lines: list[str] = []

    if has_sounds:
        lines.append(f"Sounds detected: {_format_sound_summary(sounds)}.")

    if turns and len({t.get("speaker") for t in turns}) >= 2:
        lines.append("Conversation by speaker:")
        for t in turns:
            who = t.get("speaker", "Speaker")
            text = (t.get("text") or "").strip()
            if text:
                lines.append(f"  {who}: {text}")
    elif has_speech:
        display = speech
        if (lang != "en" or lang == "multi" or len(langs) > 1) and transcript_original.strip():
            display = transcript_original.strip()
        lines.append(f'Speech heard: "{display}"')
    elif "[vocalization]" in speech or speech:
        lines.append("Non-speech vocalizations are present (e.g. barking or background noise).")

    if emo and emo.lower() not in {"unknown", "neutral"}:
        lines.append(f"Speaker emotion: {emo}.")
    elif has_speech:
        lines.append(f"Speaker emotion: {emo}.")

    if not lines:
        return "No clear speech or identifiable sounds were detected in this clip."

    return "\n".join(lines)


def answer_question_from_context(
    context: str,
    question: str,
    model_id: str = "Qwen/Qwen2-0.5B-Instruct",
    max_new_tokens: int = 32,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3,
    device: Optional[torch.device] = None,
    response_language: str = "en",
    languages: list[str] | None = None,
) -> str:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model = _get_llm(model_id, device)

    if len(context) > 800:
        context = context[:800] + "…"

    lang = (response_language or "en").lower()
    lang_list = languages or []
    if len(lang_list) > 1:
        names = ", ".join(language_label(code) for code in lang_list)
        lang_instruction = f"Answer using the same languages detected in the audio ({names})."
    elif lang == "en":
        lang_instruction = "Answer in English."
    else:
        lang_name = language_label(lang)
        lang_instruction = f"Answer in {lang_name} ({lang}). Do not answer in English."

    prompt = (
        f"{lang_instruction} Answer briefly from this audio context.\n\n"
        "Context:\n{context}\n\n"
        "Q: {question}\nA:"
    ).format(context=context, question=question)

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id or pad_id

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        repetition_penalty=repetition_penalty,
    )
    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    with torch.no_grad():
        out_ids = model.generate(**enc, **gen_kwargs)

    new_tokens = out_ids[0, enc["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text or "[No answer generated]"
