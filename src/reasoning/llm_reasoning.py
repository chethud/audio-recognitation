"""LLM answers from structured audio context (Qwen / similar)."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Optional

import torch

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


LANGUAGE_LABELS = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "ur": "Urdu",
}


def language_label(code: str) -> str:
    return LANGUAGE_LABELS.get((code or "en").lower(), code or "English")


def answer_from_context_fast(
    context: str,
    question: str,
    *,
    language: str = "en",
    transcript_original: str = "",
) -> str:
    """Instant answer without loading the LLM (fast_mode)."""
    ctx = (context or "").strip()
    q = (question or "What can be inferred from the audio?").strip()
    lang = (language or "en").lower()

    if "[No speech detected]" in ctx and "[No environmental sounds detected]" in ctx:
        if lang == "en":
            return "No clear speech or identifiable sounds were detected in this clip."
        if transcript_original.strip():
            return transcript_original.strip()
        return "No clear speech or identifiable sounds were detected in this clip."

    if lang != "en" and transcript_original.strip():
        return transcript_original.strip()

    compact = " ".join(line.strip() for line in ctx.splitlines() if line.strip())
    return f'Answer to "{q}": {compact}'


def answer_question_from_context(
    context: str,
    question: str,
    model_id: str = "Qwen/Qwen2-0.5B-Instruct",
    max_new_tokens: int = 32,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3,
    device: Optional[torch.device] = None,
    response_language: str = "en",
) -> str:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model = _get_llm(model_id, device)

    # Keep prompt short for faster generation on CPU.
    if len(context) > 800:
        context = context[:800] + "…"

    lang = (response_language or "en").lower()
    lang_name = language_label(lang)
    lang_instruction = (
        "Answer in English."
        if lang == "en"
        else f"Answer in {lang_name} ({lang}). Do not answer in English."
    )

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
