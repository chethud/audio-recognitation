"""LLM answers from structured audio context (Qwen / similar)."""
from __future__ import annotations

from typing import Optional

import torch


def answer_question_from_context(
    context: str,
    question: str,
    model_id: str = "Qwen/Qwen2-0.5B-Instruct",
    max_new_tokens: int = 256,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 4,
    device: Optional[torch.device] = None,
) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    prompt = (
        "You are an assistant that answers questions about an audio scene based on "
        "the following context (speech transcript and environmental sounds).\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ).format(context=context, question=question)

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
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
