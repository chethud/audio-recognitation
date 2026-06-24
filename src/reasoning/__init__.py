"""LLM reasoning over text context."""
from .llm_reasoning import answer_from_context_fast, answer_question_from_context

__all__ = ["answer_from_context_fast", "answer_question_from_context"]
