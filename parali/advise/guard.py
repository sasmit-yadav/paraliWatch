"""Guardrails for the farmer assistant.

1. Input: questions that try to change the assistant's instructions are answered as normal
   questions, never obeyed (the documents and question are wrapped as data in the prompt).
2. Grounding: if retrieval finds nothing relevant, the assistant says so instead of guessing.
3. Numbers: every number in an answer (amounts, percentages, dates) must appear in the sources it
   cites. Money and subsidy figures are where a wrong answer hurts a farmer most.
"""

from __future__ import annotations

import re

from parali.advise.text import numbers

INJECTION = re.compile(
    r"(ignore (all |any |the )?(previous|above|prior) (instructions|rules)|system prompt|you are now|"
    r"act as|disregard (the )?(rules|instructions)|jailbreak|developer mode)",
    re.IGNORECASE,
)
CITATION = re.compile(r"\[(\d+)\]")
DIGITS = re.compile(r"\d+")


def looks_like_injection(text: str) -> bool:
    return bool(INJECTION.search(text))


def cited_indices(answer: str, n_sources: int) -> list[int]:
    found = []
    for m in CITATION.findall(answer):
        i = int(m)
        if 1 <= i <= n_sources and i not in found:
            found.append(i)
    return found


def unsupported_numbers(answer: str, sources: list[str], question: str = "") -> set[str]:
    """Numbers in the answer that appear in neither the cited sources nor the question.
    Citation markers like [2] are ignored."""
    clean = CITATION.sub(" ", answer)
    texts = [*sources, question]
    # allow numbers as written ("1,000") and every digit group inside them ("01.11.2026" -> "2026")
    allowed = set().union(*(numbers(t) for t in texts)) | {d for t in texts for d in DIGITS.findall(t)}
    return {n for n in numbers(clean) if n not in allowed}
