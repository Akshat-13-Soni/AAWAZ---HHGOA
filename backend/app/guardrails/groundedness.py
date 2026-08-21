"""Groundedness check — runs AFTER generation, before the answer is returned.
Verifies the generated answer is actually supported by the retrieved chunks,
rather than trusting the LLM not to hallucinate. Two signals, combined:

1. Lexical overlap (fast, cheap, catches answers that introduce entities/
   numbers absent from context — a strong hallucination signal for factual QA)
2. Embedding similarity between answer and retrieved context (catches
   paraphrased hallucination that lexical overlap misses)

If either signal is too low, the pipeline should refuse rather than return
an ungrounded answer — that's requirement #6's core ask.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class GroundednessResult:
    is_grounded: bool
    lexical_overlap: float
    semantic_similarity: float
    reason: str


def _content_words(text: str) -> set[str]:
    # crude but language-agnostic-ish: strip punctuation, lowercase, drop very
    # short tokens (articles/particles contribute noise, not signal)
    tokens = re.findall(r"\w+", text.lower())
    return {t for t in tokens if len(t) > 2}


class GroundednessChecker:
    def __init__(self, embedder=None, min_lexical_overlap: float = 0.15,
                 min_semantic_similarity: float = 0.35):
        """
        Args:
            embedder: shared Embedder instance, used for the semantic-similarity
                      signal. If None, only the lexical check runs.
            min_lexical_overlap: minimum fraction of the answer's content words
                      that must appear somewhere in the retrieved context.
            min_semantic_similarity: minimum cosine similarity between the
                      answer and the concatenated retrieved context.
        """
        self.embedder = embedder
        self.min_lexical_overlap = min_lexical_overlap
        self.min_semantic_similarity = min_semantic_similarity

    def check(self, answer: str, context_texts: list[str]) -> GroundednessResult:
        if not context_texts:
            return GroundednessResult(
                is_grounded=False, lexical_overlap=0.0, semantic_similarity=0.0,
                reason="no retrieved context to ground against",
            )

        combined_context = " ".join(context_texts)

        answer_words = _content_words(answer)
        context_words = _content_words(combined_context)
        overlap = (len(answer_words & context_words) / len(answer_words)) if answer_words else 0.0

        semantic_sim = 1.0  # default pass-through if no embedder configured
        if self.embedder is not None:
            vecs = self.embedder.encode([answer, combined_context])
            denom = np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])
            semantic_sim = float(np.dot(vecs[0], vecs[1]) / denom) if denom else 0.0

        lexical_ok = overlap >= self.min_lexical_overlap
        semantic_ok = semantic_sim >= self.min_semantic_similarity

        is_grounded = lexical_ok and semantic_ok
        reason = "grounded" if is_grounded else (
            f"failed threshold(s): "
            f"{'lexical' if not lexical_ok else ''}{' + ' if not lexical_ok and not semantic_ok else ''}"
            f"{'semantic' if not semantic_ok else ''}"
        )

        return GroundednessResult(
            is_grounded=is_grounded,
            lexical_overlap=overlap,
            semantic_similarity=semantic_sim,
            reason=reason,
        )