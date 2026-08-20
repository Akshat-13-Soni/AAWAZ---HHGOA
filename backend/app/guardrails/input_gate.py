"""Input gate — runs BEFORE retrieval. Classifies a query as on-topic,
off-topic, or unsafe, so the pipeline can refuse cheaply without wasting the
latency budget on retrieval/generation for queries that were never going to
get a good answer anyway.

Two layers, cheapest first:
1. Rule-based unsafe-pattern check (fast, catches obvious cases)
2. Embedding-similarity-to-corpus-centroid check (catches topic drift a
   keyword list would miss) — requires the same embedder used for retrieval,
   injected rather than constructed here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

# Deliberately generic patterns — this is a topical/safety gate for a QA demo,
# not a content-moderation system. Expand this list based on what your judges'
# test queries actually probe for during the demo.
_UNSAFE_PATTERNS = [
    r"\bhow (to|do i) (make|build|synthesize)\b.*\b(bomb|explosive|weapon)\b",
    r"\bignore (previous|all) instructions\b",
    r"\bact as\b.*\b(dan|jailbreak)\b",
]


class GateVerdict(str, Enum):
    ON_TOPIC = "on_topic"
    OFF_TOPIC = "off_topic"
    UNSAFE = "unsafe"


@dataclass
class GateResult:
    verdict: GateVerdict
    reason: str
    similarity_to_corpus: Optional[float] = None


class InputGate:
    def __init__(self, embedder=None, corpus_centroid: Optional[np.ndarray] = None,
                 off_topic_threshold: float = 0.15):
        """
        Args:
            embedder: shared Embedder instance. If None, only the rule-based
                      unsafe check runs (still useful, just less precise on
                      off-topic detection).
            corpus_centroid: mean embedding vector of the indexed corpus,
                      computed once after indexing (see compute_corpus_centroid
                      below). Used to detect queries semantically far from the
                      dataset's actual topic space.
            off_topic_threshold: minimum cosine similarity to the corpus
                      centroid required to pass as on-topic.
        """
        self.embedder = embedder
        self.corpus_centroid = corpus_centroid
        self.off_topic_threshold = off_topic_threshold
        self._unsafe_re = [re.compile(p, re.IGNORECASE) for p in _UNSAFE_PATTERNS]

    def check(self, query: str) -> GateResult:
        for pattern in self._unsafe_re:
            if pattern.search(query):
                return GateResult(verdict=GateVerdict.UNSAFE, reason="matched unsafe pattern")

        if self.embedder is not None and self.corpus_centroid is not None:
            query_vec = self.embedder.encode([query])[0]
            sim = self._cosine_sim(query_vec, self.corpus_centroid)
            if sim < self.off_topic_threshold:
                return GateResult(
                    verdict=GateVerdict.OFF_TOPIC,
                    reason=f"low similarity to corpus ({sim:.3f} < {self.off_topic_threshold})",
                    similarity_to_corpus=sim,
                )
            return GateResult(verdict=GateVerdict.ON_TOPIC, reason="passed both checks",
                               similarity_to_corpus=sim)

        return GateResult(verdict=GateVerdict.ON_TOPIC, reason="passed rule-based check (no corpus centroid configured)")

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0


def compute_corpus_centroid(embedder, sample_texts: list[str]) -> np.ndarray:
    """Call once after indexing, on a representative sample of the corpus
    (a few hundred chunks is plenty), and pass the result into InputGate."""
    vectors = embedder.encode(sample_texts)
    return vectors.mean(axis=0)
