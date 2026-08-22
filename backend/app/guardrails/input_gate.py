from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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
    top_retrieval_score: Optional[float] = None
    retrieved: list = field(default_factory=list)


class InputGate:
    def __init__(self, retriever=None, retrieval_score_floor: float = 0.02, gate_top_k: int = 5):
        self.retriever = retriever
        self.retrieval_score_floor = retrieval_score_floor
        self.gate_top_k = gate_top_k
        self._unsafe_re = [re.compile(p, re.IGNORECASE) for p in _UNSAFE_PATTERNS]

    def check(self, query: str) -> GateResult:
        for pattern in self._unsafe_re:
            if pattern.search(query):
                return GateResult(verdict=GateVerdict.UNSAFE, reason="matched unsafe pattern")

        if self.retriever is not None:
            retrieved = self.retriever.search(query, top_k=self.gate_top_k)
            top_score = retrieved[0].rrf_score if retrieved else 0.0
            if not retrieved or top_score < self.retrieval_score_floor:
                return GateResult(
                    verdict=GateVerdict.OFF_TOPIC,
                    reason=f"top retrieval score too low ({top_score:.4f} < {self.retrieval_score_floor})",
                    top_retrieval_score=top_score,
                    retrieved=retrieved,
                )
            return GateResult(verdict=GateVerdict.ON_TOPIC, reason="passed both checks",
                               top_retrieval_score=top_score, retrieved=retrieved)

        return GateResult(verdict=GateVerdict.ON_TOPIC, reason="passed rule-based check (no retriever configured)")