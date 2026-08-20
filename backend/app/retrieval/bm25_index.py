"""BM25 sparse keyword index — the other half of hybrid retrieval alongside
FAISS dense search. BM25 catches exact keyword/entity matches that embedding
similarity sometimes misses (product names, numbers, rare terms), which
matters a lot for a QA dataset like MS MARCO."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoredChunkRef:
    chunk_id: str
    score: float


class BM25Index:
    def __init__(self):
        self._chunk_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25 = None  # built lazily once all docs are added

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def add(self, ids: list[str], texts: list[str]) -> None:
        self._chunk_ids.extend(ids)
        self._tokenized_corpus.extend(self._tokenize(t) for t in texts)
        self._bm25 = None  # invalidate; rebuilt on next search or explicit build()

    def build(self) -> None:
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunkRef]:
        if self._bm25 is None:
            self.build()
        scores = self._bm25.get_scores(self._tokenize(query))
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            ScoredChunkRef(chunk_id=self._chunk_ids[i], score=float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    def __len__(self) -> int:
        return len(self._chunk_ids)
