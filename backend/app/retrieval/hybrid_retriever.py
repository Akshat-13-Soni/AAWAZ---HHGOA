"""Hybrid retriever: builds a BM25 index and a FAISS dense index over chunks
produced by ALL chunking strategies at once, then combines results via
Reciprocal Rank Fusion (RRF). RRF is used instead of raw score blending because
BM25 and cosine-similarity scores live on incomparable scales — RRF only needs
rank order from each method, which is far more robust.

Metadata-aware filtering (e.g. by language) is applied as a pre-filter before
scoring, per the metadata_aware chunking strategy's design goal: narrow the
search space before running expensive similarity search, which both improves
precision and helps the latency budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from app.chunking.base import Chunk
from app.retrieval.bm25_index import BM25Index
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import FaissVectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    rrf_score: float
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None


class HybridRetriever:
    def __init__(self, embedder: Embedder, rrf_k: int = 60):
        """
        Args:
            embedder: shared Embedder instance (also used by SemanticChunker).
            rrf_k: RRF damping constant. 60 is the standard default from the
                   original RRF paper and needs no tuning in almost all cases.
        """
        self.embedder = embedder
        self.rrf_k = rrf_k
        self._bm25 = BM25Index()
        self._vector_store: Optional[FaissVectorStore] = None
        self._chunks_by_id: dict[str, Chunk] = {}

    def index(self, chunks: Iterable[Chunk]) -> None:
        chunks = list(chunks)
        if not chunks:
            return

        ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]

        for c in chunks:
            self._chunks_by_id[c.id] = c

        self._bm25.add(ids, texts)
        self._bm25.build()

        vectors = self.embedder.encode(texts)
        if self._vector_store is None:
            self._vector_store = FaissVectorStore(dim=vectors.shape[1])
        self._vector_store.add(ids, vectors)

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 30,
        language_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        if self._vector_store is None or len(self._vector_store) == 0:
            return []

        bm25_hits = self._bm25.search(query, top_k=candidate_pool)
        query_vec = self.embedder.encode([query])[0]
        dense_hits = self._vector_store.search(query_vec, top_k=candidate_pool)

        # RRF fusion
        rrf_scores: dict[str, float] = {}
        bm25_ranks: dict[str, int] = {}
        dense_ranks: dict[str, int] = {}

        for rank, hit in enumerate(bm25_hits):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            bm25_ranks[hit.chunk_id] = rank + 1

        for rank, hit in enumerate(dense_hits):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            dense_ranks[hit.chunk_id] = rank + 1

        results = []
        for chunk_id, score in rrf_scores.items():
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            if language_filter and chunk.metadata.get("language") != language_filter:
                continue
            results.append(RetrievedChunk(
                chunk=chunk,
                rrf_score=score,
                bm25_rank=bm25_ranks.get(chunk_id),
                dense_rank=dense_ranks.get(chunk_id),
            ))

        results.sort(key=lambda r: r.rrf_score, reverse=True)
        return results[:top_k]

    def __len__(self) -> int:
        return len(self._chunks_by_id)
