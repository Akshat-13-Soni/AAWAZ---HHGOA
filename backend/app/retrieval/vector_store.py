"""In-process FAISS vector store. Chosen over a networked vector DB (Qdrant/
Pinecone/etc.) specifically because the 200ms latency budget can't absorb a
network round trip per query — FAISS running in the same process as the API
gives the best shot at hitting that target. HNSW index for sub-linear search
as the corpus grows.

If you later want a networked DB for other reasons (persistence across
restarts, multi-instance sharing), swap this class's internals — the
`add` / `search` interface is intentionally DB-agnostic so callers don't change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ScoredChunkRef:
    chunk_id: str
    score: float


class FaissVectorStore:
    def __init__(self, dim: int, use_hnsw: bool = True, hnsw_m: int = 32):
        import faiss

        self.dim = dim
        self._ids: list[str] = []  # index position -> chunk_id
        if use_hnsw:
            self._index = faiss.IndexHNSWFlat(dim, hnsw_m)
            self._index.hnsw.efConstruction = 40
            self._index.hnsw.efSearch = 32
        else:
            self._index = faiss.IndexFlatIP(dim)  # inner product == cosine on normalized vecs

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        assert vectors.shape[1] == self.dim, f"expected dim {self.dim}, got {vectors.shape[1]}"
        self._index.add(vectors)
        self._ids.extend(ids)

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[ScoredChunkRef]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        scores, indices = self._index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(ScoredChunkRef(chunk_id=self._ids[idx], score=float(score)))
        return results

    def __len__(self) -> int:
        return len(self._ids)
