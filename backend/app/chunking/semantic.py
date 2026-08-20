"""Semantic chunking — splits text at sentence boundaries where meaning shifts,
detected via a drop in embedding similarity between consecutive sentences,
rather than at a fixed token count."""
from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np

from app.chunking.base import Chunk, ChunkingStrategy
from app.dataset import Passage

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")  # includes Devanagari danda ।


def _split_sentences(text: str) -> List[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or [text]


class SemanticChunker(ChunkingStrategy):
    name = "semantic"

    def __init__(self, embedder=None, similarity_drop_threshold: float = 0.25,
                 min_sentences_per_chunk: int = 1, max_sentences_per_chunk: int = 12):
        """
        Args:
            embedder: object exposing .encode(list[str]) -> np.ndarray. Injected
                      rather than constructed here so this module has no hard
                      dependency and stays fast to import/test. Pass a
                      sentence-transformers model at call time in production.
            similarity_drop_threshold: how much cosine similarity must drop
                      between consecutive sentences to trigger a chunk break.
        """
        self.embedder = embedder
        self.similarity_drop_threshold = similarity_drop_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_sentences_per_chunk = max_sentences_per_chunk

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def chunk(self, passages: Iterable[Passage]) -> Iterable[Chunk]:
        if self.embedder is None:
            raise RuntimeError(
                "SemanticChunker requires an embedder — inject a "
                "sentence-transformers model, e.g. SemanticChunker(embedder=model)"
            )

        for passage in passages:
            sentences = _split_sentences(passage.text)
            if len(sentences) <= 1:
                yield Chunk(
                    id=f"{passage.id}_semantic_0",
                    text=passage.text,
                    source_passage_id=passage.id,
                    strategy=self.name,
                    metadata={"language": passage.language, **passage.metadata},
                )
                continue

            embeddings = self.embedder.encode(sentences)
            current: List[str] = [sentences[0]]
            piece_num = 0

            for i in range(1, len(sentences)):
                sim = self._cosine_sim(embeddings[i - 1], embeddings[i])
                breakpoint_hit = (
                    sim < (1 - self.similarity_drop_threshold)
                    and len(current) >= self.min_sentences_per_chunk
                )
                size_cap_hit = len(current) >= self.max_sentences_per_chunk

                if breakpoint_hit or size_cap_hit:
                    yield Chunk(
                        id=f"{passage.id}_semantic_{piece_num}",
                        text=" ".join(current),
                        source_passage_id=passage.id,
                        strategy=self.name,
                        metadata={"language": passage.language, **passage.metadata},
                    )
                    piece_num += 1
                    current = [sentences[i]]
                else:
                    current.append(sentences[i])

            if current:
                yield Chunk(
                    id=f"{passage.id}_semantic_{piece_num}",
                    text=" ".join(current),
                    source_passage_id=passage.id,
                    strategy=self.name,
                    metadata={"language": passage.language, **passage.metadata},
                )
