"""Metadata-aware chunking. Wraps fixed-size chunking but promotes MSMARCO-XI's
own fields (language, paired query, any dataset-provided category/id fields)
into first-class, filterable metadata on every chunk — so retrieval can pre-filter
by language or source before running any vector search, which both improves
precision and cuts the search space (helping the latency budget)."""
from __future__ import annotations

from typing import Iterable

from app.chunking.base import Chunk, ChunkingStrategy
from app.chunking.fixed_size import FixedSizeChunker
from app.dataset import Passage


class MetadataAwareChunker(ChunkingStrategy):
    name = "metadata_aware"

    def __init__(self, chunk_size_tokens: int = 128, overlap_tokens: int = 24):
        self._base = FixedSizeChunker(chunk_size_tokens, overlap_tokens)

    def chunk(self, passages: Iterable[Passage]) -> Iterable[Chunk]:
        for base_chunk in self._base.chunk(passages):
            # Re-tag with this strategy's name and promote/normalize metadata
            # that the retrieval layer will filter on.
            meta = dict(base_chunk.metadata)
            meta["has_paired_query"] = bool(meta.get("query"))
            yield Chunk(
                id=base_chunk.id.replace("_fixed_", "_metaaware_"),
                text=base_chunk.text,
                source_passage_id=base_chunk.source_passage_id,
                strategy=self.name,
                metadata=meta,
            )
