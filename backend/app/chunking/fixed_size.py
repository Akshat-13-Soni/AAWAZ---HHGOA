"""Fixed-size chunking with overlap — the baseline strategy. Splits on
whitespace-token count, not characters, so chunk sizes are meaningful across
languages with different average word lengths."""
from __future__ import annotations

from typing import Iterable

from app.chunking.base import Chunk, ChunkingStrategy
from app.dataset import Passage


class FixedSizeChunker(ChunkingStrategy):
    name = "fixed_size"

    def __init__(self, chunk_size_tokens: int = 128, overlap_tokens: int = 24):
        if overlap_tokens >= chunk_size_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, passages: Iterable[Passage]) -> Iterable[Chunk]:
        step = self.chunk_size_tokens - self.overlap_tokens
        for passage in passages:
            tokens = passage.text.split()
            if not tokens:
                continue
            idx = 0
            piece_num = 0
            while idx < len(tokens):
                piece_tokens = tokens[idx: idx + self.chunk_size_tokens]
                text = " ".join(piece_tokens)
                yield Chunk(
                    id=f"{passage.id}_fixed_{piece_num}",
                    text=text,
                    source_passage_id=passage.id,
                    strategy=self.name,
                    metadata={
                        "language": passage.language,
                        "chunk_size_tokens": self.chunk_size_tokens,
                        "overlap_tokens": self.overlap_tokens,
                        **passage.metadata,
                    },
                )
                idx += step
                piece_num += 1
