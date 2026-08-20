"""Sentence-window chunking. Each chunk is indexed by ONE sentence (for precise
retrieval matching) but carries a wider window of surrounding sentences as
context, which is what actually gets passed to the generator. This tends to
retrieve more precisely than fixed-size chunks while still giving the LLM
enough surrounding context to answer well."""
from __future__ import annotations

from typing import Iterable, List

from app.chunking.base import Chunk, ChunkingStrategy
from app.chunking.semantic import _split_sentences
from app.dataset import Passage


class SentenceWindowChunker(ChunkingStrategy):
    name = "sentence_window"

    def __init__(self, window_size: int = 2):
        """window_size: number of sentences on each side of the anchor sentence
        included in the context window."""
        self.window_size = window_size

    def chunk(self, passages: Iterable[Passage]) -> Iterable[Chunk]:
        for passage in passages:
            sentences: List[str] = _split_sentences(passage.text)
            for i, anchor in enumerate(sentences):
                lo = max(0, i - self.window_size)
                hi = min(len(sentences), i + self.window_size + 1)
                window_text = " ".join(sentences[lo:hi])
                yield Chunk(
                    id=f"{passage.id}_window_{i}",
                    text=window_text,          # what the LLM sees
                    source_passage_id=passage.id,
                    strategy=self.name,
                    metadata={
                        "language": passage.language,
                        "anchor_sentence": anchor,   # what gets embedded/matched
                        "anchor_index": i,
                        **passage.metadata,
                    },
                )
