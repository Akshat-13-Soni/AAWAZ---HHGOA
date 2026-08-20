"""Common interface every chunking strategy implements, so retrieval can treat
them interchangeably (or combine them)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

from app.dataset import Passage


@dataclass
class Chunk:
    id: str
    text: str
    source_passage_id: str
    strategy: str
    metadata: dict = field(default_factory=dict)


class ChunkingStrategy(ABC):
    """All strategies take an iterable of Passages and yield Chunks."""

    name: str = "base"

    @abstractmethod
    def chunk(self, passages: Iterable[Passage]) -> Iterable[Chunk]:
        ...
