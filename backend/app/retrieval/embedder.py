"""Thin wrapper around a sentence-transformers model, shared by dense retrieval
and semantic chunking. Kept as a small class (not a bare function) so it's easy
to swap models or inject a mock in tests."""
from __future__ import annotations

import numpy as np


class Embedder:
    """Wraps a small/fast sentence-transformers model for latency headroom.
    bge-small / e5-small class models are a good default: strong enough
    retrieval quality at a fraction of the latency of large embedding models."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype="float32")
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings.astype("float32")
