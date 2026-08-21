"""Thin wrapper around a sentence-transformers model, shared by dense retrieval
and semantic chunking. Kept as a small class (not a bare function) so it's easy
to swap models or inject a mock in tests."""
from __future__ import annotations

import hashlib
import re

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


class HashEmbedder:
    """Small deterministic embedder for resource-constrained demo deployments.

    It preserves the shared ``encode`` interface used by dense retrieval and
    guardrails without loading PyTorch or a transformer model. It is intended
    for the hosted preview corpus; local/full evaluation should use ``Embedder``.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                vectors[row, bucket] += sign

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=vectors, where=norms != 0)
