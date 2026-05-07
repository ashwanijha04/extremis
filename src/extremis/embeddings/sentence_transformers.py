from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class SentenceTransformerEmbedder:
    """
    Local embeddings via sentence-transformers.
    Model is loaded lazily on first use to keep import time fast.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        return self._load().get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        return self._load().encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._load().encode(texts, convert_to_numpy=True).tolist()
