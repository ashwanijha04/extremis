"""
OpenAI embeddings adapter.

Removes the 90 MB sentence-transformers model download for developers
who already have an OpenAI key.

Install: pip install "extremis[openai]"

Usage via config:
    EXTREMIS_EMBEDDER=text-embedding-3-small
    OPENAI_API_KEY=sk-...
"""
from __future__ import annotations

import os
from typing import Optional

_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 100,
    ) -> None:
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("OpenAI embedder requires: pip install 'extremis[openai]'") from None

        self._model = model
        self._batch_size = batch_size
        self._client = _openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))

    @property
    def dim(self) -> int:
        return _DIMS.get(self._model, 1536)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.embeddings.create(input=batch, model=self._model)
            results.extend(item.embedding for item in response.data)
        return results
