"""Embedding model wrappers supporting sentence-transformers and OpenAI backends."""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import numpy as np
import structlog

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingModel(Protocol):
    """Common interface for all embedding backends."""

    model_name: str
    dim: int
    model_revision: str

    def embed(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Sentence-Transformers backend
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder:
    """Embeds text using a local sentence-transformers model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        """Load the sentence-transformer model and determine embedding dimension."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self._model: SentenceTransformer = SentenceTransformer(
            model_name, device=device
        )
        self.dim: int = self._model.get_sentence_embedding_dimension()
        # Capture the HuggingFace revision SHA so consumers can pin/verify the
        # exact model snapshot used to build a downstream index.  Best-effort:
        # if the lookup fails (offline, private repo, transient network) we
        # leave model_revision as an empty string rather than blocking model
        # construction.  Downstream consistency checks treat "" as "unknown".
        self.model_revision: str = ""
        try:
            from huggingface_hub import HfApi  # type: ignore[import]

            info = HfApi().model_info(model_name)
            self.model_revision = str(getattr(info, "sha", "") or "")
        except Exception as exc:  # pragma: no cover — best-effort metadata
            log.warning(
                "SentenceTransformerEmbedder.revision_lookup_failed",
                model=model_name,
                error=str(exc),
            )
        log.info(
            "SentenceTransformerEmbedder ready",
            model=model_name,
            device=device,
            dim=self.dim,
            revision=self.model_revision,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (N, dim) float32 array of embeddings for the given texts."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        result = embeddings.astype(np.float32)
        log.debug("embedded batch", n=len(texts), model=self.model_name)
        return result

    def embed_query(self, text: str) -> np.ndarray:
        """Return (dim,) float32 array for a single query string."""
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

_OPENAI_DIM_MAP: dict[str, int] = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}
_OPENAI_BATCH_SIZE = 100


class OpenAIEmbedder:
    """Embeds text using the OpenAI Embeddings API."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: str | None = None,
    ) -> None:
        """Initialise the OpenAI client; falls back to OPENAI_API_KEY env var."""
        try:
            import openai  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAIEmbedder. "
                "Install it with: pip install openai"
            ) from exc

        self.model_name = model_name
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "An OpenAI API key must be provided either via the api_key argument "
                "or the OPENAI_API_KEY environment variable."
            )
        self._client = openai.OpenAI(api_key=resolved_key)
        self.dim: int = _OPENAI_DIM_MAP.get(model_name, 1536)
        # OpenAI does not expose a per-snapshot revision the way HF does;
        # leave as "" so the downstream consistency check skips the revision
        # comparison (it only enforces when both sides are non-empty).
        self.model_revision: str = ""
        log.info(
            "OpenAIEmbedder ready",
            model=model_name,
            dim=self.dim,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (N, dim) float32 array by batching calls to the OpenAI API."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        all_embeddings: list[list[float]] = []
        for batch_start in range(0, len(texts), _OPENAI_BATCH_SIZE):
            batch = texts[batch_start : batch_start + _OPENAI_BATCH_SIZE]
            response = self._client.embeddings.create(
                input=batch,
                model=self.model_name,
            )
            batch_vecs = [item.embedding for item in response.data]
            all_embeddings.extend(batch_vecs)
            log.debug(
                "openai embed batch",
                batch_start=batch_start,
                batch_size=len(batch),
                model=self.model_name,
            )

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Return (dim,) float32 array for a single query string."""
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_embedder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    **kwargs,
) -> EmbeddingModel:
    """Return the right embedder based on model_name prefix."""
    if model_name.startswith("text-embedding-"):
        log.info("selecting OpenAIEmbedder", model=model_name)
        return OpenAIEmbedder(model_name=model_name, **kwargs)
    log.info("selecting SentenceTransformerEmbedder", model=model_name)
    return SentenceTransformerEmbedder(model_name=model_name, **kwargs)
