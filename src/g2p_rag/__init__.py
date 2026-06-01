"""g2p-rag: Retrieval-Augmented Generation over the Broad Institute G2P portal.

Public API (stable — see STABILITY.md):
    G2PRetriever   — facade for hybrid retrieval over the local ChromaDB index
    RetrievedChunk — Pydantic model for a single retrieval result

Everything else in this package (fetch, chunk, embed, retrieve, generate, cli,
integrations) is considered internal and may change between minor versions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

__version__ = "0.1.2"
__author__ = "Amit Shenoy"
__license__ = "GPL-3.0"

__all__ = ["G2PRetriever", "RetrievedChunk", "__version__"]


class RetrievedChunk(BaseModel):
    """A single chunk returned by the retriever."""

    text: str = Field(description="Full text of the chunk as stored in the index.")
    gene: str = Field(description="HGNC gene symbol (e.g. 'BRCA1').")
    uniprot_id: str = Field(description="UniProt accession (e.g. 'P38398').")
    chunk_type: Literal[
        "domain",
        "variant_cluster",
        "protein_summary",
        # Added in v0.1.1 — UniProt comment-derived biology chunks.
        # Additive extension; pre-existing types are unchanged.
        "function",
        "pathway",
        "subunit",
        "disease",
        # Added in v0.1.2 — G2P /api/gene/ cross-reference chunks.
        # Additive only; existing types continue to be emitted unchanged.
        "cross_references",
        "structures",
        "diseases",
    ] = Field(
        description="Granularity of the chunk."
    )
    residue_range: str = Field(
        description="Inclusive residue span as 'start-end' (e.g. '2-64'). "
                    "Empty string for protein-level chunks.",
    )
    source_url: str = Field(
        description="G2P portal URL for the protein page."
    )
    score: float = Field(description="RRF fusion score (higher is more relevant).")


class G2PRetriever:
    """Facade for hybrid dense+sparse retrieval over a pre-built G2P index.

    Downstream packages should import only this class and RetrievedChunk.
    The underlying ChromaDB + BM25 mechanics are internal.

    Args:
        persist_dir: Path to the ChromaDB directory written by ``g2p-rag ingest``.
        embedding_model: Sentence-transformers or OpenAI model name. Must match
            the model used during ingest.
        collection_name: ChromaDB collection name (default: "g2p_proteins").

    Raises:
        ImportError: If required dependencies are not installed.
        g2p_rag._internal.retrieve.CollectionEmptyError: If the index is empty.

    Example::

        from g2p_rag import G2PRetriever
        retriever = G2PRetriever(persist_dir="./data/chroma")
        results = retriever.retrieve("What domains in BRCA1 overlap with pathogenic variants?")
        for chunk in results:
            print(chunk.gene, chunk.chunk_type, chunk.residue_range, chunk.score)
    """

    def __init__(
        self,
        persist_dir: str | Path = "./data/chroma",
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "g2p_proteins",
    ) -> None:
        """Store configuration; index is loaded lazily on first retrieve() call."""
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self._retriever = None

    def _ensure_loaded(self) -> None:
        """Load the ChromaDB index and BM25 index if not already loaded."""
        if self._retriever is not None:
            return
        from g2p_rag.retrieve import CollectionEmptyError, load_embedder, load_retriever  # noqa: F401

        embedder = load_embedder(self.embedding_model)
        self._retriever = load_retriever(
            Path(self.persist_dir),
            embedder=embedder,
            collection_name=self.collection_name,
        )

    def retrieve(
        self,
        query: str,
        k: int = 5,
        gene_filter: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Run hybrid retrieval and return the top-k chunks.

        Args:
            query: Natural-language question or keyword query.
            k: Number of chunks to return.
            gene_filter: Optional list of HGNC symbols to restrict results to.

        Returns:
            List of RetrievedChunk ordered by descending RRF score.
        """
        self._ensure_loaded()
        raw_results = self._retriever.search(query, k=k, gene_filter=gene_filter)

        chunks: list[RetrievedChunk] = []
        for result in raw_results:
            c = result.chunk
            residue_range = (
                f"{c.residue_start}-{c.residue_end}" if c.residue_start else ""
            )
            chunks.append(
                RetrievedChunk(
                    text=c.text,
                    gene=c.gene,
                    uniprot_id=c.uniprot_id,
                    chunk_type=c.chunk_type,
                    residue_range=residue_range,
                    source_url=f"https://g2p.broadinstitute.org/protein?gene={c.gene}",
                    score=result.score,
                )
            )
        return chunks
