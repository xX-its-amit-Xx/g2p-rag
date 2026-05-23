"""Hybrid dense+sparse retrieval with Reciprocal Rank Fusion over genomic chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
import structlog

from g2p_rag.chunk import Chunk
from g2p_rag.embed import EmbeddingModel, get_embedder

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single retrieval result with its originating chunk, score, and provenance."""

    chunk: Chunk
    score: float
    rank: int
    source: str  # "dense", "sparse", or "hybrid"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metadata_to_chunk(doc: str, meta: dict[str, Any]) -> Chunk:
    """Reconstruct a Chunk from a ChromaDB document string and its metadata dict.

    The ``doc`` is the raw chunk text; ``meta`` must contain at minimum the keys
    gene, uniprot_id, chunk_type, residue_start, and residue_end.  Any additional
    keys are passed through as extra_metadata.
    """
    core_keys = {"gene", "uniprot_id", "chunk_type", "residue_start", "residue_end"}
    extra = {k: v for k, v in meta.items() if k not in core_keys}
    return Chunk(
        text=doc,
        gene=meta.get("gene", ""),
        uniprot_id=meta.get("uniprot_id", ""),
        chunk_type=meta.get("chunk_type", ""),
        residue_start=int(meta.get("residue_start", 0)),
        residue_end=int(meta.get("residue_end", 0)),
        metadata=extra,
    )


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


class VectorStore:
    """ChromaDB-backed persistent vector store for genomic chunks."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "g2p_proteins",
    ) -> None:
        """Initialise a persistent ChromaDB client and get-or-create the collection.

        Args:
            persist_dir: Directory on disk where ChromaDB persists its data.
            collection_name: Name of the ChromaDB collection.
        """
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._col = self._client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "vector_store.ready",
            persist_dir=str(persist_dir),
            collection=collection_name,
            existing_count=self._col.count(),
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        chunks: list[Chunk],
        embedder: EmbeddingModel,
        batch_size: int = 100,
    ) -> None:
        """Embed *chunks* and upsert them into ChromaDB.

        Uses deterministic IDs so re-running is idempotent.  Existing documents
        are silently overwritten via upsert.

        Args:
            chunks: Chunks to embed and store.
            embedder: EmbeddingModel used to produce dense vectors.
            batch_size: Number of chunks to embed and upsert per iteration.
        """
        total = len(chunks)
        log.info("vector_store.ingest.start", total=total)

        for batch_start in range(0, total, batch_size):
            batch = chunks[batch_start : batch_start + batch_size]

            texts = [c.text for c in batch]
            embeddings: list[list[float]] = embedder.embed(texts)

            ids: list[str] = []
            metadatas: list[dict[str, Any]] = []
            for chunk in batch:
                chunk_id = (
                    f"{chunk.gene}_{chunk.chunk_type}"
                    f"_{chunk.residue_start}_{chunk.residue_end}"
                    f"_{hash(chunk.text) & 0xFFFFFF:06x}"
                )
                ids.append(chunk_id)
                meta: dict[str, Any] = {
                    "gene": chunk.gene,
                    "uniprot_id": chunk.uniprot_id,
                    "chunk_type": chunk.chunk_type,
                    "residue_start": chunk.residue_start,
                    "residue_end": chunk.residue_end,
                }
                # Flatten any extra metadata (ChromaDB requires scalar values)
                for k, v in (chunk.metadata or {}).items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    else:
                        meta[k] = str(v)
                metadatas.append(meta)

            self._col.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            ingested_so_far = min(batch_start + batch_size, total)
            if ingested_so_far % 100 == 0 or ingested_so_far == total:
                log.info(
                    "vector_store.ingest.progress",
                    ingested=ingested_so_far,
                    total=total,
                )

        log.info("vector_store.ingest.done", total=total)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        embedder: EmbeddingModel,
        k: int = 20,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Dense vector search for *query* using cosine similarity.

        Args:
            query: Natural-language query string.
            embedder: EmbeddingModel used to embed the query.
            k: Number of results to return.
            where: Optional ChromaDB metadata filter (e.g. ``{"gene": {"$in": ["BRCA1"]}}``).

        Returns:
            List of SearchResult sorted by descending similarity score.
        """
        q_vec: list[float] = embedder.embed([query])[0]
        query_kwargs: dict = dict(
            query_embeddings=[q_vec],
            n_results=min(k, self._col.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            query_kwargs["where"] = where
        results = self._col.query(**query_kwargs)

        search_results: list[SearchResult] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for rank_idx, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
            chunk = _metadata_to_chunk(doc, meta)
            score = 1.0 - float(dist)  # cosine distance → similarity
            search_results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    rank=rank_idx + 1,
                    source="dense",
                )
            )

        log.debug("vector_store.search.done", query=query[:60], hits=len(search_results))
        return search_results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of documents currently stored in the collection."""
        return self._col.count()


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------


class BM25Index:
    """In-memory BM25 sparse index over genomic chunk texts."""

    def __init__(self) -> None:
        """Initialise an empty BM25 index (call :meth:`build` before searching)."""
        self._index: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Tokenise *chunks* and fit a BM25Okapi model.

        Args:
            chunks: Full corpus of chunks to index.
        """
        log.info("bm25.build.start", corpus_size=len(chunks))
        tokenized_corpus = [c.text.lower().split() for c in chunks]
        self._index = BM25Okapi(tokenized_corpus)
        self._chunks = list(chunks)
        log.info("bm25.build.done", corpus_size=len(chunks))

    def search(self, query: str, k: int = 20) -> list[SearchResult]:
        """Return the top-*k* chunks by BM25 score for *query*.

        Args:
            query: Natural-language query string.
            k: Number of results to return.

        Returns:
            List of SearchResult sorted by descending BM25 score.

        Raises:
            RuntimeError: If :meth:`build` has not been called yet.
        """
        if self._index is None:
            raise RuntimeError("BM25Index.build() must be called before search().")

        scores: np.ndarray = self._index.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:k]

        results: list[SearchResult] = []
        for rank_idx, idx in enumerate(top_indices):
            results.append(
                SearchResult(
                    chunk=self._chunks[int(idx)],
                    score=float(scores[int(idx)]),
                    rank=rank_idx + 1,
                    source="sparse",
                )
            )

        log.debug("bm25.search.done", query=query[:60], hits=len(results))
        return results


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Combine dense and sparse retrieval with Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedder: EmbeddingModel,
        rrf_k: int = 60,
    ) -> None:
        """Initialise the hybrid retriever.

        Args:
            vector_store: Populated VectorStore for dense retrieval.
            bm25_index: Fitted BM25Index for sparse retrieval.
            embedder: EmbeddingModel shared between dense retrieval and query embedding.
            rrf_k: RRF smoothing constant (default 60, per the original paper).
        """
        self._vs = vector_store
        self._bm25 = bm25_index
        self._embedder = embedder
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        k: int = 5,
        dense_k: int = 20,
        sparse_k: int = 20,
        gene_filter: list[str] | None = None,
    ) -> list[SearchResult]:
        """Retrieve the top-*k* chunks using RRF over dense and sparse rankings.

        For each unique document *d* appearing in either ranked list the RRF score
        is computed as ``sum(1 / (rrf_k + rank_i))`` where ``rank_i`` is the
        1-indexed position in list *i*.  Documents absent from a list receive no
        contribution from that list.

        Args:
            query: Natural-language query string.
            k: Final number of results to return.
            dense_k: Candidate pool size for the dense retrieval leg.
            sparse_k: Candidate pool size for the sparse retrieval leg.
            gene_filter: Optional list of gene symbols to restrict results to.

        Returns:
            Top-*k* SearchResult objects with source="hybrid", ranked by RRF score.
        """
        chroma_where = {"gene": {"$in": gene_filter}} if gene_filter else None
        dense_results = self._vs.search(query, self._embedder, k=dense_k, where=chroma_where)
        sparse_results = self._bm25.search(query, k=sparse_k)
        if gene_filter:
            gene_set = set(gene_filter)
            sparse_results = [r for r in sparse_results if r.chunk.gene in gene_set]

        # Map chunk identity → RRF accumulator.
        # Use (gene, chunk_type, residue_start, residue_end, text_hash) as key
        # to handle the case where different retrieval legs return the same chunk.
        rrf_scores: dict[tuple, float] = {}
        chunk_map: dict[tuple, Chunk] = {}

        def _chunk_key(c: Chunk) -> tuple:
            return (c.gene, c.chunk_type, c.residue_start, c.residue_end, hash(c.text))

        for result in dense_results:
            key = _chunk_key(result.chunk)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self._rrf_k + result.rank)
            chunk_map[key] = result.chunk

        for result in sparse_results:
            key = _chunk_key(result.chunk)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self._rrf_k + result.rank)
            chunk_map[key] = result.chunk

        # Sort by descending RRF score and take top-k
        sorted_keys = sorted(rrf_scores, key=lambda k_: rrf_scores[k_], reverse=True)[:k]

        fused: list[SearchResult] = [
            SearchResult(
                chunk=chunk_map[key],
                score=rrf_scores[key],
                rank=rank_idx + 1,
                source="hybrid",
            )
            for rank_idx, key in enumerate(sorted_keys)
        ]

        log.info(
            "hybrid_retriever.search.done",
            query=query[:60],
            dense_candidates=len(dense_results),
            sparse_candidates=len(sparse_results),
            fused_hits=len(fused),
        )
        return fused


# ---------------------------------------------------------------------------
# Top-level convenience functions
# ---------------------------------------------------------------------------


def build_index(
    chunks: list[Chunk],
    persist_dir: Path,
    embedder: EmbeddingModel | None = None,
    collection_name: str = "g2p_proteins",
) -> HybridRetriever:
    """Ingest *chunks* into ChromaDB, build a BM25 index, and return a HybridRetriever.

    Ingestion is idempotent: existing documents with matching deterministic IDs are
    silently overwritten via upsert.

    Args:
        chunks: Genomic chunks to embed and index.
        persist_dir: Directory used for ChromaDB persistence.
        embedder: Optional pre-built EmbeddingModel; ``get_embedder()`` is used when
            *None* is passed.
        collection_name: ChromaDB collection name.

    Returns:
        A fully initialised HybridRetriever ready to answer queries.
    """
    if embedder is None:
        embedder = get_embedder()

    vs = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
    vs.ingest(chunks, embedder)

    bm25 = BM25Index()
    bm25.build(chunks)

    log.info("build_index.done", chunks=len(chunks), collection=collection_name)
    return HybridRetriever(vector_store=vs, bm25_index=bm25, embedder=embedder)


class CollectionEmptyError(RuntimeError):
    """Raised when the ChromaDB collection has no documents."""


def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingModel:
    """Return an EmbeddingModel for the given model name (convenience wrapper)."""
    return get_embedder(model_name)


def index_stats(persist_dir: Path, collection_name: str = "g2p_proteins") -> dict[str, Any]:
    """Return a dict of stats about the persisted ChromaDB collection.

    Raises CollectionEmptyError if the collection does not exist or is empty.
    """
    vs = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
    count = vs.count()
    if count == 0:
        raise CollectionEmptyError(f"Collection '{collection_name}' in {persist_dir} is empty.")
    # Peek at metadata to enumerate genes and chunk types
    peek = vs._col.peek(min(count, 500))
    genes: set[str] = set()
    chunk_types: set[str] = set()
    for meta in (peek.get("metadatas") or []):
        if meta:
            genes.add(meta.get("gene", ""))
            chunk_types.add(meta.get("chunk_type", ""))
    return {
        "total_chunks": count,
        "genes": sorted(g for g in genes if g),
        "chunk_types": sorted(ct for ct in chunk_types if ct),
        "persist_dir": str(persist_dir),
        "collection_name": collection_name,
    }


def load_retriever(
    persist_dir: Path,
    embedder: EmbeddingModel | None = None,
    chunks: list[Chunk] | None = None,
    collection_name: str = "g2p_proteins",
) -> HybridRetriever:
    """Load an existing ChromaDB store and (optionally) rebuild the BM25 index.

    When *chunks* is omitted the BM25 index is reconstructed from the documents
    stored in ChromaDB itself so the retriever works without the original corpus.
    """
    if embedder is None:
        embedder = get_embedder()

    vs = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
    count = vs.count()
    if count == 0:
        raise CollectionEmptyError(
            f"Collection '{collection_name}' in {persist_dir} is empty. "
            "Run `g2p-rag ingest` first."
        )

    bm25 = BM25Index()
    if chunks:
        bm25.build(chunks)
    else:
        # Rebuild BM25 from stored documents (slower but self-contained)
        log.info("load_retriever.rebuilding_bm25_from_chroma", count=count)
        all_docs = vs._col.get(include=["documents", "metadatas"])
        corpus_chunks = [
            _metadata_to_chunk(doc, meta)
            for doc, meta in zip(
                all_docs.get("documents") or [],
                all_docs.get("metadatas") or [],
            )
        ]
        bm25.build(corpus_chunks)

    log.info("load_retriever.done", persist_dir=str(persist_dir), vector_count=count)
    return HybridRetriever(vector_store=vs, bm25_index=bm25, embedder=embedder)
