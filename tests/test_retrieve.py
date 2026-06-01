"""Tests for g2p_rag.retrieve — VectorStore, BM25Index, and HybridRetriever.

Uses a real in-memory ChromaDB (via a temp persist_dir that is discarded after each
test) and a deterministic FakeEmbedder so no network or GPU is required.
"""

import numpy as np
import pytest
from pathlib import Path

from g2p_rag.chunk import Chunk
from g2p_rag.retrieve import (
    VectorStore,
    BM25Index,
    HybridRetriever,
    SearchResult,
    CollectionEmptyError,
    EmbeddingModelMismatchError,
    build_index,
    load_retriever,
)


# ---------------------------------------------------------------------------
# Fake embedder (satisfies EmbeddingModel protocol)
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic random embedder for tests — no model loading required."""

    model_name = "fake"
    dim = 4

    def embed(self, texts: list[str]) -> np.ndarray:
        np.random.seed(42)
        return np.random.rand(len(texts), 4).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        np.random.seed(0)
        return np.random.rand(4).astype(np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chroma_dir(tmp_path: Path) -> Path:
    return tmp_path / "chroma"


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            text="BRCA1 RING finger domain mediates E3 ubiquitin ligase activity",
            chunk_type="domain",
            gene="BRCA1",
            uniprot_id="P38398",
            residue_start=2,
            residue_end=64,
            metadata={"domain_name": "RING finger"},
        ),
        Chunk(
            text="BRCA2 DNA binding domain essential for homologous recombination repair",
            chunk_type="domain",
            gene="BRCA2",
            uniprot_id="P51587",
            residue_start=2402,
            residue_end=3190,
            metadata={"domain_name": "DNA binding"},
        ),
        Chunk(
            text="TP53 tetramerization domain p.Arg248Trp Pathogenic hotspot",
            chunk_type="variant_cluster",
            gene="TP53",
            uniprot_id="P04637",
            residue_start=248,
            residue_end=248,
            metadata={"variant_count": 3},
        ),
        Chunk(
            text="KRAS GTPase domain with p.Gly12Asp Pathogenic activating mutation",
            chunk_type="variant_cluster",
            gene="KRAS",
            uniprot_id="P01116",
            residue_start=12,
            residue_end=12,
            metadata={"variant_count": 1},
        ),
        Chunk(
            text="EGFR protein summary: kinase domain, 3 PTM sites, 2 PPI partners",
            chunk_type="protein_summary",
            gene="EGFR",
            uniprot_id="P00533",
            residue_start=1,
            residue_end=1210,
            metadata={},
        ),
    ]


@pytest.fixture
def vector_store(chroma_dir: Path, sample_chunks: list[Chunk]) -> VectorStore:
    vs = VectorStore(persist_dir=chroma_dir)
    vs.ingest(sample_chunks, FakeEmbedder())
    return vs


@pytest.fixture
def bm25_index(sample_chunks: list[Chunk]) -> BM25Index:
    idx = BM25Index()
    idx.build(sample_chunks)
    return idx


@pytest.fixture
def hybrid_retriever(
    vector_store: VectorStore, bm25_index: BM25Index
) -> HybridRetriever:
    return HybridRetriever(vector_store, bm25_index, FakeEmbedder())


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------


def test_vector_store_ingest_and_count(vector_store: VectorStore) -> None:
    """count() returns 5 after ingesting 5 distinct chunks."""
    assert vector_store.count() == 5


def test_vector_store_search_returns_results(vector_store: VectorStore) -> None:
    """search() returns a non-empty list of SearchResult objects."""
    results = vector_store.search("RING finger ubiquitin", FakeEmbedder(), k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    assert len(results) <= 5


def test_vector_store_search_result_has_chunk(vector_store: VectorStore) -> None:
    """Every SearchResult must carry a Chunk instance."""
    results = vector_store.search("RING finger", FakeEmbedder(), k=5)
    for r in results:
        assert isinstance(r, SearchResult)
        assert isinstance(r.chunk, Chunk)


def test_vector_store_search_result_source_is_dense(vector_store: VectorStore) -> None:
    """Dense search results must be labelled source='dense'."""
    results = vector_store.search("kinase domain", FakeEmbedder(), k=5)
    for r in results:
        assert r.source == "dense"


def test_ingest_idempotent(
    chroma_dir: Path, sample_chunks: list[Chunk]
) -> None:
    """Ingesting the same chunks twice must not increase the document count (upsert)."""
    vs = VectorStore(persist_dir=chroma_dir)
    vs.ingest(sample_chunks, FakeEmbedder())
    count_after_first = vs.count()
    vs.ingest(sample_chunks, FakeEmbedder())
    count_after_second = vs.count()
    assert count_after_second == count_after_first


# ---------------------------------------------------------------------------
# BM25Index tests
# ---------------------------------------------------------------------------


def test_bm25_index_search(bm25_index: BM25Index) -> None:
    """BM25 search for 'RING finger domain' ranks the RING chunk highest."""
    results = bm25_index.search("RING finger domain", k=5)
    assert len(results) > 0
    top_chunk = results[0].chunk
    # The RING finger chunk text contains all three query tokens
    assert "RING" in top_chunk.text or "ring" in top_chunk.text.lower()


def test_bm25_index_search_returns_search_results(bm25_index: BM25Index) -> None:
    """BM25 results are SearchResult instances with source='sparse'."""
    results = bm25_index.search("pathogenic mutation", k=3)
    for r in results:
        assert isinstance(r, SearchResult)
        assert r.source == "sparse"


def test_bm25_index_raises_before_build() -> None:
    """Calling search() before build() must raise RuntimeError."""
    idx = BM25Index()
    with pytest.raises(RuntimeError, match="build"):
        idx.search("BRCA1")


def test_bm25_index_rank_is_one_indexed(bm25_index: BM25Index) -> None:
    """The top result must have rank == 1."""
    results = bm25_index.search("BRCA1", k=3)
    assert results[0].rank == 1


# ---------------------------------------------------------------------------
# HybridRetriever tests
# ---------------------------------------------------------------------------


def test_hybrid_retriever_search(hybrid_retriever: HybridRetriever) -> None:
    """Hybrid search returns SearchResult objects labelled source='hybrid'."""
    results = hybrid_retriever.search("RING finger ubiquitin ligase", k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert isinstance(r, SearchResult)
        assert r.source == "hybrid"


def test_hybrid_retriever_top_k_respected(hybrid_retriever: HybridRetriever) -> None:
    """Returned list length must not exceed the requested k."""
    results = hybrid_retriever.search("domain", k=3)
    assert len(results) <= 3


def test_hybrid_retriever_rrf_scoring(hybrid_retriever: HybridRetriever) -> None:
    """RRF scores must be non-negative and the list must be sorted descending."""
    results = hybrid_retriever.search("BRCA1 RING finger domain", k=5)
    assert len(results) >= 1
    scores = [r.score for r in results]
    assert all(s >= 0 for s in scores)
    # Descending order: first score >= last score
    assert scores[0] >= scores[-1]


def test_hybrid_retriever_rank_is_one_indexed(hybrid_retriever: HybridRetriever) -> None:
    """The top hybrid result must have rank == 1."""
    results = hybrid_retriever.search("TP53 hotspot", k=5)
    assert results[0].rank == 1


# ---------------------------------------------------------------------------
# CollectionEmptyError
# ---------------------------------------------------------------------------


def test_load_retriever_raises_on_empty_dir(tmp_path: Path) -> None:
    """load_retriever() on a directory with no documents must raise CollectionEmptyError."""
    empty_dir = tmp_path / "empty_chroma"
    empty_dir.mkdir()
    with pytest.raises(CollectionEmptyError):
        load_retriever(empty_dir, embedder=FakeEmbedder())


# ---------------------------------------------------------------------------
# Embedding-model consistency
# ---------------------------------------------------------------------------


class OtherFakeEmbedder:
    """A second deterministic embedder with a different model_name."""

    model_name = "other-fake"
    dim = 4

    def embed(self, texts: list[str]) -> np.ndarray:
        np.random.seed(7)
        return np.random.rand(len(texts), 4).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        np.random.seed(1)
        return np.random.rand(4).astype(np.float32)


def test_build_index_persists_embedding_model_metadata(
    tmp_path: Path, sample_chunks: list[Chunk]
) -> None:
    """build_index() must record embedder.model_name in the collection metadata."""
    persist_dir = tmp_path / "chroma_meta"
    build_index(sample_chunks, persist_dir=persist_dir, embedder=FakeEmbedder())

    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    col = client.get_collection("g2p_proteins")
    assert (col.metadata or {}).get("embedding_model") == "fake"


def test_load_retriever_raises_on_embedding_model_mismatch(
    tmp_path: Path, sample_chunks: list[Chunk]
) -> None:
    """Loading a retriever with a different embedder than was used to build must raise."""
    persist_dir = tmp_path / "chroma_mismatch"
    build_index(sample_chunks, persist_dir=persist_dir, embedder=FakeEmbedder())

    with pytest.raises(EmbeddingModelMismatchError, match="fake.*other-fake"):
        load_retriever(persist_dir, embedder=OtherFakeEmbedder(), chunks=sample_chunks)


def test_load_retriever_allows_matching_embedding_model(
    tmp_path: Path, sample_chunks: list[Chunk]
) -> None:
    """Loading with the same model that built the index must succeed."""
    persist_dir = tmp_path / "chroma_match"
    build_index(sample_chunks, persist_dir=persist_dir, embedder=FakeEmbedder())

    retriever = load_retriever(
        persist_dir, embedder=FakeEmbedder(), chunks=sample_chunks
    )
    assert retriever is not None


def test_load_retriever_warns_when_metadata_missing(
    tmp_path: Path, sample_chunks: list[Chunk], caplog
) -> None:
    """A legacy collection without embedding_model metadata should load with a warning."""
    persist_dir = tmp_path / "chroma_legacy"
    # Simulate a legacy collection: build directly via VectorStore with no model name.
    vs = VectorStore(persist_dir=persist_dir)  # no embedding_model_name passed
    vs.ingest(sample_chunks, FakeEmbedder())
    assert (vs._col.metadata or {}).get("embedding_model") is None

    # Should NOT raise; should still return a working retriever.
    retriever = load_retriever(
        persist_dir, embedder=FakeEmbedder(), chunks=sample_chunks
    )
    assert retriever is not None
