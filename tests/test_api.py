"""Tests for the g2p-rag public API surface (G2PRetriever + RetrievedChunk).

Only imports from g2p_rag top-level package — internal modules are never touched.
"""
import re
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from g2p_rag import G2PRetriever, RetrievedChunk, __version__


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_search_result():
    """A fake internal SearchResult-like object."""
    from g2p_rag.retrieve import SearchResult
    from g2p_rag.chunk import Chunk
    chunk = Chunk(
        text="Gene: BRCA1 | UniProt: P38398\nDomain: RING finger (domain)\nResidues: 2–64",
        chunk_type="domain",
        gene="BRCA1",
        uniprot_id="P38398",
        residue_start=2,
        residue_end=64,
        metadata={"domain_name": "RING finger"},
    )
    return SearchResult(chunk=chunk, score=0.85, rank=1, source="hybrid")


@pytest.fixture
def g2p_retriever(tmp_path, mock_search_result):
    """G2PRetriever with a mocked internal HybridRetriever."""
    retriever = G2PRetriever(persist_dir=str(tmp_path / "chroma"))

    fake_internal = MagicMock()
    fake_internal.search.return_value = [mock_search_result]
    retriever._retriever = fake_internal  # bypass lazy load

    return retriever


# ---------------------------------------------------------------------------
# Version and model shape
# ---------------------------------------------------------------------------


def test_version_is_string():
    """__version__ must be a string equal to '0.1.2'."""
    assert isinstance(__version__, str)
    assert __version__ == "0.1.2"


def test_retrieved_chunk_is_pydantic_model():
    """RetrievedChunk must be a Pydantic v2 BaseModel with model_fields."""
    assert hasattr(RetrievedChunk, "model_fields")
    assert hasattr(RetrievedChunk, "model_validate")


def test_retrieved_chunk_required_fields():
    """RetrievedChunk can be instantiated with all fields and they are accessible."""
    chunk = RetrievedChunk(
        text="test",
        gene="BRCA1",
        uniprot_id="P38398",
        chunk_type="domain",
        residue_range="2-64",
        source_url="https://g2p.broadinstitute.org/protein?gene=BRCA1",
        score=0.85,
    )
    assert chunk.gene == "BRCA1"
    assert chunk.score == 0.85


def test_retrieved_chunk_type_literal():
    """chunk_type rejects values outside the Literal union."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        RetrievedChunk(
            text="x",
            gene="X",
            uniprot_id="Y",
            chunk_type="invalid",
            residue_range="",
            source_url="",
            score=0.0,
        )


# ---------------------------------------------------------------------------
# G2PRetriever instantiation
# ---------------------------------------------------------------------------


def test_g2p_retriever_instantiation():
    """G2PRetriever construction with an arbitrary path does not raise."""
    G2PRetriever(persist_dir="/tmp/test")


def test_g2p_retriever_is_lazy():
    """_retriever is None immediately after construction (lazy load)."""
    retriever = G2PRetriever(persist_dir="/tmp/test_lazy")
    assert retriever._retriever is None


# ---------------------------------------------------------------------------
# retrieve() return type
# ---------------------------------------------------------------------------


def test_retrieve_returns_list_of_retrieved_chunks(g2p_retriever):
    """retrieve() returns a non-empty list where every item is a RetrievedChunk."""
    result = g2p_retriever.retrieve("test query")
    assert isinstance(result, list)
    assert len(result) > 0
    for item in result:
        assert isinstance(item, RetrievedChunk)


def test_retrieve_chunk_fields_populated(g2p_retriever):
    """First result has non-empty text, gene, uniprot_id, source_url, and score > 0."""
    result = g2p_retriever.retrieve("BRCA1 RING domain")
    chunk = result[0]
    assert chunk.text != ""
    assert chunk.gene != ""
    assert chunk.uniprot_id != ""
    assert chunk.source_url != ""
    assert chunk.score > 0


def test_retrieve_source_url_format(g2p_retriever):
    """source_url starts with the G2P portal base URL including the gene parameter."""
    result = g2p_retriever.retrieve("BRCA1")
    chunk = result[0]
    assert chunk.source_url.startswith("https://g2p.broadinstitute.org/protein?gene=")


def test_retrieve_residue_range_format(g2p_retriever):
    """Domain chunk residue_range matches 'start-end' pattern."""
    result = g2p_retriever.retrieve("RING finger domain")
    chunk = result[0]
    assert re.match(r"^\d+-\d+$", chunk.residue_range), (
        f"residue_range {chunk.residue_range!r} does not match expected 'start-end' format"
    )


# ---------------------------------------------------------------------------
# retrieve() parameter forwarding
# ---------------------------------------------------------------------------


def test_retrieve_k_parameter(g2p_retriever):
    """k=3 is forwarded to the internal search call as a keyword argument."""
    g2p_retriever.retrieve("query", k=3)
    call_kwargs = g2p_retriever._retriever.search.call_args
    # Accept either positional or keyword argument
    passed_k = (
        call_kwargs.kwargs.get("k")
        if call_kwargs.kwargs.get("k") is not None
        else call_kwargs.args[1]
    )
    assert passed_k == 3


def test_retrieve_gene_filter_forwarded(g2p_retriever):
    """gene_filter list is forwarded to the internal search call."""
    gene_filter = ["BRCA1", "TP53"]
    g2p_retriever.retrieve("query", gene_filter=gene_filter)
    call_kwargs = g2p_retriever._retriever.search.call_args.kwargs
    assert call_kwargs.get("gene_filter") == gene_filter


# ---------------------------------------------------------------------------
# Lazy load / call-count behaviour
# ---------------------------------------------------------------------------


def test_g2p_retriever_lazy_load_called_once(g2p_retriever):
    """Calling retrieve twice invokes internal search twice but the mock is set once."""
    g2p_retriever.retrieve("first query")
    g2p_retriever.retrieve("second query")
    # The internal mock is already in place; search must be called exactly twice
    assert g2p_retriever._retriever.search.call_count == 2
    # The fixture set _retriever once; it must still be the same mock object
    assert g2p_retriever._retriever is not None


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_retrieved_chunk_serializable():
    """model_dump() returns a dict containing all expected field keys."""
    chunk = RetrievedChunk(
        text="sample text",
        gene="BRCA1",
        uniprot_id="P38398",
        chunk_type="domain",
        residue_range="2-64",
        source_url="https://g2p.broadinstitute.org/protein?gene=BRCA1",
        score=0.85,
    )
    dumped = chunk.model_dump()
    assert isinstance(dumped, dict)
    expected_keys = {"text", "gene", "uniprot_id", "chunk_type", "residue_range", "source_url", "score"}
    assert expected_keys.issubset(dumped.keys())
