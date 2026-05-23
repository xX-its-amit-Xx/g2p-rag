"""Tests for g2p_rag.chunk — ProteinChunker methods and Chunk helpers."""

import pytest
from g2p_rag.chunk import ProteinChunker, Chunk, chunk_all
from g2p_rag.fetch import (
    ClinVarVariant,
    GeneStructureMap,
    ProteinFeatures,
    ProteinDomain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_variant(variant_id: str, position: int, clinsig: str = "Pathogenic") -> ClinVarVariant:
    return ClinVarVariant(
        variant_id=variant_id,
        gene_symbol="BRCA1",
        position=position,
        protein_change=f"p.Xxx{position}Yyy",
        clinical_significance=clinsig,
        review_status="criteria provided",
    )


# ---------------------------------------------------------------------------
# domain_chunks
# ---------------------------------------------------------------------------


def test_domain_chunks_returns_chunk_per_domain(sample_structure: GeneStructureMap) -> None:
    """One Chunk is produced for each domain in the structure."""
    chunker = ProteinChunker()
    chunks = chunker.domain_chunks(sample_structure)
    n_domains = len(sample_structure.features.domains)
    assert len(chunks) == n_domains
    assert n_domains == 1  # fixture has exactly one domain


def test_domain_chunks_contains_domain_name(sample_structure: GeneStructureMap) -> None:
    """The domain name must appear in the produced chunk text."""
    chunker = ProteinChunker()
    chunk = chunker.domain_chunks(sample_structure)[0]
    assert "RING finger" in chunk.text


def test_domain_chunk_chunk_type(sample_structure: GeneStructureMap) -> None:
    """Domain chunks must have chunk_type == 'domain'."""
    chunker = ProteinChunker()
    for chunk in chunker.domain_chunks(sample_structure):
        assert chunk.chunk_type == "domain"


def test_domain_chunk_residue_range(sample_structure: GeneStructureMap) -> None:
    """residue_start / residue_end on the chunk must match the domain boundaries."""
    chunker = ProteinChunker()
    chunk = chunker.domain_chunks(sample_structure)[0]
    domain = sample_structure.features.domains[0]
    assert chunk.residue_start == domain.start
    assert chunk.residue_end == domain.end


# ---------------------------------------------------------------------------
# variant_cluster_chunks
# ---------------------------------------------------------------------------


def test_variant_cluster_chunks_clusters_nearby() -> None:
    """variant_cluster_chunks groups spatially nearby variants into clusters."""
    chunker = ProteinChunker()
    variants = [
        _make_variant("v1", 61),
        _make_variant("v2", 64),
        _make_variant("v3", 70),
        _make_variant("v4", 200),
    ]
    # Variants 61,64,70 are within window=10; variant at 200 is in its own cluster.
    chunks = chunker.variant_cluster_chunks("BRCA1", "P38398", variants, window=10)
    assert len(chunks) == 2
    assert all(c.chunk_type == "variant_cluster" for c in chunks)
    # First cluster covers 61–70
    assert chunks[0].residue_start == 61
    assert chunks[0].residue_end == 70


def test_variant_cluster_chunks_chunk_type() -> None:
    """Cluster chunks must have chunk_type == 'variant_cluster'."""
    chunker = ProteinChunker()
    variants = [_make_variant("v1", 61), _make_variant("v2", 64)]
    chunks = chunker.variant_cluster_chunks("BRCA1", "P38398", variants)
    assert all(c.chunk_type == "variant_cluster" for c in chunks)


def test_variant_cluster_skips_zero_position() -> None:
    """Variants with position == 0 are excluded; others produce one cluster."""
    chunker = ProteinChunker()
    variants = [
        _make_variant("v_zero", 0),
        _make_variant("v_good", 50),
    ]
    chunks = chunker.variant_cluster_chunks("BRCA1", "P38398", variants)
    # Only v_good (position=50) should survive; v_zero is dropped.
    assert len(chunks) == 1
    assert chunks[0].residue_start == 50


def test_variant_cluster_empty_returns_empty() -> None:
    """All-zero (unpositioned) variant list must return empty list."""
    chunker = ProteinChunker()
    variants = [_make_variant("v1", 0), _make_variant("v2", 0)]
    chunks = chunker.variant_cluster_chunks("BRCA1", "P38398", variants)
    assert chunks == []


# ---------------------------------------------------------------------------
# protein_summary_chunk
# ---------------------------------------------------------------------------


def test_protein_summary_chunk(
    sample_structure: GeneStructureMap,
    sample_variants: list[ClinVarVariant],
) -> None:
    """Summary chunk has correct chunk_type, gene, and residue_start."""
    chunker = ProteinChunker()
    chunk = chunker.protein_summary_chunk(sample_structure, sample_variants)
    assert chunk.chunk_type == "protein_summary"
    assert chunk.gene == "BRCA1"
    assert chunk.residue_start == 1


def test_protein_summary_chunk_mentions_gene(
    sample_structure: GeneStructureMap,
    sample_variants: list[ClinVarVariant],
) -> None:
    """The word 'BRCA1' must appear somewhere in the summary text."""
    chunker = ProteinChunker()
    chunk = chunker.protein_summary_chunk(sample_structure, sample_variants)
    assert "BRCA1" in chunk.text


# ---------------------------------------------------------------------------
# chunk_all
# ---------------------------------------------------------------------------


def test_chunk_all_returns_list(
    sample_structure: GeneStructureMap,
    sample_variants: list[ClinVarVariant],
) -> None:
    """chunk_all accepts gene_data dict and returns a list of Chunk objects."""
    gene_data = {
        "BRCA1": {
            "structure": sample_structure,
            "variants": sample_variants,
        }
    }
    result = chunk_all(gene_data)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(c, Chunk) for c in result)


def test_chunk_all_skips_bad_entry(sample_structure: GeneStructureMap) -> None:
    """chunk_all must not crash when a gene entry has no 'variants' key."""
    gene_data = {"BRCA1": {"structure": sample_structure}}
    result = chunk_all(gene_data)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Chunk.to_chroma_metadata
# ---------------------------------------------------------------------------


def test_to_chroma_metadata_is_flat(sample_chunk: Chunk) -> None:
    """to_chroma_metadata() must return only scalar values (no nested dicts)."""
    meta = sample_chunk.to_chroma_metadata()
    for value in meta.values():
        assert isinstance(value, (str, int, float, bool)), (
            f"Non-scalar metadata value: {value!r} (type {type(value).__name__})"
        )


def test_to_chroma_metadata_contains_gene(sample_chunk: Chunk) -> None:
    """The 'gene' key must be present in the flattened metadata."""
    meta = sample_chunk.to_chroma_metadata()
    assert "gene" in meta
    assert meta["gene"] == "BRCA1"


def test_to_chroma_metadata_extra_fields_are_strings(sample_chunk: Chunk) -> None:
    """Extra metadata keys from the Chunk.metadata dict are coerced to strings."""
    meta = sample_chunk.to_chroma_metadata()
    assert "domain_name" in meta
    assert isinstance(meta["domain_name"], str)
