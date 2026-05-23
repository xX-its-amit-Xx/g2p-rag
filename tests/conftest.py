"""Shared fixtures for the g2p-rag test suite."""
import pytest
from pathlib import Path

from g2p_rag.fetch import (
    GeneStructureMap,
    ProteinFeatures,
    ProteinDomain,
    PTMSite,
    PPInteraction,
    PocketAnnotation,
    MaveDBScore,
    ClinVarVariant,
)
from g2p_rag.chunk import Chunk


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache


# ---------------------------------------------------------------------------
# Protein-feature building blocks
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_domain() -> ProteinDomain:
    return ProteinDomain(
        name="RING finger",
        start=2,
        end=64,
        domain_type="domain",
        description="E3 ubiquitin ligase activity",
    )


@pytest.fixture
def sample_ptm_sites() -> list[PTMSite]:
    return [
        PTMSite(position=100, ptm_type="phosphorylation", evidence="experimental"),
        PTMSite(position=200, ptm_type="ubiquitination", evidence="experimental"),
        PTMSite(position=300, ptm_type="acetylation", evidence="predicted"),
    ]


@pytest.fixture
def sample_ppi() -> list[PPInteraction]:
    return [
        PPInteraction(partner="BRCA2", interaction_type="direct", evidence="yeast-2-hybrid"),
        PPInteraction(partner="RAD51", interaction_type="direct", evidence="co-IP"),
    ]


@pytest.fixture
def sample_pockets() -> list[PocketAnnotation]:
    return [
        PocketAnnotation(
            pocket_id="P1",
            residues=[10, 11, 12, 20],
            druggability_score=0.85,
        )
    ]


@pytest.fixture
def sample_mavedb() -> list[MaveDBScore]:
    return [
        MaveDBScore(variant="p.Ala2Val", score=-1.2, functional_class="LOF"),
        MaveDBScore(variant="p.Gly3Arg", score=0.3, functional_class="INT"),
        MaveDBScore(variant="p.Leu5Pro", score=-2.1, functional_class="LOF"),
    ]


@pytest.fixture
def sample_features(
    sample_domain: ProteinDomain,
    sample_ptm_sites: list[PTMSite],
    sample_ppi: list[PPInteraction],
    sample_pockets: list[PocketAnnotation],
    sample_mavedb: list[MaveDBScore],
) -> ProteinFeatures:
    return ProteinFeatures(
        uniprot_id="P38398",
        sequence=(
            "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQ"
            "CPLCKNDITKRSLQESTRFSQLVEELLKIICAFQLDTGLEI"
        ),
        domains=[sample_domain],
        ptm_sites=sample_ptm_sites,
        ppi=sample_ppi,
        pockets=sample_pockets,
        mavedb_scores=sample_mavedb,
    )


@pytest.fixture
def sample_structure(sample_features: ProteinFeatures) -> GeneStructureMap:
    sequence = sample_features.sequence
    return GeneStructureMap(
        gene_symbol="BRCA1",
        uniprot_id="P38398",
        transcript_id="ENST00000357654",
        protein_id="ENSP00000350283",
        sequence=sequence,
        length=len(sequence),
        features=sample_features,
    )


@pytest.fixture
def sample_variants() -> list[ClinVarVariant]:
    return [
        ClinVarVariant(
            variant_id="1001",
            gene_symbol="BRCA1",
            position=61,
            protein_change="p.Cys61Gly",
            clinical_significance="Pathogenic",
            review_status="criteria provided",
        ),
        ClinVarVariant(
            variant_id="1002",
            gene_symbol="BRCA1",
            position=64,
            protein_change="p.Cys64Arg",
            clinical_significance="Likely pathogenic",
            review_status="criteria provided",
        ),
        ClinVarVariant(
            variant_id="1003",
            gene_symbol="BRCA1",
            position=70,
            protein_change="p.Arg70Trp",
            clinical_significance="Uncertain significance",
            review_status="no assertion criteria",
        ),
        ClinVarVariant(
            variant_id="1004",
            gene_symbol="BRCA1",
            position=120,
            protein_change="p.Glu120Lys",
            clinical_significance="Pathogenic",
            review_status="criteria provided",
        ),
        ClinVarVariant(
            variant_id="1005",
            gene_symbol="BRCA1",
            position=125,
            protein_change="p.Val125Gly",
            clinical_significance="Benign",
            review_status="criteria provided",
        ),
    ]


# ---------------------------------------------------------------------------
# Chunk fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        text="Test chunk for BRCA1 domain",
        chunk_type="domain",
        gene="BRCA1",
        uniprot_id="P38398",
        residue_start=2,
        residue_end=64,
        metadata={"domain_name": "RING finger"},
    )
