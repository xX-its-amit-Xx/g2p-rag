"""Tests for g2p_rag.fetch — G2PClient and ClinVarClient with mocked HTTP."""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from g2p_rag.fetch import (
    G2PClient,
    ClinVarClient,
    GeneStructureMap,
    ProteinFeatures,
    ClinVarVariant,
    _parse_pdb_information,
    _parse_hgnc_aliases,
    _parse_gencc_diseases,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(status_code: int, payload: dict) -> MagicMock:
    """Return a mock httpx.Response-like object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _make_http_error(status_code: int) -> MagicMock:
    """Return a mock that raises an httpx.HTTPStatusError on raise_for_status."""
    import httpx

    resp = MagicMock()
    resp.status_code = status_code
    error = httpx.HTTPStatusError(
        f"{status_code} error",
        request=MagicMock(),
        response=MagicMock(status_code=status_code),
    )
    resp.raise_for_status.side_effect = error
    return resp


# ---------------------------------------------------------------------------
# G2PClient fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def g2p_client(tmp_cache_dir: Path) -> G2PClient:
    client = G2PClient(cache_dir=tmp_cache_dir)
    yield client
    client._client.close()
    client._cache.close()


# ---------------------------------------------------------------------------
# G2PClient tests
# ---------------------------------------------------------------------------


def test_g2p_client_get_structure_map_success(g2p_client: G2PClient) -> None:
    """Happy-path: HTTP 200 response is parsed into a GeneStructureMap."""
    payload = {
        "uniprot_id": "P38398",
        "transcript_id": "ENST00000357654",
        "protein_id": "ENSP00000350283",
        "sequence": "MDLS",
    }
    with patch.object(g2p_client._client, "get", return_value=_make_response(200, payload)):
        result = g2p_client.get_gene_structure_map("BRCA1")

    assert isinstance(result, GeneStructureMap)
    assert result.gene_symbol == "BRCA1"
    assert result.uniprot_id == "P38398"
    assert result.transcript_id == "ENST00000357654"
    assert result.sequence == "MDLS"


def test_g2p_client_uses_cache(g2p_client: G2PClient) -> None:
    """Second call for the same gene must use the cache — HTTP only called once."""
    payload = {
        "uniprot_id": "P38398",
        "transcript_id": "ENST00000357654",
        "protein_id": "ENSP00000350283",
        "sequence": "MDLS",
    }
    mock_get = MagicMock(return_value=_make_response(200, payload))
    with patch.object(g2p_client._client, "get", mock_get):
        g2p_client.get_gene_structure_map("BRCA1")
        g2p_client.get_gene_structure_map("BRCA1")

    assert mock_get.call_count == 1


def test_g2p_client_handles_404(g2p_client: G2PClient) -> None:
    """A 404 must not raise — graceful degradation returns empty GeneStructureMap."""
    with patch.object(g2p_client._client, "get", return_value=_make_http_error(404)):
        result = g2p_client.get_gene_structure_map("BRCA1")

    assert isinstance(result, GeneStructureMap)
    assert result.gene_symbol == "BRCA1"
    assert result.uniprot_id == ""


def test_g2p_client_parses_protein_features(g2p_client: G2PClient) -> None:
    """Protein-features endpoint response is fully parsed into ProteinFeatures."""
    payload = {
        "sequence": "MDLSALRV",
        "domains": [{"name": "RING", "start": 2, "end": 64, "type": "domain", "description": "E3"}],
        "ptm_sites": [{"position": 100, "type": "phosphorylation", "evidence": "exp"}],
        "ppi": [{"partner": "BRCA2", "interaction_type": "direct", "evidence": "yeast-2-hybrid"}],
        "pockets": [{"id": "P1", "residues": [10, 11, 12], "score": 0.85}],
        "mavedb_scores": [{"variant": "p.Ala2Val", "score": -1.2, "class": "LOF"}],
    }
    with patch.object(g2p_client._client, "get", return_value=_make_response(200, payload)):
        result = g2p_client.get_protein_features("P38398")

    assert isinstance(result, ProteinFeatures)
    assert result.uniprot_id == "P38398"
    assert len(result.domains) == 1
    assert result.domains[0].name == "RING"
    assert len(result.ptm_sites) == 1
    assert result.ptm_sites[0].position == 100
    assert len(result.ppi) == 1
    assert result.ppi[0].partner == "BRCA2"
    assert len(result.pockets) == 1
    assert result.pockets[0].pocket_id == "P1"
    assert len(result.mavedb_scores) == 1
    assert result.mavedb_scores[0].variant == "p.Ala2Val"


# ---------------------------------------------------------------------------
# ClinVarClient fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clinvar_client(tmp_cache_dir: Path) -> ClinVarClient:
    client = ClinVarClient(cache_dir=tmp_cache_dir)
    yield client
    client._client.close()
    client._cache.close()


# ---------------------------------------------------------------------------
# ClinVar helpers
# ---------------------------------------------------------------------------


def _esearch_payload(ids: list[str]) -> dict:
    return {"esearchresult": {"idlist": ids}}


def _esummary_payload(entries: list[dict]) -> dict:
    uids = [e["uid"] for e in entries]
    result: dict = {"uids": uids}
    for e in entries:
        result[e["uid"]] = {
            "title": e["title"],
            "germline_classification": {"description": e.get("clinsig", "Pathogenic")},
            "review_status": e.get("review_status", "criteria provided"),
            "variation_set": [],
            "molecular_consequence": "",
        }
    return {"result": result}


# ---------------------------------------------------------------------------
# ClinVarClient tests
# ---------------------------------------------------------------------------


def test_clinvar_client_get_variants_success(clinvar_client: ClinVarClient) -> None:
    """Two-step esearch + esummary flow returns 2 ClinVarVariant instances."""
    search_resp = _make_response(200, _esearch_payload(["12345", "67890"]))
    summary_resp = _make_response(
        200,
        _esummary_payload([
            {
                "uid": "12345",
                "title": "NM_007294.4(BRCA1):c.181T>G (p.Cys61Gly)",
                "clinsig": "Pathogenic",
                "review_status": "criteria provided",
            },
            {
                "uid": "67890",
                "title": "NM_007294.4(BRCA1):c.211A>G (p.Arg71Gly)",
                "clinsig": "Pathogenic",
                "review_status": "criteria provided",
            },
        ]),
    )

    responses = [search_resp, summary_resp]
    with patch.object(clinvar_client._client, "get", side_effect=responses):
        results = clinvar_client.get_variants("BRCA1")

    assert len(results) == 2
    assert all(isinstance(v, ClinVarVariant) for v in results)
    protein_changes = {v.protein_change for v in results}
    assert "p.Cys61Gly" in protein_changes
    assert "p.Arg71Gly" in protein_changes


def test_clinvar_client_empty_gene(clinvar_client: ClinVarClient) -> None:
    """Empty idlist from esearch returns an empty list with no further HTTP calls."""
    search_resp = _make_response(200, _esearch_payload([]))
    with patch.object(clinvar_client._client, "get", return_value=search_resp) as mock_get:
        results = clinvar_client.get_variants("UNKNOWN_GENE_XYZ")

    assert results == []
    # Only esearch should have been called; esummary must not be reached
    assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# v0.1.2 — G2P /api/gene/ cross-reference parsers
# ---------------------------------------------------------------------------


def test_parse_pdb_information_splits_entries() -> None:
    """PDB delimited string: entries by '&', columns by ';'."""
    raw = "1JM7;NMR;N/A;A=1-110&1JNX;X-ray;2.5Å;A=1646-1859"
    pdbs = _parse_pdb_information(raw)
    assert len(pdbs) == 2
    assert pdbs[0].pdb_id == "1JM7"
    assert pdbs[0].method == "NMR"
    assert pdbs[0].chain_range == "A=1-110"
    assert pdbs[1].pdb_id == "1JNX"
    assert pdbs[1].resolution == "2.5Å"


def test_parse_pdb_information_skips_blank_and_missing_columns() -> None:
    """Blank entries are skipped; rows with missing trailing columns survive."""
    raw = "&1ABC;X-ray&"
    pdbs = _parse_pdb_information(raw)
    assert len(pdbs) == 1
    assert pdbs[0].pdb_id == "1ABC"
    assert pdbs[0].method == "X-ray"
    assert pdbs[0].resolution == ""
    assert pdbs[0].chain_range == ""


def test_parse_pdb_information_handles_not_available() -> None:
    """'not-available' and empty string both yield an empty list."""
    assert _parse_pdb_information("") == []
    assert _parse_pdb_information("not-available") == []


def test_parse_hgnc_aliases_splits_commas() -> None:
    """Comma-separated aliases are split and stripped."""
    aliases = _parse_hgnc_aliases("FANCS, RNF53,  BRCC1")
    assert aliases == ["FANCS", "RNF53", "BRCC1"]


def test_parse_hgnc_aliases_handles_not_available() -> None:
    """'not-available' / '' / empty all yield an empty list."""
    assert _parse_hgnc_aliases("") == []
    assert _parse_hgnc_aliases("not-available") == []


def test_parse_gencc_diseases_from_json_string() -> None:
    """JSON-encoded gencc list is parsed into GenccDisease records."""
    raw = json.dumps([
        {
            "disease_title": "hereditary breast ovarian cancer syndrome",
            "disease_curie": "MONDO:0011450",
            "classification_title": "Definitive",
            "moi_title": "Autosomal dominant",
        },
        {
            "disease_title": "Fanconi anemia",
            "disease_curie": "MONDO:0019391",
        },
    ])
    diseases = _parse_gencc_diseases(raw)
    assert len(diseases) == 2
    assert diseases[0].mondo_id == "MONDO:0011450"
    assert diseases[0].moi == "Autosomal dominant"
    assert diseases[1].disease_name == "Fanconi anemia"


def test_parse_gencc_diseases_handles_bad_json() -> None:
    """Malformed JSON yields an empty list (no exception)."""
    assert _parse_gencc_diseases("{not-json") == []
    assert _parse_gencc_diseases("") == []
    assert _parse_gencc_diseases("not-available") == []


def test_parse_gencc_diseases_accepts_list_input() -> None:
    """Some snapshots ship gencc as a real list rather than a JSON string."""
    raw = [{"disease": "X", "MONDO": "MONDO:0000001"}]
    diseases = _parse_gencc_diseases(raw)
    assert len(diseases) == 1
    assert diseases[0].disease_name == "X"
    assert diseases[0].mondo_id == "MONDO:0000001"


def test_g2p_client_extracts_high_value_fields(g2p_client: G2PClient) -> None:
    """_parse_structure_map pulls AlphaFold / ChEMBL / PDB / HGNC alias / GenCC."""
    payload = {
        "status": "success",
        "data": [{
            "GeneCard": "BRCA1",
            "UniprotKB_Entry": "P38398",
            "Canonical_Protein_Isoform": "P38398-1",
            "AlphaFold": "P38398",
            "ChEMBL": "CHEMBL5990",
            "DrugBank": "not-available",
            "OMIM_id": "113705",
            "Orphanet_id": "145",
            "HGNC_alias": "FANCS,RNF53",
            "PDBinformation": "1JM7;NMR;N/A;A=1-110&1JNX;X-ray;2.5Å;A=1646-1859",
            "genccDiseases": json.dumps([
                {"disease_title": "HBOC", "disease_curie": "MONDO:0011450"},
            ]),
        }],
    }
    with patch.object(g2p_client._client, "get",
                       return_value=_make_response(200, payload)):
        result = g2p_client.get_gene_structure_map("BRCA1")

    assert result.alphafold_id == "P38398"
    assert result.chembl_id == "CHEMBL5990"
    assert result.drugbank_id == ""  # 'not-available' is filtered
    assert result.omim_id == "113705"
    assert result.orphanet_id == "145"
    assert result.hgnc_aliases == ["FANCS", "RNF53"]
    assert len(result.pdb_structures) == 2
    assert result.pdb_structures[0].pdb_id == "1JM7"
    assert len(result.gencc_diseases) == 1
    assert result.gencc_diseases[0].mondo_id == "MONDO:0011450"


def test_clinvar_client_uses_cache(clinvar_client: ClinVarClient) -> None:
    """Repeated call for same gene uses disk cache — HTTP called once per step."""
    search_resp = _make_response(200, _esearch_payload(["12345"]))
    summary_resp = _make_response(
        200,
        _esummary_payload([
            {
                "uid": "12345",
                "title": "NM_007294.4(BRCA1):c.181T>G (p.Cys61Gly)",
                "clinsig": "Pathogenic",
                "review_status": "criteria provided",
            }
        ]),
    )

    # First call uses 2 HTTP requests; second should be served entirely from cache
    with patch.object(
        clinvar_client._client, "get", side_effect=[search_resp, summary_resp]
    ) as mock_get:
        clinvar_client.get_variants("BRCA1")
        clinvar_client.get_variants("BRCA1")

    # Both esearch and esummary are cached after the first call
    assert mock_get.call_count == 2
