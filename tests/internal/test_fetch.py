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
    """Happy-path: HTTP 200 response is parsed into a GeneStructureMap.

    The current G2P /api/gene/{symbol} endpoint (2026-05) wraps the record
    in a {status, data: [...]} envelope and uses GeneCard-style field names
    (UniprotKB_Entry, Canonical_Protein_Isoform, …). It does NOT return a
    transcript_id or sequence — those fields are populated downstream via
    UniProt in get_protein_features.
    """
    payload = {
        "status": "success",
        "data": [{
            "GeneCard": "BRCA1",
            "UniprotKB_Entry": "P38398",
            "Canonical_Protein_Isoform": "P38398-1",
            "AlphaFold": "P38398",
            "ChEMBL": "CHEMBL5990",
            "DrugBank": "not-available",
            "HGNC_alias": "FANCS,RNF53",
            "PDBinformation": "1JM7;NMR;N/A;A=1-110",
        }],
    }
    with patch.object(g2p_client._client, "get", return_value=_make_response(200, payload)):
        result = g2p_client.get_gene_structure_map("BRCA1")

    assert isinstance(result, GeneStructureMap)
    assert result.gene_symbol == "BRCA1"
    assert result.uniprot_id == "P38398"
    assert result.protein_id == "P38398-1"
    # /api/gene/ no longer carries transcript_id or per-residue sequence —
    # those are now sourced from UniProt by get_protein_features.
    assert result.transcript_id == ""
    assert result.sequence == ""


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


@pytest.mark.skip(
    reason="legacy G2P endpoint retired upstream; per-residue features now "
    "via _uniprot_features_sync — see test_uniprot_features_* once added"
)
def test_g2p_client_parses_protein_features(g2p_client: G2PClient) -> None:
    """Protein-features endpoint response is fully parsed into ProteinFeatures.

    The G2P `/protein-features/{uniprot}` endpoint was retired upstream
    (2026-05). G2PClient.get_protein_features now delegates to UniProt's
    REST API via the module-level `_uniprot_fetch_raw` /
    `_uniprot_features_sync` helpers, so this httpx-client-patching test
    no longer exercises the live path. Skipping until we add coverage
    against the UniProt fallback shape.
    """
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
