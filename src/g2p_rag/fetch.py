"""Fetch gene structure maps, protein features, and ClinVar variants."""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

import httpx
import structlog
from diskcache import Cache
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

GENE_LIST: list[str] = [
    "BRCA1", "BRCA2", "TP53", "KRAS", "EGFR", "MUC1", "UMOD", "SERPING1",
    "KLKB1", "TMED9", "LDLR", "APOE", "HTT", "SOD1", "CFTR", "DMD", "MEN1",
    "NF1", "PTEN", "RB1", "MLH1", "MSH2", "APC", "BRAF", "PIK3CA",
]
G2P_BASE_URL = "https://g2p.broadinstitute.org/api"
CLINVAR_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ProteinDomain(BaseModel):
    """A discrete structural or functional domain within a protein sequence."""

    name: str
    start: int
    end: int
    domain_type: str = ""
    description: str = ""


class PTMSite(BaseModel):
    """A post-translational modification site on a protein."""

    position: int
    ptm_type: str
    evidence: str = ""


class PPInteraction(BaseModel):
    """A protein-protein interaction record."""

    partner: str
    interaction_type: str = ""
    evidence: str = ""


class PocketAnnotation(BaseModel):
    """A predicted druggable binding pocket on a protein."""

    pocket_id: str
    residues: list[int]
    druggability_score: float = 0.0


class MaveDBScore(BaseModel):
    """A variant functional score from MaveDB deep mutational scanning data."""

    variant: str
    score: float
    functional_class: str = ""


class ProteinFeatures(BaseModel):
    """Aggregated structural and functional annotations for a single protein."""

    uniprot_id: str
    sequence: str = ""
    length: int = 0
    domains: list[ProteinDomain] = []
    ptm_sites: list[PTMSite] = []
    ppi: list[PPInteraction] = []
    pockets: list[PocketAnnotation] = []
    mavedb_scores: list[MaveDBScore] = []


class GeneStructureMap(BaseModel):
    """Gene-to-transcript-to-protein isoform structure for a gene symbol."""

    gene_symbol: str
    uniprot_id: str = ""
    transcript_id: str = ""
    protein_id: str = ""
    sequence: str = ""
    length: int = 0
    features: ProteinFeatures = Field(
        default_factory=lambda: ProteinFeatures(uniprot_id="")
    )


class ClinVarVariant(BaseModel):
    """A ClinVar variant record with clinical significance and protein change."""

    variant_id: str
    gene_symbol: str
    position: int = 0
    ref: str = ""
    alt: str = ""
    clinical_significance: str = ""
    review_status: str = ""
    protein_change: str = ""
    hgvs: str = ""
    molecular_consequence: str = ""


# ---------------------------------------------------------------------------
# G2P client
# ---------------------------------------------------------------------------


class G2PClient:
    """HTTP client for the Broad Institute G2P portal API."""

    def __init__(self, cache_dir: Path, ttl: int = 86400) -> None:
        """Initialise the client with an on-disk response cache and HTTP session."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Cache = Cache(str(cache_dir / "g2p"))
        self._ttl = ttl
        self._client = httpx.Client(
            base_url=G2P_BASE_URL,
            timeout=30.0,
            headers={"Accept": "application/json"},
        )
        self._last_request: float = 0.0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "G2PClient":
        """Enter the runtime context; returns self."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Close the underlying HTTP client and cache on context exit."""
        self._client.close()
        self._cache.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Sleep if necessary to honour the 2 req/s rate limit."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_request = time.monotonic()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def _get(self, endpoint: str, params: dict | None = None) -> Any:
        """Perform a GET request with retry logic and disk-cache support."""
        cache_key = f"g2p:{endpoint}:{sorted((params or {}).items())}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            log.debug("g2p.cache_hit", endpoint=endpoint)
            return cached

        self._rate_limit()
        log.debug("g2p.fetch", endpoint=endpoint, params=params)
        response = self._client.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        self._cache.set(cache_key, data, expire=self._ttl)
        return data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_gene_structure_map(self, gene_symbol: str) -> GeneStructureMap:
        """Fetch the gene-transcript-protein isoform structure map for a gene."""
        endpoint = f"/gene-transcript-protein-isoform-structure-map/{gene_symbol}"
        log.info("g2p.get_gene_structure_map", gene=gene_symbol)
        try:
            data = self._get(endpoint)
            return self._parse_structure_map(gene_symbol, data)
        except Exception as exc:
            log.warning("g2p.get_gene_structure_map.failed", gene=gene_symbol, error=str(exc))
            return GeneStructureMap(gene_symbol=gene_symbol)

    def get_protein_features(self, uniprot_id: str) -> ProteinFeatures:
        """Fetch all structural and functional annotations for a UniProt accession."""
        endpoint = f"/protein-features/{uniprot_id}"
        log.info("g2p.get_protein_features", uniprot_id=uniprot_id)
        try:
            data = self._get(endpoint)
            return self._parse_protein_features(uniprot_id, data)
        except Exception as exc:
            log.warning("g2p.get_protein_features.failed", uniprot_id=uniprot_id, error=str(exc))
            return ProteinFeatures(uniprot_id=uniprot_id)

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_structure_map(self, gene_symbol: str, data: dict) -> GeneStructureMap:
        """Parse a raw G2P structure-map response into a GeneStructureMap."""
        # Unwrap optional top-level envelope
        if isinstance(data, dict):
            payload: dict = data.get("data") or data.get("results") or data
        else:
            payload = {}

        def _pick(*keys: str) -> str:
            for k in keys:
                v = payload.get(k)
                if v:
                    return str(v)
            return ""

        uniprot_id = _pick("uniprot_id", "uniprotId", "uniprotID")
        transcript_id = _pick("transcript_id", "transcriptId", "transcriptID")
        protein_id = _pick("protein_id", "proteinId", "proteinID")
        sequence = _pick("sequence", "proteinSequence", "seq")
        length = len(sequence) if sequence else int(payload.get("length", 0))

        return GeneStructureMap(
            gene_symbol=gene_symbol,
            uniprot_id=uniprot_id,
            transcript_id=transcript_id,
            protein_id=protein_id,
            sequence=sequence,
            length=length,
        )

    def _parse_protein_features(self, uniprot_id: str, data: dict) -> ProteinFeatures:
        """Parse a raw G2P protein-features response into a ProteinFeatures model."""
        if isinstance(data, dict):
            payload: dict = data.get("data") or data.get("results") or data
        else:
            payload = {}

        sequence = str(payload.get("sequence", ""))
        length = len(sequence) if sequence else int(payload.get("length", 0))

        # Domains
        domains: list[ProteinDomain] = []
        for d in payload.get("domains", []):
            try:
                domains.append(
                    ProteinDomain(
                        name=str(d.get("name", "")),
                        start=int(d.get("start") or d.get("startPos", 0)),
                        end=int(d.get("end") or d.get("endPos", 0)),
                        domain_type=str(d.get("type", d.get("domain_type", ""))),
                        description=str(d.get("description", "")),
                    )
                )
            except Exception as exc:
                log.warning("g2p.parse_domain.skip", error=str(exc))

        # PTM sites
        ptm_sites: list[PTMSite] = []
        for p in payload.get("ptm_sites", payload.get("ptmSites", [])):
            try:
                ptm_sites.append(
                    PTMSite(
                        position=int(p.get("position") or p.get("pos", 0)),
                        ptm_type=str(p.get("type", p.get("ptm_type", ""))),
                        evidence=str(p.get("evidence", "")),
                    )
                )
            except Exception as exc:
                log.warning("g2p.parse_ptm.skip", error=str(exc))

        # Protein-protein interactions
        ppi: list[PPInteraction] = []
        for i in payload.get("ppi", payload.get("interactions", [])):
            try:
                ppi.append(
                    PPInteraction(
                        partner=str(i.get("partner") or i.get("partnerGene", "")),
                        interaction_type=str(i.get("interaction_type", i.get("type", ""))),
                        evidence=str(i.get("evidence", "")),
                    )
                )
            except Exception as exc:
                log.warning("g2p.parse_ppi.skip", error=str(exc))

        # Pockets
        pockets: list[PocketAnnotation] = []
        for pk in payload.get("pockets", []):
            try:
                raw_residues = pk.get("residues", [])
                pockets.append(
                    PocketAnnotation(
                        pocket_id=str(pk.get("id") or pk.get("pocket_id", "")),
                        residues=[int(r) for r in raw_residues],
                        druggability_score=float(
                            pk.get("score") or pk.get("druggability_score", 0.0)
                        ),
                    )
                )
            except Exception as exc:
                log.warning("g2p.parse_pocket.skip", error=str(exc))

        # MaveDB scores
        mavedb_scores: list[MaveDBScore] = []
        for m in payload.get("mavedb_scores", payload.get("mavedbScores", [])):
            try:
                mavedb_scores.append(
                    MaveDBScore(
                        variant=str(m.get("variant", "")),
                        score=float(m.get("score", 0.0)),
                        functional_class=str(
                            m.get("functional_class") or m.get("class", "")
                        ),
                    )
                )
            except Exception as exc:
                log.warning("g2p.parse_mavedb.skip", error=str(exc))

        return ProteinFeatures(
            uniprot_id=uniprot_id,
            sequence=sequence,
            length=length,
            domains=domains,
            ptm_sites=ptm_sites,
            ppi=ppi,
            pockets=pockets,
            mavedb_scores=mavedb_scores,
        )


# ---------------------------------------------------------------------------
# ClinVar client
# ---------------------------------------------------------------------------


class ClinVarClient:
    """HTTP client for NCBI ClinVar via the E-utilities REST API."""

    _BATCH_SIZE = 200

    def __init__(
        self,
        cache_dir: Path,
        ttl: int = 86400,
        api_key: str = "",
    ) -> None:
        """Initialise the client; api_key is stored in memory but never logged."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Cache = Cache(str(cache_dir / "clinvar"))
        self._ttl = ttl
        self._api_key = api_key  # never surfaced in logs or cache keys
        self._rate_interval = 0.1 if api_key else 0.34
        self._client = httpx.Client(
            base_url=CLINVAR_BASE_URL,
            timeout=30.0,
            headers={"Accept": "application/json"},
        )
        self._last_request: float = 0.0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ClinVarClient":
        """Enter the runtime context; returns self."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Close the underlying HTTP client and cache on context exit."""
        self._client.close()
        self._cache.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Sleep if necessary to honour NCBI rate limits."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._rate_interval:
            time.sleep(self._rate_interval - elapsed)
        self._last_request = time.monotonic()

    def _cache_key(self, endpoint: str, params: dict) -> str:
        """Build a cache key that excludes the api_key parameter."""
        safe_params = {k: v for k, v in params.items() if k != "api_key"}
        raw = f"clinvar:{endpoint}:{sorted(safe_params.items())}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def _get(self, endpoint: str, params: dict) -> Any:
        """Perform a GET request with retry logic and disk-cache support."""
        cache_key = self._cache_key(endpoint, params)
        cached = self._cache.get(cache_key)
        if cached is not None:
            log.debug("clinvar.cache_hit", endpoint=endpoint)
            return cached

        # Inject api_key at request time; it must not reach the cache key
        request_params = dict(params)
        if self._api_key:
            request_params["api_key"] = self._api_key

        self._rate_limit()
        log.debug("clinvar.fetch", endpoint=endpoint)
        response = self._client.get(endpoint, params=request_params)
        response.raise_for_status()
        data = response.json()
        self._cache.set(cache_key, data, expire=self._ttl)
        return data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_variants(
        self, gene_symbol: str, max_results: int = 500
    ) -> list[ClinVarVariant]:
        """Search ClinVar for pathogenic/likely-pathogenic variants for a gene."""
        search_term = (
            f"{gene_symbol}[gene] AND "
            "(pathogenic[clinsig] OR likely_pathogenic[clinsig])"
        )
        log.info("clinvar.get_variants", gene=gene_symbol)

        # Step 1 — ESearch to obtain a list of UIDs
        try:
            search_data = self._get(
                "/esearch.fcgi",
                {
                    "db": "clinvar",
                    "term": search_term,
                    "retmax": max_results,
                    "retmode": "json",
                },
            )
        except Exception as exc:
            log.warning("clinvar.esearch.failed", gene=gene_symbol, error=str(exc))
            return []

        id_list: list[str] = (
            search_data.get("esearchresult", {}).get("idlist", [])
        )
        if not id_list:
            log.info("clinvar.no_results", gene=gene_symbol)
            return []

        # Step 2 — ESummary in batches of 200
        variants: list[ClinVarVariant] = []
        for offset in range(0, len(id_list), self._BATCH_SIZE):
            batch = id_list[offset : offset + self._BATCH_SIZE]
            try:
                summary_data = self._get(
                    "/esummary.fcgi",
                    {
                        "db": "clinvar",
                        "id": ",".join(batch),
                        "retmode": "json",
                    },
                )
                variants.extend(self._parse_variants(gene_symbol, summary_data))
            except Exception as exc:
                log.warning(
                    "clinvar.esummary.batch_failed",
                    gene=gene_symbol,
                    offset=offset,
                    error=str(exc),
                )

        log.info("clinvar.variants_fetched", gene=gene_symbol, count=len(variants))
        return variants

    # ------------------------------------------------------------------
    # Parser
    # ------------------------------------------------------------------

    def _parse_variants(
        self, gene_symbol: str, data: dict
    ) -> list[ClinVarVariant]:
        """Parse an ESummary JSON response into a list of ClinVarVariant records."""
        result_section = data.get("result", {})
        # 'uids' contains the ordered list of IDs; all other keys are summaries
        uids: list[str] = result_section.get("uids", [])
        variants: list[ClinVarVariant] = []

        for uid in uids:
            entry = result_section.get(uid, {})
            if not entry:
                continue
            try:
                # Clinical significance
                germline = entry.get("germline_classification", {})
                clinical_significance = germline.get("description", "")
                review_status = entry.get("review_status", "")

                # Title may contain protein change, e.g. "NM_007294.4(BRCA1):c.5266dupC (p.Gln1756ProfsTer74)"
                title: str = entry.get("title", "")

                # Protein change — match HGVS p. notation
                protein_change = ""
                pc_match = re.search(r"(p\.[A-Za-z]+\d+[A-Za-z*]+)", title)
                if pc_match:
                    protein_change = pc_match.group(1)

                # Amino-acid position from protein change
                position = 0
                if protein_change:
                    pos_match = re.search(r"(\d+)", protein_change)
                    if pos_match:
                        position = int(pos_match.group(1))

                # HGVS c. notation
                hgvs = ""
                variation_set = entry.get("variation_set", [])
                if variation_set and isinstance(variation_set, list):
                    vs = variation_set[0]
                    hgvs = (
                        vs.get("cdna_change")
                        or vs.get("hgvs")
                        or vs.get("nucleotide_change")
                        or ""
                    )

                molecular_consequence = entry.get("molecular_consequence", "")

                variants.append(
                    ClinVarVariant(
                        variant_id=uid,
                        gene_symbol=gene_symbol,
                        position=position,
                        clinical_significance=clinical_significance,
                        review_status=review_status,
                        protein_change=protein_change,
                        hgvs=hgvs,
                        molecular_consequence=molecular_consequence,
                    )
                )
            except Exception as exc:
                log.warning("clinvar.parse_variant.skip", uid=uid, error=str(exc))

        return variants


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------


def fetch_all_genes(
    genes: list[str] = GENE_LIST,
    cache_dir: Path = Path("data"),
) -> dict[str, dict[str, Any]]:
    """Fetch structure maps, protein features, and ClinVar variants for all genes."""
    results: dict[str, dict[str, Any]] = {}

    with G2PClient(cache_dir=cache_dir) as g2p, ClinVarClient(cache_dir=cache_dir) as clinvar:
        for gene in genes:
            log.info("fetch_all_genes.processing", gene=gene)
            structure = g2p.get_gene_structure_map(gene)

            # Enrich with protein features when a UniProt ID is available
            if structure.uniprot_id:
                features = g2p.get_protein_features(structure.uniprot_id)
                structure = structure.model_copy(update={"features": features})

            variants = clinvar.get_variants(gene)

            results[gene] = {
                "structure": structure,
                "variants": variants,
            }
            log.info(
                "fetch_all_genes.done",
                gene=gene,
                variants=len(variants),
            )

    return results
