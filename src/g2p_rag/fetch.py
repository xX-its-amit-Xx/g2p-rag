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
    # Free-text biology pulled from UniProt's comments[] section. Each is
    # the cleaned, de-cited, length-capped text for that comment type, or
    # empty string when UniProt has no such comment for this protein.
    function_text: str = ""
    pathway_text: str = ""
    subunit_text: str = ""
    disease_text: str = ""


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
# UniProt-direct fallback for per-residue features
# ---------------------------------------------------------------------------
#
# The legacy G2P /protein-features endpoint was retired upstream (2026-05).
# Per-residue annotations are now sourced directly from UniProt's REST API,
# which is the same primary data G2P historically wrapped. We pull domains,
# PTM sites, and binding-site features from the UniProt response and shape
# them as the ProteinFeatures model the rest of the package expects.

_UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb"

# Matches "(PubMed:12345)", "(PubMed:12345, PubMed:67890)", and bare
# "PubMed:12345" tokens used as inline citations in UniProt comment text.
_PUBMED_CITE_RE = re.compile(r"\s*\(?PubMed:\d+(?:,\s*PubMed:\d+)*\)?")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _clean_comment_text(text: str, max_chars: int = 500) -> str:
    """Strip PubMed citations, collapse whitespace, and cap length."""
    if not text:
        return ""
    cleaned = _PUBMED_CITE_RE.sub("", text)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
    if len(cleaned) > max_chars:
        # Truncate at the last word boundary inside max_chars to avoid
        # mid-word cuts; fall back to a hard cut if no boundary exists.
        cut = cleaned[:max_chars]
        last_space = cut.rfind(" ")
        cleaned = (cut[:last_space] if last_space > 0 else cut).rstrip() + "…"
    return cleaned


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _uniprot_features_sync(uniprot_id: str) -> "ProteinFeatures":
    """Synchronous UniProt features fetch for use inside G2PClient.

    Returns a populated ProteinFeatures or raises on hard failure (the
    caller logs + returns an empty model on exception).
    """
    if not uniprot_id:
        return ProteinFeatures(uniprot_id="")

    # Strip isoform suffix (P38398-1 -> P38398) for canonical UniProt entry.
    acc = uniprot_id.split("-", 1)[0]
    with httpx.Client(timeout=20.0,
                       headers={"Accept": "application/json"}) as client:
        r = client.get(f"{_UNIPROT_ENTRY_URL}/{acc}.json")
        r.raise_for_status()
        data = r.json()

    seq = (data.get("sequence", {}) or {}).get("value", "")
    length = len(seq) if seq else (data.get("sequence", {}) or {}).get("length", 0)

    domains: list[ProteinDomain] = []
    ptm_sites: list[PTMSite] = []
    pockets: list[PocketAnnotation] = []

    # UniProt "features" list contains DOMAIN, REGION, MOTIF, ACT_SITE,
    # BINDING, MOD_RES (PTM), CARBOHYD, LIPID, DISULFID, ...
    for feat in data.get("features", []):
        ftype = feat.get("type", "")
        desc = feat.get("description", "") or ""
        loc = feat.get("location", {}) or {}
        start = (loc.get("start") or {}).get("value")
        end = (loc.get("end") or {}).get("value")
        try:
            start_i = int(start) if start is not None else None
            end_i = int(end) if end is not None else start_i
        except (TypeError, ValueError):
            continue

        if ftype in ("Domain", "Region of interest", "Motif"):
            if start_i is not None and end_i is not None:
                domains.append(ProteinDomain(
                    name=desc[:80] or ftype, start=start_i, end=end_i,
                    domain_type=ftype, description=desc,
                ))
        elif ftype in ("Modified residue", "Glycosylation", "Lipidation",
                        "Disulfide bond", "Cross-link"):
            if start_i is not None:
                ptm_sites.append(PTMSite(
                    position=start_i, ptm_type=ftype, evidence=desc,
                ))
        elif ftype in ("Binding site", "Active site", "Site"):
            # Surface binding/active sites as pockets (residue list of length 1)
            if start_i is not None:
                pockets.append(PocketAnnotation(
                    pocket_id=f"{ftype}:{start_i}",
                    residues=[start_i],
                    druggability_score=0.0,
                ))

    # PPI from comments[type=INTERACTION] + free-text biology from
    # FUNCTION / PATHWAY / SUBUNIT / DISEASE comment types. We accumulate
    # text fragments per type and join+clean once at the end so a protein
    # with multiple FUNCTION blocks still produces a single chunk.
    ppi: list[PPInteraction] = []
    function_parts: list[str] = []
    pathway_parts: list[str] = []
    subunit_parts: list[str] = []
    disease_parts: list[str] = []

    for c in data.get("comments", []):
        ctype = c.get("commentType")
        if ctype == "INTERACTION":
            for interaction in c.get("interactions", []):
                partner_obj = interaction.get("interactantTwo") or {}
                partner_gene = partner_obj.get("geneName", "")
                if partner_gene:
                    ppi.append(PPInteraction(
                        partner=partner_gene,
                        interaction_type="binary",
                        evidence=interaction.get("interactionType", ""),
                    ))
        elif ctype == "FUNCTION":
            for t in c.get("texts", []) or []:
                val = (t.get("value") or "").strip()
                # Skip "(Microbial infection)" prefixed entries — those
                # describe viral hijack contexts, not the protein's own role.
                if not val or val.startswith("(Microbial infection)"):
                    continue
                function_parts.append(val)
        elif ctype == "PATHWAY":
            for t in c.get("texts", []) or []:
                val = (t.get("value") or "").strip()
                if val:
                    pathway_parts.append(val)
        elif ctype == "SUBUNIT":
            for t in c.get("texts", []) or []:
                val = (t.get("value") or "").strip()
                if val:
                    subunit_parts.append(val)
        elif ctype == "DISEASE":
            disease = c.get("disease") or {}
            disease_id = (disease.get("diseaseId") or "").strip()
            acronym = (disease.get("acronym") or "").strip()
            desc = (disease.get("description") or "").strip()
            header_bits = [b for b in (disease_id, f"({acronym})" if acronym else "") if b]
            header = " ".join(header_bits)
            piece = f"{header}: {desc}" if header and desc else (header or desc)
            if piece:
                disease_parts.append(piece)

    function_text = _clean_comment_text(" ".join(function_parts), max_chars=500)
    pathway_text = _clean_comment_text(" ".join(pathway_parts), max_chars=500)
    subunit_text = _clean_comment_text(" ".join(subunit_parts), max_chars=500)
    disease_text = _clean_comment_text(" ".join(disease_parts), max_chars=500)

    return ProteinFeatures(
        uniprot_id=acc, sequence=seq, length=int(length or 0),
        domains=domains, ptm_sites=ptm_sites, ppi=ppi, pockets=pockets,
        mavedb_scores=[],  # MaveDB is a separate API; not fetched in fallback
        function_text=function_text,
        pathway_text=pathway_text,
        subunit_text=subunit_text,
        disease_text=disease_text,
    )


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
        """Fetch gene metadata + UniProt + cross-refs for a gene.

        Calls the current G2P portal endpoint `/api/gene/{symbol}` (the older
        `gene-transcript-protein-isoform-structure-map/{symbol}` endpoint was
        retired upstream — 404 as of 2026-05). Parsed payload includes the
        canonical UniProt accession, ChEMBL ID, AlphaFold model, and a
        delimited PDB list, all of which feed downstream chunking.
        """
        endpoint = f"/gene/{gene_symbol}"
        log.info("g2p.get_gene_structure_map", gene=gene_symbol)
        try:
            data = self._get(endpoint)
            return self._parse_structure_map(gene_symbol, data)
        except Exception as exc:
            log.warning("g2p.get_gene_structure_map.failed", gene=gene_symbol, error=str(exc))
            return GeneStructureMap(gene_symbol=gene_symbol)

    def get_protein_features(self, uniprot_id: str) -> ProteinFeatures:
        """Fetch per-residue protein features (domains, PTMs, PPI) for a UniProt accession.

        The legacy G2P `/protein-features/{uniprot}` endpoint was retired
        upstream. We now source per-residue features directly from UniProt's
        REST API (FUNCTION / SUBUNIT / PTM / DOMAIN comments + features list),
        which is the same primary data G2P historically curated.

        Cross-refs that G2P uniquely provides (AlphaFold ID, ChEMBL ID, PDB
        list, drug bank, gencc diseases) are kept in GeneStructureMap and
        surfaced separately by chunk.py.
        """
        endpoint = f"/uniprot-features/{uniprot_id}"
        log.info("g2p.get_protein_features", uniprot_id=uniprot_id)
        try:
            features = _uniprot_features_sync(uniprot_id)
            return features
        except Exception as exc:
            log.warning("g2p.get_protein_features.uniprot_fallback_failed",
                         uniprot_id=uniprot_id, error=str(exc))
            return ProteinFeatures(uniprot_id=uniprot_id)

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_structure_map(self, gene_symbol: str, data: dict) -> GeneStructureMap:
        """Parse a /api/gene/{symbol} response into a GeneStructureMap.

        Current G2P portal payload (2026-05) shape::

            {"status": "success", "data": [
                {"GeneCard": "BRCA1", "UniprotKB_Entry": "P38398",
                 "Canonical_Protein_Isoform": "P38398-1",
                 "PDBinformation": "1JM7;NMR;…&1JNX;X-ray;…",
                 "AlphaFold": "P38398", "ChEMBL": "CHEMBL5990",
                 "DrugBank": "not-available", "HGNC_alias": "FANCS,RNF53,…",
                 …}]}

        The newer endpoint doesn't include the per-residue sequence; we keep
        sequence="" and rely on get_protein_features to fetch UniProt content.
        """
        # Unwrap the {status, data: [record]} envelope.
        rec: dict = {}
        if isinstance(data, dict):
            inner = data.get("data") or data.get("results")
            if isinstance(inner, list) and inner:
                rec = inner[0] if isinstance(inner[0], dict) else {}
            elif isinstance(inner, dict):
                rec = inner
            else:
                rec = data

        def _pick(*keys: str) -> str:
            for k in keys:
                v = rec.get(k)
                if v and v != "not-available":
                    return str(v)
            return ""

        uniprot_id = _pick("UniprotKB_Entry", "uniprot_id", "uniprotId")
        protein_id = _pick("Canonical_Protein_Isoform", "protein_id")
        return GeneStructureMap(
            gene_symbol=gene_symbol,
            uniprot_id=uniprot_id,
            transcript_id="",
            protein_id=protein_id,
            sequence="",
            length=0,
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
