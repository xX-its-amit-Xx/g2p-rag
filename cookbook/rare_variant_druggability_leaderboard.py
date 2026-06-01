"""
Cookbook: Rare-Variant Druggability Leaderboard — All-Gene Ranking
(citation-disciplined, all 48 indexed genes)

Drug-discovery question:
    If we had to triage all 48 indexed genes for "rare-variant druggability"
    using only what the g2p-rag ChromaDB index says, which 10 would top the
    leaderboard? "Druggable" here is a synthetic composite built ENTIRELY
    from chunk-level signals — not from training-data drug-name recall:

      (A) Variant pile-up density
            = (Pathogenic + Likely-pathogenic ClinVar count) / sequence_length
          taken verbatim from each gene's `protein_summary` chunk text.
      (B) Structural coverage proxy
            = number of "Druggable pockets" enumerated in `protein_summary`
              (the index already aggregates UniProt Binding-site / Active-site
              / Site annotations + AlphaFold/PDB-derived pockets into a single
              count there). NB: the index does NOT carry raw PDB chain counts
              or per-residue AF confidence; the chunk-level "Druggable
              pockets" count is the closest stand-in the corpus actually
              contains.
      (C) Domain breadth
            = sum of residues covered by indexed `domain` chunks.
              A gene with zero domain chunks (intrinsically disordered or
              non-indexed fold) loses this column. This is the structural
              proxy for "are there PDB-resolvable folded regions to drug?"
      (D) Clinical / disease track record
            = number of distinct disease entries parsed out of the `disease`
              chunk (each block begins with "Name (ACRONYM):"). Stand-in for
              "# approved drugs + ongoing trials per gencc_diseases entry" —
              the index carries the disease list, NOT the drug list, so the
              column is honest about what it measures.

    The composite is the rank-sum of (A), (B), (C), (D) — lower rank-sum =
    more druggable. Rank-sum is robust to the wildly different scales of
    these four signals (per-residue density vs. raw counts).

    NO hard-coded drug names, DrugBank IDs, MONDO ids, or mechanism phrases
    appear in the printed output. Each printed factual line is either tied
    to a specific retrieved chunk (Cited(text, chunk)) or explicitly tagged
    TEXTBOOK_CONTEXT (source=None, label="TEXTBOOK_CONTEXT").

Run with the project venv:
    $env:PYTHONIOENCODING="utf-8"
    d:/Users/ashenoy00000/.windsurf/g2p-rag/.venv/Scripts/python.exe `
        d:/Users/ashenoy00000/.windsurf/g2p-rag/cookbook/rare_variant_druggability_leaderboard.py
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Unicode-safe stdout on Windows consoles BEFORE any print() runs.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Allow running directly from the cookbook/ directory without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _citation import Cited, assert_supported, find_in_chunks, print_index_manifest  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The 48 genes currently indexed in the g2p-rag ChromaDB collection
# (see CHUNK_INDEX_VERSION v0.11 reingest list). Hard-coding the gene LIST
# is allowed — these are NOT facts about each gene, just identifiers for the
# retrieval scope. The script derives every numeric column from chunk text.
INDEXED_GENES: tuple[str, ...] = (
    "ACVR2A", "AKT1", "ALAS1", "APOE", "APP", "BCL11A", "BDKRB2", "BMPR2",
    "BRAF", "C5", "CALCA", "CALCRL", "CFB", "CFTR", "CHRM4", "CRHR1",
    "CXCR4", "CYP21A2", "DMD", "EDN1", "EDNRA", "EGFR", "ERBB2", "F12",
    "GLA", "GLP1R", "HBB", "HMBS", "HTT", "IDH1", "IL13", "KLKB1", "LDLR",
    "MC4R", "MUC1", "PCSK9", "PIGA", "PIK3CA", "POMC", "SERPING1", "SMN1",
    "SMN2", "SOD1", "THRB", "TMED9", "TNF", "TP53", "UMOD",
)


# ---------------------------------------------------------------------------
# Environment / .env loader
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Load .env from the g2p-rag project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv  # type: ignore[import]

        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded environment variables from {env_path}")
        else:
            print("No .env file found - using existing environment variables.")
    except ImportError:
        print("python-dotenv not installed; skipping .env load.")


# ---------------------------------------------------------------------------
# Parsers — every regex below extracts ONLY what the chunk text literally
# contains. No external lookups, no hard-coded numbers per gene.
# ---------------------------------------------------------------------------

_SEQ_LEN_RE = re.compile(r"Canonical sequence length:\s*(\d+)\s*aa", re.IGNORECASE)
_PATH_LP_RE = re.compile(r"Pathogenic/Likely pathogenic:\s*(\d+)", re.IGNORECASE)
_VUS_RE = re.compile(r"VUS:\s*(\d+)", re.IGNORECASE)
_BENIGN_RE = re.compile(r"Benign/Likely benign:\s*(\d+)", re.IGNORECASE)
_CLINVAR_TOTAL_RE = re.compile(r"ClinVar variants:\s*(\d+)\s*total", re.IGNORECASE)
_DRUG_POCKETS_RE = re.compile(r"Druggable pockets\s*\((\d+)\)", re.IGNORECASE)
_DOMAINS_COUNT_RE = re.compile(r"Domains\s*\((\d+)\)", re.IGNORECASE)
_PPI_COUNT_RE = re.compile(r"PPI partners\s*\((\d+)\)", re.IGNORECASE)
# Disease block headers in the disease chunk look like "Name (ACRONYM):" — the
# acronym is 2-12 uppercase/digits, followed by a colon. Robust to the en-dash
# chunk-body encoding.
_DISEASE_ENTRY_RE = re.compile(r"\(([A-Z][A-Z0-9]{1,11})\):")
# Residue range strings — the chunk_type=domain chunks expose this as
# 'start-end' in metadata, but the chunk body uses 'start–end' (en-dash). The
# retriever normalises metadata to ASCII hyphens, so this regex hits the
# RetrievedChunk.residue_range field.
_RES_RANGE_RE = re.compile(r"(\d+)\s*[-–]\s*(\d+)")


def _parse_int(rx: re.Pattern[str], text: str) -> int:
    """Return the first capture group as int, or 0 if not found."""
    m = rx.search(text)
    return int(m.group(1)) if m else 0


def _parse_range(residue_range: str) -> tuple[int, int] | None:
    """Parse a 'start-end' (or en-dashed) residue range into a tuple."""
    if not residue_range:
        return None
    m = _RES_RANGE_RE.search(residue_range)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# ---------------------------------------------------------------------------
# Per-gene chunk fetching
# ---------------------------------------------------------------------------

def _fetch_chunks_for_gene(retriever, gene: str) -> dict[str, list]:
    """Fetch a representative bundle of chunks for one gene, bucketed by type.

    We issue several targeted queries that the v0.11 hybrid retriever resolves
    well for each chunk type. All queries are gene-filtered, so cross-gene
    bleed is impossible.
    """
    buckets: dict[str, list] = defaultdict(list)

    # protein_summary: anchor chunk with the headline druggability metrics.
    for chunk in retriever.retrieve(
        f"{gene} protein summary canonical sequence length druggable pockets ClinVar",
        k=4,
        gene_filter=[gene],
    ):
        buckets[chunk.chunk_type].append(chunk)

    # If protein_summary did not surface in the top-k of the headline query
    # (some genes — e.g. LDLR with 11 domain chunks — push the summary out of
    # the top-4), retry with a wider net specifically aimed at the summary.
    if not buckets.get("protein_summary"):
        for chunk in retriever.retrieve(
            f"{gene} Protein summary Canonical sequence length",
            k=15,
            gene_filter=[gene],
        ):
            if chunk.chunk_type == "protein_summary":
                buckets["protein_summary"].append(chunk)
                break

    # domain chunks: structural coverage. Use a generous k so genes with many
    # domains (LDLR has 11) get full coverage in the merged-span sum.
    for chunk in retriever.retrieve(
        f"{gene} domain residues fold catalytic binding",
        k=15,
        gene_filter=[gene],
    ):
        buckets[chunk.chunk_type].append(chunk)

    # disease chunks: indication track record.
    for chunk in retriever.retrieve(
        f"{gene} disease indication clinical phenotype syndrome",
        k=3,
        gene_filter=[gene],
    ):
        buckets[chunk.chunk_type].append(chunk)

    # Deduplicate per type — the retriever can return the same chunk via
    # multiple queries.
    for ct, lst in buckets.items():
        seen: set[tuple[str, str]] = set()
        deduped = []
        for c in lst:
            key = (c.residue_range, c.text[:80])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        buckets[ct] = deduped

    return dict(buckets)


def _pick_protein_summary(buckets: dict[str, list]):
    """Pick the highest-scoring protein_summary chunk (or None)."""
    ps_list = buckets.get("protein_summary", [])
    if not ps_list:
        return None
    return max(ps_list, key=lambda c: c.score)


def _pick_disease(buckets: dict[str, list]):
    """Pick the highest-scoring disease chunk (or None)."""
    dl = buckets.get("disease", [])
    if not dl:
        return None
    return max(dl, key=lambda c: c.score)


def _sum_domain_residues(buckets: dict[str, list]) -> tuple[int, int]:
    """Return (n_domain_chunks, total_residues_covered) from domain chunks.

    Overlapping ranges are merged so a gene doesn't get artificially inflated
    by re-indexed paralogous domains. The arithmetic shows its work in the
    leaderboard footer.
    """
    domain_chunks = buckets.get("domain", [])
    spans: list[tuple[int, int]] = []
    for c in domain_chunks:
        rng = _parse_range(c.residue_range)
        if rng is not None:
            spans.append(rng)
    if not spans:
        return 0, 0
    # Merge overlapping ranges before summing.
    spans.sort()
    merged: list[tuple[int, int]] = [spans[0]]
    for s, e in spans[1:]:
        ms, me = merged[-1]
        if s <= me + 1:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))
    total = sum(e - s + 1 for s, e in merged)
    return len(domain_chunks), total


def _count_disease_entries(disease_chunk) -> int:
    """Count distinct '(ACRONYM):' headers inside a disease chunk's text.

    The G2P disease chunk template uses 'Disease Name (ACRONYM): description'
    for each indexed condition. The acronym + colon pattern is the most
    chunk-faithful entry counter we can build without external ontologies.
    """
    if disease_chunk is None:
        return 0
    acronyms = set(_DISEASE_ENTRY_RE.findall(disease_chunk.text))
    return len(acronyms)


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------

class GeneFeatures:
    """All chunk-derived features for one gene. Numbers ONLY come from chunks."""

    __slots__ = (
        "gene", "ps_chunk", "disease_chunk",
        "seq_len", "n_pathogenic_lp", "n_vus", "n_benign", "clinvar_total",
        "n_drug_pockets", "n_domains_field", "n_ppi_partners",
        "n_domain_chunks", "domain_residues_covered",
        "n_disease_entries",
        "variant_density",  # (Pathogenic + LP) / seq_len
    )

    def __init__(self, gene: str) -> None:
        self.gene = gene
        self.ps_chunk = None
        self.disease_chunk = None
        self.seq_len = 0
        self.n_pathogenic_lp = 0
        self.n_vus = 0
        self.n_benign = 0
        self.clinvar_total = 0
        self.n_drug_pockets = 0
        self.n_domains_field = 0
        self.n_ppi_partners = 0
        self.n_domain_chunks = 0
        self.domain_residues_covered = 0
        self.n_disease_entries = 0
        self.variant_density = 0.0


def _extract_features(gene: str, buckets: dict[str, list]) -> GeneFeatures:
    """Parse all four druggability columns from chunk text. No external data."""
    f = GeneFeatures(gene)

    ps = _pick_protein_summary(buckets)
    f.ps_chunk = ps
    if ps is not None:
        t = ps.text
        f.seq_len = _parse_int(_SEQ_LEN_RE, t)
        f.n_pathogenic_lp = _parse_int(_PATH_LP_RE, t)
        f.n_vus = _parse_int(_VUS_RE, t)
        f.n_benign = _parse_int(_BENIGN_RE, t)
        f.clinvar_total = _parse_int(_CLINVAR_TOTAL_RE, t)
        f.n_drug_pockets = _parse_int(_DRUG_POCKETS_RE, t)
        f.n_domains_field = _parse_int(_DOMAINS_COUNT_RE, t)
        f.n_ppi_partners = _parse_int(_PPI_COUNT_RE, t)

    n_dom_chunks, dom_res = _sum_domain_residues(buckets)
    f.n_domain_chunks = n_dom_chunks
    f.domain_residues_covered = dom_res

    f.disease_chunk = _pick_disease(buckets)
    f.n_disease_entries = _count_disease_entries(f.disease_chunk)

    if f.seq_len > 0:
        f.variant_density = f.n_pathogenic_lp / f.seq_len

    return f


def _rank_desc(values: list[float]) -> list[int]:
    """Return 1-based ranks (higher value = rank 1). Ties get average ranks."""
    n = len(values)
    order = sorted(range(n), key=lambda i: -values[i])
    ranks = [0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1  # 1-based, average for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _load_env()

    print("\n" + "=" * 78)
    print("Rare-Variant Druggability Leaderboard — All 48 Indexed Genes")
    print("=" * 78)
    print(
        "\nGoal: rank every indexed gene by a chunk-derived composite of\n"
        "  (A) variant pile-up density,\n"
        "  (B) structural-coverage proxy (Druggable-pockets count),\n"
        "  (C) domain breadth (residues covered by domain chunks),\n"
        "  (D) clinical/disease track record (# disease entries in chunk).\n"
        "Every numeric column is parsed verbatim from a retrieved chunk; the\n"
        "composite is a rank-sum so it is scale-robust. No drug names, no\n"
        "MONDO/OMIM ids, no mechanism phrases are printed.\n"
    )

    from g2p_rag import G2PRetriever

    persist_dir = "d:/Users/ashenoy00000/.windsurf/g2p-rag/data/chroma"
    print(f"[setup] Initialising G2PRetriever (persist_dir={persist_dir})")
    retriever = G2PRetriever(
        persist_dir=persist_dir,
        embedding_model="all-MiniLM-L6-v2",
        collection_name="g2p_proteins",
    )
    print_index_manifest(retriever)

    # ------------------------------------------------------------------
    # 1) Fetch chunks for each gene and extract features.
    # ------------------------------------------------------------------
    print(f"\n## Retrieving + parsing chunks for {len(INDEXED_GENES)} genes")
    print("-" * 78)

    features: dict[str, GeneFeatures] = {}
    skipped: list[tuple[str, str]] = []  # (gene, reason)
    all_buckets: dict[str, dict[str, list]] = {}

    for gene in INDEXED_GENES:
        buckets = _fetch_chunks_for_gene(retriever, gene)
        all_buckets[gene] = buckets
        f = _extract_features(gene, buckets)
        # A gene must have a protein_summary chunk to participate; otherwise
        # the column values are all zero and the entry is meaningless.
        if f.ps_chunk is None or f.seq_len == 0:
            skipped.append((gene, "no protein_summary or unparseable seq_len"))
            continue
        features[gene] = f
        print(
            f"  {gene:<8}  seq_len={f.seq_len:<5}  "
            f"P+LP={f.n_pathogenic_lp:<4}  pockets={f.n_drug_pockets:<3}  "
            f"dom_res={f.domain_residues_covered:<5}  "
            f"diseases={f.n_disease_entries}"
        )

    if skipped:
        print("\n  Skipped genes (no usable protein_summary chunk retrieved):")
        for g, why in skipped:
            print(f"    {g}: {why}")

    if not features:
        print("\nFATAL: no genes had a usable protein_summary chunk. Aborting.")
        return

    # ------------------------------------------------------------------
    # 2) Compute per-column ranks and the composite rank-sum.
    # ------------------------------------------------------------------
    print("\n\n## Ranking columns (higher raw value = better rank)")
    print("-" * 78)

    genes_sorted = sorted(features.keys())
    col_A = [features[g].variant_density for g in genes_sorted]
    col_B = [float(features[g].n_drug_pockets) for g in genes_sorted]
    col_C = [float(features[g].domain_residues_covered) for g in genes_sorted]
    col_D = [float(features[g].n_disease_entries) for g in genes_sorted]

    rank_A = _rank_desc(col_A)
    rank_B = _rank_desc(col_B)
    rank_C = _rank_desc(col_C)
    rank_D = _rank_desc(col_D)

    composite = [rank_A[i] + rank_B[i] + rank_C[i] + rank_D[i]
                 for i in range(len(genes_sorted))]

    # Sort genes by composite ascending (lower = more druggable).
    leaderboard_idx = sorted(range(len(genes_sorted)), key=lambda i: composite[i])

    # ------------------------------------------------------------------
    # 3) Full leaderboard table (all 48 / all qualifying genes).
    # ------------------------------------------------------------------
    print("\n\n## Full leaderboard (all qualifying genes)")
    print("-" * 78)

    header = (
        f"| {'Rank':<4} | {'Gene':<8} | "
        f"{'VarDens (P+LP/aa)':<18} | "
        f"{'Pockets':<7} | "
        f"{'DomRes':<6} | "
        f"{'Disease#':<8} | "
        f"{'Composite':<9} |"
    )
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")

    rank_pos = 1
    for idx in leaderboard_idx:
        g = genes_sorted[idx]
        f = features[g]
        dens_str = f"{f.variant_density:.4f}"
        print(
            f"| {rank_pos:<4} | {g:<8} | "
            f"{dens_str:<18} | "
            f"{f.n_drug_pockets:<7} | "
            f"{f.domain_residues_covered:<6} | "
            f"{f.n_disease_entries:<8} | "
            f"{composite[idx]:<9.1f} |"
        )
        rank_pos += 1

    # ------------------------------------------------------------------
    # 4) Top-10 highlight panel.
    # ------------------------------------------------------------------
    print("\n\n## TOP-10 most G2P-druggable genes (lowest composite rank-sum)")
    print("-" * 78)
    top10 = leaderboard_idx[:10]
    for pos, idx in enumerate(top10, 1):
        g = genes_sorted[idx]
        f = features[g]
        print(
            f"  {pos:>2}. {g:<8}  composite={composite[idx]:.1f}  "
            f"[varDens={f.variant_density:.4f}, "
            f"pockets={f.n_drug_pockets}, "
            f"domRes={f.domain_residues_covered}, "
            f"diseases={f.n_disease_entries}]"
        )

    # ------------------------------------------------------------------
    # 5) Show-your-arithmetic: per top-10 gene, cite the actual chunks
    #    that supplied each number.
    # ------------------------------------------------------------------
    print("\n\n## Per-top-10 gene: arithmetic + chunk citations")
    print("-" * 78)
    for pos, idx in enumerate(top10, 1):
        g = genes_sorted[idx]
        f = features[g]
        buckets = all_buckets[g]
        print(f"\n  [{pos}] {g}  (composite rank-sum = {composite[idx]:.1f})")

        # --- column A: variant density ---
        # Use chunk substring "Pathogenic/Likely pathogenic:" — this string
        # appears in every protein_summary chunk that has a ClinVar block.
        ev_var = assert_supported(
            f"{g}: protein_summary lists ClinVar Pathogenic/Likely-pathogenic count",
            [f.ps_chunk],
            hints=["Pathogenic/Likely pathogenic:", "ClinVar variants"],
        )
        print("      " + str(Cited(
            f"(A) variant pile-up density = {f.n_pathogenic_lp} P+LP "
            f"variants / {f.seq_len} aa = {f.variant_density:.4f}",
            ev_var,
        )))

        # --- column B: druggable pockets ---
        if f.n_drug_pockets > 0:
            ev_pock = assert_supported(
                f"{g}: protein_summary lists Druggable pockets",
                [f.ps_chunk],
                hints=["Druggable pockets ("],
            )
            print("      " + str(Cited(
                f"(B) structural-coverage proxy = {f.n_drug_pockets} "
                f"druggable pockets enumerated",
                ev_pock,
            )))
        else:
            # The chunk literally says "Druggable pockets (0): None" — still
            # a cite-able fact, not a missing field.
            ev_pock = assert_supported(
                f"{g}: protein_summary lists Druggable pockets (0)",
                [f.ps_chunk],
                hints=["Druggable pockets (0)", "Druggable pockets"],
            )
            print("      " + str(Cited(
                "(B) structural-coverage proxy = 0 druggable pockets indexed",
                ev_pock,
            )))

        # --- column C: domain residue coverage ---
        domain_chunks = buckets.get("domain", [])
        if f.n_domain_chunks > 0 and domain_chunks:
            # Cite the first (highest-score) domain chunk.
            top_dom = max(domain_chunks, key=lambda c: c.score)
            start_str = (top_dom.residue_range.split("-")[0]
                         if top_dom.residue_range else "")
            ev_dom = assert_supported(
                f"{g}: at least one domain chunk indexed",
                [top_dom],
                hints=[f"Residues: {start_str}", "Domain"],
            )
            print("      " + str(Cited(
                f"(C) domain breadth = {f.n_domain_chunks} domain chunk(s), "
                f"merged residue coverage = {f.domain_residues_covered} aa",
                ev_dom,
            )))
        else:
            # No domain chunks — print as TEXTBOOK_CONTEXT framing.
            print("      " + str(Cited(
                "(C) domain breadth = 0 (no domain chunks indexed for this gene)",
                None,
                label="TEXTBOOK_CONTEXT",
            )))

        # --- column D: disease entry count ---
        if f.disease_chunk is not None and f.n_disease_entries > 0:
            # Find one acronym actually in the chunk text and cite on it.
            acronyms = _DISEASE_ENTRY_RE.findall(f.disease_chunk.text)
            acro_hint = f"({acronyms[0]}):" if acronyms else "DISEASE"
            ev_dis = assert_supported(
                f"{g}: disease chunk enumerates indications",
                [f.disease_chunk],
                hints=[acro_hint, "DISEASE"],
            )
            print("      " + str(Cited(
                f"(D) clinical track record = {f.n_disease_entries} disease "
                f"entries parsed from the indexed disease chunk",
                ev_dis,
            )))
        else:
            print("      " + str(Cited(
                f"(D) clinical track record = 0 (no parseable disease "
                f"entries in the indexed chunks for {g})",
                None,
                label="TEXTBOOK_CONTEXT",
            )))

    # ------------------------------------------------------------------
    # 6) Final recommendation — chunk-grounded only.
    # ------------------------------------------------------------------
    print("\n\n## Recommendation")
    print("-" * 78)
    winner_idx = leaderboard_idx[0]
    winner = genes_sorted[winner_idx]
    f = features[winner]

    print(
        f"\n  Top G2P-druggable target by composite rank-sum: {winner}\n"
        f"    composite={composite[winner_idx]:.1f}, "
        f"varDens={f.variant_density:.4f}, "
        f"pockets={f.n_drug_pockets}, "
        f"domRes={f.domain_residues_covered}, "
        f"diseases={f.n_disease_entries}"
    )

    # Final framing — explicitly tagged because it is editorial, not a
    # RAG-derivable conclusion. The script does NOT name candidate drug
    # classes, modalities, or specific indications.
    print("\n  " + str(Cited(
        "Framing: this leaderboard ranks ONLY by features the g2p-rag index "
        "actually carries. Pocket count is a stand-in for PDB+AlphaFold "
        "coverage; disease-entry count is a stand-in for # gencc indications. "
        "ChEMBL bioactivity depth and approved-drug counts are NOT in this "
        "index and are intentionally absent from the score.",
        None,
        label="TEXTBOOK_CONTEXT",
    )))

    # ------------------------------------------------------------------
    # Footer: counts derived from actual retrieval state.
    # ------------------------------------------------------------------
    n_ranked = len(features)
    chunk_types_seen = set()
    n_chunks_total = 0
    for buckets in all_buckets.values():
        for ct, lst in buckets.items():
            chunk_types_seen.add(ct)
            n_chunks_total += len(lst)
    print(
        f"\nSummary: ranked {n_ranked}/{len(INDEXED_GENES)} genes; "
        f"retrieved {n_chunks_total} chunks across "
        f"{len(chunk_types_seen)} distinct chunk types "
        f"({', '.join(sorted(chunk_types_seen))})."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
