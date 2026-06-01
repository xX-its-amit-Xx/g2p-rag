"""
Cookbook: PDB-vs-AlphaFold-vs-Variant Coverage Map — chunk-grounded.

Drug-discovery question
-----------------------
For each gene, overlay three data layers:
  (a) residue ranges covered by experimental PDB structures
      (would come from ``structures`` chunks),
  (b) residue ranges covered by AlphaFold predictions with confidence
      pLDDT >= 80  (would come from ``cross_references`` chunks),
  (c) residue ranges hit by pathogenic / likely-pathogenic variant
      clusters  (from ``variant_cluster`` chunks).

Then identify *structural gaps* — regions with HIGH pathogenic variant
density BUT neither PDB nor high-confidence AlphaFold coverage. Those are
high-value crystallography targets for drug discovery.

Honest-discipline note
----------------------
The current g2p-rag index does NOT contain ``structures`` or
``cross_references`` chunks (only ``variant_cluster``, ``domain``,
``protein_summary``, ``function``, ``subunit``, ``disease``, ``pathway``
were ingested). The discipline says: *if a fact is not in a chunk, it
does not get printed.* So this cookbook does NOT invent PDB IDs or
pLDDT scores. Instead it:

  - Reports, per gene, that the ``structures`` and ``cross_references``
    chunk types are absent (this absence is itself a chunk-supported
    factual claim — supported by the retrieval call returning none of
    those types).
  - Uses the ``domain`` chunks (curated residue ranges that mark
    structurally-characterized regions in the index) as the available
    proxy for "structural coverage".
  - Computes, per ``variant_cluster``, whether it overlaps any indexed
    domain range. Pathogenic burden inside no-domain stretches is the
    *structural gap* signal.
  - Ranks the 10 requested genes by gap burden and emits an
    actionable crystallography priority list.

Every printed factual claim is either ``Cited(text, chunk)`` (grounded
in a retrieved chunk whose text contains a hint substring, gated by
``assert_supported``) or ``Cited(text, None, label="TEXTBOOK_CONTEXT")``
for framing sentences.
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Unicode-safe stdout for the table glyphs used below on Windows consoles.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Allow running directly from the cookbook/ directory without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _citation import Cited, assert_supported, find_in_chunks  # noqa: E402


# ---------------------------------------------------------------------------
# Targets (per the cookbook spec). BRCA1 is intentionally kept in the list
# so the script can demonstrate its handling of "gene absent from index" as
# a chunk-supported negative fact rather than silently dropping it.
# ---------------------------------------------------------------------------

TARGET_GENES: list[str] = [
    "BRCA1", "TP53", "EGFR", "CFTR", "DMD",
    "HBB", "PCSK9", "APP", "IDH1", "BRAF",
]


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Load .env from the project root if python-dotenv is available."""
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
# Parsing helpers — all operate on chunk text or chunk.residue_range only.
# ---------------------------------------------------------------------------

_RES_RANGE_RE = re.compile(r"(\d+)\s*[-–]\s*(\d+)")
# Domain listing inside a protein_summary chunk looks like:
#   "Domains (2): Inhibitor I9 (77-149), Peptidase S8 (155-461)"
# Use a non-greedy label + 'open paren digits dash digits close paren' tail.
_DOMAIN_LINE_RE = re.compile(r"Domains\s*\(\d+\)\s*:\s*(.+)", re.IGNORECASE)
_DOMAIN_ITEM_RE = re.compile(
    r"([^,()]+?)\s*\((\d+)\s*[-–]\s*(\d+)\)"
)
_CANON_LEN_RE = re.compile(r"Canonical sequence length:\s*(\d+)\s*aa", re.IGNORECASE)
_VAR_CLUSTER_HEADER_RE = re.compile(
    r"Variant cluster:\s*(\d+)\s*[-–]\s*(\d+)\s*\((\d+)\s*variants?\)",
    re.IGNORECASE,
)


def _parse_range(residue_range: str | None) -> tuple[int, int] | None:
    """Parse a 'start-end' (or 'start-end' with en-dash) string into (int,int)."""
    if not residue_range:
        return None
    m = _RES_RANGE_RE.search(residue_range)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True iff inclusive residue ranges (a_start,a_end) and (b_start,b_end) overlap."""
    return not (a[1] < b[0] or b[1] < a[0])


def _count_pathogenic_in_text(chunk_text: str) -> tuple[int, int, int, int]:
    """Return (P, LP, VUS, B/LB) counts parsed from a variant_cluster chunk text.

    We count parenthetical clinical-significance literals exactly as the
    chunk writes them (so this stays a pure read of the chunk text).
    """
    text_lower = chunk_text.lower()
    n_path = text_lower.count("(pathogenic)")
    n_likely = text_lower.count("(likely pathogenic)")
    n_vus = text_lower.count("(uncertain significance)")
    # Benign + likely benign + conflicting — used only for total denominator
    # context, never to derive a recommendation.
    n_other = (
        text_lower.count("(benign)")
        + text_lower.count("(likely benign)")
        + text_lower.count("(conflicting classifications of pathogenicity)")
    )
    return n_path, n_likely, n_vus, n_other


def _parse_canonical_length(summary_text: str) -> int | None:
    """Extract canonical sequence length (aa) from a protein_summary chunk."""
    m = _CANON_LEN_RE.search(summary_text)
    return int(m.group(1)) if m else None


def _parse_domains_from_summary(
    summary_text: str,
) -> list[tuple[str, tuple[int, int]]]:
    """Parse the 'Domains (N): label1 (s-e), label2 (s-e), ...' line.

    Returns a list of (label, (start, end)). Empty list if the summary
    has 'Domains (0): None'.
    """
    out: list[tuple[str, tuple[int, int]]] = []
    line_match = _DOMAIN_LINE_RE.search(summary_text)
    if not line_match:
        return out
    listing = line_match.group(1)
    if listing.strip().lower() == "none":
        return out
    for m in _DOMAIN_ITEM_RE.finditer(listing):
        label = m.group(1).strip().rstrip(",").strip()
        if not label:
            continue
        s, e = int(m.group(2)), int(m.group(3))
        out.append((label, (s, e)))
    return out


# ---------------------------------------------------------------------------
# Retrieval bundle for a single gene
# ---------------------------------------------------------------------------

class GeneBundle:
    """All chunks (per type) the retriever returned for one gene.

    Attributes
    ----------
    gene:
        Gene symbol (e.g. ``"PCSK9"``).
    chunks_by_type:
        ``chunk_type -> list[RetrievedChunk]`` for every type touched.
    """

    __slots__ = ("gene", "chunks_by_type")

    def __init__(self, gene: str) -> None:
        self.gene = gene
        self.chunks_by_type: dict[str, list] = defaultdict(list)

    def add(self, chunks: list) -> None:
        for c in chunks:
            # Only retain chunks for THIS gene — retriever may return
            # neighbours from the same vector space.
            if c.gene != self.gene:
                continue
            # Deduplicate within a chunk_type by (residue_range, text head).
            key = (c.residue_range, c.text[:80])
            existing = self.chunks_by_type[c.chunk_type]
            if any((x.residue_range, x.text[:80]) == key for x in existing):
                continue
            existing.append(c)

    @property
    def all_chunks(self) -> list:
        return [c for lst in self.chunks_by_type.values() for c in lst]


def _retrieve_gene_bundle(retriever, gene: str) -> GeneBundle:
    """Drive the four chunk types we need into a single bundle."""
    bundle = GeneBundle(gene)

    # protein_summary — domain coordinates + canonical length live here.
    bundle.add(
        retriever.retrieve(
            f"{gene} protein summary canonical sequence length domains PTM",
            k=4,
            gene_filter=[gene],
        )
    )

    # domain — explicit per-domain residue ranges.
    bundle.add(
        retriever.retrieve(
            f"{gene} domain residues description sequence span",
            k=8,
            gene_filter=[gene],
        )
    )

    # variant_cluster — the pathogenic-burden layer.
    bundle.add(
        retriever.retrieve(
            f"{gene} variant cluster pathogenic likely pathogenic uncertain significance",
            k=12,
            gene_filter=[gene],
        )
    )

    # cross_references / structures — these chunk types do not exist in
    # v0.11 ingest. The retrieve calls below are issued so the *absence*
    # of returned chunks of those types is itself a fact derived from
    # the index, not from external knowledge.
    bundle.add(
        retriever.retrieve(
            f"{gene} PDB experimental structure crystal X-ray NMR resolution chain",
            k=4,
            gene_filter=[gene],
        )
    )
    bundle.add(
        retriever.retrieve(
            f"{gene} AlphaFold model cross-references ChEMBL DrugBank OMIM",
            k=4,
            gene_filter=[gene],
        )
    )

    return bundle


# ---------------------------------------------------------------------------
# Coverage map computation — purely arithmetic over chunk contents.
# ---------------------------------------------------------------------------

def _domain_ranges(bundle: GeneBundle) -> list[tuple[str, tuple[int, int]]]:
    """Collect domain (label, range) tuples from BOTH domain chunks AND
    the 'Domains (...)' line in the protein_summary chunk."""
    out: list[tuple[str, tuple[int, int]]] = []

    # From protein_summary
    for ps in bundle.chunks_by_type.get("protein_summary", []):
        for lbl, rng in _parse_domains_from_summary(ps.text):
            out.append((lbl, rng))

    # From standalone domain chunks
    for dc in bundle.chunks_by_type.get("domain", []):
        rng = _parse_range(dc.residue_range)
        if rng is None:
            continue
        # Pull a label from the chunk body if present.
        label = "(unnamed)"
        for line in dc.text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("domain:"):
                label = stripped.split(":", 1)[1].strip()
                break
        out.append((label, rng))

    # De-duplicate by (label, range)
    seen: set = set()
    deduped: list[tuple[str, tuple[int, int]]] = []
    for lbl, rng in out:
        key = (lbl.lower(), rng)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((lbl, rng))
    return deduped


def _variant_cluster_rows(
    bundle: GeneBundle,
) -> list[dict]:
    """One row per variant_cluster chunk, with parsed counts and range."""
    rows: list[dict] = []
    for vc in bundle.chunks_by_type.get("variant_cluster", []):
        rng = _parse_range(vc.residue_range)
        if rng is None:
            # Fall back to the in-text header line.
            m = _VAR_CLUSTER_HEADER_RE.search(vc.text)
            if m:
                rng = (int(m.group(1)), int(m.group(2)))
        if rng is None:
            continue
        p, lp, vus, other = _count_pathogenic_in_text(vc.text)
        rows.append({
            "range": rng,
            "p": p,
            "lp": lp,
            "vus": vus,
            "other": other,
            "chunk": vc,
        })
    # Sort by range start for stable, readable tables.
    rows.sort(key=lambda r: r["range"][0])
    return rows


def _compute_gene_coverage(bundle: GeneBundle) -> dict:
    """Build the per-gene coverage report.

    Returns a dict with keys:
        ``gene``, ``canonical_length``, ``domains``, ``variant_rows``,
        ``has_structures_chunk``, ``has_xrefs_chunk``, ``summary_chunk``.
    """
    summary_chunk = None
    summaries = bundle.chunks_by_type.get("protein_summary", [])
    if summaries:
        summary_chunk = summaries[0]

    canon_len: int | None = None
    if summary_chunk is not None:
        canon_len = _parse_canonical_length(summary_chunk.text)

    domains = _domain_ranges(bundle)
    var_rows = _variant_cluster_rows(bundle)
    has_struct = bool(bundle.chunks_by_type.get("structures"))
    has_xref = bool(bundle.chunks_by_type.get("cross_references"))

    return {
        "gene": bundle.gene,
        "canonical_length": canon_len,
        "domains": domains,
        "variant_rows": var_rows,
        "has_structures_chunk": has_struct,
        "has_xrefs_chunk": has_xref,
        "summary_chunk": summary_chunk,
    }


def _classify_cluster(
    cluster_range: tuple[int, int],
    domains: list[tuple[str, tuple[int, int]]],
) -> tuple[bool, list[str]]:
    """Return (covered_by_any_domain, list_of_overlapping_domain_labels)."""
    labels: list[str] = []
    for lbl, drng in domains:
        if _overlap(cluster_range, drng):
            labels.append(f"{lbl} [{drng[0]}-{drng[1]}]")
    return (bool(labels), labels)


def _gap_score(report: dict) -> tuple[int, int, int]:
    """Compute a crystallography-priority score for one gene.

    Returns ``(gap_burden, total_pathogenic, total_clusters)`` where
    ``gap_burden`` is the sum of (P + LP) counts across variant_cluster
    rows that have NO overlapping indexed domain. Higher gap_burden
    means more pathogenic mass falling outside any structurally-curated
    region in the index — the crystallography target.
    """
    if not report["variant_rows"]:
        return (0, 0, 0)
    gap_burden = 0
    total_path = 0
    for row in report["variant_rows"]:
        covered, _labels = _classify_cluster(row["range"], report["domains"])
        burden = row["p"] + row["lp"]
        total_path += burden
        if not covered:
            gap_burden += burden
    return (gap_burden, total_path, len(report["variant_rows"]))


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_bundle_overview(bundle: GeneBundle) -> None:
    print(f"\n  {bundle.gene}: chunk-type tally")
    types = sorted(bundle.chunks_by_type.keys())
    if not types:
        print("    (no chunks retrieved for this gene)")
        return
    for t in types:
        print(f"    {t:<18}  n={len(bundle.chunks_by_type[t])}")


def _print_coverage_table(reports: list[dict]) -> None:
    """The deliverable table the cookbook is asked for."""
    print(
        "\n| Gene | Cluster range | Path/LikelyPath/VUS | "
        "Overlapping domain coverage (indexed) | PDB chunk? | "
        "AlphaFold chunk? | Gap | Crystallography priority |"
    )
    print(
        "|------|---------------|---------------------|"
        "---------------------------------------|------------|"
        "------------------|-----|-------------------------|"
    )
    for rep in reports:
        gene = rep["gene"]
        pdb_cell = "yes" if rep["has_structures_chunk"] else "NO (no structures chunk in index)"
        af_cell = "yes" if rep["has_xrefs_chunk"] else "NO (no cross_references chunk in index)"
        if not rep["variant_rows"]:
            print(
                f"| {gene} | (no variant_cluster chunks) | - | - | "
                f"{pdb_cell} | {af_cell} | - | - |"
            )
            continue
        for row in rep["variant_rows"]:
            rng = row["range"]
            covered, labels = _classify_cluster(rng, rep["domains"])
            burden = row["p"] + row["lp"]
            dom_cell = ", ".join(labels) if labels else "(none)"
            gap = "no" if covered else ("HIGH" if burden >= 1 else "low")
            if not covered and burden >= 1:
                prio = "crystallography candidate"
            elif not covered:
                prio = "watch (VUS only)"
            else:
                prio = "covered by indexed domain"
            print(
                f"| {gene} | {rng[0]}-{rng[1]} | "
                f"{row['p']}/{row['lp']}/{row['vus']} | "
                f"{dom_cell} | {pdb_cell} | {af_cell} | {gap} | {prio} |"
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the coverage-map cookbook end-to-end."""
    _load_env()

    print("\n" + "=" * 78)
    print("PDB-vs-AlphaFold-vs-Variant Coverage Map (citation-disciplined)")
    print("=" * 78)
    print(
        "\nQuestion: overlay PDB / AlphaFold (pLDDT>=80) / pathogenic-variant\n"
        "residue ranges per gene, surface regions with HIGH pathogenic burden\n"
        "but NO structural coverage, recommend crystallography priorities.\n"
        "\nHonesty note (printed as the script's first chunk-supported claim\n"
        "below): the v0.11 index has no 'structures' or 'cross_references'\n"
        "chunks; we therefore use the 'Domains (...)' listing in protein_summary\n"
        "plus the standalone 'domain' chunks as the indexed proxy for\n"
        "structurally-characterized residue ranges.\n"
    )

    # ------------------------------------------------------------------
    # 1. Load the retriever
    # ------------------------------------------------------------------
    from g2p_rag import G2PRetriever

    persist_dir = "d:/Users/ashenoy00000/.windsurf/g2p-rag/data/chroma"
    print(f"[setup] Initialising G2PRetriever  (persist_dir={persist_dir})")
    retriever = G2PRetriever(
        persist_dir=persist_dir,
        embedding_model="all-MiniLM-L6-v2",
        collection_name="g2p_proteins",
    )

    # ------------------------------------------------------------------
    # 2. Pull bundles for every target gene
    # ------------------------------------------------------------------
    print("\n## Retrieval (per-gene chunk-type tallies)")
    print("-" * 78)
    bundles: dict[str, GeneBundle] = {}
    for gene in TARGET_GENES:
        bundles[gene] = _retrieve_gene_bundle(retriever, gene)
        _print_bundle_overview(bundles[gene])

    # ------------------------------------------------------------------
    # 3. Compute coverage report per gene
    # ------------------------------------------------------------------
    reports: list[dict] = []
    for gene in TARGET_GENES:
        reports.append(_compute_gene_coverage(bundles[gene]))

    # ------------------------------------------------------------------
    # 4. Print arithmetic per gene
    # ------------------------------------------------------------------
    print("\n\n## Per-gene arithmetic (parsed from chunk text)")
    print("-" * 78)
    for rep in reports:
        gene = rep["gene"]
        cl = rep["canonical_length"]
        n_dom = len(rep["domains"])
        n_var = len(rep["variant_rows"])
        gap_burden, total_path, _ = _gap_score(rep)
        print(
            f"  {gene:<6}  canonical_len={cl if cl is not None else 'unknown'}  "
            f"domains_indexed={n_dom}  "
            f"variant_clusters={n_var}  "
            f"sum(P+LP)={total_path}  "
            f"gap_burden(P+LP outside any indexed domain)={gap_burden}"
        )

    # ------------------------------------------------------------------
    # 5. The deliverable table
    # ------------------------------------------------------------------
    print("\n\n## Coverage map table")
    print("-" * 78)
    _print_coverage_table(reports)

    # ------------------------------------------------------------------
    # 6. Citation-disciplined synthesis
    # ------------------------------------------------------------------
    print("\n\n## Synthesis (citation-disciplined)")
    print("-" * 78)

    claims: list[Cited] = []

    # --- (0) framing on chunk-type availability ---
    # This is meta-evidence: the absence of structures/cross_references
    # chunks is what forces the domain-proxy. We attest it from a chunk
    # we DO have (any protein_summary) that proves the index is alive
    # for these genes — the negative inference (no structures chunk) is
    # then a counted-zero from the bundles, which is purely script
    # arithmetic, not a TEXTBOOK claim.
    pooled_summary_chunks: list = []
    for rep in reports:
        if rep["summary_chunk"] is not None:
            pooled_summary_chunks.append(rep["summary_chunk"])
    if pooled_summary_chunks:
        anchor = assert_supported(
            "index has at least one protein_summary chunk to anchor coordinates",
            pooled_summary_chunks,
            hints=["Canonical sequence length", "Protein summary"],
        )
        claims.append(
            Cited(
                "0. protein_summary chunks anchor the residue coordinate system "
                "for every gene listed below; the index's 'Domains (...)' line "
                "in those summaries is the structural proxy used in lieu of "
                "PDB / AlphaFold chunks (which are absent from this index).",
                anchor,
            )
        )

    # --- (1..N) per-gene claims, one Cited per gene, plus a recommendation ---
    ranked: list[tuple[int, int, int, dict]] = []
    for rep in reports:
        gap_burden, total_path, n_clusters = _gap_score(rep)
        ranked.append((gap_burden, total_path, n_clusters, rep))
    ranked.sort(key=lambda t: (-t[0], -t[1], t[3]["gene"]))

    for gap_burden, total_path, n_clusters, rep in ranked:
        gene = rep["gene"]
        n_dom = len(rep["domains"])

        # 1a. Anchor each gene-level claim to a real retrieved chunk.
        anchor_chunks = bundles[gene].all_chunks
        if not anchor_chunks:
            # Gene absent from the index: that is itself a fact derived
            # from the retrieval call returning nothing for this gene,
            # which is script arithmetic — emit as TEXTBOOK_CONTEXT to
            # avoid faking a chunk citation.
            claims.append(
                Cited(
                    f"{gene}: retriever returned 0 chunks (gene not present in v0.11 index).",
                    source=None,
                    label="TEXTBOOK_CONTEXT",
                )
            )
            continue

        # Prefer a variant_cluster chunk as the citation for the gap claim
        # (its text is what the burden numbers come from). Fall back to
        # protein_summary, then any chunk.
        cite_chunk = None
        for vc in bundles[gene].chunks_by_type.get("variant_cluster", []):
            cite_chunk = vc
            break
        if cite_chunk is None:
            cite_chunk = anchor_chunks[0]

        # Verify the citation actually contains substring evidence for
        # what we are about to assert.
        assert_supported(
            f"{gene} has indexed variant or summary content",
            [cite_chunk],
            hints=[
                f"Gene: {gene}",
                gene,
                "Variant cluster",
                "Protein summary",
                "Canonical sequence length",
            ],
        )

        if total_path == 0 and n_clusters == 0:
            verdict = (
                f"{gene}: variant_cluster chunks indexed = 0; cannot compute "
                "gap; not a crystallography priority from this index."
            )
        elif gap_burden == 0:
            verdict = (
                f"{gene}: variant_clusters={n_clusters}, sum(P+LP)={total_path}, "
                f"gap_burden=0 — every pathogenic-bearing cluster overlaps at "
                f"least one of {n_dom} indexed domain(s). Crystallography "
                "priority: LOW (existing structural annotation already covers "
                "the pathogenic burden in the index)."
            )
        else:
            # Find the worst gap row to name explicitly.
            worst = None
            for row in rep["variant_rows"]:
                covered, _ = _classify_cluster(row["range"], rep["domains"])
                if covered:
                    continue
                burden = row["p"] + row["lp"]
                if burden <= 0:
                    continue
                if worst is None or burden > (worst["p"] + worst["lp"]):
                    worst = row
            wrng = worst["range"] if worst else (0, 0)
            wburden = (worst["p"] + worst["lp"]) if worst else 0
            verdict = (
                f"{gene}: variant_clusters={n_clusters}, sum(P+LP)={total_path}, "
                f"gap_burden={gap_burden} (P+LP outside any of {n_dom} indexed "
                f"domain(s)). Worst un-covered cluster: residues "
                f"{wrng[0]}-{wrng[1]} with P+LP={wburden}. "
                "Crystallography priority: HIGH — recommend experimental "
                "structure or AlphaFold pLDDT>=80 refinement spanning this "
                "range; the index carries no structures/cross_references "
                "chunk for this gene to rule the gap out."
            )
        claims.append(Cited(verdict, cite_chunk))

    # --- closing framing ---
    claims.append(
        Cited(
            "Framing: 'PDB coverage' and 'AlphaFold pLDDT' columns are reported "
            "as the literal presence/absence of the corresponding chunk types in "
            "this index (structures, cross_references), not as a claim about the "
            "underlying public databases. A 'NO' cell means the index has no "
            "such chunk to consult — the question is left open to wet-lab follow-up.",
            source=None,
            label="TEXTBOOK_CONTEXT",
        )
    )

    for c in claims:
        print(str(c))

    # ------------------------------------------------------------------
    # 7. Top-3 ranked crystallography targets (the actionable output)
    # ------------------------------------------------------------------
    print("\n\n## Top crystallography priorities (ranked by gap_burden)")
    print("-" * 78)
    top = [t for t in ranked if t[0] > 0][:3]
    if not top:
        print("  No gene in this run shows pathogenic burden outside indexed domains.")
    else:
        for i, (gap_burden, total_path, n_clusters, rep) in enumerate(top, 1):
            gene = rep["gene"]
            cl = rep["canonical_length"]
            print(
                f"  {i}. {gene} (canonical_len={cl}, gap_burden={gap_burden} of "
                f"sum(P+LP)={total_path} across {n_clusters} clusters; "
                f"indexed_domains={len(rep['domains'])}). "
                "Recommended action: prioritise PDB campaign or AlphaFold "
                "pLDDT-refinement over the un-covered cluster range listed in "
                "the per-gene verdict above."
            )

    print("\n" + "=" * 78)
    print("End of cookbook.")
    print("=" * 78)


if __name__ == "__main__":
    main()
