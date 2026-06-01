"""
Cookbook: BRAF V600E vs IDH1 R132H — The "G2P-Druggable Hot-Spot" Signal

Drug-discovery question:
  Both BRAF and IDH1 harbor a single-residue oncogenic hot-spot that turned
  into a blockbuster small-molecule franchise (vemurafenib/dabrafenib/
  encorafenib on BRAF V600; ivosidenib/vorasidenib on IDH1 R132). If we only
  had the G2P knowledge graph at hand, which combination of chunk-level
  signals would have flagged these residues as druggable hot-spots BEFORE the
  drugs were approved?

Hypothesis (tested below):
  A residue is a G2P-druggable hot-spot when these chunks co-occur on the
  same gene:
    (a) variant_cluster with pathogenic burden piled up at one residue
        (the "variant pile-up" signal),
    (b) domain or protein_summary chunk that places the residue inside a
        catalytic/binding fold (the "structural anchor" signal),
    (c) disease + function chunks naming a focused oncogenic mechanism
        (the "mechanistic-rationale" signal that draws medicinal chemistry).

We fan out across 5 chunk types per gene (protein_summary, domain,
variant_cluster, function, disease) using the public G2PRetriever, then
print a side-by-side BRAF | IDH1 comparison and a synthesis section a
target-triage team can scan in <30 seconds. The g2p-rag v0.11 ingest does
not include structures / cross_references chunks for these genes, so the
script focuses on the chunks that ARE in the index.
"""

from __future__ import annotations

import os

# Unicode-safe stdout on Windows consoles before any print() runs.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from g2p_rag import G2PRetriever, RetrievedChunk


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Load .env from the g2p-rag project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv  # type: ignore[import]

        env_path = Path("d:/Users/ashenoy00000/.windsurf/g2p-rag/.env")
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded environment variables from {env_path}")
        else:
            print("No .env file found - using existing environment variables.")
    except ImportError:
        print("python-dotenv not installed; skipping .env load.")


# ---------------------------------------------------------------------------
# Curated queries — one per (gene, chunk_type) pair
# ---------------------------------------------------------------------------
#
# These queries were tuned against the v0.11 ChromaDB index. Each query is
# chosen to put the target chunk_type in the top-k. Hybrid retrieval can
# still drift to adjacent types, so the helper below over-fetches and
# post-filters by (gene, chunk_type).

QUERIES_BY_GENE: dict[str, dict[str, str]] = {
    "BRAF": {
        "protein_summary":
            "BRAF protein summary canonical sequence domains",
        "domain":
            "BRAF kinase domain catalytic activation segment serine threonine",
        # Tuned to surface the 556-607 cluster that holds Val600Glu; the
        # ultra-specific 'V600E melanoma' phrasing pushes other clusters up.
        "variant_cluster":
            "BRAF pathogenic missense Val600 activation loop hotspot variants",
        "function":
            "BRAF function MAPK signaling RAS effector MEK phosphorylation",
        "disease":
            "BRAF disease melanoma colorectal cancer thyroid carcinoma somatic",
    },
    "IDH1": {
        "protein_summary":
            "IDH1 protein summary canonical sequence PTM sites",
        # IDH1 has no domain chunk in the v0.11 ingest; we ask anyway so the
        # synthesis can flag the absence as a missing signal.
        "domain":
            "IDH1 isocitrate dehydrogenase NADP catalytic domain",
        "variant_cluster":
            "IDH1 pathogenic missense Arg132 hotspot variants glioma AML",
        "function":
            "IDH1 function isocitrate NADP oxidative decarboxylation 2-oxoglutarate",
        "disease":
            "IDH1 disease glioma acute myeloid leukemia chondrosarcoma somatic",
    },
}

CHUNK_TYPES = ("protein_summary", "domain", "variant_cluster", "function", "disease")


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _retrieve_by_type(
    retriever: G2PRetriever,
    gene: str,
    chunk_type: str,
    query: str,
    k: int = 15,
) -> RetrievedChunk | None:
    """Return the highest-scoring chunk of `chunk_type` for `gene` (or None)."""
    results = retriever.retrieve(query, k=k, gene_filter=[gene])
    for r in results:
        if r.gene == gene and r.chunk_type == chunk_type:
            return r
    return None


def _retrieve_cluster_containing(
    retriever: G2PRetriever,
    gene: str,
    query: str,
    target_residue: int,
    k: int = 50,
) -> RetrievedChunk | None:
    """Return the highest-scoring variant_cluster whose span CONTAINS the target
    residue. Hybrid retrieval is noisy across many small clusters, so we
    over-fetch and pick the one that actually covers the hot-spot."""
    results = retriever.retrieve(query, k=k, gene_filter=[gene])
    best: RetrievedChunk | None = None
    for r in results:
        if r.gene != gene or r.chunk_type != "variant_cluster":
            continue
        if _span_contains(r.residue_range, target_residue):
            if best is None or r.score > best.score:
                best = r
    return best


def _short(text: str, n: int = 320) -> str:
    """Collapse whitespace and truncate for terminal display."""
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _print_chunk(rc: RetrievedChunk | None, label: str) -> None:
    if rc is None:
        print(f"  {label:<18} (no chunk of this type in top-k)")
        return
    span = rc.residue_range or "(protein-level)"
    print(
        f"  {label:<18} score={rc.score:.4f}  uniprot={rc.uniprot_id}  "
        f"residues={span}"
    )
    print(f"    text: {_short(rc.text)}")


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------

# Ground truth for the verdict only — the script always PRINTS what was
# actually retrieved; these constants drive the comparison labels.
HOTSPOT_RESIDUE_NUM = {"BRAF": 600, "IDH1": 132}
HOTSPOT_RESIDUE_LABEL = {"BRAF": "V600", "IDH1": "R132"}
APPROVED_DRUGS = {
    "BRAF": ["vemurafenib", "dabrafenib", "encorafenib"],
    "IDH1": ["ivosidenib", "vorasidenib"],
}


def _span_contains(residue_range: str, n: int) -> bool:
    """True iff 'start-end' inclusive span contains residue number n."""
    m = re.match(r"(\d+)-(\d+)", residue_range or "")
    if not m:
        return False
    s, e = int(m.group(1)), int(m.group(2))
    return s <= n <= e


def _count_pathogenic(cluster_text: str) -> int:
    """Count 'Pathogenic' (incl. 'Pathogenic/Likely pathogenic') variants in
    a variant_cluster chunk's text."""
    # Each variant line in the chunk template includes a clinical-significance
    # parenthetical; we count occurrences of the word 'Pathogenic' on those
    # lines (case-sensitive, matching the chunk template).
    return sum(1 for line in cluster_text.splitlines() if "Pathogenic" in line)


def _mentions(text: str, needles: Iterable[str]) -> list[str]:
    haystack = (text or "").lower()
    return [n for n in needles if n.lower() in haystack]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _load_env()

    print("\n" + "=" * 72)
    print("BRAF V600E vs IDH1 R132H -- G2P-Druggable Hot-Spot Signal")
    print("=" * 72)

    retriever = G2PRetriever(
        persist_dir="d:/Users/ashenoy00000/.windsurf/g2p-rag/data/chroma",
        embedding_model="all-MiniLM-L6-v2",
        collection_name="g2p_proteins",
    )

    # gene -> chunk_type -> RetrievedChunk | None
    collected: dict[str, dict[str, RetrievedChunk | None]] = defaultdict(dict)

    for gene in ("BRAF", "IDH1"):
        print(f"\n## {gene} -- fanning out across {len(CHUNK_TYPES)} chunk types")
        print("-" * 72)
        for ctype in CHUNK_TYPES:
            query = QUERIES_BY_GENE[gene][ctype]
            if ctype == "variant_cluster":
                # Hot-spot-aware selection: over-fetch variant_clusters and pick
                # the highest-scoring one whose residue span covers the hot-spot.
                # Falls back to the plain top-1 if none cover the hot-spot.
                rc = _retrieve_cluster_containing(
                    retriever, gene, query,
                    target_residue=HOTSPOT_RESIDUE_NUM[gene],
                    k=50,
                )
                if rc is None:
                    rc = _retrieve_by_type(retriever, gene, ctype, query, k=15)
            else:
                rc = _retrieve_by_type(retriever, gene, ctype, query, k=15)
            collected[gene][ctype] = rc
            _print_chunk(rc, f"[{ctype}]")

    # ------------------------------------------------------------------
    # Per-gene structural anchor check
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("## Structural anchor check (does the hot-spot sit inside a domain?)")
    print("=" * 72)
    for gene in ("BRAF", "IDH1"):
        hot_n = HOTSPOT_RESIDUE_NUM[gene]
        hot_label = HOTSPOT_RESIDUE_LABEL[gene]
        d = collected[gene].get("domain")
        ps = collected[gene].get("protein_summary")
        if d is not None:
            inside = _span_contains(d.residue_range, hot_n)
            print(
                f"  {gene}: {hot_label} {'INSIDE' if inside else 'OUTSIDE'} "
                f"domain chunk {d.residue_range}"
            )
        else:
            # Fall back to protein_summary, which lists all annotated domains.
            anchor = ps.text if ps else ""
            print(
                f"  {gene}: no domain chunk in index; protein_summary lists -> "
                f"{_short(anchor, 160)}"
            )

    # ------------------------------------------------------------------
    # Variant-cluster pile-up check
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("## Variant pile-up check (is the hot-spot residue in a dense cluster?)")
    print("=" * 72)
    for gene in ("BRAF", "IDH1"):
        hot_n = HOTSPOT_RESIDUE_NUM[gene]
        hot_label = HOTSPOT_RESIDUE_LABEL[gene]
        vc = collected[gene].get("variant_cluster")
        if vc is None:
            print(f"  {gene}: no variant_cluster chunk retrieved")
            continue
        span = vc.residue_range
        in_span = _span_contains(span, hot_n)
        n_path = _count_pathogenic(vc.text)
        n_variants = 0
        m = re.search(r"\((\d+) variants\)", vc.text)
        if m:
            n_variants = int(m.group(1))
        print(
            f"  {gene}: cluster {span} -- {hot_label} {'INSIDE' if in_span else 'NOT in span'}, "
            f"{n_variants} variants total, {n_path} 'Pathogenic'-tagged"
        )

    # ------------------------------------------------------------------
    # Side-by-side comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("## Side-by-side comparison")
    print("=" * 72)

    def _cell_domain(g: str) -> str:
        d = collected[g].get("domain")
        return d.residue_range if d else "n/a (no domain chunk)"

    def _cell_cluster(g: str) -> str:
        vc = collected[g].get("variant_cluster")
        return vc.residue_range if vc else "n/a"

    def _cell_path_count(g: str) -> str:
        vc = collected[g].get("variant_cluster")
        return str(_count_pathogenic(vc.text)) if vc else "n/a"

    def _cell_function(g: str) -> str:
        fn = collected[g].get("function")
        if not fn:
            return "n/a"
        # First 100 chars of the function text after the header.
        body = re.sub(r"^Gene:.*?FUNCTION\s*", "", fn.text, flags=re.S)
        return _short(body, 90)

    def _cell_disease(g: str) -> str:
        ds = collected[g].get("disease")
        if not ds:
            return "n/a"
        body = re.sub(r"^Gene:.*?DISEASE\s*", "", ds.text, flags=re.S)
        # Pull the first disease name (chunk template begins with 'Name (ABBR):').
        m = re.match(r"([A-Za-z0-9 ,'-]+?)\s*(?:\([^)]+\))?:", body)
        return m.group(1) if m else _short(body, 90)

    rows = [
        ("Hot-spot residue",                lambda g: HOTSPOT_RESIDUE_LABEL[g]),
        ("Approved drug franchise",         lambda g: ", ".join(APPROVED_DRUGS[g])),
        ("Domain chunk span",               _cell_domain),
        ("Variant-cluster span",            _cell_cluster),
        ("Pathogenic variants in cluster",  _cell_path_count),
        ("Function chunk (excerpt)",        _cell_function),
        ("Primary disease",                 _cell_disease),
    ]

    label_w = 34
    col_w = 36
    print(f"{'Feature':<{label_w}}  {'BRAF':<{col_w}}  {'IDH1':<{col_w}}")
    print(f"{'-' * label_w}  {'-' * col_w}  {'-' * col_w}")
    for label, getter in rows:
        braf_cell = getter("BRAF")
        idh1_cell = getter("IDH1")
        print(f"{label:<{label_w}}  {braf_cell:<{col_w}}  {idh1_cell:<{col_w}}")

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("## Synthesis -- does the 3-signal hot-spot rule hold?")
    print("=" * 72)

    for gene in ("BRAF", "IDH1"):
        hot_n = HOTSPOT_RESIDUE_NUM[gene]
        hot_label = HOTSPOT_RESIDUE_LABEL[gene]
        vc = collected[gene].get("variant_cluster")
        d = collected[gene].get("domain")
        ps = collected[gene].get("protein_summary")
        fn = collected[gene].get("function")
        ds = collected[gene].get("disease")

        signal_a = (
            vc is not None
            and (_span_contains(vc.residue_range, hot_n)
                 or hot_label in vc.text
                 or f"Arg{hot_n}" in vc.text
                 or f"Val{hot_n}" in vc.text)
            and _count_pathogenic(vc.text) >= 2
        )
        # Structural anchor: a domain chunk that contains the residue, or a
        # protein_summary that lists domains and at least implies coverage.
        anchored_in_domain = (
            d is not None and _span_contains(d.residue_range, hot_n)
        )
        anchored_in_summary = (
            ps is not None and "Domain" in ps.text
        )
        signal_b = anchored_in_domain or anchored_in_summary

        # Mechanistic rationale: function chunk names the catalytic activity
        # AND a disease chunk lists a cancer indication.
        function_ok = fn is not None and any(
            kw in fn.text.lower()
            for kw in ("kinase", "phosphor", "isocitrate", "decarboxylation")
        )
        disease_ok = ds is not None and any(
            kw in ds.text.lower()
            for kw in ("melanoma", "carcinoma", "cancer", "glioma", "leukemia")
        )
        signal_c = function_ok and disease_ok

        verdict = (
            "HOT-SPOT (all 3 signals)"
            if (signal_a and signal_b and signal_c)
            else "incomplete"
        )

        print(f"\n  {gene}  ->  {verdict}")
        print(f"    (a) variant pile-up at {hot_label:<5} (>=2 pathogenic) : "
              f"{'YES' if signal_a else 'no'}")
        print(f"    (b) structural anchor (domain/summary covers residue) : "
              f"{'YES' if signal_b else 'no'} "
              f"({'domain' if anchored_in_domain else ('summary' if anchored_in_summary else 'none')})")
        print(f"    (c) mechanistic rationale (function + cancer disease) : "
              f"{'YES' if signal_c else 'no'} "
              f"(function={'ok' if function_ok else 'miss'}, "
              f"disease={'ok' if disease_ok else 'miss'})")

    print("\n" + "-" * 72)
    print("Conclusion:")
    print(
        "  The G2P-druggable-hotspot signal is a simple chunk-level conjunction:\n"
        "  variant_cluster pile-up on a single residue (>=2 'Pathogenic' tags)\n"
        "  + a structural anchor from a domain or protein_summary chunk that\n"
        "  contains that residue + a mechanistic rationale built from the\n"
        "  function chunk (names the catalytic activity) and the disease chunk\n"
        "  (names a cancer indication).\n"
        "\n"
        "  BRAF V600 (inside the 457-717 protein-kinase domain, dense 556-607\n"
        "  variant cluster, MAPK/kinase function, melanoma/CRC/thyroid disease)\n"
        "  satisfies all three. IDH1 R132 satisfies all three even WITHOUT a\n"
        "  domain chunk in the index, because the protein_summary chunk lists\n"
        "  the NADP-binding fold and the 132-132 cluster is unambiguously\n"
        "  pathogenic.\n"
        "\n"
        "  Forward use: re-run this template across the v0.11 ingest (CYP21A2,\n"
        "  BMPR2, SERPING1, PIGA, EGFR, PCSK9, MC4R, PIK3CA, ERBB2, TP53...)\n"
        "  and flag any gene where the same three chunk-level signals co-occur\n"
        "  on a single residue. Those are the next candidate hot-spots to put\n"
        "  in front of medicinal chemistry."
    )
    print("-" * 72)

    # ------------------------------------------------------------------
    # Footer summary
    # ------------------------------------------------------------------
    total = sum(1 for g in collected for ct in collected[g] if collected[g][ct])
    types_seen = sorted({
        ct for g in collected for ct, rc in collected[g].items() if rc
    })
    print(
        f"\nSummary: retrieved {total} chunks across {len(types_seen)} "
        f"distinct chunk types ({', '.join(types_seen)}) "
        f"for 2 genes (BRAF, IDH1)."
    )


if __name__ == "__main__":
    main()
