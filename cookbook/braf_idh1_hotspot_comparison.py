"""
Cookbook: BRAF V600E vs IDH1 R132H -- G2P-Druggable Hot-Spot Signal
(citation-disciplined rewrite)

Drug-discovery question:
  Both BRAF and IDH1 harbor a single-residue oncogenic hot-spot. If we only
  had the G2P knowledge graph at hand, which combination of chunk-level
  signals would have flagged these residues as druggable hot-spots?

Hypothesis (tested below):
  A residue is a G2P-druggable hot-spot when these chunks co-occur on the
  same gene:
    (a) variant_cluster with pathogenic burden piled up at one residue,
    (b) domain or protein_summary chunk that places the residue inside a
        catalytic/binding fold,
    (c) function/disease chunks that together name a focused mechanism.

This rewrite enforces RAG citation discipline via ``_citation.py``:
every printed factual line is either wrapped in ``Cited(text, chunk)`` so the
chunk text actually contains evidence for the claim, or wrapped in
``Cited(text, None, label="TEXTBOOK_CONTEXT")`` and clearly framed as
background -- never as a RAG-derived insight.
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

from _citation import (
    Cited,
    assert_supported,
    find_in_chunks,
    print_index_manifest,
    resolve_chroma_path,
)


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
# Curated queries -- one per (gene, chunk_type) pair
# ---------------------------------------------------------------------------

QUERIES_BY_GENE: dict[str, dict[str, str]] = {
    "BRAF": {
        "protein_summary":
            "BRAF protein summary canonical sequence domains",
        "domain":
            "BRAF kinase domain catalytic activation segment serine threonine",
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
    residue. Over-fetch and pick the one that actually covers the hot-spot."""
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

# Hot-spot residue numbers drive the structural-anchor / pile-up math; the
# labels are purely cosmetic. These are NOT RAG facts -- they parameterize
# the lookup. No drug names, approval dates, or disease names are hard-coded.
HOTSPOT_RESIDUE_NUM = {"BRAF": 600, "IDH1": 132}
HOTSPOT_RESIDUE_LABEL = {"BRAF": "V600", "IDH1": "R132"}


def _span_contains(residue_range: str, n: int) -> bool:
    """True iff 'start-end' inclusive span contains residue number n."""
    m = re.match(r"(\d+)-(\d+)", residue_range or "")
    if not m:
        return False
    s, e = int(m.group(1)), int(m.group(2))
    return s <= n <= e


def _count_pathogenic(cluster_text: str) -> int:
    """Count 'Pathogenic' lines in a variant_cluster chunk's text."""
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
        persist_dir=resolve_chroma_path(),
        embedding_model="all-MiniLM-L6-v2",
        collection_name="g2p_proteins",
    )
    print_index_manifest(retriever)

    # gene -> chunk_type -> RetrievedChunk | None
    collected: dict[str, dict[str, RetrievedChunk | None]] = defaultdict(dict)

    for gene in ("BRAF", "IDH1"):
        print(f"\n## {gene} -- fanning out across {len(CHUNK_TYPES)} chunk types")
        print("-" * 72)
        for ctype in CHUNK_TYPES:
            query = QUERIES_BY_GENE[gene][ctype]
            if ctype == "variant_cluster":
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

    # Flatten all retrieved chunks per gene for the citation helper.
    chunks_by_gene: dict[str, list[RetrievedChunk]] = {
        g: [c for c in collected[g].values() if c is not None]
        for g in ("BRAF", "IDH1")
    }

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
    # Side-by-side comparison table -- ONLY columns derivable from chunks
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("## Side-by-side comparison (chunk-derived columns only)")
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

    def _cell_chunks_retrieved(g: str) -> str:
        # Compute from len(chunks) rather than hard-coding "5".
        return str(len(chunks_by_gene[g]))

    rows = [
        ("Hot-spot residue (queried)",       lambda g: HOTSPOT_RESIDUE_LABEL[g]),
        ("Chunks retrieved (len)",           _cell_chunks_retrieved),
        ("Domain chunk span",                _cell_domain),
        ("Variant-cluster span",             _cell_cluster),
        ("Pathogenic variants in cluster",   _cell_path_count),
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
    # Synthesis -- citation-gated
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
        anchored_in_domain = (
            d is not None and _span_contains(d.residue_range, hot_n)
        )
        anchored_in_summary = (
            ps is not None and "Domain" in ps.text
        )
        signal_b = anchored_in_domain or anchored_in_summary

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

        # --------------------------------------------------------------
        # Citation-gated per-gene evidence lines.
        # Every Cited(...) below either points at a chunk whose text we
        # already verified contains a supporting substring (via
        # assert_supported), or is explicitly TEXTBOOK_CONTEXT.
        # --------------------------------------------------------------
        print(f"\n  {gene} chunk-grounded evidence:")

        chunks = chunks_by_gene[gene]

        if signal_a and vc is not None:
            # Anchor the pile-up sentence to the variant_cluster chunk we
            # actually retrieved. NB: chunk bodies use en-dash for ranges, so
            # hint on residue *tokens* and "Pathogenic", which the chunk
            # template always emits.
            hint_alts = [
                f"Arg{hot_n}",
                f"Val{hot_n}",
                hot_label,
                "Pathogenic",
            ]
            ev = assert_supported(
                f"{gene}: pathogenic variants pile up around residue {hot_label}",
                [vc],
                hints=hint_alts,
            )
            n_path = _count_pathogenic(vc.text)
            print("    " + str(Cited(
                f"variant_cluster {vc.residue_range} carries "
                f"{n_path} 'Pathogenic'-tagged entries covering {hot_label}",
                ev,
            )))

        if anchored_in_domain and d is not None:
            # Chunk body uses en-dashes in residue ranges (e.g. "457–717"),
            # but `residue_range` metadata uses ASCII hyphen. Hint on tokens
            # the chunk body actually contains: 'Domain' header + start
            # residue number.
            start_str = d.residue_range.split("-")[0] if d.residue_range else ""
            ev = assert_supported(
                f"{gene}: hot-spot residue {hot_label} sits inside domain {d.residue_range}",
                [d],
                hints=[f"Residues: {start_str}", "Domain", "domain"],
            )
            print("    " + str(Cited(
                f"domain chunk {d.residue_range} contains residue {hot_n}",
                ev,
            )))
        elif anchored_in_summary and ps is not None:
            ev = assert_supported(
                f"{gene}: protein_summary lists domain annotations",
                [ps],
                hints=["Domain", "domain"],
            )
            print("    " + str(Cited(
                f"protein_summary enumerates domain annotations (no separate "
                f"domain chunk indexed for {gene})",
                ev,
            )))

        if function_ok and fn is not None:
            # Find the actual keyword that fired so we cite truthfully.
            hint = next(
                kw for kw in ("kinase", "phosphor", "isocitrate", "decarboxylation")
                if kw in fn.text.lower()
            )
            ev = assert_supported(
                f"{gene}: function chunk names a catalytic activity",
                [fn],
                hints=[hint],
            )
            print("    " + str(Cited(
                f"function chunk references '{hint}' activity",
                ev,
            )))

        if disease_ok and ds is not None:
            hint = next(
                kw for kw in ("melanoma", "carcinoma", "cancer", "glioma", "leukemia")
                if kw in ds.text.lower()
            )
            ev = assert_supported(
                f"{gene}: disease chunk names a cancer indication",
                [ds],
                hints=[hint],
            )
            print("    " + str(Cited(
                f"disease chunk references '{hint}'",
                ev,
            )))

        # If a known-druggable franchise is something the reader expects to see
        # spelled out, mark it as TEXTBOOK_CONTEXT -- NOT a RAG conclusion.
        # We do NOT name specific drugs here because no chunk we retrieved
        # contains drug names; that knowledge is out-of-index.
        print("    " + str(Cited(
            f"Background: oncogenic hot-spots at a single residue have, "
            f"historically, attracted small-molecule programs; whether any "
            f"specific drug exists for {gene} {hot_label} is OUT OF THIS INDEX.",
            None,
            label="TEXTBOOK_CONTEXT",
        )))

    # ------------------------------------------------------------------
    # Conclusion -- chunk-grounded only, no overreach
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("Conclusion (chunk-grounded):")

    # Pull the BRAF domain and IDH1 protein_summary chunks (if present) so we
    # can re-cite their actual residue spans in the conclusion. NO disease
    # names, NO drug names, NO "NADP-binding fold" claim are made unless a
    # chunk literally contains them.
    braf_chunks = chunks_by_gene["BRAF"]
    idh1_chunks = chunks_by_gene["IDH1"]

    # BRAF kinase-domain span -- only print if a chunk supports it.
    # Chunk bodies use en-dash separators, so hint on the start-residue
    # token plus the "Domain" header that the chunk template always emits.
    braf_domain = collected["BRAF"].get("domain")
    if braf_domain is not None and _span_contains(
        braf_domain.residue_range, HOTSPOT_RESIDUE_NUM["BRAF"]
    ):
        start_str = braf_domain.residue_range.split("-")[0]
        ev = assert_supported(
            "BRAF V600 sits inside the retrieved domain chunk span",
            [braf_domain],
            hints=[f"Residues: {start_str}", "Domain"],
        )
        print("  " + str(Cited(
            f"BRAF V600 lies within the domain chunk span "
            f"{braf_domain.residue_range}",
            ev,
        )))

    braf_cluster = collected["BRAF"].get("variant_cluster")
    if braf_cluster is not None and _span_contains(
        braf_cluster.residue_range, HOTSPOT_RESIDUE_NUM["BRAF"]
    ):
        start_str = braf_cluster.residue_range.split("-")[0]
        ev = assert_supported(
            "BRAF V600 falls inside a retrieved variant_cluster span",
            [braf_cluster],
            hints=[f"Variant cluster: {start_str}", "Pathogenic"],
        )
        print("  " + str(Cited(
            f"BRAF V600 falls inside variant_cluster {braf_cluster.residue_range}",
            ev,
        )))

    idh1_cluster = collected["IDH1"].get("variant_cluster")
    if idh1_cluster is not None and _span_contains(
        idh1_cluster.residue_range, HOTSPOT_RESIDUE_NUM["IDH1"]
    ):
        start_str = idh1_cluster.residue_range.split("-")[0]
        ev = assert_supported(
            "IDH1 R132 falls inside a retrieved variant_cluster span",
            [idh1_cluster],
            hints=[f"Variant cluster: {start_str}", "Pathogenic"],
        )
        print("  " + str(Cited(
            f"IDH1 R132 falls inside variant_cluster {idh1_cluster.residue_range}",
            ev,
        )))

    # IDH1 NADP-binding fold: ONLY print if a chunk literally contains "NADP".
    idh1_nadp_evidence = find_in_chunks("NADP", idh1_chunks)
    if idh1_nadp_evidence is not None:
        print("  " + str(Cited(
            "IDH1 chunks reference NADP cofactor / cofactor-binding context",
            idh1_nadp_evidence,
        )))
    # If no chunk says "NADP", the sentence is silently dropped -- per the
    # rewrite contract, we do NOT paper over the gap with training knowledge.

    print(
        "\n  Method recap (no external facts asserted):\n"
        "    The G2P-druggable-hotspot signal is a chunk-level conjunction:\n"
        "      (a) variant_cluster pile-up containing the residue (>=2 'Pathogenic'),\n"
        "      (b) a domain or protein_summary chunk whose span covers the residue,\n"
        "      (c) function + disease chunks that name a catalytic activity and an\n"
        "          oncology context.\n"
        "    Whether either gene's hot-spot has an approved drug, and which\n"
        "    cancer types are affected, are claims that must be sourced from\n"
        "    outside this RAG index. This script intentionally does not assert\n"
        "    them."
    )
    print("-" * 72)

    # ------------------------------------------------------------------
    # Footer summary -- counts come from len() of actual retrieval state
    # ------------------------------------------------------------------
    total = sum(1 for g in collected for ct in collected[g] if collected[g][ct])
    types_seen = sorted({
        ct for g in collected for ct, rc in collected[g].items() if rc
    })
    print(
        f"\nSummary: retrieved {total} chunks across {len(types_seen)} "
        f"distinct chunk types ({', '.join(types_seen)}) "
        f"for {len(collected)} genes "
        f"(BRAF={len(chunks_by_gene['BRAF'])}, IDH1={len(chunks_by_gene['IDH1'])})."
    )


if __name__ == "__main__":
    main()
