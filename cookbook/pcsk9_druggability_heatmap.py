"""
Cookbook: PCSK9 Druggability Heatmap — chunk-grounded synthesis

Drug-discovery question:
    Where in PCSK9 do ClinVar-curated variants cluster, what protein-level
    features (domains, oligomerization, function) overlap those clusters,
    and what does the indexed evidence say about therapeutic engagement?
    This cookbook composes g2p-rag chunk types (domain, variant_cluster,
    protein_summary, function, subunit, disease) over a single gene and
    enforces strict citation discipline: every printed factual sentence
    either points at a retrieved chunk that contains the supporting
    substring (Cited(text, chunk)) or is explicitly tagged
    TEXTBOOK_CONTEXT with source=None and the [NO_RAG_SOURCE] marker.

    NOTE: BRCA1 is NOT in the v0.11 reingest list. PCSK9 carries the full
    chunk-type complement (domain + variant_cluster + protein_summary +
    function + subunit + disease) and is used as the stand-in for the
    BRCA1-style druggability synthesis.
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

from _citation import (  # noqa: E402
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
# Helpers
# ---------------------------------------------------------------------------

_RES_RANGE_RE = re.compile(r"(\d+)\s*[-–]\s*(\d+)")


def _parse_range(residue_range: str) -> tuple[int, int] | None:
    """Parse a 'start-end' (or 'start–end') string into an (int, int) tuple."""
    if not residue_range:
        return None
    m = _RES_RANGE_RE.search(residue_range)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _count_pathogenic_in_text(chunk_text: str) -> tuple[int, int, int]:
    """Return (pathogenic, likely_pathogenic, vus) counts from a variant_cluster blob."""
    text_lower = chunk_text.lower()
    n_path = text_lower.count("(pathogenic)")
    n_likely = text_lower.count("(likely pathogenic)")
    n_vus = text_lower.count("(uncertain significance)")
    return n_path, n_likely, n_vus


def _print_chunks(label: str, chunks: list, max_chars: int = 360) -> None:
    """Pretty-print a list of RetrievedChunk with a short text preview."""
    print(f"\n{label}")
    print("-" * 70)
    if not chunks:
        print("  (no chunks retrieved)")
        return
    for rank, c in enumerate(chunks, 1):
        preview = c.text.replace("\n", " ").strip()
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "..."
        residues = c.residue_range or "(protein-level)"
        print(
            f"  [{rank}] gene={c.gene}  type={c.chunk_type}  "
            f"residues={residues}  score={c.score:.4f}"
        )
        print(f"      {preview}")


def _bucket(buckets: dict, chunks: list) -> None:
    """Append retrieved chunks into a chunk_type -> [chunks] dict (single-gene scope)."""
    for c in chunks:
        buckets[c.chunk_type].append(c)


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return True if residue ranges (a_start, a_end) and (b_start, b_end) overlap."""
    return not (a[1] < b[0] or b[1] < a[0])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the PCSK9 druggability-heatmap cookbook end-to-end."""

    _load_env()

    print("\n" + "=" * 70)
    print("PCSK9 Druggability Heatmap Cookbook (citation-disciplined)")
    print("=" * 70)
    print(
        "\nQuestion: For PCSK9, which residue ranges carry the ClinVar variant\n"
        "burden, which protein domains overlap those ranges, and what does\n"
        "the indexed chunk evidence say? Each printed factual sentence is\n"
        "either Cited(text, chunk) (grounded) or Cited(text, None,\n"
        "label='TEXTBOOK_CONTEXT') (framing, no RAG support).\n"
    )

    # ------------------------------------------------------------------
    # 1. Load the retriever
    # ------------------------------------------------------------------
    from g2p_rag import G2PRetriever

    persist_dir = resolve_chroma_path()
    print(f"[setup] Initialising G2PRetriever  (persist_dir={persist_dir})")
    retriever = G2PRetriever(
        persist_dir=persist_dir,
        embedding_model="all-MiniLM-L6-v2",
        collection_name="g2p_proteins",
    )
    print_index_manifest(retriever)

    buckets: dict = defaultdict(list)
    all_chunks: list = []

    # ------------------------------------------------------------------
    # 2. protein_summary
    # ------------------------------------------------------------------
    print("\n## Protein-level orientation")

    summary = retriever.retrieve(
        "PCSK9 proprotein convertase subtilisin/kexin type 9 canonical sequence length",
        k=3,
        gene_filter=["PCSK9"],
    )
    _print_chunks("protein_summary + nearby chunks", summary)
    _bucket(buckets, summary)
    all_chunks.extend(summary)

    # ------------------------------------------------------------------
    # 3. function + subunit
    # ------------------------------------------------------------------
    print("\n## Mechanism: function + subunit")

    fn_chunks = retriever.retrieve(
        "PCSK9 function LDL receptor binding lysosomal degradation cholesterol homeostasis",
        k=4,
        gene_filter=["PCSK9"],
    )
    _print_chunks("function chunks (mechanism of action)", fn_chunks)
    _bucket(buckets, fn_chunks)
    all_chunks.extend(fn_chunks)

    su_chunks = retriever.retrieve(
        "PCSK9 subunit oligomeric state monomer dimer self-association",
        k=3,
        gene_filter=["PCSK9"],
    )
    _print_chunks("subunit chunks (oligomeric state)", su_chunks)
    _bucket(buckets, su_chunks)
    all_chunks.extend(su_chunks)

    # ------------------------------------------------------------------
    # 4. domain
    # ------------------------------------------------------------------
    print("\n## Structural features: domain")

    dom_chunks = retriever.retrieve(
        "PCSK9 domain Inhibitor I9 prodomain Peptidase S8 catalytic subtilisin fold",
        k=6,
        gene_filter=["PCSK9"],
    )
    _print_chunks("domain chunks (Inhibitor I9, Peptidase S8, ...)", dom_chunks)
    _bucket(buckets, dom_chunks)
    all_chunks.extend(dom_chunks)

    # ------------------------------------------------------------------
    # 5. variant_cluster
    # ------------------------------------------------------------------
    print("\n## Variant burden: variant_cluster (ClinVar-curated)")

    var_chunks = retriever.retrieve(
        "PCSK9 pathogenic missense gain-of-function variant cluster familial "
        "hypercholesterolemia catalytic prodomain",
        k=8,
        gene_filter=["PCSK9"],
    )
    _print_chunks("variant_cluster chunks", var_chunks)
    _bucket(buckets, var_chunks)
    all_chunks.extend(var_chunks)

    # ------------------------------------------------------------------
    # 6. disease
    # ------------------------------------------------------------------
    print("\n## Indication: disease")

    dis_chunks = retriever.retrieve(
        "PCSK9 familial hypercholesterolemia 3 FHCL3 LDL atherosclerosis "
        "premature coronary heart disease",
        k=3,
        gene_filter=["PCSK9"],
    )
    _print_chunks("disease chunks", dis_chunks)
    _bucket(buckets, dis_chunks)
    all_chunks.extend(dis_chunks)

    # ------------------------------------------------------------------
    # 7. Chunk-type coverage report
    # ------------------------------------------------------------------
    print("\n\n## Retrieval coverage")
    print("-" * 70)
    chunk_types_seen = sorted(buckets.keys())
    total_chunks = sum(len(v) for v in buckets.values())
    print(f"  Chunk types touched: {chunk_types_seen}")
    print(f"  Total chunks pulled: {total_chunks}")
    print("  Per chunk_type counts:")
    for ct in chunk_types_seen:
        seen = set()
        unique_chunks = []
        for c in buckets[ct]:
            key = (c.residue_range, c.text[:80])
            if key in seen:
                continue
            seen.add(key)
            unique_chunks.append(c)
        buckets[ct] = unique_chunks
        print(f"    {ct:<18}  n={len(unique_chunks)}  (unique)")

    # ------------------------------------------------------------------
    # 8. Build the heatmap rows from indexed evidence only
    # ------------------------------------------------------------------
    print("\n\n## Domain x variant_cluster overlap (from chunks)")
    print("-" * 70)

    # Domain rows: residue range + label, parsed straight from chunk text.
    domain_rows: list[tuple[tuple[int, int], str, "object"]] = []
    for dc in buckets.get("domain", []):
        rng = _parse_range(dc.residue_range)
        if rng is None:
            continue
        label = "(unnamed)"
        for line in dc.text.splitlines():
            line = line.strip()
            if line.lower().startswith("domain:"):
                label = line.split(":", 1)[1].strip()
                break
        domain_rows.append((rng, label, dc))

    # Variant burden per residue range, from chunk text only.
    variant_burden_by_range: dict[tuple[int, int], tuple[int, int, int]] = {}
    variant_chunk_by_range: dict[tuple[int, int], object] = {}
    for vc in buckets.get("variant_cluster", []):
        rng = _parse_range(vc.residue_range)
        if rng is None:
            continue
        variant_burden_by_range[rng] = _count_pathogenic_in_text(vc.text)
        variant_chunk_by_range[rng] = vc

    # Header
    print(
        f"  {'Domain (residues)':<40} | "
        f"{'Path / LikelyPath / VUS in overlap':<36} | Overlapping variant ranges"
    )
    print("  " + "-" * 110)

    for (drng, dlabel, _dchunk) in domain_rows:
        n_path = n_likely = n_vus = 0
        overlapping = []
        for vrng, (p, lp, vus) in variant_burden_by_range.items():
            if _overlap(drng, vrng):
                n_path += p
                n_likely += lp
                n_vus += vus
                overlapping.append(f"{vrng[0]}-{vrng[1]}")
        burden_cell = f"{n_path} / {n_likely} / {n_vus}"
        dom_cell = f"{dlabel} [{drng[0]}-{drng[1]}]"
        ov_cell = ", ".join(overlapping) if overlapping else "(none)"
        print(f"  {dom_cell[:40]:<40} | {burden_cell:<36} | {ov_cell}")

    # ------------------------------------------------------------------
    # 9. Citation-disciplined synthesis
    # ------------------------------------------------------------------
    print("\n\n## Synthesis (citation-disciplined)")
    print("-" * 70)

    claims: list[Cited] = []

    # --- (1) protein_summary anchor: PCSK9 gene / canonical sequence ---
    summary_evidence = assert_supported(
        "PCSK9 has an indexed protein summary",
        buckets.get("protein_summary", []) + summary,
        hints=["PCSK9", "proprotein convertase"],
    )
    claims.append(
        Cited(
            "1. protein_summary anchors PCSK9 as the indexed gene "
            "(coordinate system for the residue ranges used below).",
            summary_evidence,
        )
    )

    # --- (2) function: LDLR binding + lysosomal degradation ---
    ldlr_evidence = find_in_chunks("LDLR", buckets.get("function", []))
    if ldlr_evidence is None:
        ldlr_evidence = find_in_chunks(
            "low-density lipoprotein receptor", buckets.get("function", [])
        )
    if ldlr_evidence is None:
        ldlr_evidence = find_in_chunks(
            "low density lipoprotein receptor", buckets.get("function", [])
        )
    if ldlr_evidence is not None:
        claims.append(
            Cited(
                "2. function chunks describe PCSK9 binding to the LDL receptor.",
                ldlr_evidence,
            )
        )
    else:
        claims.append(
            Cited(
                "2. function chunks for PCSK9 did not surface an LDLR-binding "
                "string; mechanism statement is framing only.",
                source=None,
                label="TEXTBOOK_CONTEXT",
            )
        )

    # Lysosomal-degradation sub-claim: only printed if a chunk actually says it.
    lyso_evidence = find_in_chunks("lysosom", buckets.get("function", []))
    if lyso_evidence is None:
        lyso_evidence = find_in_chunks("degradation", buckets.get("function", []))
    if lyso_evidence is not None:
        claims.append(
            Cited(
                "   PCSK9 routes bound receptor toward lysosomal degradation "
                "(per indexed function chunk).",
                lyso_evidence,
            )
        )

    # --- (3) subunit: oligomeric state, only if chunks say something ---
    subunit_chunks = buckets.get("subunit", [])
    subunit_evidence = None
    for hint in ("monomer", "oligomer", "self-association", "dimer", "subunit"):
        subunit_evidence = find_in_chunks(hint, subunit_chunks)
        if subunit_evidence is not None:
            break
    if subunit_evidence is not None:
        claims.append(
            Cited(
                "3. subunit chunks describe the oligomeric-state evidence indexed "
                "for PCSK9 (used verbatim, no extrapolation to drug-dose modeling).",
                subunit_evidence,
            )
        )
    else:
        claims.append(
            Cited(
                "3. subunit chunks did not surface an oligomeric-state string; "
                "no subunit conclusion stated.",
                source=None,
                label="TEXTBOOK_CONTEXT",
            )
        )

    # --- (4) domain: enumerate exactly what the chunks named ---
    if domain_rows:
        # Use the highest-scoring domain chunk as the citation.
        best_dom_chunk = domain_rows[0][2]
        domain_str = "; ".join(
            f"{lbl} [{r[0]}-{r[1]}]" for (r, lbl, _c) in domain_rows
        )
        claims.append(
            Cited(
                f"4. domain chunks resolve: {domain_str}. (Names and ranges "
                "taken verbatim from indexed domain chunks; no inference about "
                "which domain holds which surface patch.)",
                best_dom_chunk,
            )
        )
    else:
        claims.append(
            Cited(
                "4. No domain chunks were retrieved.",
                source=None,
                label="TEXTBOOK_CONTEXT",
            )
        )

    # --- (5) variant_cluster: residue ranges + counts ---
    cluster_lines = []
    cluster_evidence = None
    for vrng in sorted(variant_burden_by_range.keys()):
        p, lp, vus = variant_burden_by_range[vrng]
        cluster_lines.append(
            f"{vrng[0]}-{vrng[1]} (P={p}, LP={lp}, VUS={vus})"
        )
        if cluster_evidence is None:
            cluster_evidence = variant_chunk_by_range[vrng]
    if cluster_evidence is not None:
        cluster_summary = "; ".join(cluster_lines)
        claims.append(
            Cited(
                f"5. variant_cluster chunks give residue-resolved ClinVar burden: "
                f"{cluster_summary}.",
                cluster_evidence,
            )
        )

        # Optional: domain x variant overlap, only as observed in the chunks.
        overlap_observed = False
        for (drng, dlabel, dchunk) in domain_rows:
            for vrng in variant_burden_by_range:
                if _overlap(drng, vrng):
                    overlap_observed = True
                    claims.append(
                        Cited(
                            f"   Overlap observed: domain {dlabel} [{drng[0]}-{drng[1]}] "
                            f"overlaps variant_cluster [{vrng[0]}-{vrng[1]}].",
                            dchunk,
                        )
                    )
                    break
            if overlap_observed:
                break
    else:
        claims.append(
            Cited(
                "5. No variant_cluster chunks with parseable residue ranges.",
                source=None,
                label="TEXTBOOK_CONTEXT",
            )
        )

    # --- (6) disease: indication, only what the chunk literally says ---
    disease_chunks = buckets.get("disease", [])
    disease_evidence = None
    for hint in (
        "hypercholesterolemia",
        "familial hypercholesterolemia",
        "FHCL3",
        "LDL",
    ):
        disease_evidence = find_in_chunks(hint, disease_chunks)
        if disease_evidence is not None:
            break
    if disease_evidence is not None:
        # Print the chunk's own residue_range / disease label as the claim
        # body, not a textbook gloss.
        disease_label = "indication chunk present"
        for line in disease_evidence.text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("disease:"):
                disease_label = stripped
                break
        claims.append(
            Cited(
                f"6. disease chunk indexed for PCSK9: '{disease_label}'.",
                disease_evidence,
            )
        )
    else:
        claims.append(
            Cited(
                "6. No disease chunk for PCSK9 matched the searched indication hints.",
                source=None,
                label="TEXTBOOK_CONTEXT",
            )
        )

    # --- (7) framing closer: explicitly marked TEXTBOOK_CONTEXT ---
    claims.append(
        Cited(
            "Framing: PCSK9 is clinically engaged by multiple therapeutic "
            "modalities in the literature; this script only reports which "
            "chunks the index returned for each chunk_type, not editorial "
            "drug-class conclusions.",
            source=None,
            label="TEXTBOOK_CONTEXT",
        )
    )

    for c in claims:
        print(str(c))

    print("\n" + "=" * 70)
    print("End of cookbook.")
    print("=" * 70)


if __name__ == "__main__":
    main()
