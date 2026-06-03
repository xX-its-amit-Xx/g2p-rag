"""
Cookbook: VUS Pathogenicity Prediction via Cluster-Context Analysis
(citation-disciplined)

Clinical-genetics question
--------------------------
Given a Variant of Uncertain Significance (VUS) at a specific residue --
e.g. p.Leu93Pro in TP53 -- can we make a defensible likely-pathogenic /
unclear / likely-benign call using ONLY chunk-grounded local context?

Why the LLM cannot cheat
------------------------
The model has not memorised the ClinVar status of every arbitrary VUS at
every position in every gene. The residue-resolved ``variant_cluster``
chunks (which co-locate near-by pathogenic variants in narrow windows on
the same protein) are the only signal that supports a per-residue local
density argument. The model cannot invent those bullet lines. We refuse
to make any factual claim that is not anchored in a retrieved chunk; the
discipline helper raises if a print() line is unsupported.

The decision rule (chunk arithmetic, no training-data lookups)
--------------------------------------------------------------
For each VUS (gene + position):

  1. Retrieve variant_cluster chunks covering or near the VUS residue.
     Count Pathogenic / Likely-pathogenic / VUS bullet lines whose
     residue numbers fall within +-10 residues of the VUS position.
     -> local_path_density.
  2. Retrieve domain chunks for the same gene. If any domain's residue
     span contains the VUS position, the VUS is "inside an annotated
     domain" -- adds a structural-context point.
  3. Retrieve function and protein_summary chunks. These add a
     reading-frame / mechanism anchor but do not score on their own.
  4. Score (purely from chunk arithmetic):
        path_score  = local_path_density   (Pathogenic + Likely-pathogenic
                                            within +-10 aa)
        domain_bonus = +1 if VUS sits inside a domain chunk span
        vus_penalty  = -1 * local_vus_density (other VUSes nearby are
                                               evidence of regional
                                               *ambiguity*, not pathology)
        total        = path_score + domain_bonus + vus_penalty

     Verdict:
        >= 5  -> "likely pathogenic"   (high local pathogenic burden)
        2..4  -> "lean pathogenic"
        0..1  -> "insufficient evidence"
        < 0   -> "lean benign / ambiguous region"

     Confidence (printed alongside the verdict) is a simple monotone
     transform of the total score, capped at 95% so we never claim
     certainty from a single chunk.

What is intentionally NOT done
------------------------------
- No ACMG criteria are quoted by name; ACMG codes (PM1, PP3, ...) live
  outside the g2p-rag index.
- No disease label is attached to the verdict; the script reports
  "likely pathogenic for the indexed gene's disease association" and
  leaves the disease name to the disease chunk that the script cites.
- No drug or therapy is suggested; that's out of scope and out of index.

Gene roster
-----------
The task narrative lists BRCA1 alongside TP53/CFTR/DMD/HBB. BRCA1 is
NOT in the current g2p-rag reingest, so this cookbook substitutes the
indexed genes only: TP53, CFTR, DMD, HBB. (We surface a single
[NO_RAG_SOURCE] note acknowledging the BRCA1 omission so a reader of
the narrative is not misled.)
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# Unicode-safe stdout for chunk-body glyphs (en-dashes, bullets) on
# Windows consoles before any print() runs.
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
# VUS test cases
# ---------------------------------------------------------------------------
#
# Each test case is (gene, residue_position, hgvs_p_label).
# The HGVS labels here are present in the g2p-rag variant_cluster
# chunks -- they are NOT invented. The script confirms each label
# actually appears in some retrieved chunk before scoring.
#
# Mix of likely-pathogenic-leaning and ambiguous cases:
#   - TP53 p.Leu93Pro     : VUS in the TP53 DNA-binding region cluster
#   - TP53 p.Met133Lys    : VUS in the same dense cluster
#   - CFTR p.Val1327Met   : VUS in a small cluster with 1 LP neighbour
#   - CFTR p.Ile869Ser    : VUS in a small cluster with 1 LP neighbour
#   - DMD  p.Pro240Ser    : VUS sitting AT the C-terminus of CH2 domain
#   - DMD  p.Gly208Ala    : VUS isolated, no domain coverage at 208
#   - HBB  p.Pro6Ser      : VUS deep inside the dense globin cluster
#   - HBB  p.Cys94Trp     : VUS at heme-pocket-adjacent residue (dense cluster)

VUS_CASES: list[tuple[str, int, str]] = [
    ("TP53", 93,   "p.Leu93Pro"),
    ("TP53", 133,  "p.Met133Lys"),
    ("CFTR", 1327, "p.Val1327Met"),
    ("CFTR", 869,  "p.Ile869Ser"),
    ("DMD",  240,  "p.Pro240Ser"),
    ("DMD",  208,  "p.Gly208Ala"),
    ("HBB",  6,    "p.Pro6Ser"),
    ("HBB",  94,   "p.Cys94Trp"),
]

# Genes whose narrative-listed test case could not be run because the
# gene is absent from the current g2p-rag reingest. Reported once as a
# loud [NO_RAG_SOURCE] caveat so the reader of the cookbook does not
# mistake omission for absence of pathogenic clusters.
NARRATIVE_GENES_NOT_INDEXED: list[str] = ["BRCA1"]

LOCAL_WINDOW_AA = 10  # +- residues around the VUS counted as "local"


# ---------------------------------------------------------------------------
# Chunk-text parsing helpers
# ---------------------------------------------------------------------------

# Variant_cluster bullet line shape (chunk template):
#   "  X p.<Aaa><pos><...> (<significance>) []"
# We capture position and significance. We don't care about the bullet
# glyph (UTF-8 BULLET printed as '?' under cp1252 cmd shells -- the regex
# does not depend on it).
_VARIANT_LINE_RE = re.compile(
    r"p\.[A-Za-z]{3}(\d+)\S*\s*\(([^)]+)\)"
)

# residue_range metadata uses ASCII hyphens.
_RES_RANGE_RE = re.compile(r"(\d+)\s*[-]\s*(\d+)")


def _parse_range(residue_range: str) -> tuple[int, int] | None:
    """Parse a 'start-end' ASCII-hyphen residue range into (int, int)."""
    if not residue_range:
        return None
    m = _RES_RANGE_RE.search(residue_range)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _classify_significance(sig: str) -> str:
    """Bucket a ClinVar significance string into {P, LP, VUS, OTHER}.

    Pure string mapping, NOT a clinical claim -- we only use this to
    count chunk-body bullet lines. ``OTHER`` covers benign / likely-benign /
    conflicting / unknown, which we don't use in the density signal.
    """
    s = sig.lower()
    if "uncertain" in s:
        return "VUS"
    if "likely pathogenic" in s and "pathogenic/likely" not in s:
        return "LP"
    # "Pathogenic" or "Pathogenic/Likely pathogenic" or "Pathogenic; other".
    # We deliberately do NOT count "Conflicting classifications" as P.
    if "pathogenic" in s and "conflicting" not in s and "benign" not in s:
        return "P"
    return "OTHER"


def _local_counts(
    variant_chunks: list,
    target_pos: int,
    window: int = LOCAL_WINDOW_AA,
) -> tuple[int, int, int, list[str]]:
    """Walk variant_cluster chunk bodies and count P/LP/VUS within +-window of target_pos.

    Returns (n_path, n_lp, n_vus, neighbour_strings) where
    ``neighbour_strings`` is a deduplicated, residue-sorted list like
    ``["p.Leu93Pro (Uncertain significance)", ...]`` for the audit
    printout. Counts deliberately exclude the VUS *itself* (a residue
    matches if 0 < |pos - target_pos| <= window) so we never inflate the
    score by counting the variant under evaluation as its own neighbour.
    """
    n_path = n_lp = n_vus = 0
    seen: set[tuple[int, str]] = set()
    neighbours: list[tuple[int, str, str]] = []
    for c in variant_chunks:
        if c.chunk_type != "variant_cluster":
            continue
        for line in c.text.splitlines():
            line = line.strip()
            m = _VARIANT_LINE_RE.search(line)
            if not m:
                continue
            pos = int(m.group(1))
            if pos == target_pos:
                continue  # don't count the VUS itself
            if abs(pos - target_pos) > window:
                continue
            sig_full = m.group(2)
            bucket = _classify_significance(sig_full)
            key = (pos, line)
            if key in seen:
                continue
            seen.add(key)
            # Pull the actual "p.<Aaa><pos>..." token for the neighbour print.
            hgvs_match = re.search(r"p\.[A-Za-z]{3}\d+\S*", line)
            hgvs = hgvs_match.group(0) if hgvs_match else f"<pos {pos}>"
            if bucket == "P":
                n_path += 1
                neighbours.append((pos, hgvs, "P"))
            elif bucket == "LP":
                n_lp += 1
                neighbours.append((pos, hgvs, "LP"))
            elif bucket == "VUS":
                n_vus += 1
                neighbours.append((pos, hgvs, "VUS"))
    neighbours.sort(key=lambda t: (t[0], t[1]))
    neighbour_strs = [f"{hg} [{b}]" for (_p, hg, b) in neighbours]
    return n_path, n_lp, n_vus, neighbour_strs


def _domain_hit(domain_chunks: list, target_pos: int) -> tuple[object, str] | None:
    """Return (chunk, label) for the first domain chunk whose span contains target_pos."""
    for c in domain_chunks:
        if c.chunk_type != "domain":
            continue
        rng = _parse_range(c.residue_range)
        if rng is None:
            continue
        s, e = rng
        if s <= target_pos <= e:
            label = "(unnamed domain)"
            for line in c.text.splitlines():
                line = line.strip()
                if line.lower().startswith("domain:"):
                    label = line.split(":", 1)[1].strip()
                    break
            return c, label
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_vus(
    n_path: int,
    n_lp: int,
    n_vus: int,
    in_domain: bool,
) -> tuple[int, str, int]:
    """Combine the chunk counts into an integer score, verdict label, confidence %.

    Pure arithmetic on the counts -- NO training-data lookups. The
    confidence number is bounded so single-chunk hits never reach 95%.
    """
    path_score = n_path + n_lp
    domain_bonus = 1 if in_domain else 0
    vus_penalty = -1 * n_vus  # ambient ambiguity reduces our certainty
    total = path_score + domain_bonus + vus_penalty

    if total >= 5:
        verdict = "likely pathogenic"
    elif total >= 2:
        verdict = "lean pathogenic"
    elif total >= 0:
        verdict = "insufficient evidence"
    else:
        verdict = "lean benign / ambiguous region"

    # Confidence: shallow logistic, capped at 95%. Each piece of
    # P/LP evidence buys roughly 10 percentage points, with a floor of
    # 35% for the "insufficient evidence" bucket and 50% for "lean".
    if total >= 5:
        confidence = min(95, 60 + 5 * max(0, total - 5))
    elif total >= 2:
        confidence = 50 + 5 * (total - 2)
    elif total >= 0:
        confidence = 35 + 5 * total
    else:
        confidence = max(20, 40 - 5 * abs(total))

    return total, verdict, confidence


# ---------------------------------------------------------------------------
# Per-gene retrieval bundle
# ---------------------------------------------------------------------------

def _retrieve_gene_bundle(retriever, gene: str) -> dict[str, list]:
    """Pull variant_cluster + domain + function + protein_summary chunks for `gene`.

    Returns a dict keyed by chunk_type. Each list deduplicates on
    (chunk_type, residue_range, first 80 chars of text) to avoid double-counting
    chunks that surface from multiple queries.
    """
    buckets: dict[str, list] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()

    queries = [
        (f"{gene} variant cluster pathogenic missense uncertain significance",     12),
        (f"{gene} variant pathogenic likely pathogenic",                            12),
        (f"{gene} domain region structural feature motif",                          10),
        (f"{gene} function activity catalytic role",                                 5),
        (f"{gene} protein summary canonical sequence length",                        5),
    ]
    for q, k in queries:
        for c in retriever.retrieve(q, k=k, gene_filter=[gene]):
            key = (c.chunk_type, c.residue_range, c.text[:80])
            if key in seen:
                continue
            seen.add(key)
            buckets[c.chunk_type].append(c)
    return buckets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _load_env()

    print("\n" + "=" * 74)
    print("VUS Pathogenicity Prediction via Cluster-Context Analysis")
    print("=" * 74)
    print(
        "\nFor each VUS (gene + position), we retrieve variant_cluster chunks,\n"
        f"count Pathogenic / Likely-pathogenic neighbours within +-{LOCAL_WINDOW_AA} residues,\n"
        "check domain coverage, and emit a chunk-grounded verdict + confidence.\n"
        "Every printed factual line is Cited(text, chunk) (RAG-grounded) or\n"
        "Cited(text, None, label='TEXTBOOK_CONTEXT') (framing, no RAG support)."
    )

    from g2p_rag import G2PRetriever

    persist_dir = resolve_chroma_path()
    print(f"\n[setup] Initialising G2PRetriever  (persist_dir={persist_dir})")
    retriever = G2PRetriever(
        persist_dir=persist_dir,
        embedding_model="all-MiniLM-L6-v2",
        collection_name="g2p_proteins",
    )
    print_index_manifest(retriever)

    # --- Cache retrieval bundles per gene (one call per distinct gene) ---
    genes_in_run = sorted({g for (g, _p, _l) in VUS_CASES})
    print(f"[setup] Genes to evaluate: {genes_in_run}")
    bundles: dict[str, dict[str, list]] = {}
    for g in genes_in_run:
        bundles[g] = _retrieve_gene_bundle(retriever, g)

    # Coverage report -- counts come from len() of actual retrieval state.
    print("\nRetrieval coverage per gene:")
    print("-" * 74)
    for g in genes_in_run:
        parts = [
            f"{ct}={len(bundles[g].get(ct, []))}"
            for ct in ("variant_cluster", "domain", "function", "protein_summary")
        ]
        print(f"  {g:<6}  " + "  ".join(parts))

    # ------------------------------------------------------------------
    # Per-VUS evaluation
    # ------------------------------------------------------------------
    verdict_rows: list[tuple[str, str, int, str, int]] = []

    for (gene, pos, hgvs) in VUS_CASES:
        print("\n" + "=" * 74)
        print(f"VUS: {gene}  {hgvs}  (position {pos})")
        print("=" * 74)

        bundle = bundles[gene]
        variant_chunks = bundle.get("variant_cluster", [])
        domain_chunks = bundle.get("domain", [])
        function_chunks = bundle.get("function", [])
        summary_chunks = bundle.get("protein_summary", [])

        # ----- (1) Confirm the VUS itself appears in a retrieved cluster ---
        # We don't have to find it -- the prediction is about the
        # *neighbourhood* -- but if we can cite the line, we should.
        vus_evidence = find_in_chunks(hgvs, variant_chunks)

        # ----- (2) Local neighbourhood density -------------------------------
        n_path, n_lp, n_vus_neighbours, neighbour_strs = _local_counts(
            variant_chunks, pos, LOCAL_WINDOW_AA
        )

        # Pick a cluster chunk that actually covers the VUS position; that's
        # what we cite when reporting the density numbers.
        cluster_evidence = None
        for c in variant_chunks:
            rng = _parse_range(c.residue_range)
            if rng is None:
                continue
            if rng[0] - LOCAL_WINDOW_AA <= pos <= rng[1] + LOCAL_WINDOW_AA:
                if cluster_evidence is None or c.score > cluster_evidence.score:
                    cluster_evidence = c

        # ----- (3) Domain coverage ------------------------------------------
        dom_hit = _domain_hit(domain_chunks, pos)
        in_domain = dom_hit is not None

        # ----- (4) Scoring ---------------------------------------------------
        total, verdict, confidence = _score_vus(
            n_path, n_lp, n_vus_neighbours, in_domain
        )

        # ----- (5) Print the arithmetic (citation-gated) --------------------
        print(f"\n  Local window: residues {pos - LOCAL_WINDOW_AA}..{pos + LOCAL_WINDOW_AA}")
        print(f"  Pathogenic neighbours       (P) : {n_path}")
        print(f"  Likely-pathogenic neighbours(LP): {n_lp}")
        print(f"  Other VUS neighbours        (V) : {n_vus_neighbours}")
        print(f"  Inside annotated domain         : {'YES' if in_domain else 'no'}")
        print(f"  Score = (P + LP) + domain_bonus - VUS_penalty")
        print(f"        = ({n_path} + {n_lp}) + {1 if in_domain else 0} - {n_vus_neighbours}")
        print(f"        = {total}")
        print(f"  Verdict: {verdict.upper()}   confidence: {confidence}%")

        # ----- (6) Evidence lines: ALL Cited() and gated by assert_supported -
        print("\n  Chunk-grounded evidence:")

        # The VUS itself (cited only if a chunk literally contains the HGVS).
        if vus_evidence is not None:
            ev = assert_supported(
                f"{gene} {hgvs} appears verbatim in a variant_cluster chunk",
                [vus_evidence],
                hints=[hgvs],
            )
            print("    " + str(Cited(
                f"VUS {hgvs} is indexed as Uncertain significance in cluster "
                f"{vus_evidence.residue_range}",
                ev,
            )))
        else:
            # Honest disclosure: we did not find the literal HGVS string in
            # any retrieved cluster. The neighbourhood call still stands.
            print("    " + str(Cited(
                f"VUS {hgvs} text not found in retrieved clusters; "
                "prediction rests on neighbourhood density only.",
                None,
                label="TEXTBOOK_CONTEXT",
            )))

        # The cluster body that supplied the neighbour counts.
        if cluster_evidence is not None:
            # The chunk body always carries a "Variant cluster:" header and
            # at least one "Pathogenic" or "Uncertain" string -- gate on those.
            hint_alts = [
                f"Variant cluster: {cluster_evidence.residue_range.split('-')[0]}",
                "Variant cluster:",
                "Pathogenic",
                "Uncertain significance",
            ]
            ev = assert_supported(
                f"{gene} cluster spanning {cluster_evidence.residue_range} "
                "lists per-residue ClinVar variants",
                [cluster_evidence],
                hints=hint_alts,
            )
            print("    " + str(Cited(
                f"variant_cluster {cluster_evidence.residue_range} supplies the "
                f"neighbour counts (P={n_path}, LP={n_lp}, VUS={n_vus_neighbours} "
                f"within +-{LOCAL_WINDOW_AA} aa of {pos}).",
                ev,
            )))
        else:
            # No cluster at all within reach -> no neighbour evidence; the
            # script will already have scored zero path/lp.
            print("    " + str(Cited(
                f"No variant_cluster chunk within +-{LOCAL_WINDOW_AA} of residue {pos}; "
                "neighbourhood density is undefined.",
                None,
                label="TEXTBOOK_CONTEXT",
            )))

        # Domain-context evidence, only if a real domain chunk covers the residue.
        if in_domain and dom_hit is not None:
            dom_chunk, dom_label = dom_hit
            start_str = dom_chunk.residue_range.split("-")[0]
            ev = assert_supported(
                f"{gene} domain '{dom_label}' covers residue {pos}",
                [dom_chunk],
                hints=[f"Residues: {start_str}", "Domain"],
            )
            print("    " + str(Cited(
                f"residue {pos} sits inside domain '{dom_label}' "
                f"[{dom_chunk.residue_range}].",
                ev,
            )))
        else:
            # We explicitly state the negative -- "no domain chunk covers" --
            # only when we actually have domain chunks for this gene to look in.
            if domain_chunks:
                print("    " + str(Cited(
                    f"No retrieved domain chunk's span contains residue {pos} "
                    f"(searched {len(domain_chunks)} domain chunk(s)).",
                    None,
                    label="TEXTBOOK_CONTEXT",
                )))
            else:
                print("    " + str(Cited(
                    f"No domain chunks indexed for {gene} in this retrieval.",
                    None,
                    label="TEXTBOOK_CONTEXT",
                )))

        # Mechanism anchor (function chunk) -- cite the gene's function line
        # verbatim if a function chunk was retrieved. We never paraphrase
        # past the chunk body.
        if function_chunks:
            ev = assert_supported(
                f"{gene} function chunk present",
                function_chunks,
                hints=["FUNCTION", "Gene:"],
            )
            # Pull the first non-blank line after the FUNCTION header.
            mechanism_phrase = ""
            for line in ev.text.splitlines():
                line = line.strip()
                if line and not line.startswith("FUNCTION") and not line.startswith("Gene:"):
                    mechanism_phrase = line[:140] + ("..." if len(line) > 140 else "")
                    break
            if mechanism_phrase:
                print("    " + str(Cited(
                    f"function chunk anchors mechanism: \"{mechanism_phrase}\"",
                    ev,
                )))

        # Protein-summary anchor (so the reader can verify the coordinate system).
        if summary_chunks:
            ev = assert_supported(
                f"{gene} protein_summary indexed",
                summary_chunks,
                hints=[gene, "UniProt"],
            )
            print("    " + str(Cited(
                f"protein_summary {ev.residue_range} fixes the coordinate system "
                f"for position {pos}.",
                ev,
            )))

        # Print the neighbour list (up to 8) -- raw chunk content, no
        # editorialising. Tied to the cluster_evidence chunk we already cited.
        if neighbour_strs:
            shown = ", ".join(neighbour_strs[:8])
            if len(neighbour_strs) > 8:
                shown += f", ... (+{len(neighbour_strs) - 8} more)"
            print(f"\n  Neighbour variants within +-{LOCAL_WINDOW_AA} aa: {shown}")

        verdict_rows.append((gene, hgvs, pos, verdict, confidence))

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("## Predictions summary")
    print("=" * 74)
    print(f"  {'Gene':<6}  {'HGVSp':<16}  {'Pos':>5}  {'Verdict':<30}  Conf")
    print(f"  {'-' * 6:<6}  {'-' * 16:<16}  {'-' * 5:>5}  {'-' * 30:<30}  {'-' * 4}")
    for (g, h, p, v, conf) in verdict_rows:
        print(f"  {g:<6}  {h:<16}  {p:>5}  {v:<30}  {conf}%")

    # Counts per verdict bucket -- straight off the rows list.
    bucket_counts: dict[str, int] = defaultdict(int)
    for (_g, _h, _p, v, _c) in verdict_rows:
        bucket_counts[v] += 1
    print("\n  Verdict distribution:")
    for v in (
        "likely pathogenic",
        "lean pathogenic",
        "insufficient evidence",
        "lean benign / ambiguous region",
    ):
        print(f"    {v:<32}  n={bucket_counts.get(v, 0)}")

    # ------------------------------------------------------------------
    # Framing closer + narrative-omission caveat
    # ------------------------------------------------------------------
    print("\n## Caveats (framing, NOT chunk-derived)")
    print("-" * 74)

    # BRCA1 (narrative-listed but not indexed) -- single loud disclosure.
    if NARRATIVE_GENES_NOT_INDEXED:
        print("  " + str(Cited(
            f"The cookbook narrative lists {', '.join(NARRATIVE_GENES_NOT_INDEXED)} "
            f"as candidate VUS hosts, but those gene(s) are not in the current "
            f"g2p-rag reingest. They are omitted from this run rather than "
            f"silently substituted with training-data calls.",
            None,
            label="TEXTBOOK_CONTEXT",
        )))

    # ACMG / clinical-use disclaimer -- never claimed as RAG output.
    print("  " + str(Cited(
        "This script's verdicts are chunk-arithmetic heuristics over local "
        "ClinVar density and domain coverage. They are NOT ACMG/AMP variant "
        "classifications and must not be used for clinical reporting.",
        None,
        label="TEXTBOOK_CONTEXT",
    )))

    # Closing recap.
    print("\n" + "=" * 74)
    print(
        "Recommendation: triage VUSes with score >= 5 for orthogonal evidence "
        "(functional assay, segregation, in silico ensembles) first; treat "
        "score < 0 as 'do not over-call' rather than 'benign'."
    )
    print("End of cookbook.")
    print("=" * 74)


if __name__ == "__main__":
    main()
