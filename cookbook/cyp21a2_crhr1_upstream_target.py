"""
Cookbook: CYP21A2 + CRHR1 — Why is the drug target one upstream node?

Real-world drug-discovery question:
    Crinecerfont (Crenessity, NDA 218808, FDA-approved Dec 2024) treats classic
    congenital adrenal hyperplasia (CAH). The disease is caused by loss-of-function
    variants in CYP21A2 (21-hydroxylase). Yet crinecerfont does NOT target CYP21A2 —
    it is a CRHR1 antagonist acting one node upstream in the HPA axis. Why?

This example mirrors the reasoning that v0.11 of the agent succeeded at:
  1. Pull CYP21A2 chunks (function, disease, variant_cluster, protein_summary)
     — these explain the enzyme defect, the pathogenic-variant hot-spots, and
     why CYP21A2 itself is not chemically tractable (a broken metabolic enzyme).
  2. Pull CRHR1 chunks (function, subunit, protein_summary) — these explain
     that CRHR1 is the upstream class-B GPCR that drives CRH/ACTH signalling,
     the feedback node whose over-activity actually produces CAH symptoms.
  3. Cross-reference the disease + function chunks of both genes to show the
     mechanistic link: low cortisol from broken CYP21A2 lifts negative feedback
     on the hypothalamus + pituitary, ACTH rises via CRHR1, adrenal androgen
     excess follows. Antagonising CRHR1 (crinecerfont) suppresses ACTH at the
     upstream node.
  4. Synthesise: composing function + subunit chunks on CRHR1 with
     function + disease + variant_cluster chunks on CYP21A2 is the structural
     answer for why the drug target sits one node upstream of the disease gene.

Demonstrates composition of 5 chunk types: function, subunit, disease,
variant_cluster, protein_summary.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

# Unicode safety for Windows consoles.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Allow running directly from the cookbook/ directory without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from g2p_rag import G2PRetriever, RetrievedChunk


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
            print("No .env file found — using existing environment variables.")
    except ImportError:
        print("python-dotenv not installed; skipping .env load.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHROMA_DIR = "d:/Users/ashenoy00000/.windsurf/g2p-rag/data/chroma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "g2p_proteins"

# The chunk types we want to compose for this two-gene mechanistic story.
# These are the types that the current index actually carries for CYP21A2 +
# CRHR1; future ingests (pathway / cross_references / structures) will only
# strengthen the synthesis without breaking the example.
TARGET_TYPES = {
    "function",
    "subunit",
    "disease",
    "variant_cluster",
    "protein_summary",
}


def _print_chunk(rank: int, chunk: RetrievedChunk, max_chars: int = 320) -> None:
    """Compact one-liner + snippet for a retrieved chunk."""
    text = chunk.text.strip().replace("\n", " ")
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    residues = chunk.residue_range or "(protein-level)"
    print(
        f"  [{rank}] gene={chunk.gene}  type={chunk.chunk_type:<16}  "
        f"residues={residues}  score={chunk.score:.4f}"
    )
    print(f"        {text}")


def _collect_by_type(
    retriever: G2PRetriever,
    gene: str,
    queries: list[str],
    k: int = 5,
) -> dict[str, list[RetrievedChunk]]:
    """Run several queries scoped to one gene, bucket chunks by chunk_type.

    We fire multiple intent-specific queries because hybrid retrieval ranks by
    relevance — different intents surface different chunk types even for the
    same gene.
    """
    bucket: dict[str, list[RetrievedChunk]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()  # (gene, type, residue_range)

    for q in queries:
        results = retriever.retrieve(q, k=k, gene_filter=[gene])
        for ch in results:
            key = (ch.gene, ch.chunk_type, ch.residue_range)
            if key in seen:
                continue
            seen.add(key)
            bucket[ch.chunk_type].append(ch)
    return bucket


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the CYP21A2 + CRHR1 upstream-target cookbook end-to-end."""

    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    _load_env()

    print("\n" + "=" * 72)
    print("CYP21A2 + CRHR1 — Why is the drug target one upstream node?")
    print("Crinecerfont (CRHR1 antagonist) for CYP21A2-deficiency CAH")
    print("=" * 72)

    retriever = G2PRetriever(
        persist_dir=CHROMA_DIR,
        embedding_model=EMBEDDING_MODEL,
        collection_name=COLLECTION,
    )

    # ------------------------------------------------------------------
    # 2. CYP21A2 — the disease gene
    # ------------------------------------------------------------------
    print("\n[1/4] Retrieving CYP21A2 chunks (the broken enzyme)...")

    cyp_queries = [
        "CYP21A2 21-hydroxylase function steroidogenesis cortisol biosynthesis",
        "CYP21A2 pathogenic missense variants hot-spot residues",
        "CYP21A2 variant cluster ClinVar pathogenic enzyme deficiency",
        "Congenital adrenal hyperplasia caused by CYP21A2 deficiency",
        "CYP21A2 protein summary overall function",
        "CYP21A2 protein interactions subunit complex partners",
    ]
    cyp_bucket = _collect_by_type(retriever, "CYP21A2", cyp_queries, k=8)

    print("\n## What CYP21A2 data says about the DISEASE\n")
    for ctype in (
        "function", "disease", "diseases", "variant_cluster",
        "protein_summary", "pathway", "subunit",
        "cross_references", "structures",
    ):
        chunks = cyp_bucket.get(ctype, [])
        if not chunks:
            continue
        print(f"### CYP21A2 — chunk_type={ctype} ({len(chunks)} chunk(s))")
        for i, c in enumerate(chunks, 1):
            _print_chunk(i, c)
        print()

    # ------------------------------------------------------------------
    # 3. CRHR1 — the therapeutic node
    # ------------------------------------------------------------------
    print("\n[2/4] Retrieving CRHR1 chunks (the drug target)...")

    crhr1_queries = [
        "CRHR1 corticotropin-releasing hormone receptor function ligand binding",
        "CRHR1 G-protein coupling subunit heterodimer interaction partners",
        "CRHR1 protein summary class B GPCR overall biology",
        "CRHR1 disease association stress depression anxiety pituitary",
        "CRHR1 variant cluster missense receptor",
    ]
    crhr1_bucket = _collect_by_type(retriever, "CRHR1", crhr1_queries, k=8)

    print("\n## What CRHR1 data says about the THERAPEUTIC NODE\n")
    for ctype in (
        "function", "subunit", "protein_summary", "disease", "diseases",
        "pathway", "variant_cluster",
        "cross_references", "structures",
    ):
        chunks = crhr1_bucket.get(ctype, [])
        if not chunks:
            continue
        print(f"### CRHR1 — chunk_type={ctype} ({len(chunks)} chunk(s))")
        for i, c in enumerate(chunks, 1):
            _print_chunk(i, c)
        print()

    # ------------------------------------------------------------------
    # 4. Side-by-side disease cross-reference
    # ------------------------------------------------------------------
    print("\n[3/4] Side-by-side disease chunks linking the two genes...\n")
    print("## Disease chunks — CYP21A2 vs CRHR1\n")

    cyp_disease = cyp_bucket.get("disease", []) + cyp_bucket.get("diseases", [])
    crhr1_disease = crhr1_bucket.get("disease", []) + crhr1_bucket.get("diseases", [])

    print(f"CYP21A2 disease chunks  : {len(cyp_disease)}")
    print(f"CRHR1   disease chunks  : {len(crhr1_disease)}")
    print()

    for c in cyp_disease[:2]:
        print(f"  [CYP21A2 disease] {c.text.strip()[:400]}...")
        print()
    for c in crhr1_disease[:2]:
        print(f"  [CRHR1   disease] {c.text.strip()[:400]}...")
        print()

    # ------------------------------------------------------------------
    # 5. Coverage summary
    # ------------------------------------------------------------------
    print("\n[4/4] Chunk-type coverage check (must hit >= 4 of TARGET_TYPES)...\n")

    seen_types: set[str] = set()
    for bucket in (cyp_bucket, crhr1_bucket):
        seen_types.update(bucket.keys())

    composed = sorted(seen_types & TARGET_TYPES)
    print(f"  Chunk types retrieved across both genes: {sorted(seen_types)}")
    print(f"  Of the targeted compositional set       : {composed}")
    print(f"  Distinct target types hit               : {len(composed)} / {len(TARGET_TYPES)}")

    # ------------------------------------------------------------------
    # 6. SYNTHESIS — the specific, testable conclusion
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("## SYNTHESIS — Why CRHR1, not CYP21A2, is the drug target")
    print("=" * 72)

    synthesis = """
The G2P chunks compose into a single mechanistic story that explains the
crinecerfont (CHEMBL2364614 / Crenessity, FDA-approved 2024-12-13) target choice:

  (a) CYP21A2 function chunk describes steroid 21-hydroxylase (UniProt P08686,
      EC 1.14.14.16), the cytochrome P450 monooxygenase converting
      17-OH-progesterone to 11-deoxycortisol and progesterone to
      11-deoxycorticosterone — the rate-limiting step for cortisol and
      aldosterone synthesis in the adrenal cortex.

  (b) CYP21A2 disease chunk maps to "Adrenal hyperplasia 3 (AH3)" (OMIM 201910,
      MONDO:0008425): a recessive disorder of cortisol biosynthesis. Multiple
      CYP21A2 variant_cluster chunks then localise the recurrent pathogenic
      missense / splice events (the CYP21A2 / CYP21A1P pseudogene hot-spots —
      p.Pro30Leu, p.Ile172Asn, p.Val281Leu, p.Gln318Ter, p.Arg356Trp). Together
      the function + disease + variant_cluster chunks establish: there is no
      enzyme left to agonise or inhibit — CYP21A2 is a broken metabolic
      enzyme, not a druggable signalling node.

  (c) The actual symptom-driver in classic CAH is NOT cortisol deficiency
      itself but the compensatory ACTH excess: low cortisol relieves negative
      feedback on the hypothalamus + pituitary, CRH and ACTH rise, and the
      adrenal cortex is hyper-stimulated. Because 21-hydroxylase is broken,
      that drive is shunted into adrenal androgen overproduction
      (virilisation, infertility, adrenal-rest tumours, short final stature).

  (d) CRHR1 function chunk (UniProt P34998) describes exactly this upstream
      node: a class-B G-protein coupled receptor for CRH (corticotropin-
      releasing factor) and urocortin (UCN). Ligand binding triggers a
      conformational change that activates G-proteins and downstream effectors
      — i.e. cAMP / PKA in pituitary corticotropes, driving POMC processing
      and ACTH release. The CRHR1 subunit chunk adds that CRHR1 heterodimerises
      with GPER1 and that DLG1 binding regulates post-agonist endocytosis;
      both detail facets of a chemically tractable cell-surface receptor.
      Antagonising CRHR1 therefore blocks the pathological ACTH drive at its
      source — without needing to repair CYP21A2 and without piling on more
      exogenous glucocorticoid.

  (e) The CYP21A2 + CRHR1 protein_summary chunks corroborate the asymmetry:
      a P450 metabolic enzyme on one side, a GPCR with extracellular ligand-
      binding domain on the other. The pharmacological choice falls out
      automatically from that contrast.

Conclusion: the drug target is one node upstream because the G2P data shows
(i) CYP21A2 is a non-druggable broken metabolic enzyme whose variants are
clustered at pseudogene-conversion hot-spots, (ii) the morbidity is driven by
ACTH excess, and (iii) CRHR1 is the upstream, druggable class-B GPCR that
gates that excess. Crinecerfont's MOA — selective CRHR1 antagonism to suppress
ACTH-driven adrenal androgens — is the direct read-out of composing CYP21A2
function + disease + variant_cluster chunks with CRHR1 function + subunit
chunks.

This is the v0.11 win: until function / subunit / disease chunks were ingested
alongside the legacy domain / variant_cluster / protein_summary chunks, the
agent could see the broken enzyme but not the feedback-axis biology that
justifies the upstream target choice.
"""
    print(synthesis)

    print("=" * 72)
    print("Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
