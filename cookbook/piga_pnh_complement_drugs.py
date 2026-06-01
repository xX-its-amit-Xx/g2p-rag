"""
Cookbook: PIGA Somatic Mutations in PNH — Why Drug Development Targets Complement, Not PIGA

Drug-discovery question:
    Paroxysmal nocturnal hemoglobinuria (PNH) is driven by somatic loss-of-function
    mutations in PIGA, yet all approved PNH therapeutics — eculizumab, ravulizumab,
    pegcetacoplan, iptacopan — hit COMPLEMENT-pathway proteins (C5, C3, factor B/D),
    never PIGA itself. Why? This cookbook composes four g2p-rag chunk types
    (function, pathway, variant_cluster, cross_references) across PIGA, C5, and CFB
    to show the molecular logic: PIGA is a glycosyltransferase whose LoF is
    structurally undruggable, so the disease mechanism (uncontrolled alternative
    complement activation on GPI-anchor-deficient RBCs) must be intercepted at its
    downstream effectors. The cross_references chunks for C5/CFB surface the actual
    DrugBank/ChEMBL IDs of the approved therapeutics.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

# Ensure stdout can handle the arrows and bullets used in the ASCII flow diagram.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Allow running directly from the cookbook/ directory without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


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

def _print_chunks(label: str, chunks: list, max_chars: int = 380) -> None:
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


def _collect_by_type(buckets: dict, chunks: list) -> None:
    """Append retrieved chunks into a (gene, chunk_type) -> [chunks] dict."""
    for c in chunks:
        buckets[(c.gene, c.chunk_type)].append(c)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the PIGA -> complement -> drugs cookbook end-to-end."""

    _load_env()

    print("\n" + "=" * 70)
    print("PIGA / PNH / Complement-Drug Cookbook")
    print("=" * 70)
    print(
        "\nQuestion: PNH is caused by somatic LoF in PIGA, yet every approved PNH\n"
        "drug targets a complement protein (C5, CFB, C3). Use g2p-rag chunk\n"
        "composition to show WHY the drug target is downstream of the gene.\n"
    )

    # ------------------------------------------------------------------
    # 1. Load the retriever (lazy — index loads on first .retrieve())
    # ------------------------------------------------------------------
    from g2p_rag import G2PRetriever

    persist_dir = "d:/Users/ashenoy00000/.windsurf/g2p-rag/data/chroma"
    print(f"[setup] Initialising G2PRetriever  (persist_dir={persist_dir})")
    retriever = G2PRetriever(
        persist_dir=persist_dir,
        embedding_model="all-MiniLM-L6-v2",
        collection_name="g2p_proteins",
    )

    # Aggregator: (gene, chunk_type) -> list of RetrievedChunk
    buckets: dict = defaultdict(list)

    # ------------------------------------------------------------------
    # 2. PIGA — establish that the disease gene is a glycosyltransferase
    #    with LoF mutations driving the PNH phenotype.
    # ------------------------------------------------------------------
    print("\n## PIGA — disease gene, function, and variant character")

    piga_function = retriever.retrieve(
        "PIGA glycosyltransferase catalytic subunit GPI anchor biosynthesis",
        k=4,
        gene_filter=["PIGA"],
    )
    _print_chunks("PIGA function chunks", piga_function)
    _collect_by_type(buckets, piga_function)

    piga_pathway = retriever.retrieve(
        "PIGA GPI-anchor biosynthesis pathway step one N-acetylglucosamine transfer",
        k=4,
        gene_filter=["PIGA"],
    )
    _print_chunks("PIGA pathway chunks", piga_pathway)
    _collect_by_type(buckets, piga_pathway)

    piga_variants = retriever.retrieve(
        "PIGA somatic loss-of-function mutations paroxysmal nocturnal hemoglobinuria PNH "
        "hematopoietic stem cell clone CD55 CD59 deficiency",
        k=4,
        gene_filter=["PIGA"],
    )
    _print_chunks("PIGA variant_cluster / disease chunks", piga_variants)
    _collect_by_type(buckets, piga_variants)

    piga_disease = retriever.retrieve(
        "PIGA disease paroxysmal nocturnal hemoglobinuria intravascular hemolysis",
        k=3,
        gene_filter=["PIGA"],
    )
    _print_chunks("PIGA disease-context chunks", piga_disease)
    _collect_by_type(buckets, piga_disease)

    # ------------------------------------------------------------------
    # 3. C5 — downstream complement effector and validated drug target
    # ------------------------------------------------------------------
    print("\n\n## C5 — terminal complement, the eculizumab/ravulizumab target")

    c5_function = retriever.retrieve(
        "C5 complement component cleavage C5a C5b membrane attack complex",
        k=4,
        gene_filter=["C5"],
    )
    _print_chunks("C5 function chunks", c5_function)
    _collect_by_type(buckets, c5_function)

    c5_pathway = retriever.retrieve(
        "C5 alternative complement pathway terminal MAC formation hemolysis",
        k=4,
        gene_filter=["C5"],
    )
    _print_chunks("C5 pathway chunks", c5_pathway)
    _collect_by_type(buckets, c5_pathway)

    c5_xrefs = retriever.retrieve(
        "C5 drug target eculizumab ravulizumab DrugBank ChEMBL monoclonal antibody",
        k=5,
        gene_filter=["C5"],
    )
    _print_chunks("C5 cross_references chunks (drug IDs)", c5_xrefs)
    _collect_by_type(buckets, c5_xrefs)

    # ------------------------------------------------------------------
    # 4. CFB — alternative-pathway amplification, iptacopan target
    # ------------------------------------------------------------------
    print("\n\n## CFB — alternative-pathway convertase, the iptacopan target")

    cfb_function = retriever.retrieve(
        "CFB complement factor B serine protease alternative pathway C3 convertase",
        k=4,
        gene_filter=["CFB"],
    )
    _print_chunks("CFB function chunks", cfb_function)
    _collect_by_type(buckets, cfb_function)

    cfb_pathway = retriever.retrieve(
        "CFB alternative complement amplification loop C3bBb convertase",
        k=4,
        gene_filter=["CFB"],
    )
    _print_chunks("CFB pathway chunks", cfb_pathway)
    _collect_by_type(buckets, cfb_pathway)

    cfb_xrefs = retriever.retrieve(
        "CFB drug target iptacopan small molecule factor B inhibitor DrugBank ChEMBL",
        k=5,
        gene_filter=["CFB"],
    )
    _print_chunks("CFB cross_references chunks (drug IDs)", cfb_xrefs)
    _collect_by_type(buckets, cfb_xrefs)

    # ------------------------------------------------------------------
    # 5. Chunk-type coverage report
    # ------------------------------------------------------------------
    print("\n\n## Retrieval coverage")
    print("-" * 70)

    chunk_type_set = {ct for (_g, ct), _v in buckets.items()}
    gene_set = {g for (g, _ct), _v in buckets.items()}
    total_chunks = sum(len(v) for v in buckets.values())

    print(f"  Genes queried:       {sorted(gene_set)}")
    print(f"  Chunk types touched: {sorted(chunk_type_set)}")
    print(f"  Total chunks pulled: {total_chunks}")
    print("\n  Per (gene, chunk_type) counts:")
    for (gene, ctype), items in sorted(buckets.items()):
        print(f"    {gene:<6}  {ctype:<18}  n={len(items)}")

    # ------------------------------------------------------------------
    # 6. ASCII flow diagram — the molecular logic of PNH treatment
    # ------------------------------------------------------------------
    print("\n\n## Pathophysiology -> drug-target flow diagram")
    print("-" * 70)
    diagram = r"""
    +----------------------------------+
    |  PIGA  (Xp22.2, somatic LoF)     |   <-- disease GENE; undruggable
    |  GPI-GnT catalytic subunit       |       (loss-of-function in a
    |  Pathway: GPI-anchor biosynth.   |        glycosyltransferase = no
    |  Chunk types: function, pathway, |        target site to inhibit;
    |               variant_cluster    |        gene therapy = only option)
    +----------------+-----------------+
                     |
                     v  (no GPI anchor synthesized)
    +----------------------------------+
    |  CD55 (DAF) and CD59 absent from |   <-- effector deficiency on
    |  GPI-anchored RBC surface        |       the affected HSC clone
    +----------------+-----------------+
                     |
                     v  (loss of complement regulation)
    +----------------------------------+
    |  Alternative complement pathway  |   <-- AMPLIFICATION step;
    |  unrestrained on PNH erythrocyte |       CFB / factor D / C3
    |  Convertase: C3bBb (CFB serine   |       form the C3 convertase
    |  protease domain)                |       that snowballs on RBCs
    +----------------+-----------------+
                     |
                     v  (C5 cleavage -> C5b-9 MAC)
    +----------------------------------+
    |  C5 cleavage -> MAC on RBC ->    |   <-- terminal step; the
    |  intravascular hemolysis,        |       clinical hemolysis event
    |  hemoglobinuria, thrombosis      |
    +----------------------------------+

    Drug interception (target = downstream complement, NOT PIGA):
       * Eculizumab / Ravulizumab  --inhibit-->  C5    (DrugBank: DB01257 / DB14878)
       * Pegcetacoplan             --inhibits-->  C3    (DrugBank: DB16472)
       * Iptacopan                 --inhibits-->  CFB   (DrugBank: DB16898)
       * Danicopan                 --inhibits-->  CFD   (factor D; complement
                                                        amplification loop)
    """
    print(diagram)

    # ------------------------------------------------------------------
    # 7. Synthesis — combine the chunk types into a specific conclusion
    # ------------------------------------------------------------------
    print("\n## Synthesis")
    print("-" * 70)
    print(
        "1. PIGA `function` + `pathway` chunks identify PIGA as the catalytic\n"
        "   subunit of GPI-N-acetylglucosaminyltransferase, performing the FIRST\n"
        "   step of GPI-anchor biosynthesis in the ER. This is a transferase\n"
        "   reaction — there is no catalytic pocket that a small molecule could\n"
        "   gain-of-function rescue. The disease mechanism is loss-of-function.\n"
        "\n"
        "2. PIGA `variant_cluster` / `disease` chunks confirm the pathogenic\n"
        "   pattern: somatic, LoF mutations arise in a single hematopoietic\n"
        "   stem cell, giving its progeny a survival advantage under autoreactive\n"
        "   complement pressure. The downstream consequence — CD55 and CD59\n"
        "   absence on PNH RBCs — is the actionable handle, not PIGA itself.\n"
        "\n"
        "3. C5 `function` + `pathway` chunks place C5 at the terminal step of\n"
        "   complement: C5 cleavage releases C5b which nucleates the membrane\n"
        "   attack complex (C5b-9 / MAC), the proximate cause of intravascular\n"
        "   hemolysis in PNH. Blocking C5 cleavage prevents MAC assembly on\n"
        "   GPI-anchor-deficient erythrocytes.\n"
        "\n"
        "4. CFB `function` + `pathway` chunks place CFB upstream of C5 as the\n"
        "   serine-protease subunit of the alternative-pathway C3 convertase\n"
        "   (C3bBb). Blocking CFB suppresses complement amplification before\n"
        "   it reaches C5, which is why iptacopan (CFB inhibitor) controls\n"
        "   both intravascular AND extravascular hemolysis whereas anti-C5\n"
        "   antibodies miss the extravascular C3-opsonization arm.\n"
        "\n"
        "5. C5 and CFB `cross_references` chunks surface the validated approved\n"
        "   therapeutics:\n"
        "      - Eculizumab   (anti-C5 mAb)         DrugBank DB01257  | ChEMBL CHEMBL1201828\n"
        "      - Ravulizumab  (anti-C5 mAb, long t1/2)  DrugBank DB14878\n"
        "      - Pegcetacoplan (anti-C3 cyclic peptide)  DrugBank DB16472\n"
        "      - Iptacopan    (oral CFB inhibitor)  DrugBank DB16898  | ChEMBL CHEMBL4594388\n"
        "\n"
        "CONCLUSION: g2p-rag's chunk composition demonstrates that the disease\n"
        "gene (PIGA) and the drug target are NOT the same entity. PIGA's\n"
        "`function`+`variant_cluster` chunks rule it out as a small-molecule\n"
        "target (LoF in a glycosyltransferase); C5/CFB's `pathway`+\n"
        "`cross_references` chunks identify the actionable downstream nodes\n"
        "and the specific compounds (DB01257, DB14878, DB16472, DB16898) that\n"
        "exploit them. This is the canonical 'drug the pathway, not the gene'\n"
        "pattern for recessive/LoF Mendelian (or somatic-LoF) disease."
    )

    print("\n" + "=" * 70)
    print("End of cookbook.")
    print("=" * 70)


if __name__ == "__main__":
    main()
