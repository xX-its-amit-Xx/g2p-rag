"""
Cookbook: PIGA / complement chunks — what the index actually says.

Drug-discovery question (framing only — NOT a RAG-grounded conclusion):
    Paroxysmal nocturnal hemoglobinuria (PNH) is associated with PIGA, but
    approved PNH therapeutics target the complement cascade. Can g2p-rag's
    chunk types, on their own, support the molecular logic for that
    "drug the pathway, not the gene" pattern?

Citation discipline (this rewrite)
----------------------------------
This script was rewritten under a strict citation contract. Every printed
factual claim is either:

  (a) wrapped in ``Cited(text, chunk)`` where ``chunk`` came from a real
      ``r.retrieve(...)`` call AND ``assert_supported(...)`` confirmed the
      chunk text literally contains a substring supporting the claim, OR
  (b) wrapped in ``Cited(text, None, label="TEXTBOOK_CONTEXT")`` and clearly
      framed as background, NOT as a RAG-derived insight.

The previous version of this script printed a long synthesis paragraph
asserting things the index does NOT contain — CD55/CD59 surface loss, clonal
HSC selection, intravascular vs extravascular hemolysis, eculizumab /
ravulizumab / pegcetacoplan / iptacopan as approved drugs, and specific
DrugBank/ChEMBL IDs (DB01257, DB14878, DB16472, DB16898, CHEMBL1201828,
CHEMBL4594388). None of those appear in the retrieved chunks for the
queries below, so they have been DELETED from the synthesis (not papered
over). What remains is only what the chunks actually attest.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _citation import Cited, assert_supported, find_in_chunks  # noqa: E402


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
    """Run the PIGA / C5 / CFB retrieval cookbook end-to-end."""

    _load_env()

    print("\n" + "=" * 70)
    print("PIGA / C5 / CFB chunk-composition cookbook")
    print("=" * 70)
    print(
        "\nQuestion (framing): PNH is associated with PIGA, but approved PNH\n"
        "therapeutics target complement proteins. What do g2p-rag chunks for\n"
        "PIGA, C5 and CFB literally say about each protein's function and\n"
        "pathway role? (No drug-name / DrugBank-ID conclusions are drawn unless\n"
        "they appear verbatim in a retrieved chunk.)\n"
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
    # Flat pool of every chunk seen — used by assert_supported in the
    # synthesis section.
    all_chunks: list = []

    def _extend(chunks: list) -> list:
        all_chunks.extend(chunks)
        return chunks

    # ------------------------------------------------------------------
    # 2. PIGA — establish what the index says about PIGA's function,
    #    pathway membership, and disease association.
    # ------------------------------------------------------------------
    print("\n## PIGA — function, pathway, disease")

    piga_function = _extend(retriever.retrieve(
        "PIGA glycosyltransferase catalytic subunit GPI anchor biosynthesis",
        k=4,
        gene_filter=["PIGA"],
    ))
    _print_chunks("PIGA function chunks", piga_function)
    _collect_by_type(buckets, piga_function)

    piga_pathway = _extend(retriever.retrieve(
        "PIGA GPI-anchor biosynthesis pathway step one N-acetylglucosamine transfer",
        k=4,
        gene_filter=["PIGA"],
    ))
    _print_chunks("PIGA pathway chunks", piga_pathway)
    _collect_by_type(buckets, piga_pathway)

    piga_variants = _extend(retriever.retrieve(
        "PIGA somatic loss-of-function mutations paroxysmal nocturnal hemoglobinuria PNH "
        "hematopoietic stem cell clone CD55 CD59 deficiency",
        k=4,
        gene_filter=["PIGA"],
    ))
    _print_chunks("PIGA variant_cluster / disease chunks", piga_variants)
    _collect_by_type(buckets, piga_variants)

    piga_disease = _extend(retriever.retrieve(
        "PIGA disease paroxysmal nocturnal hemoglobinuria intravascular hemolysis",
        k=3,
        gene_filter=["PIGA"],
    ))
    _print_chunks("PIGA disease-context chunks", piga_disease)
    _collect_by_type(buckets, piga_disease)

    # ------------------------------------------------------------------
    # 3. C5 — what the index says about C5's role
    # ------------------------------------------------------------------
    print("\n\n## C5 — complement component")

    c5_function = _extend(retriever.retrieve(
        "C5 complement component cleavage C5a C5b membrane attack complex",
        k=4,
        gene_filter=["C5"],
    ))
    _print_chunks("C5 function chunks", c5_function)
    _collect_by_type(buckets, c5_function)

    c5_pathway = _extend(retriever.retrieve(
        "C5 alternative complement pathway terminal MAC formation hemolysis",
        k=4,
        gene_filter=["C5"],
    ))
    _print_chunks("C5 pathway chunks", c5_pathway)
    _collect_by_type(buckets, c5_pathway)

    c5_xrefs = _extend(retriever.retrieve(
        "C5 drug target eculizumab ravulizumab DrugBank ChEMBL monoclonal antibody",
        k=5,
        gene_filter=["C5"],
    ))
    _print_chunks("C5 'cross_references' query chunks", c5_xrefs)
    _collect_by_type(buckets, c5_xrefs)

    # ------------------------------------------------------------------
    # 4. CFB — what the index says about complement factor B
    # ------------------------------------------------------------------
    print("\n\n## CFB — complement factor B")

    cfb_function = _extend(retriever.retrieve(
        "CFB complement factor B serine protease alternative pathway C3 convertase",
        k=4,
        gene_filter=["CFB"],
    ))
    _print_chunks("CFB function chunks", cfb_function)
    _collect_by_type(buckets, cfb_function)

    cfb_pathway = _extend(retriever.retrieve(
        "CFB alternative complement amplification loop C3bBb convertase",
        k=4,
        gene_filter=["CFB"],
    ))
    _print_chunks("CFB pathway chunks", cfb_pathway)
    _collect_by_type(buckets, cfb_pathway)

    cfb_xrefs = _extend(retriever.retrieve(
        "CFB drug target iptacopan small molecule factor B inhibitor DrugBank ChEMBL",
        k=5,
        gene_filter=["CFB"],
    ))
    _print_chunks("CFB 'cross_references' query chunks", cfb_xrefs)
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

    # Audit note: the queries asked for `cross_references` content (drug
    # names, DrugBank IDs) but the actual retrieved chunk_types are checked
    # here so the script doesn't silently assume those chunks exist.
    has_xrefs = "cross_references" in chunk_type_set
    print(
        f"\n  cross_references chunks in pool? {has_xrefs}"
        + ("" if has_xrefs else "  <-- drug-ID claims will NOT be made")
    )

    # ------------------------------------------------------------------
    # 6. ASCII flow diagram — textbook context, NOT chunk-grounded
    # ------------------------------------------------------------------
    print("\n\n## Pathophysiology cartoon (textbook context, NOT RAG-derived)")
    print("-" * 70)
    diagram_context = Cited(
        text=(
            "The diagram below is general complement-cascade textbook context "
            "to orient the reader. It is NOT derived from retrieved chunks; "
            "any node-by-node claim it implies should be read as background."
        ),
        source=None,
        label="TEXTBOOK_CONTEXT",
    )
    print(diagram_context)
    diagram = r"""
    +----------------------------------+
    |  PIGA  (catalytic subunit of     |
    |  GPI-GnT; first step of GPI-     |   [chunk-grounded — see synthesis]
    |  anchor biosynthesis)            |
    +----------------+-----------------+
                     |
                     v
    +----------------------------------+
    |  Alternative complement pathway: |   [chunk-grounded for CFB role:
    |  CFB Bb is the catalytic         |    C3 convertase = C3bBb;
    |  component of the C3 convertase  |    C5 convertase = C3bBb3b]
    |  (C3bBb) and C5 convertase       |
    |  (C3bBb3b)                       |
    +----------------+-----------------+
                     |
                     v
    +----------------------------------+
    |  C5 -> C5a + C5b; C5b is a       |   [chunk-grounded — see synthesis]
    |  component of the membrane       |
    |  attack complex (MAC)            |
    +----------------------------------+
    """
    print(diagram)

    # ------------------------------------------------------------------
    # 7. Synthesis — strictly chunk-grounded conclusions
    # ------------------------------------------------------------------
    print("\n## Synthesis (chunk-grounded only)")
    print("-" * 70)

    # ---- (1) PIGA function: catalytic subunit of GPI-GnT --------------
    piga_func_chunk = assert_supported(
        claim="PIGA is the catalytic subunit of the GPI-GnT complex.",
        chunks=all_chunks,
        hints=[
            "Catalytic subunit of the glycosylphosphatidylinositol-N-acetylglucosaminyltransferase",
            "GPI-GnT",
        ],
    )
    print(Cited(
        "PIGA is the catalytic subunit of the GPI-N-acetylglucosaminyltransferase "
        "(GPI-GnT) complex.",
        piga_func_chunk,
    ))

    # ---- (2) PIGA reaction: transfers GlcNAc from UDP-GlcNAc to PI ----
    piga_rxn_chunk = assert_supported(
        claim="PIGA catalyses transfer of GlcNAc from UDP-GlcNAc to phosphatidylinositol.",
        chunks=all_chunks,
        hints=[
            "transfer of N-acetylglucosamine from UDP-N-acetylglucosamine to phosphatidylinositol",
        ],
    )
    print(Cited(
        "It catalyses transfer of N-acetylglucosamine from UDP-N-acetylglucosamine "
        "to phosphatidylinositol.",
        piga_rxn_chunk,
    ))

    # ---- (3) PIGA position in pathway: first step of GPI biosynthesis -
    piga_first_step_chunk = assert_supported(
        claim="PIGA acts in the first step of GPI biosynthesis.",
        chunks=all_chunks,
        hints=[
            "participates in the first step of GPI biosynthesis",
            "first step of GPI biosynthesis",
        ],
    )
    print(Cited(
        "This reaction is the FIRST step of GPI biosynthesis.",
        piga_first_step_chunk,
    ))

    # ---- (4) PIGA pathway chunk: GPI-anchor biosynthesis --------------
    piga_pathway_chunk = assert_supported(
        claim="PIGA's pathway annotation is GPI-anchor biosynthesis (glycolipid biosynthesis).",
        chunks=all_chunks,
        hints=[
            "glycosylphosphatidylinositol-anchor biosynthesis",
            "Glycolipid biosynthesis",
        ],
    )
    print(Cited(
        "PIGA's `pathway` chunk records: 'Glycolipid biosynthesis; "
        "glycosylphosphatidylinositol-anchor biosynthesis'.",
        piga_pathway_chunk,
    ))

    # ---- (5) PIGA disease association: PNH1 ---------------------------
    piga_disease_chunk = assert_supported(
        claim="PIGA is associated with Paroxysmal nocturnal hemoglobinuria 1 (PNH1).",
        chunks=all_chunks,
        hints=[
            "Paroxysmal nocturnal hemoglobinuria 1",
            "PNH1",
        ],
    )
    print(Cited(
        "PIGA's `disease` chunk associates it with Paroxysmal nocturnal "
        "hemoglobinuria 1 (PNH1).",
        piga_disease_chunk,
    ))

    # ---- (6) PNH disease description from chunk only ------------------
    piga_disease_desc_chunk = assert_supported(
        claim="The chunk describes PNH as hemolytic anemia with hemoglobinuria and thromboses.",
        chunks=all_chunks,
        hints=[
            "hemolytic anemia with hemoglobinuria",
        ],
    )
    print(Cited(
        "The same chunk describes PNH1 as 'hemolytic anemia with hemoglobinuria, "
        "thromboses in large vessels, and a deficiency in hematopoiesis'.",
        piga_disease_desc_chunk,
    ))

    # ---- (7) C5: precursor of C5a and C5b -----------------------------
    c5_prec_chunk = assert_supported(
        claim="C5 is the precursor of C5a anaphylatoxin and C5b.",
        chunks=all_chunks,
        hints=[
            "Precursor of the C5a anaphylatoxin and complement C5b",
        ],
    )
    print(Cited(
        "C5 is the precursor of the C5a anaphylatoxin and the C5b component "
        "of complement.",
        c5_prec_chunk,
    ))

    # ---- (8) C5 is a MAC component ------------------------------------
    c5_mac_chunk = assert_supported(
        claim="C5 is a component of the membrane attack complex.",
        chunks=all_chunks,
        hints=[
            "Component of the membrane attack complex (MAC)",
            "membrane attack complex",
        ],
    )
    print(Cited(
        "C5 is a component of the membrane attack complex (MAC), the pore-forming "
        "multiprotein complex of the complement cascade.",
        c5_mac_chunk,
    ))

    # ---- (9) C5 is activated downstream of all four complement arms ---
    c5_downstream_chunk = assert_supported(
        claim="C5 is activated downstream of classical, alternative, lectin and GZMK pathways.",
        chunks=all_chunks,
        hints=[
            "Activated downstream of classical, alternative, lectin and GZMK complement pathways",
            "downstream of classical, alternative",
        ],
    )
    print(Cited(
        "C5 is activated downstream of the classical, alternative, lectin and "
        "GZMK complement pathways — i.e. it sits at a convergent step.",
        c5_downstream_chunk,
    ))

    # ---- (10) CFB is the catalytic component of C3 convertase ---------
    cfb_c3conv_chunk = assert_supported(
        claim="CFB Bb is the catalytic component of the C3 convertase (C3bBb) of the alternative pathway.",
        chunks=all_chunks,
        hints=[
            "Catalytic component of the C3 convertase of the alternative complement pathway, also named C3bBb",
            "C3bBb",
        ],
    )
    print(Cited(
        "CFB Bb is the catalytic component of the alternative-pathway C3 "
        "convertase (C3bBb).",
        cfb_c3conv_chunk,
    ))

    # ---- (11) CFB is also catalytic component of C5 convertase --------
    cfb_c5conv_chunk = assert_supported(
        claim="CFB Bb is the catalytic component of the C5 convertase (C3bBb3b) of the alternative pathway.",
        chunks=all_chunks,
        hints=[
            "C5 convertase of the alternative complement pathway, also named C3bBb3b",
            "C3bBb3b",
        ],
    )
    print(Cited(
        "CFB Bb is also the catalytic component of the alternative-pathway C5 "
        "convertase (C3bBb3b).",
        cfb_c5conv_chunk,
    ))

    # ---- (12) Alternative pathway acts as an amplification loop -------
    cfb_amp_chunk = assert_supported(
        claim="The alternative complement pathway acts as an amplification loop.",
        chunks=all_chunks,
        hints=[
            "alternative complement pathway acts as an amplification loop",
            "amplification loop",
        ],
    )
    print(Cited(
        "The alternative complement pathway acts as an amplification loop that "
        "enhances the other complement pathways.",
        cfb_amp_chunk,
    ))

    # ---- (13) CFB is cleaved/activated by CFD -------------------------
    cfb_cfd_chunk = assert_supported(
        claim="CFB is cleaved and activated by CFD.",
        chunks=all_chunks,
        hints=[
            "CFB is cleaved and activated by CFD",
        ],
    )
    print(Cited(
        "CFB is cleaved and activated by CFD (complement factor D).",
        cfb_cfd_chunk,
    ))

    # ---- (14) CFB Peptidase S1 domain is annotated in the index -------
    # We make NO claim about whether this domain is the drug-binding site;
    # the chunks only state the protein contains it.
    cfb_pep_chunk = assert_supported(
        claim="CFB contains a Peptidase S1 domain.",
        chunks=all_chunks,
        hints=[
            "Peptidase S1",
        ],
    )
    print(Cited(
        "CFB's protein_summary chunk lists a Peptidase S1 domain (residues "
        "477-757) — the index records the domain's existence; it does NOT "
        "state which residues are drug-targeted.",
        cfb_pep_chunk,
    ))

    # ------------------------------------------------------------------
    # 8. What the chunks do NOT support — explicit deletions
    # ------------------------------------------------------------------
    print("\n\n## What the retrieved chunks do NOT support (explicit deletions)")
    print("-" * 70)

    # Helper: report which prior-version claims have no chunk evidence.
    unsupported_topics: list[tuple[str, list[str]]] = [
        (
            "CD55 / CD59 are GPI-anchored and absent from PNH red cells",
            ["CD55", "CD59", "decay-accelerating factor"],
        ),
        (
            "Clonal HSC selection / survival advantage in PNH",
            ["clone", "clonal", "hematopoietic stem cell", "survival advantage"],
        ),
        (
            "Intravascular vs extravascular hemolysis distinction",
            ["intravascular", "extravascular"],
        ),
        (
            "Eculizumab / Ravulizumab / Pegcetacoplan / Iptacopan / Danicopan "
            "as approved PNH drugs",
            [
                "eculizumab",
                "ravulizumab",
                "pegcetacoplan",
                "iptacopan",
                "danicopan",
            ],
        ),
        (
            "DrugBank IDs (DB01257 / DB14878 / DB16472 / DB16898) for PNH drugs",
            ["DB01257", "DB14878", "DB16472", "DB16898"],
        ),
        (
            "ChEMBL IDs (CHEMBL1201828 / CHEMBL4594388) for PNH drugs",
            ["CHEMBL1201828", "CHEMBL4594388"],
        ),
        (
            "PIGA mutations are somatic / loss-of-function",
            ["somatic", "loss-of-function", "loss of function"],
        ),
        (
            "Glycosyltransferases are 'structurally undruggable'",
            ["undruggable", "no catalytic pocket"],
        ),
    ]
    for topic, hints in unsupported_topics:
        hit = None
        for h in hints:
            hit = find_in_chunks(h, all_chunks)
            if hit is not None:
                break
        if hit is None:
            print(
                Cited(
                    f"No chunk supports: {topic}. "
                    f"(searched hints={hints!r})",
                    source=None,
                    label="TEXTBOOK_CONTEXT",
                )
            )
        else:
            # If a hint DOES appear somewhere we missed, surface it honestly
            # rather than claim deletion.
            print(
                Cited(
                    f"NOTE — at least one hint for '{topic}' was found in a "
                    f"chunk ({hit.gene}/{hit.chunk_type}); claim is partially "
                    "supported and was NOT auto-deleted.",
                    source=hit,
                )
            )

    # ------------------------------------------------------------------
    # 9. Closing remark — framing only
    # ------------------------------------------------------------------
    print("\n\n## Closing remark (framing, NOT a RAG conclusion)")
    print("-" * 70)
    print(
        Cited(
            "g2p-rag's PIGA / C5 / CFB chunks attest the gene functions, the "
            "pathway memberships, and the convertase architecture — but in this "
            "snapshot of the index they do NOT name approved PNH drugs or carry "
            "DrugBank/ChEMBL identifiers for them. Any 'drug the pathway, not "
            "the gene' conclusion therefore requires an external source on top "
            "of this retrieval.",
            source=None,
            label="TEXTBOOK_CONTEXT",
        )
    )

    print("\n" + "=" * 70)
    print("End of cookbook.")
    print("=" * 70)


if __name__ == "__main__":
    main()
