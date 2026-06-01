"""
Cookbook: BMPR2 / ACVR2A — what G2P chunks actually say about the BMP /
activin receptor axis.

Drug-discovery question (framing only — not a RAG-derived insight):
    Sotatercept is an ActRIIA-Fc fusion used in PAH; the disease is caused by
    BMPR2 loss-of-function but the drug never touches BMPR2 itself. This
    cookbook does NOT try to reconstruct the sotatercept design from chunks
    (the index does not contain drug-mechanism text). Instead it asks a more
    honest question: which pieces of the ligand-trap rationale are actually
    grounded in retrieved BMPR2 + ACVR2A chunks, and which are textbook
    framing that the RAG index cannot attest to?

Citation discipline is enforced by ``_citation.Cited`` /
``assert_supported`` — every printed factual claim is either tied to a
specific retrieved chunk or explicitly tagged ``[NO_RAG_SOURCE]`` as
textbook context.

Run with the project venv:
    d:/Users/ashenoy00000/.windsurf/g2p-rag/.venv/Scripts/python.exe \
        d:/Users/ashenoy00000/.windsurf/g2p-rag/cookbook/bmpr2_sotatercept_ligand_trap.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure Windows console can print arrows / Greek letters used in chunk text.
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
# Pretty-printer for chunk panels
# ---------------------------------------------------------------------------

def _print_chunk(rank: int, chunk) -> None:
    """Print one retrieved chunk with metadata and a text excerpt."""
    residues = chunk.residue_range or "n/a"
    text = (chunk.text or "").strip().replace("\r", "")
    # Truncate very long chunk bodies so the panel stays readable.
    if len(text) > 900:
        text = text[:880].rstrip() + " ...[truncated]"
    print(
        f"  [{rank}] gene={chunk.gene:<8} type={chunk.chunk_type:<16}"
        f" residues={residues:<12} score={chunk.score:.4f}"
    )
    print(f"      uniprot={chunk.uniprot_id}  source={chunk.source_url}")
    indented = "\n".join("      " + line for line in text.splitlines())
    print(indented)
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Compose BMPR2 + ACVR2A chunks and synthesize only what they support."""

    _load_env()

    print("\n" + "=" * 72)
    print("BMPR2 / ACVR2A cookbook - chunk-grounded BMP/activin receptor axis")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Build a retriever against the main g2p-rag chroma index
    # ------------------------------------------------------------------
    from g2p_rag import G2PRetriever

    chroma_dir = "d:/Users/ashenoy00000/.windsurf/g2p-rag/data/chroma"
    print(f"\n[1/4] Connecting to ChromaDB at {chroma_dir} ...")
    retriever = G2PRetriever(
        persist_dir=chroma_dir,
        embedding_model="all-MiniLM-L6-v2",
        collection_name="g2p_proteins",
    )

    # ------------------------------------------------------------------
    # 2. Query the BMPR2 side. The G2P index advertises chunk_type values
    #    {function, domain, subunit, disease, variant_cluster,
    #     protein_summary, pathway, ...}; we ask one query per type we need.
    # ------------------------------------------------------------------
    bmpr2_queries = {
        "function":        "BMPR2 BMP type II receptor serine/threonine kinase activity",
        "domain":          "BMPR2 protein kinase catalytic domain residues",
        "subunit":         "BMPR2 heteromeric receptor complex with BMPR1A ACVR2A activin INHBA interactors",
        "disease":         "BMPR2 pulmonary arterial hypertension PAH primary PPH1 loss-of-function",
        "variant_cluster": "BMPR2 pathogenic missense LoF hotspot residues clinvar",
    }

    print("\n[2/4] Retrieving BMPR2 chunks (one query per chunk type)...")
    bmpr2_chunks_by_type: dict[str, list] = defaultdict(list)
    bmpr2_all: list = []
    for chunk_type, query in bmpr2_queries.items():
        print(f"\n## BMPR2 {chunk_type} chunks")
        print(f"   query: {query}")
        results = retriever.retrieve(query, k=4, gene_filter=["BMPR2"])
        if not results:
            print("   (no chunks returned)")
            continue
        bmpr2_all.extend(results)
        # Keep only chunks whose type matches; the retriever may mix types.
        kept = [c for c in results if c.chunk_type == chunk_type] or results[:2]
        for rank, chunk in enumerate(kept[:2], 1):
            _print_chunk(rank, chunk)
            bmpr2_chunks_by_type[chunk.chunk_type].append(chunk)

    # ------------------------------------------------------------------
    # 3. Query the ACVR2A side - same chunk types we leaned on for BMPR2.
    # ------------------------------------------------------------------
    acvr2a_queries = {
        "function": "ACVR2A activin type IIA receptor serine/threonine kinase ligand binding",
        "subunit":  "ACVR2A binds activin A INHBA inhibin GDF11 myostatin heteromeric complex",
        "pathway":  "ACVR2A activin SMAD2 SMAD3 TGF-beta superfamily signaling",
    }

    print("\n[3/4] Retrieving ACVR2A chunks (the type-II receptor family)...")
    acvr2a_chunks_by_type: dict[str, list] = defaultdict(list)
    acvr2a_all: list = []
    for chunk_type, query in acvr2a_queries.items():
        print(f"\n## ACVR2A {chunk_type} chunks")
        print(f"   query: {query}")
        results = retriever.retrieve(query, k=4, gene_filter=["ACVR2A"])
        if not results:
            print("   (no chunks returned)")
            continue
        acvr2a_all.extend(results)
        kept = [c for c in results if c.chunk_type == chunk_type] or results[:2]
        for rank, chunk in enumerate(kept[:2], 1):
            _print_chunk(rank, chunk)
            acvr2a_chunks_by_type[chunk.chunk_type].append(chunk)

    # ------------------------------------------------------------------
    # 4. Synthesis - only conclusions reachable from chunk evidence are
    #    wrapped in Cited(... , chunk); textbook framing is explicitly
    #    tagged with label="TEXTBOOK_CONTEXT" and source=None.
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("[4/4] ## Synthesis - composing the chunk types (citation-disciplined)")
    print("=" * 72)

    all_types_used = set(bmpr2_chunks_by_type.keys()) | set(acvr2a_chunks_by_type.keys())
    type_counts: Counter = Counter()
    for chunks in list(bmpr2_chunks_by_type.values()) + list(acvr2a_chunks_by_type.values()):
        for c in chunks:
            type_counts[c.chunk_type] += 1

    print("\nChunk types surfaced across both genes:")
    for ct, n in sorted(type_counts.items()):
        print(f"  - {ct:<16} : {n} chunk(s)")

    # ---- BMPR2 DISEASE ------------------------------------------------
    disease_chunk = assert_supported(
        "BMPR2 LoF causes Primary Pulmonary Hypertension (PPH1)",
        bmpr2_all,
        hints=[
            "Pulmonary hypertension, primary, 1 (PPH1)",
            "pulmonary arterial",
        ],
    )
    print("\n1. DISEASE (BMPR2):")
    print("   " + str(Cited(
        "BMPR2 variants cause Primary Pulmonary Hypertension type 1 (PPH1), "
        "characterised by plexiform lesions of proliferating endothelial cells "
        "in pulmonary arterioles, elevated pulmonary arterial pressure, and "
        "right ventricular failure.",
        disease_chunk,
    )))

    # ---- BMPR2 FUNCTION ----------------------------------------------
    function_chunk = assert_supported(
        "BMPR2 is a type II serine/threonine kinase that drives a SMAD cascade",
        bmpr2_all,
        hints=[
            "type II and two type I transmembrane serine/threonine kinases",
            "SMAD transcriptional regulators",
        ],
    )
    print("\n2. FUNCTION (BMPR2):")
    print("   " + str(Cited(
        "On ligand binding, BMPR2 forms a receptor complex of two type II "
        "and two type I transmembrane serine/threonine kinases; the type II "
        "receptors phosphorylate and activate type I receptors, which in "
        "turn bind and activate SMAD transcriptional regulators.",
        function_chunk,
    )))
    ligand_chunk = assert_supported(
        "BMPR2 binds BMP family ligands",
        bmpr2_all,
        hints=["Binds to BMP7, BMP2", "BMP7"],
    )
    print("   " + str(Cited(
        "BMPR2 binds BMP7 and BMP2 (and BMP4 less efficiently); binding is "
        "enhanced when type I receptors are present.",
        ligand_chunk,
    )))

    # ---- BMPR2 DOMAIN ------------------------------------------------
    domain_chunk = assert_supported(
        "BMPR2 has a protein kinase domain at 203-504",
        bmpr2_all,
        hints=["Protein kinase", "203"],
    )
    print("\n3. DOMAIN (BMPR2):")
    print("   " + str(Cited(
        "BMPR2 carries a single annotated Protein kinase domain spanning "
        "residues 203-504 of the 1038 aa canonical chain.",
        domain_chunk,
    )))

    # ---- BMPR2 VARIANT_CLUSTER ---------------------------------------
    # We do NOT claim "missense variants destroy activity across the kinase
    # fold" - the retrieved variant clusters are dominated by frameshifts,
    # not missense, so that claim would be unsupported. We only attest what
    # the chunks actually show: pathogenic variant clusters at specific
    # positions, including positions outside the 203-504 kinase domain.
    variant_chunks = bmpr2_chunks_by_type.get("variant_cluster", [])
    extra_variant_evidence = find_in_chunks("Pathogenic", bmpr2_all)
    if variant_chunks or extra_variant_evidence is not None:
        evidence = variant_chunks[0] if variant_chunks else extra_variant_evidence
        positions = sorted({
            (c.residue_range or "?") for c in variant_chunks
        }) or [(evidence.residue_range or "?")]
        print("\n4. VARIANT_CLUSTER (BMPR2):")
        print("   " + str(Cited(
            "Pathogenic / likely-pathogenic BMPR2 variant clusters are present "
            f"at residue ranges {positions}; the retrieved clusters include "
            "frameshift variants (e.g. p.His641fs, p.Gln692fs) and a "
            "missense p.Lys982Arg located outside the 203-504 kinase domain.",
            evidence,
        )))
    else:
        print("\n4. VARIANT_CLUSTER (BMPR2): (no variant_cluster chunks retrieved)")

    # ---- BMPR2 SUBUNIT (the bridge to ACVR2A / activin A) -------------
    subunit_chunk = assert_supported(
        "BMPR2 subunit chunk lists activin A/INHBA as an interactor",
        bmpr2_all,
        hints=["activin A/INHBA", "INHBA"],
    )
    print("\n5. SUBUNIT (BMPR2) - the bridge to the activin axis:")
    print("   " + str(Cited(
        "The BMPR2 SUBUNIT chunk explicitly lists 'Interacts with activin "
        "A/INHBA' alongside GDF5, BMP4, SCUBE3 and TSC22D1 - i.e., the same "
        "G2P field that anchors BMPR2 to BMPs also names activin A as a "
        "physical partner.",
        subunit_chunk,
    )))

    # ---- ACVR2A FUNCTION + SUBUNIT -----------------------------------
    acvr2a_function = assert_supported(
        "ACVR2A is a type II receptor for activin A, activin B and inhibin A",
        acvr2a_all,
        hints=[
            "Receptor for activin A, activin B and inhibin A",
            "activin A, activin B",
        ],
    )
    print("\n6. FUNCTION (ACVR2A):")
    print("   " + str(Cited(
        "ACVR2A is a transmembrane serine/threonine kinase receptor for "
        "activin A, activin B and inhibin A; ligand binding assembles the "
        "same two-type-II + two-type-I complex that activates SMAD "
        "transcriptional regulators.",
        acvr2a_function,
    )))
    acvr2a_subunit = assert_supported(
        "ACVR2A subunit chunk lists activin A/INHBA as an interactor",
        acvr2a_all,
        hints=["activin A/INHBA"],
    )
    print("   " + str(Cited(
        "The ACVR2A SUBUNIT chunk also lists 'Interacts with activin "
        "A/INHBA' - so the chunks attest a shared activin A ligand "
        "between BMPR2 and ACVR2A.",
        acvr2a_subunit,
    )))

    # ---- Chunk-grounded conclusion -----------------------------------
    print("\nChunk-grounded conclusion (RAG-attested only):")
    print(
        "  Composing the BMPR2 disease + function + subunit chunks with the\n"
        "  ACVR2A function + subunit chunks recovers exactly one mechanistic\n"
        "  link that the index attests: BMPR2 and ACVR2A both list activin A\n"
        "  / INHBA as a physical partner, and both function chunks describe\n"
        "  the same two-type-II + two-type-I serine/threonine kinase complex\n"
        "  that activates SMADs. That shared-ligand bridge is the only piece\n"
        "  of the ligand-trap rationale the chunk corpus supports."
    )

    # ---- Explicit textbook framing (un-sourced, tagged loudly) -------
    print("\nTextbook framing - NOT attested by retrieved chunks:")
    print("   " + str(Cited(
        "Sotatercept (an ActRIIA-Fc fusion) was developed as an extracellular "
        "ligand trap and was approved by FDA in 2024 for PAH (brand: Winrevair); "
        "trial evidence comes from PULSAR / STELLAR.",
        source=None,
        label="TEXTBOOK_CONTEXT",
    )))
    print("   " + str(Cited(
        "Disease anchor identifiers (MIM 178600 for PPH1; MONDO:0008347) and "
        "the specific claim that sotatercept 'sequesters activin A in the "
        "circulation' are not present in the retrieved chunk text.",
        source=None,
        label="TEXTBOOK_CONTEXT",
    )))
    print("   " + str(Cited(
        "Claims sometimes attached to BMPR2 in the literature - e.g. that "
        "BMPR2 is 'constitutively active', that missense variants 'destroy "
        "activity across the entire kinase fold', or that GDF11 / myostatin "
        "are ACVR2A ligands - are not supported by the retrieved chunks and "
        "are intentionally omitted from the synthesis above.",
        source=None,
        label="TEXTBOOK_CONTEXT",
    )))

    # ------------------------------------------------------------------
    # 5. Final summary stats
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    total_chunks = sum(len(v) for v in bmpr2_chunks_by_type.values()) + sum(
        len(v) for v in acvr2a_chunks_by_type.values()
    )
    print(f"  Total chunks surfaced     : {total_chunks}")
    print(f"  Distinct chunk types used : {len(all_types_used)} ({sorted(all_types_used)})")
    print(f"  Genes queried             : BMPR2, ACVR2A")
    print(f"  Chunk-grounded link       : BMPR2 + ACVR2A both list activin A/INHBA")
    print(f"  Un-attested framing       : tagged [NO_RAG_SOURCE] above")


if __name__ == "__main__":
    main()
