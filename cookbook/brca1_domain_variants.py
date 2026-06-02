"""
Cookbook: BRCA1 — Correlating ClinVar Pathogenic Variants with Protein Domains

Demonstrates:
  1. Fetching BRCA1 structure + ClinVar data
  2. Chunking into domain/variant-cluster/summary chunks
  3. Building a hybrid retrieval index
  4. Querying: which BRCA1 domains are most affected by pathogenic variants?
  5. Generating an LLM-paraphrased summary grounded in retrieved chunks
     (wrapped in ``if is_llm_available()`` so the script still produces useful
     output when no LLM backend is reachable).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Allow running directly from the cookbook/ directory without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _citation import print_index_manifest  # noqa: E402
from _llm import get_llm, is_llm_available  # noqa: E402


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


def _format_context(results) -> str:
    """Render retrieval results as a labelled context block for the LLM prompt."""
    sections = []
    for i, r in enumerate(results, 1):
        header = (
            f"--- Context {i} "
            f"[Gene:{r.chunk.gene}"
            f"|UniProt:{r.chunk.uniprot_id}"
            f"|{r.chunk.chunk_type}"
            f"|Residues:{r.chunk.residue_start}-{r.chunk.residue_end}] ---"
        )
        sections.append(f"{header}\n{r.chunk.text}")
    return "\n\n".join(sections)


def _citation_tokens(results) -> list:
    """Build canonical [Gene:UniProt:ChunkType:Residues] tokens for results."""
    tokens = []
    for r in results:
        tok = (
            f"{r.chunk.gene}:{r.chunk.uniprot_id}:"
            f"{r.chunk.chunk_type}:"
            f"{r.chunk.residue_start}-{r.chunk.residue_end}"
        )
        tokens.append(tok)
    return tokens


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the BRCA1 domain-variant cookbook end-to-end."""

    # ------------------------------------------------------------------
    # 1. Setup: data directory + environment variables
    # ------------------------------------------------------------------
    _load_env()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    chroma_dir = data_dir / "chroma_brca1_cookbook"  # isolated from main index
    data_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("BRCA1 Domain-Variant Cookbook")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 2. Fetch BRCA1 data (structure map + ClinVar variants)
    # ------------------------------------------------------------------
    from g2p_rag.fetch import fetch_all_genes

    print("\n[1/5] Fetching BRCA1 data from G2P portal and ClinVar...")
    gene_data = fetch_all_genes(["BRCA1"], cache_dir=data_dir)

    brca1_entry = gene_data["BRCA1"]
    structure = brca1_entry["structure"]
    variants = brca1_entry["variants"]

    # ------------------------------------------------------------------
    # 3. Print a quick data summary
    # ------------------------------------------------------------------
    features = structure.features
    n_domains = len(getattr(features, "domains", []))
    n_variants = len(variants)

    print(f"\n  Gene:             {structure.gene_symbol}")
    print(f"  UniProt ID:       {structure.uniprot_id or '(not resolved)'}")
    print(f"  Sequence length:  {structure.length} aa")
    print(f"  Annotated domains:{n_domains}")
    print(f"  ClinVar variants: {n_variants}")

    # ------------------------------------------------------------------
    # 4. Chunk the data
    # ------------------------------------------------------------------
    from g2p_rag.chunk import ProteinChunker

    print("\n[2/5] Chunking BRCA1 data...")
    chunker = ProteinChunker()
    chunks = chunker.chunk_gene(structure, variants)

    # Count chunks by type
    type_counts: Counter = Counter(c.chunk_type for c in chunks)
    print(f"  Total chunks: {len(chunks)}")
    for chunk_type, count in sorted(type_counts.items()):
        print(f"    {chunk_type:<20}: {count}")

    # ------------------------------------------------------------------
    # 5. Build hybrid retrieval index (ChromaDB + BM25)
    # ------------------------------------------------------------------
    from g2p_rag.embed import get_embedder
    from g2p_rag.retrieve import build_index

    print("\n[3/5] Building hybrid retrieval index...")
    embedder = get_embedder("sentence-transformers/all-MiniLM-L6-v2")
    retriever = build_index(
        chunks=chunks,
        persist_dir=chroma_dir,
        embedder=embedder,
        collection_name="brca1_cookbook",
    )
    print_index_manifest(retriever)
    print(f"  Index ready - {len(chunks)} chunks embedded and stored.")

    # ------------------------------------------------------------------
    # 6. Build LLM via the shared _llm fallback adapter
    # ------------------------------------------------------------------
    print("\n[4/5] Initialising LLM (Anthropic -> Llama -> NoOp fallback)...")
    llm = get_llm()
    llm_backend = getattr(llm, "backend", "noop")
    print(f"  LLM backend selected: {llm_backend}")

    # ------------------------------------------------------------------
    # 7. Run three queries
    # ------------------------------------------------------------------
    queries = [
        "What domains in BRCA1 are most affected by pathogenic missense variants?",
        "What PTM sites are near the BRCA1 RING domain?",
        "Are there druggable pockets in BRCA1 that overlap with pathogenic variant clusters?",
    ]

    print("\n[5/5] Running queries...\n")
    print("=" * 70)

    all_retrieved_chunks: list = []
    retrieved_chunk_types: set = set()

    for query_idx, query in enumerate(queries, 1):
        print(f"\nQuery {query_idx}: {query}")
        print("-" * 70)

        # Retrieve top-3 results - pure retrieval, always runs
        results = retriever.search(query, k=3)

        print(f"  Top-3 retrieved chunks:")
        for rank, r in enumerate(results, 1):
            print(
                f"    [{rank}] gene={r.chunk.gene}"
                f"  type={r.chunk.chunk_type}"
                f"  residues={r.chunk.residue_start}-{r.chunk.residue_end}"
                f"  score={r.score:.4f}"
            )
            all_retrieved_chunks.append(r.chunk)
            retrieved_chunk_types.add(r.chunk.chunk_type)

        # Citation-helper section - always runs
        cite_tokens = _citation_tokens(results)
        print(f"\n  Citation tokens: {cite_tokens}")

        # LLM-dependent synthesis - only when a real LLM is reachable
        if is_llm_available():
            context = _format_context(results)
            prompt = (
                "You are a precise genomics assistant. Answer ONLY from the context, "
                "citing each claim with [Gene:UniProt:ChunkType:Residues].\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )
            print(f"\n  Generating LLM-paraphrased answer...")
            answer = llm(prompt, max_tokens=512)
            print(f"\n  Answer:\n{answer}")
        else:
            print(
                "\n  (LLM synthesis skipped - no Anthropic key and no reachable Llama "
                "model. Retrieval results above are the cookbook output.)"
            )

        print("\n" + "=" * 70)

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print(f"\nSummary:")
    print(
        f"  Retrieved {len(all_retrieved_chunks)} chunks across "
        f"{len(retrieved_chunk_types)} chunk type(s): "
        f"{', '.join(sorted(retrieved_chunk_types))}."
    )
    print(f"  LLM backend used: {llm_backend}")


if __name__ == "__main__":
    main()
