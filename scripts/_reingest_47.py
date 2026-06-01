"""One-shot fresh re-ingest of all 47 benchmark genes (F1).

Invoke directly with the project venv::

    $env:PYTHONIOENCODING="utf-8"
    d:/Users/ashenoy00000/.windsurf/g2p-rag/.venv/Scripts/python.exe `
        d:/Users/ashenoy00000/.windsurf/g2p-rag/scripts/_reingest_47.py

The gene list is duplicated from cookbook/rare_variant_druggability_leaderboard.py
on purpose — that file is the canonical source of "what's in the v0.1.2 index"
and this script needs to stay byte-exact with it for the manifest to line up.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from g2p_rag import fetch, chunk, retrieve  # noqa: E402

INDEXED_GENES: tuple[str, ...] = (
    "ACVR2A", "AKT1", "ALAS1", "APOE", "APP", "BCL11A", "BDKRB2", "BMPR2",
    "BRAF", "C5", "CALCA", "CALCRL", "CFB", "CFTR", "CHRM4", "CRHR1",
    "CXCR4", "CYP21A2", "DMD", "EDN1", "EDNRA", "EGFR", "ERBB2", "F12",
    "GLA", "GLP1R", "HBB", "HMBS", "HTT", "IDH1", "IL13", "KLKB1", "LDLR",
    "MC4R", "MUC1", "PCSK9", "PIGA", "PIK3CA", "POMC", "SERPING1", "SMN1",
    "SMN2", "SOD1", "THRB", "TMED9", "TNF", "TP53", "UMOD",
)


def main() -> int:
    print(f"Re-ingesting {len(INDEXED_GENES)} benchmark genes (fresh, no cache).")
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir = data_dir / "chroma"

    print(f"  data_dir   = {data_dir}")
    print(f"  chroma_dir = {chroma_dir}")
    print()
    print("Step 1/3: fetching from G2P + UniProt + ClinVar...")
    gene_data = fetch.fetch_all_genes(
        list(INDEXED_GENES), data_dir,
        force_refetch=True,  # fresh — no stale 404s
        write_snapshots=True,
    )
    fetched = [g for g, d in gene_data.items() if d.get("structure")]
    print(f"  fetched {len(fetched)}/{len(INDEXED_GENES)} genes.")
    print()
    print("Step 2/3: chunking...")
    chunks = chunk.chunk_all(gene_data)
    print(f"  produced {len(chunks)} chunks.")
    print()
    print("Step 3/3: building ChromaDB index + BM25...")
    embedder = retrieve.load_embedder("sentence-transformers/all-MiniLM-L6-v2")
    retrieve.build_index(chunks, chroma_dir, embedder)
    print()

    # Verify the manifest
    stats = retrieve.index_stats(chroma_dir)
    print("Index manifest:")
    for key in (
        "g2p_rag_version", "build_utc", "gene_count", "embedding_model",
        "g2p_api_base", "build_commit", "total_chunks",
    ):
        print(f"  {key}: {stats.get(key)}")
    print(f"  chunk_types: {stats.get('chunk_types')}")
    print(f"  source_apis: {stats.get('source_apis')}")
    print()
    print(
        "If chunk_types includes pdb_structures / cross_references / gencc_diseases "
        "(via 'structures' + 'cross_references' + 'diseases') the v0.1.2 chunker is wired in."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
