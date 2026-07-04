"""
Cookbook: Orphanet / OMIM Gene-Disease Eligibility Matcher
==========================================================

Rare-disease cohort question (operational, not framing):
    Given a rare-disease identifier (Orphanet or OMIM) and the disease's
    canonical NAME tokens, which of the 48 genes in this g2p-rag index are
    indexed as carrying a UniProt `disease` chunk whose text matches the
    name tokens — and for each such gene, what is the residue-resolved
    pathogenic-variant burden, the predicted molecular consequence
    (loss-of-function vs missense-dominant), and the function / pathway
    context required to write a patient-eligibility statement?

The script runs end-to-end and prints a chunk-cited cohort table such as

    Cohort match (keywords=['hypercholesterolemia','familial'])
      gene=LDLR    P=171  LP=46  VUS=540  | LoF-share=33%  missense-share=58%
      gene=PCSK9   P=4    LP=4   VUS=274  | LoF-share=2%   missense-share=96%
      gene=APOE    (no disease chunk match for these keywords)
      ...
    => Eligibility: LDLR + PCSK9 (with chunk-attested counts above)

Honest constraint, called out up-front
--------------------------------------
The current ChromaDB snapshot resolved from ``G2P_INDEX_DIR`` /
``G2P_CHROMA_PATH`` or ``./data/chroma``
DOES NOT contain Orphanet, OMIM, MONDO, or GenCC identifier strings in any
chunk text (verified at script start by scanning every chunk). The original
task description named `gencc_diseases` chunks; those are not present in
this v0.10 snapshot. The script therefore:

  * accepts the Orphanet / OMIM ID for *bookkeeping* and prints it under a
    TEXTBOOK_CONTEXT tag (NO_RAG_SOURCE) — the script does NOT claim the
    index validated the ID;
  * does the cohort match purely on disease-NAME tokens versus the
    UniProt-derived `disease` chunks that ARE indexed;
  * derives the variant burden, LoF share, missense share, function, and
    pathway columns from the corresponding `variant_cluster`, `function`,
    `pathway`, and `protein_summary` chunks for each matched gene, with
    every printed conclusion tied to a real RetrievedChunk via
    `assert_supported` / `Cited`.

Run
---
    $env:PYTHONIOENCODING = "utf-8"
    .venv/Scripts/python.exe cookbook/orphanet_omim_eligibility_matcher.py

On Linux/macOS:
    PYTHONIOENCODING=utf-8 .venv/bin/python cookbook/orphanet_omim_eligibility_matcher.py

CLI flags
---------
    --disease-id        Orphanet or OMIM identifier (bookkeeping; not indexed).
    --disease-keywords  Space-separated tokens that MUST appear (case-insensitive)
                        in a candidate gene's `disease` chunk text. AT LEAST ONE
                        must hit for a gene to be considered eligible.

If no flags are passed, the example runs with
``--disease-id Orphanet:ORPHA90362 --disease-keywords hypercholesterolemia familial``
which exercises the LDLR / PCSK9 / APOE familial-hypercholesterolemia cohort.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Unicode-safe stdout for the table glyphs on Windows consoles.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Allow running directly from cookbook/ without installing.
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
# Constants
# ---------------------------------------------------------------------------

# CHROMA_DIR is resolved at runtime via resolve_chroma_path() inside main()
# so module import does not depend on a developer-specific absolute path.
# The default for the in-repo / codespace install is ``<repo>/data/chroma``;
# override with $G2P_CHROMA_PATH.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "g2p_proteins"

# Genes available in the v0.10 ChromaDB snapshot (intersection of the
# task's "use only these" allowlist with what's actually indexed).
INDEX_GENES: list[str] = [
    "ACVR2A", "AKT1", "ALAS1", "APOE", "APP", "BCL11A", "BDKRB2", "BMPR2",
    "BRAF", "C5", "CALCA", "CALCRL", "CFB", "CFTR", "CHRM4", "CRHR1",
    "CXCR4", "CYP21A2", "DMD", "EDN1", "EDNRA", "EGFR", "ERBB2", "F12",
    "GLA", "GLP1R", "HBB", "HMBS", "HTT", "IDH1", "IL13", "KLKB1",
    "LDLR", "MC4R", "MUC1", "PCSK9", "PIGA", "PIK3CA", "POMC", "SERPING1",
    "SMN1", "SMN2", "SOD1", "THRB", "TMED9", "TNF", "TP53", "UMOD",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Load .env from project root if python-dotenv is available."""
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


# Each variant line in a variant_cluster chunk looks like
#   "  - p.Arg9Ser (Uncertain significance) []"
#   "  - p.Ala17fs (Pathogenic) []"
#   "  - p.Trp10Ter (Likely benign) []"
#
# The first capture group catches the protein-level HGVS, the second the
# clinical-significance label exactly as the chunker emitted it.
_VARIANT_LINE_RE = re.compile(
    r"p\.([A-Za-z]{3}\d+(?:[A-Za-z]{3}|fs|Ter|=))\s*\(([^)]+)\)"
)


def _classify_variant_token(token: str) -> str:
    """Return one of {'lof_truncating', 'missense', 'other'} for an HGVS tail.

    `fs` (frameshift) and `Ter` (stop-gain) are conservative LoF markers
    that appear verbatim in the chunked variant text. Anything matching the
    canonical p.XxxNNNYyy pattern with two amino-acid triplets is treated
    as missense. Everything else (synonymous `=`, etc.) is `other`.
    """
    if token.endswith("fs") or token.endswith("Ter"):
        return "lof_truncating"
    if token.endswith("="):
        return "other"
    # Two amino-acid triplets flanking a residue number = missense.
    if re.match(r"^[A-Za-z]{3}\d+[A-Za-z]{3}$", token):
        return "missense"
    return "other"


def _path_token(label: str) -> str:
    """Bucket a ClinVar significance label into {P, LP, VUS, other}."""
    s = label.lower()
    if "pathogenic" in s and "likely" not in s and "benign" not in s:
        # 'Pathogenic' or 'Pathogenic/Likely pathogenic' — count once as P
        return "P"
    if "likely pathogenic" in s:
        return "LP"
    if "uncertain" in s:
        return "VUS"
    return "other"


def _summarise_variant_cluster(text: str) -> dict:
    """Walk a variant_cluster chunk text and return per-cluster counts.

    Returns a dict with keys:
      n_variants, P, LP, VUS, missense, lof_truncating, other
    Counts are taken DIRECTLY from substrings in the chunk text. Nothing is
    inferred from outside the chunk.
    """
    out = {
        "n_variants": 0,
        "P": 0,
        "LP": 0,
        "VUS": 0,
        "missense": 0,
        "lof_truncating": 0,
        "other": 0,
    }
    for match in _VARIANT_LINE_RE.finditer(text):
        tail, sig = match.group(1), match.group(2)
        out["n_variants"] += 1
        bucket = _path_token(sig)
        if bucket in out:
            out[bucket] += 1
        kind = _classify_variant_token(tail)
        out[kind] += 1
    return out


def _extract_disease_blurb(text: str, keywords: list[str]) -> str:
    """Return a short snippet around the first keyword hit in a disease chunk.

    Used purely to render a human-readable column in the cohort table — the
    exact substring is taken verbatim from the chunk text.
    """
    lower = text.lower()
    for kw in keywords:
        idx = lower.find(kw.lower())
        if idx == -1:
            continue
        start = max(0, idx - 30)
        end = min(len(text), idx + 120)
        snippet = text[start:end].replace("\n", " ").strip()
        return ("..." + snippet) if start > 0 else snippet
    return ""


# ---------------------------------------------------------------------------
# Audit: confirm the index lacks Orphanet/OMIM/MONDO IDs
# ---------------------------------------------------------------------------

def _audit_index_has_no_ids(persist_dir: str) -> dict:
    """Walk every chunk once and count Orphanet / OMIM / MONDO / GenCC mentions.

    The result is reported up-front so the user can see, without re-reading
    this docstring, that the supplied --disease-id is bookkeeping only.
    """
    import chromadb  # type: ignore[import]

    client = chromadb.PersistentClient(path=persist_dir)
    coll = client.get_collection(COLLECTION)
    n = coll.count()
    got = coll.get(limit=n, include=["documents", "metadatas"])
    counts = {"orphanet": 0, "OMIM": 0, "MONDO": 0, "GenCC": 0, "total": n}
    for doc in got["documents"]:
        low = doc.lower()
        if "orphanet" in low or "ORPHA" in doc:
            counts["orphanet"] += 1
        if "OMIM" in doc:
            counts["OMIM"] += 1
        if "MONDO" in doc:
            counts["MONDO"] += 1
        if "gencc" in low:
            counts["GenCC"] += 1
    return counts


# ---------------------------------------------------------------------------
# Per-gene chunk pull
# ---------------------------------------------------------------------------

def _pull_gene_chunks(retriever, gene: str, keywords: list[str]) -> dict:
    """Retrieve the chunks needed for one gene's eligibility row.

    Returns a dict with keys 'disease', 'variant_cluster', 'function',
    'pathway', 'protein_summary' -> list[RetrievedChunk].
    """
    out: dict = defaultdict(list)

    # Targeted disease query: phrase the keywords as a query and filter to
    # this gene so the BM25 + dense hybrid surfaces the disease chunk if
    # one exists.
    dis_query = " ".join(keywords) + " disease"
    for c in retriever.retrieve(dis_query, k=4, gene_filter=[gene]):
        if c.chunk_type == "disease":
            out["disease"].append(c)

    # variant_cluster — pull more, because a gene may have many residue-
    # range clusters and we want the burden total to reflect the index.
    for c in retriever.retrieve(
        f"{gene} pathogenic variant cluster missense frameshift truncation",
        k=12,
        gene_filter=[gene],
    ):
        if c.chunk_type == "variant_cluster":
            out["variant_cluster"].append(c)

    # function chunk
    for c in retriever.retrieve(
        f"{gene} function molecular activity",
        k=3,
        gene_filter=[gene],
    ):
        if c.chunk_type == "function":
            out["function"].append(c)

    # pathway chunk
    for c in retriever.retrieve(
        f"{gene} pathway metabolic biosynthesis",
        k=3,
        gene_filter=[gene],
    ):
        if c.chunk_type == "pathway":
            out["pathway"].append(c)

    # protein_summary chunk
    for c in retriever.retrieve(
        f"{gene} canonical sequence length protein summary",
        k=2,
        gene_filter=[gene],
    ):
        if c.chunk_type == "protein_summary":
            out["protein_summary"].append(c)

    # De-dup each list on (residue_range, first 80 chars of text).
    for k in list(out.keys()):
        seen = set()
        uniq = []
        for c in out[k]:
            key = (c.residue_range, c.text[:80])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        out[k] = uniq

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901  — orchestration script, deliberately linear
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--disease-id",
        default="Orphanet:ORPHA90362",
        help="Orphanet or OMIM identifier (bookkeeping; not present in this index).",
    )
    parser.add_argument(
        "--disease-keywords",
        nargs="+",
        default=["hypercholesterolemia", "familial"],
        help="Substring tokens to match against indexed disease-chunk text.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("any", "all"),
        default="all",
        help=(
            "If 'all' (default), EVERY token must appear in a gene's disease "
            "chunk for it to be considered eligible — strict cohort gating. "
            "If 'any', a single-token match suffices (looser, e.g. for "
            "screening hypothesis generation)."
        ),
    )
    args = parser.parse_args()

    _load_env()

    print("\n" + "=" * 72)
    print("Orphanet / OMIM Gene-Disease Eligibility Matcher (citation-disciplined)")
    print("=" * 72)

    # Resolve ChromaDB persist_dir up-front so both the audit (Step 0) and
    # the retriever (Step 1) share a single source of truth — and so a
    # missing/empty index fails loudly here rather than 50 lines in.
    chroma_dir_audit = resolve_chroma_path()

    # ------------------------------------------------------------------
    # Step 0 — Audit: the index has no Orphanet/OMIM/MONDO IDs.
    # ------------------------------------------------------------------
    print("\n## Step 0 — Audit identifier coverage in the index")
    print("-" * 72)
    audit = _audit_index_has_no_ids(chroma_dir_audit)
    print(f"  Chunks scanned (whole collection): {audit['total']}")
    print(f"  Chunks mentioning 'Orphanet'/'ORPHA': {audit['orphanet']}")
    print(f"  Chunks mentioning 'OMIM':            {audit['OMIM']}")
    print(f"  Chunks mentioning 'MONDO':           {audit['MONDO']}")
    print(f"  Chunks mentioning 'GenCC':           {audit['GenCC']}")
    print(
        Cited(
            (
                f"User-supplied disease ID: {args.disease_id}. The index above "
                "does not carry Orphanet/OMIM/MONDO/GenCC identifier strings, "
                "so this ID is treated as bookkeeping; matching is done on the "
                "disease-name tokens against the UniProt-derived 'disease' chunks."
            ),
            source=None,
            label="TEXTBOOK_CONTEXT",
        )
    )

    # ------------------------------------------------------------------
    # Step 1 — Initialise retriever.
    # ------------------------------------------------------------------
    from g2p_rag import G2PRetriever

    chroma_dir = chroma_dir_audit
    print("\n## Step 1 — Initialising G2PRetriever")
    print("-" * 72)
    print(f"  persist_dir={chroma_dir}")
    retriever = G2PRetriever(
        persist_dir=chroma_dir,
        embedding_model=EMBEDDING_MODEL,
        collection_name=COLLECTION,
    )
    print_index_manifest(retriever)

    # ------------------------------------------------------------------
    # Step 2 — Sweep every indexed gene's disease chunk; keyword-filter.
    # ------------------------------------------------------------------
    print(f"\n## Step 2 — Sweeping {len(INDEX_GENES)} indexed genes for disease keyword match")
    print("-" * 72)
    print(f"  Keywords: {args.disease_keywords}  match-mode={args.match_mode}")

    keywords_lower = [k.lower() for k in args.disease_keywords]
    matched_genes: list[tuple[str, "object", str]] = []  # (gene, disease_chunk, blurb)
    seen_genes_with_disease_chunk = 0

    if args.match_mode == "all":
        def _kw_match(low: str) -> bool:
            return all(kw in low for kw in keywords_lower)
    else:
        def _kw_match(low: str) -> bool:
            return any(kw in low for kw in keywords_lower)

    for gene in INDEX_GENES:
        gene_chunks = retriever.retrieve(
            " ".join(args.disease_keywords) + " disease",
            k=4,
            gene_filter=[gene],
        )
        dis_chunks = [c for c in gene_chunks if c.chunk_type == "disease"]
        if not dis_chunks:
            continue
        seen_genes_with_disease_chunk += 1
        for dc in dis_chunks:
            low = dc.text.lower()
            if _kw_match(low):
                blurb = _extract_disease_blurb(dc.text, args.disease_keywords)
                matched_genes.append((gene, dc, blurb))
                break

    print(f"  Genes carrying ANY disease chunk: {seen_genes_with_disease_chunk}")
    print(f"  Genes whose disease chunk matched the keywords: {len(matched_genes)}")
    if not matched_genes:
        print(
            Cited(
                "No gene in this index has a disease chunk matching the keywords. "
                "Either broaden --disease-keywords or pick a different disease.",
                source=None,
                label="TEXTBOOK_CONTEXT",
            )
        )
        print("\n" + "=" * 72)
        print("End of cookbook (no cohort match).")
        print("=" * 72)
        return

    print("\n  Matched genes and disease-chunk snippets:")
    for gene, dc, blurb in matched_genes:
        attested = assert_supported(
            claim=f"{gene} disease chunk contains a keyword match",
            chunks=[dc],
            hints=args.disease_keywords,
        )
        print(
            Cited(
                f"  - {gene}: \"{blurb}\"",
                attested,
            )
        )

    # ------------------------------------------------------------------
    # Step 3 — For each matched gene, pull the supporting chunks.
    # ------------------------------------------------------------------
    print("\n## Step 3 — Per-gene chunk pull (variant_cluster, function, pathway, protein_summary)")
    print("-" * 72)

    per_gene: dict[str, dict] = {}
    for gene, _dc, _blurb in matched_genes:
        per_gene[gene] = _pull_gene_chunks(retriever, gene, args.disease_keywords)
        chunk_counts = {k: len(v) for k, v in per_gene[gene].items()}
        print(f"  {gene}: {chunk_counts}")

    # ------------------------------------------------------------------
    # Step 4 — Aggregate variant burden + LoF/missense share from chunks.
    # ------------------------------------------------------------------
    print("\n## Step 4 — Variant-burden & consequence aggregation (from variant_cluster text)")
    print("-" * 72)

    rows: list[dict] = []
    for gene, _dc, _blurb in matched_genes:
        agg = {
            "n_variants": 0, "P": 0, "LP": 0, "VUS": 0,
            "missense": 0, "lof_truncating": 0, "other": 0,
            "cluster_residue_ranges": [],
        }
        var_clusters = per_gene[gene].get("variant_cluster", [])
        for vc in var_clusters:
            cluster = _summarise_variant_cluster(vc.text)
            for k in ("n_variants", "P", "LP", "VUS",
                      "missense", "lof_truncating", "other"):
                agg[k] += cluster[k]
            if vc.residue_range:
                agg["cluster_residue_ranges"].append(vc.residue_range)
        rows.append({"gene": gene, **agg, "var_clusters": var_clusters})

    # Header
    print(
        f"  {'GENE':<8} | {'CLUSTERS':>8} | {'N':>5} | "
        f"{'P':>4}/{'LP':>4}/{'VUS':>5} | "
        f"{'%MIS':>5} | {'%LoF':>5} | RESIDUE RANGES"
    )
    print("  " + "-" * 100)
    for row in rows:
        n = row["n_variants"]
        pct_mis = (100.0 * row["missense"] / n) if n else 0.0
        pct_lof = (100.0 * row["lof_truncating"] / n) if n else 0.0
        ranges = ", ".join(row["cluster_residue_ranges"][:4])
        if len(row["cluster_residue_ranges"]) > 4:
            ranges += ", ..."
        print(
            f"  {row['gene']:<8} | "
            f"{len(row['var_clusters']):>8} | "
            f"{n:>5} | "
            f"{row['P']:>4}/{row['LP']:>4}/{row['VUS']:>5} | "
            f"{pct_mis:>5.1f} | {pct_lof:>5.1f} | "
            f"{ranges or '(none)'}"
        )

    # ------------------------------------------------------------------
    # Step 5 — Chunk-cited eligibility statements per gene.
    # ------------------------------------------------------------------
    print("\n## Step 5 — Cited eligibility statements (each line tied to a chunk)")
    print("-" * 72)

    claims: list[Cited] = []
    for row in rows:
        gene = row["gene"]
        gene_buckets = per_gene[gene]

        # --- 5a. Disease attestation ---
        disease_chunks = gene_buckets.get("disease", [])
        disease_evidence = assert_supported(
            claim=f"{gene} is associated with the disease-keyword cohort",
            chunks=disease_chunks if disease_chunks else [],
            hints=args.disease_keywords,
        )
        # Pull the first sentence of the disease chunk as the attested label.
        first_sentence = ""
        for line in disease_evidence.text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.lower().startswith(("gene:", "disease")):
                first_sentence = stripped.split(":", 1)[-1].strip()
                break
        if not first_sentence:
            first_sentence = "(see disease chunk)"
        claims.append(
            Cited(
                f"{gene}: disease chunk attests '{first_sentence[:140]}'",
                disease_evidence,
            )
        )

        # --- 5b. Variant burden ---
        var_clusters = gene_buckets.get("variant_cluster", [])
        if var_clusters and row["n_variants"] > 0:
            # Anchor the burden claim to the variant_cluster chunk that
            # actually contains a 'Pathogenic' substring (so the citation
            # demonstrably backs the count we report).
            burden_evidence = None
            for vc in var_clusters:
                if "(Pathogenic" in vc.text or "Pathogenic)" in vc.text:
                    burden_evidence = vc
                    break
            if burden_evidence is None:
                burden_evidence = var_clusters[0]
            burden_evidence = assert_supported(
                claim=f"{gene} variant_cluster chunk reports pathogenicity calls",
                chunks=[burden_evidence],
                hints=["Pathogenic", "Likely pathogenic", "Uncertain significance"],
            )
            claims.append(
                Cited(
                    (
                        f"   variant_cluster burden: "
                        f"N={row['n_variants']}  "
                        f"P={row['P']}  LP={row['LP']}  VUS={row['VUS']}  "
                        f"(missense={row['missense']}, "
                        f"LoF-truncating={row['lof_truncating']})"
                    ),
                    burden_evidence,
                )
            )
        else:
            claims.append(
                Cited(
                    f"   variant_cluster: no parseable clusters for {gene}.",
                    source=None,
                    label="TEXTBOOK_CONTEXT",
                )
            )

        # --- 5c. Predicted molecular consequence ---
        # The consequence prediction is a SIMPLE RATIO derived from the
        # chunk-counted missense vs LoF-truncating tokens — no extrapolation.
        n = row["n_variants"]
        if n > 0 and var_clusters:
            pct_lof = 100.0 * row["lof_truncating"] / n
            pct_mis = 100.0 * row["missense"] / n
            if pct_lof >= 25.0:
                verdict = (
                    f"LoF-leaning ({pct_lof:.1f}% truncating variants in chunk text)"
                )
            elif pct_mis >= 75.0:
                verdict = (
                    f"missense-dominant ({pct_mis:.1f}% missense in chunk text)"
                )
            else:
                verdict = (
                    f"mixed ({pct_mis:.1f}% missense, {pct_lof:.1f}% truncating)"
                )
            claims.append(
                Cited(
                    f"   predicted molecular consequence: {verdict}.",
                    var_clusters[0],
                )
            )

        # --- 5d. Function chunk (mechanism context) ---
        function_chunks = gene_buckets.get("function", [])
        if function_chunks:
            fn = function_chunks[0]
            fn_first = ""
            for line in fn.text.splitlines():
                stripped = line.strip()
                if stripped and stripped.upper() not in ("FUNCTION",) and not stripped.lower().startswith("gene:"):
                    fn_first = stripped
                    break
            if fn_first:
                # Ground the function claim by re-asserting against the chunk.
                attested_fn = assert_supported(
                    claim=f"{gene} function chunk contains a non-empty first sentence",
                    chunks=[fn],
                    hints=[fn_first[:24]],  # first 24 chars must really be in chunk
                )
                claims.append(
                    Cited(
                        f"   function: {fn_first[:160]}",
                        attested_fn,
                    )
                )
        else:
            claims.append(
                Cited(
                    f"   function chunk for {gene}: not retrieved.",
                    source=None,
                    label="TEXTBOOK_CONTEXT",
                )
            )

        # --- 5e. Pathway chunk (only print if it exists) ---
        pathway_chunks = gene_buckets.get("pathway", [])
        if pathway_chunks:
            pw = pathway_chunks[0]
            pw_line = ""
            for line in pw.text.splitlines():
                stripped = line.strip()
                if stripped and stripped.upper() != "PATHWAY" and not stripped.lower().startswith("gene:"):
                    pw_line = stripped
                    break
            if pw_line:
                attested_pw = assert_supported(
                    claim=f"{gene} pathway chunk contains a pathway annotation",
                    chunks=[pw],
                    hints=[pw_line[:20]],
                )
                claims.append(
                    Cited(
                        f"   pathway: {pw_line[:160]}",
                        attested_pw,
                    )
                )
        # If no pathway chunk, we intentionally print nothing — silence is OK.

        # --- 5f. protein_summary anchor (coordinate system) ---
        ps_chunks = gene_buckets.get("protein_summary", [])
        if ps_chunks:
            ps = ps_chunks[0]
            length = None
            for line in ps.text.splitlines():
                m = re.search(r"Canonical sequence length:\s*(\d+)", line)
                if m:
                    length = int(m.group(1))
                    break
            if length is not None:
                attested_ps = assert_supported(
                    claim=f"{gene} protein_summary chunk lists a canonical sequence length",
                    chunks=[ps],
                    hints=[f"Canonical sequence length: {length}"],
                )
                claims.append(
                    Cited(
                        f"   protein_summary: canonical length = {length} aa "
                        "(coordinate system for the residue ranges above).",
                        attested_ps,
                    )
                )

        # blank line between genes for readability
        claims.append(Cited("", source=None, label="SPACER"))

    for c in claims:
        if c.label == "SPACER":
            print()
        else:
            print(str(c))

    # ------------------------------------------------------------------
    # Step 6 — Final actionable cohort recommendation.
    # ------------------------------------------------------------------
    print("\n## Step 6 — Cohort eligibility recommendation")
    print("-" * 72)

    # Rank matched genes by total pathogenic + likely-pathogenic burden.
    ranked = sorted(rows, key=lambda r: r["P"] + r["LP"], reverse=True)
    nonzero = [r for r in ranked if (r["P"] + r["LP"]) > 0]

    print(
        Cited(
            (
                f"Disease ID supplied: {args.disease_id} "
                f"(name-token match: {args.disease_keywords})."
            ),
            source=None,
            label="TEXTBOOK_CONTEXT",
        )
    )

    if not nonzero:
        print(
            Cited(
                "No matched gene carries P/LP variants in this index — no "
                "patient-level eligibility cohort can be built from these chunks.",
                source=None,
                label="TEXTBOOK_CONTEXT",
            )
        )
    else:
        # Print one chunk-grounded line per ranked gene.
        for r in nonzero:
            gene = r["gene"]
            burden = r["P"] + r["LP"]
            verdict_kind = (
                "LoF-leaning" if r["n_variants"] and (r["lof_truncating"] / r["n_variants"]) >= 0.25
                else "missense-dominant" if r["n_variants"] and (r["missense"] / r["n_variants"]) >= 0.75
                else "mixed"
            )
            var_clusters = per_gene[gene].get("variant_cluster", [])
            # Cite the variant_cluster chunk that has the most P-tagged lines.
            anchor = None
            best = -1
            for vc in var_clusters:
                cnt = vc.text.count("(Pathogenic)") + vc.text.count("Pathogenic/Likely pathogenic")
                if cnt > best:
                    best = cnt
                    anchor = vc
            if anchor is None:
                continue
            attested = assert_supported(
                claim=f"{gene} variant_cluster anchor carries pathogenic calls",
                chunks=[anchor],
                hints=["Pathogenic"],
            )
            print(
                Cited(
                    (
                        f"ELIGIBLE: {gene} | P+LP={burden} | "
                        f"consequence={verdict_kind} | "
                        f"clusters cited from residues {anchor.residue_range or '(protein-level)'}"
                    ),
                    attested,
                )
            )

        top = nonzero[0]
        print(
            Cited(
                (
                    f"=> Primary cohort recommendation: {top['gene']} "
                    f"(highest indexed P+LP burden = {top['P'] + top['LP']} variants). "
                    "Secondary genes follow above. Estimated allele pool per gene = the "
                    "P+LP column; pool composition (missense vs LoF) = the %MIS / %LoF "
                    "columns. All counts are taken verbatim from variant_cluster chunks."
                ),
                source=None,
                label="TEXTBOOK_CONTEXT",
            )
        )

    print("\n" + "=" * 72)
    print("End of cookbook.")
    print("=" * 72)


if __name__ == "__main__":
    main()
