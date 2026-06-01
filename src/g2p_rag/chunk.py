"""Converts structured protein data into text chunks suitable for embedding."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from g2p_rag.fetch import (  # noqa: F401
    ClinVarVariant,
    GenccDisease,
    GeneStructureMap,
    PDBStructure,
    ProteinDomain,
    PTMSite,
)

if TYPE_CHECKING:
    pass

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single text chunk ready for embedding and storage in ChromaDB."""

    text: str
    chunk_type: str
    gene: str
    uniprot_id: str
    residue_start: int = 0
    residue_end: int = 0
    metadata: dict = field(default_factory=dict)

    def to_chroma_metadata(self) -> dict:
        """Return flat dict suitable for ChromaDB metadata (no nested dicts, no None)."""
        return {
            "chunk_type": self.chunk_type,
            "gene": self.gene,
            "uniprot_id": self.uniprot_id,
            "residue_start": self.residue_start,
            "residue_end": self.residue_end,
            **{k: str(v) for k, v in self.metadata.items()},
        }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _ptms_in_range(
    ptm_sites: list[PTMSite], start: int, end: int
) -> list[PTMSite]:
    """Return PTM sites whose position falls within [start, end] inclusive."""
    return [p for p in ptm_sites if start <= p.position <= end]


def _pocket_ids_in_range(pockets: list, start: int, end: int) -> list[str]:
    """Return pocket IDs where at least one residue falls within [start, end]."""
    result: list[str] = []
    for pocket in pockets:
        residues: list[int] = getattr(pocket, "residues", [])
        if any(start <= r <= end for r in residues):
            result.append(str(pocket.pocket_id))
    return result


def _clin_sig_distribution(variants: list[ClinVarVariant]) -> str:
    """Return a human-readable clinical significance distribution string."""
    counts: dict[str, int] = {}
    for v in variants:
        sig = v.clinical_significance or "Unknown"
        counts[sig] = counts.get(sig, 0) + 1
    return ", ".join(f"{sig}: {cnt}" for sig, cnt in sorted(counts.items()))


def _extract_position(variant: ClinVarVariant) -> int:
    """Return the integer residue position from a ClinVarVariant (0 if unparseable)."""
    pos = getattr(variant, "position", 0)
    if isinstance(pos, int):
        return pos
    try:
        return int(pos)
    except (TypeError, ValueError):
        return 0


def _pathogenic_count(variants: list[ClinVarVariant]) -> int:
    """Count variants classified as Pathogenic or Likely pathogenic."""
    return sum(
        1
        for v in variants
        if v.clinical_significance
        and "pathogenic" in v.clinical_significance.lower()
    )


def _vus_count(variants: list[ClinVarVariant]) -> int:
    """Count variants classified as VUS (Uncertain significance)."""
    return sum(
        1
        for v in variants
        if v.clinical_significance
        and "uncertain" in v.clinical_significance.lower()
    )


def _benign_count(variants: list[ClinVarVariant]) -> int:
    """Count variants classified as Benign or Likely benign."""
    return sum(
        1
        for v in variants
        if v.clinical_significance
        and "benign" in v.clinical_significance.lower()
    )


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class ProteinChunker:
    """Produces text chunks from structured protein data for downstream embedding."""

    def __init__(self) -> None:
        """Initialise with no required arguments."""
        log.debug("ProteinChunker initialised")

    # ------------------------------------------------------------------
    # Domain chunks
    # ------------------------------------------------------------------

    def domain_chunks(self, structure: GeneStructureMap) -> list[Chunk]:
        """Produce one Chunk per annotated protein domain in the structure."""
        chunks: list[Chunk] = []
        features = structure.features
        domains: list[ProteinDomain] = getattr(features, "domains", [])
        ptm_sites: list[PTMSite] = getattr(features, "ptm_sites", [])
        # UniProt-backed enrichment uses `ppi` (matches ProteinFeatures model);
        # the older field name `ppi_partners` is kept as a fallback so chunks
        # don't silently drop interactions when only one source is populated.
        ppi_partners: list = (getattr(features, "ppi", []) or
                                getattr(features, "ppi_partners", []))
        pockets: list = getattr(features, "pockets", [])
        mavedb_scores: list = getattr(features, "mavedb_scores", [])

        # Prefer features.sequence (UniProt-derived) over structure.sequence
        # (blank since the G2P /api/gene/ endpoint doesn't return sequences).
        protein_seq = (getattr(features, "sequence", "")
                        or getattr(structure, "sequence", "")
                        or "")

        for domain in domains:
            # Sequence span
            if protein_seq:
                seq_span = protein_seq[domain.start - 1 : domain.end]
            else:
                seq_span = "N/A"

            # PTMs in range
            ptms_in_range = _ptms_in_range(ptm_sites, domain.start, domain.end)
            ptm_str = (
                ", ".join(f"{p.position}({p.ptm_type})" for p in ptms_in_range)
                if ptms_in_range
                else "None"
            )

            # PPI partners (first 5)
            partner_names = [
                getattr(p, "partner_name", str(p)) for p in ppi_partners[:5]
            ]
            ppi_str = ", ".join(partner_names) if partner_names else "None"

            # Pockets overlapping domain
            pocket_ids = _pocket_ids_in_range(pockets, domain.start, domain.end)
            pocket_str = ", ".join(pocket_ids) if pocket_ids else "None"

            # MaveDB scores in range
            scores_in_range = [
                s
                for s in mavedb_scores
                if domain.start <= getattr(s, "position", 0) <= domain.end
            ]
            score_count = len(scores_in_range)
            if score_count:
                raw_scores = [
                    getattr(s, "score", None) for s in scores_in_range
                ]
                numeric = [x for x in raw_scores if isinstance(x, (int, float))]
                if numeric:
                    mean_score = statistics.mean(numeric)
                    mavedb_str = (
                        f"{score_count} variants scored (mean score: {mean_score:.3f})"
                    )
                else:
                    mavedb_str = f"{score_count} variants scored"
            else:
                mavedb_str = "0 variants scored"

            text = (
                f"Gene: {structure.gene_symbol} | UniProt: {structure.uniprot_id}\n"
                f"Domain: {domain.name} ({domain.domain_type})\n"
                f"Residues: {domain.start}–{domain.end}"
                f" ({domain.end - domain.start + 1} aa)\n"
                f"Description: {domain.description}\n"
                f"\n"
                f"Sequence span: {seq_span}\n"
                f"\n"
                f"PTM sites in range: {ptm_str}\n"
                f"PPI partners (protein-level): {ppi_str}\n"
                f"Pocket annotations overlapping: {pocket_str}\n"
                f"MaveDB scores in range: {mavedb_str}"
            )

            chunk = Chunk(
                text=text,
                chunk_type="domain",
                gene=structure.gene_symbol,
                uniprot_id=structure.uniprot_id,
                residue_start=domain.start,
                residue_end=domain.end,
                metadata={
                    "domain_name": domain.name,
                    "domain_type": domain.domain_type,
                },
            )
            chunks.append(chunk)
            log.debug(
                "domain_chunk created",
                gene=structure.gene_symbol,
                domain=domain.name,
            )

        return chunks

    # ------------------------------------------------------------------
    # Variant cluster chunks
    # ------------------------------------------------------------------

    def variant_cluster_chunks(
        self,
        gene: str,
        uniprot_id: str,
        variants: list[ClinVarVariant],
        window: int = 10,
    ) -> list[Chunk]:
        """Produce one Chunk per greedy cluster of spatially proximate variants."""
        # Filter out variants with no position and sort by position
        positioned = [v for v in variants if _extract_position(v) > 0]
        positioned.sort(key=_extract_position)

        if not positioned:
            log.debug("no positioned variants", gene=gene)
            return []

        # Greedy clustering: start new cluster when gap > window
        clusters: list[list[ClinVarVariant]] = []
        current_cluster: list[ClinVarVariant] = [positioned[0]]
        for v in positioned[1:]:
            prev_pos = _extract_position(current_cluster[-1])
            curr_pos = _extract_position(v)
            if curr_pos - prev_pos <= window:
                current_cluster.append(v)
            else:
                clusters.append(current_cluster)
                current_cluster = [v]
        clusters.append(current_cluster)

        chunks: list[Chunk] = []
        for cluster in clusters:
            positions = [_extract_position(v) for v in cluster]
            min_pos = min(positions)
            max_pos = max(positions)
            cluster_len = len(cluster)

            # Variant bullet list
            variant_lines = "\n".join(
                f"  • {v.protein_change} ({v.clinical_significance})"
                f" [{v.review_status}]"
                for v in cluster
            )

            # Molecular consequences (unique)
            mol_consequences = sorted(
                {
                    getattr(v, "molecular_consequence", None)
                    for v in cluster
                    if getattr(v, "molecular_consequence", None)
                }
            )
            mol_str = (
                ", ".join(mol_consequences) if mol_consequences else "Unknown"
            )

            # Clinical significance distribution
            clin_dist = _clin_sig_distribution(cluster)

            text = (
                f"Gene: {gene} | UniProt: {uniprot_id}\n"
                f"Variant cluster: {min_pos}–{max_pos} ({cluster_len} variants)\n"
                f"\n"
                f"Variants:\n"
                f"{variant_lines}\n"
                f"\n"
                f"Molecular consequences: {mol_str}\n"
                f"Clinical significance distribution: {clin_dist}"
            )

            unique_sigs = sorted(
                {v.clinical_significance for v in cluster if v.clinical_significance}
            )
            chunk = Chunk(
                text=text,
                chunk_type="variant_cluster",
                gene=gene,
                uniprot_id=uniprot_id,
                residue_start=min_pos,
                residue_end=max_pos,
                metadata={
                    "variant_count": cluster_len,
                    "clinical_sigs": "|".join(unique_sigs),
                },
            )
            chunks.append(chunk)
            log.debug(
                "variant_cluster_chunk created",
                gene=gene,
                min_pos=min_pos,
                max_pos=max_pos,
                n=cluster_len,
            )

        return chunks

    # ------------------------------------------------------------------
    # Protein summary chunk
    # ------------------------------------------------------------------

    def protein_summary_chunk(
        self,
        structure: GeneStructureMap,
        variants: list[ClinVarVariant],
    ) -> Chunk:
        """Produce a single high-level summary Chunk for the whole protein."""
        features = structure.features
        domains: list[ProteinDomain] = getattr(features, "domains", [])
        ptm_sites: list[PTMSite] = getattr(features, "ptm_sites", [])
        # UniProt-backed enrichment uses `ppi` (matches ProteinFeatures model);
        # the older field name `ppi_partners` is kept as a fallback so chunks
        # don't silently drop interactions when only one source is populated.
        ppi_partners: list = (getattr(features, "ppi", []) or
                                getattr(features, "ppi_partners", []))
        pockets: list = getattr(features, "pockets", [])
        mavedb_scores: list = getattr(features, "mavedb_scores", [])

        # Domains
        domain_str = (
            ", ".join(f"{d.name} ({d.start}–{d.end})" for d in domains)
            if domains
            else "None"
        )

        # PTM sites (first 10)
        ptm_str = (
            ", ".join(
                f"{p.position}({p.ptm_type})" for p in ptm_sites[:10]
            )
            if ptm_sites
            else "None"
        )

        # PPI partners (first 10)
        partner_names = [
            getattr(p, "partner_name", str(p)) for p in ppi_partners[:10]
        ]
        ppi_str = ", ".join(partner_names) if partner_names else "None"

        # Pockets
        pocket_id_list = [str(getattr(pk, "pocket_id", pk)) for pk in pockets]
        pocket_str = ", ".join(pocket_id_list) if pocket_id_list else "None"

        # ClinVar counts
        total_variants = len(variants)
        pathogenic = _pathogenic_count(variants)
        vus = _vus_count(variants)
        benign = _benign_count(variants)

        # Prefer features.length (UniProt-derived, always present when features
        # were enriched) over structure.length (which is 0 since the current
        # G2P /api/gene/ endpoint doesn't return the sequence).
        protein_length = (getattr(features, "length", 0)
                            or getattr(structure, "length", 0)
                            or 0)
        text = (
            f"Gene: {structure.gene_symbol} | UniProt: {structure.uniprot_id}\n"
            f"Protein summary\n"
            f"\n"
            f"Canonical sequence length: {protein_length} aa\n"
            f"Transcript: {structure.transcript_id} | Protein: {structure.protein_id}\n"
            f"\n"
            f"Domains ({len(domains)}): {domain_str}\n"
            f"PTM sites ({len(ptm_sites)}): {ptm_str}\n"
            f"PPI partners ({len(ppi_partners)}): {ppi_str}\n"
            f"Druggable pockets ({len(pockets)}): {pocket_str}\n"
            f"MaveDB variants scored: {len(mavedb_scores)}\n"
            f"\n"
            f"ClinVar variants: {total_variants} total\n"
            f"  Pathogenic/Likely pathogenic: {pathogenic}\n"
            f"  VUS: {vus}\n"
            f"  Benign/Likely benign: {benign}"
        )

        chunk = Chunk(
            text=text,
            chunk_type="protein_summary",
            gene=structure.gene_symbol,
            uniprot_id=structure.uniprot_id,
            residue_start=1,
            residue_end=protein_length,
        )
        log.debug(
            "protein_summary_chunk created",
            gene=structure.gene_symbol,
            uniprot_id=structure.uniprot_id,
        )
        return chunk

    # ------------------------------------------------------------------
    # UniProt comment-derived biology chunks (function / pathway /
    # subunit / disease). Each emits at most one chunk per protein; we
    # return None when the upstream text is empty so chunk_gene can
    # drop it without producing a stub entry.
    # ------------------------------------------------------------------

    def _comment_chunk(
        self,
        structure: GeneStructureMap,
        text: str,
        chunk_type: str,
        category_label: str,
    ) -> "Chunk | None":
        """Build one protein-level Chunk for a UniProt comment-derived field."""
        if not text:
            return None
        body = (
            f"Gene: {structure.gene_symbol} | UniProt: {structure.uniprot_id}\n"
            f"{category_label}\n"
            f"\n"
            f"{text}"
        )
        chunk = Chunk(
            text=body,
            chunk_type=chunk_type,
            gene=structure.gene_symbol,
            uniprot_id=structure.uniprot_id,
            residue_start=0,
            residue_end=0,
        )
        log.debug(
            f"{chunk_type}_chunk created",
            gene=structure.gene_symbol,
            uniprot_id=structure.uniprot_id,
        )
        return chunk

    def function_chunk(self, structure: GeneStructureMap) -> "Chunk | None":
        """Emit the FUNCTION chunk for this protein, or None when absent."""
        text = getattr(structure.features, "function_text", "") or ""
        return self._comment_chunk(structure, text, "function", "FUNCTION")

    def pathway_chunk(self, structure: GeneStructureMap) -> "Chunk | None":
        """Emit the PATHWAY chunk for this protein, or None when absent."""
        text = getattr(structure.features, "pathway_text", "") or ""
        return self._comment_chunk(structure, text, "pathway", "PATHWAY")

    def subunit_chunk(self, structure: GeneStructureMap) -> "Chunk | None":
        """Emit the SUBUNIT chunk for this protein, or None when absent."""
        text = getattr(structure.features, "subunit_text", "") or ""
        return self._comment_chunk(structure, text, "subunit", "SUBUNIT")

    def disease_chunk(self, structure: GeneStructureMap) -> "Chunk | None":
        """Emit the DISEASE chunk for this protein, or None when absent."""
        text = getattr(structure.features, "disease_text", "") or ""
        return self._comment_chunk(structure, text, "disease", "DISEASE")

    # ------------------------------------------------------------------
    # G2P /api/gene/ cross-reference chunks (PDB, AlphaFold, ChEMBL,
    # DrugBank, OMIM, Orphanet, HGNC aliases, GenCC diseases). These
    # fields are unique to the G2P portal payload and were ignored by
    # the UniProt-direct fallback path until v0.1.2.
    # ------------------------------------------------------------------

    def cross_references_chunk(
        self, structure: GeneStructureMap
    ) -> "Chunk | None":
        """Emit a single chunk listing AlphaFold / ChEMBL / DrugBank / OMIM / Orphanet / HGNC aliases."""
        af = getattr(structure, "alphafold_id", "") or ""
        chembl = getattr(structure, "chembl_id", "") or ""
        drugbank = getattr(structure, "drugbank_id", "") or ""
        omim = getattr(structure, "omim_id", "") or ""
        orphanet = getattr(structure, "orphanet_id", "") or ""
        aliases: list[str] = list(getattr(structure, "hgnc_aliases", []) or [])

        if not (af or chembl or drugbank or omim or orphanet or aliases):
            return None

        lines: list[str] = [
            f"Gene: {structure.gene_symbol} | UniProt: {structure.uniprot_id}",
            "Cross-references",
            "",
        ]
        if af:
            lines.append(f"AlphaFold model: {af}")
        if chembl:
            lines.append(f"ChEMBL target ID: {chembl}")
        if drugbank:
            lines.append(f"DrugBank ID: {drugbank}")
        if omim:
            lines.append(f"OMIM ID: {omim}")
        if orphanet:
            lines.append(f"Orphanet ID: {orphanet}")
        if aliases:
            lines.append(f"HGNC aliases: {', '.join(aliases)}")

        chunk = Chunk(
            text="\n".join(lines),
            chunk_type="cross_references",
            gene=structure.gene_symbol,
            uniprot_id=structure.uniprot_id,
            residue_start=0,
            residue_end=0,
            metadata={
                "alphafold_id": af,
                "chembl_id": chembl,
                "drugbank_id": drugbank,
                "omim_id": omim,
                "orphanet_id": orphanet,
            },
        )
        log.debug(
            "cross_references_chunk created",
            gene=structure.gene_symbol,
            uniprot_id=structure.uniprot_id,
        )
        return chunk

    def structures_chunk(
        self, structure: GeneStructureMap
    ) -> "Chunk | None":
        """Emit one chunk listing all experimentally-resolved PDB structures."""
        pdbs: list[PDBStructure] = list(
            getattr(structure, "pdb_structures", []) or []
        )
        if not pdbs:
            return None

        lines: list[str] = [
            f"Gene: {structure.gene_symbol} | UniProt: {structure.uniprot_id}",
            f"Experimental structures ({len(pdbs)})",
            "",
        ]
        for s in pdbs:
            cells = [s.pdb_id]
            if s.method:
                cells.append(s.method)
            if s.resolution:
                cells.append(s.resolution)
            if s.chain_range:
                cells.append(s.chain_range)
            lines.append("  • " + " | ".join(cells))

        chunk = Chunk(
            text="\n".join(lines),
            chunk_type="structures",
            gene=structure.gene_symbol,
            uniprot_id=structure.uniprot_id,
            residue_start=0,
            residue_end=0,
            metadata={"pdb_count": len(pdbs)},
        )
        log.debug(
            "structures_chunk created",
            gene=structure.gene_symbol,
            n=len(pdbs),
        )
        return chunk

    def diseases_chunk(
        self, structure: GeneStructureMap
    ) -> "Chunk | None":
        """Emit one chunk listing GenCC-curated gene-disease associations."""
        diseases: list[GenccDisease] = list(
            getattr(structure, "gencc_diseases", []) or []
        )
        if not diseases:
            return None

        lines: list[str] = [
            f"Gene: {structure.gene_symbol} | UniProt: {structure.uniprot_id}",
            f"GenCC gene-disease associations ({len(diseases)})",
            "",
        ]
        for d in diseases:
            bits: list[str] = []
            if d.disease_name:
                bits.append(d.disease_name)
            if d.mondo_id:
                bits.append(f"[{d.mondo_id}]")
            if d.classification:
                bits.append(f"({d.classification})")
            if d.moi:
                bits.append(f"MOI: {d.moi}")
            if bits:
                lines.append("  • " + " ".join(bits))

        chunk = Chunk(
            text="\n".join(lines),
            chunk_type="diseases",
            gene=structure.gene_symbol,
            uniprot_id=structure.uniprot_id,
            residue_start=0,
            residue_end=0,
            metadata={"disease_count": len(diseases)},
        )
        log.debug(
            "diseases_chunk created",
            gene=structure.gene_symbol,
            n=len(diseases),
        )
        return chunk

    # ------------------------------------------------------------------
    # Combined entry point
    # ------------------------------------------------------------------

    def chunk_gene(
        self,
        structure: GeneStructureMap,
        variants: list[ClinVarVariant],
    ) -> list[Chunk]:
        """Return all chunks for a single gene: domains + variant clusters + summary + biology."""
        d_chunks = self.domain_chunks(structure)
        v_chunks = self.variant_cluster_chunks(
            structure.gene_symbol, structure.uniprot_id, variants
        )
        summary = self.protein_summary_chunk(structure, variants)
        biology_chunks = [
            c for c in (
                self.function_chunk(structure),
                self.pathway_chunk(structure),
                self.subunit_chunk(structure),
                self.disease_chunk(structure),
            ) if c is not None
        ]
        crossref_chunks = [
            c for c in (
                self.cross_references_chunk(structure),
                self.structures_chunk(structure),
                self.diseases_chunk(structure),
            ) if c is not None
        ]
        all_chunks = d_chunks + v_chunks + [summary] + biology_chunks + crossref_chunks
        log.info(
            "gene chunked",
            gene=structure.gene_symbol,
            n_domain=len(d_chunks),
            n_variant_cluster=len(v_chunks),
            n_biology=len(biology_chunks),
            n_crossref=len(crossref_chunks),
            total=len(all_chunks),
        )
        return all_chunks


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def chunk_all(gene_data: dict[str, dict]) -> list[Chunk]:
    """Chunk all genes returned by fetch_all_genes."""
    chunker = ProteinChunker()
    all_chunks: list[Chunk] = []
    for gene_symbol, data in gene_data.items():
        structure: GeneStructureMap = data["structure"]
        variants: list[ClinVarVariant] = data.get("variants", [])
        try:
            chunks = chunker.chunk_gene(structure, variants)
            all_chunks.extend(chunks)
        except Exception:
            log.exception("failed to chunk gene", gene=gene_symbol)
    log.info("chunk_all complete", total_chunks=len(all_chunks))
    return all_chunks
