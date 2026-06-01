"""Citation discipline helper for the g2p-rag cookbook.

Why this module exists
----------------------
Cookbook examples synthesize narrative text from retrieved chunks. Without a
strict gate, a stray sentence can leak training-data knowledge that has no
provenance in the retrieval index — the exact failure mode RAG is supposed to
prevent. This helper enforces three rules:

  1. Every printed factual claim is wrapped in ``Cited(...)``.
  2. ``Cited`` either points at a concrete ``RetrievedChunk`` (and prints a
     compact provenance tag) OR is explicitly marked ``[NO_RAG_SOURCE]`` so a
     human reviewer can see, at a glance, which lines are framing text rather
     than retrieved evidence.
  3. Before constructing a ``Cited`` for a claim, callers run
     ``assert_supported(...)`` with one or more substring "hints" that MUST
     appear in some retrieved chunk; if no chunk supports the claim, the
     script crashes instead of silently emitting the sentence.

Public API
----------
  - ``Cited(text, source, label="")``   — printable claim bundle
  - ``find_in_chunks(query_substr, chunks)``         — soft lookup, returns None
  - ``assert_supported(claim, chunks, hints)``       — strict gate, raises

Run ``python _citation.py`` for a self-test demo.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Allow ``python cookbook/_citation.py`` from a fresh checkout without an
# editable install — mirrors the sys.path shim used by the sibling cookbook
# scripts (see e.g. pcsk9_druggability_heatmap.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from g2p_rag import RetrievedChunk  # noqa: E402  (after sys.path shim)


__all__ = ["Cited", "find_in_chunks", "assert_supported"]


# ---------------------------------------------------------------------------
# Provenance formatting
# ---------------------------------------------------------------------------

def _short_id(chunk: RetrievedChunk) -> str:
    """Build a stable, short, human-readable id for a retrieved chunk.

    ChromaDB chunk ids aren't on the public ``RetrievedChunk`` model, so we
    derive a deterministic short id from the fields that *are* exposed:
    uniprot + chunk_type + residue_range. Truncated to keep the printed line
    scannable.
    """
    residue_part = chunk.residue_range or "full"
    raw = f"{chunk.uniprot_id}:{chunk.chunk_type}:{residue_part}"
    # Keep it short but unambiguous — uniprot ids are 6-10 chars, types are
    # short, residue ranges are short. 32 chars is plenty.
    return raw[:32]


class Cited:
    """A printable claim tied (or explicitly NOT tied) to a retrieved chunk.

    Parameters
    ----------
    text:
        The human-readable claim sentence. This is what gets printed.
    source:
        The ``RetrievedChunk`` that supports the claim. Pass ``None`` ONLY for
        textbook framing sentences that you accept will be flagged as
        unsourced — the printed line will include the loud ``[NO_RAG_SOURCE]``
        marker so reviewers see it.
    label:
        Optional free-form tag (e.g. ``"framing"``, ``"caveat"``). Currently
        retained on the instance for downstream callers / future audit logs;
        it is not included in ``__str__`` so as to keep the inline citation
        compact.
    """

    __slots__ = ("text", "source", "label")

    def __init__(
        self,
        text: str,
        source: Optional[RetrievedChunk],
        label: str = "",
    ) -> None:
        self.text = text
        self.source = source
        self.label = label

    def __str__(self) -> str:
        if self.source is None:
            return f"{self.text}  [NO_RAG_SOURCE]"
        c = self.source
        # Residue range is empty string for protein-level chunks; print a
        # placeholder so the bracket structure stays consistent.
        residue = c.residue_range if c.residue_range else "-"
        return (
            f"{self.text}  "
            f"[chunk {_short_id(c)}: {c.chunk_type} {c.gene} {residue}]"
        )

    def __repr__(self) -> str:
        src = "None" if self.source is None else _short_id(self.source)
        return f"Cited(text={self.text!r}, source={src}, label={self.label!r})"


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def find_in_chunks(
    query_substr: str,
    chunks: list[RetrievedChunk],
) -> Optional[RetrievedChunk]:
    """Return the first chunk whose ``.text`` contains ``query_substr``.

    Case-insensitive substring match. Returns ``None`` if no chunk matches —
    callers that need a hard failure should use ``assert_supported`` instead.
    """
    if not query_substr:
        return None
    needle = query_substr.lower()
    for chunk in chunks:
        if needle in chunk.text.lower():
            return chunk
    return None


def assert_supported(
    claim: str,
    chunks: list[RetrievedChunk],
    hints: list[str],
) -> RetrievedChunk:
    """Strict synthesis gate: ensure at least one hint appears in some chunk.

    Walks ``hints`` in order and returns the first chunk that contains any
    hint (case-insensitive substring). Raises ``AssertionError`` with a
    descriptive message if no hint matches any chunk — the calling cookbook
    script should be allowed to crash rather than silently print an
    unsourced sentence.

    Parameters
    ----------
    claim:
        The natural-language claim being attested. Used only for the error
        message so a failing run points at the offending sentence.
    chunks:
        The retrieved chunk corpus to search.
    hints:
        Alternative substrings that, if any appears in any chunk, prove the
        claim is grounded. Pass several wordings to be robust to paraphrasing
        in the source corpus (e.g. ``"LDLR"`` vs ``"low density lipoprotein
        receptor"``).
    """
    if not hints:
        raise AssertionError(
            f"assert_supported({claim!r}): no hints provided — "
            "refusing to attest a claim without at least one substring to look for."
        )
    if not chunks:
        raise AssertionError(
            f"assert_supported({claim!r}): empty chunk corpus — "
            "retriever returned nothing, cannot support any claim."
        )

    for hint in hints:
        match = find_in_chunks(hint, chunks)
        if match is not None:
            return match

    # Build a compact diagnostic so the failing run shows *what* was searched
    # and *where*, without dumping entire chunk bodies.
    chunk_summary = ", ".join(
        f"{c.gene}/{c.chunk_type}" for c in chunks[:5]
    ) or "<none>"
    raise AssertionError(
        f"Unsupported claim: {claim!r}\n"
        f"  Tried hints: {hints!r}\n"
        f"  Searched {len(chunks)} chunk(s): {chunk_summary}"
        f"{' ...' if len(chunks) > 5 else ''}\n"
        "  No chunk text contained any hint. Either retrieve more chunks, "
        "broaden the hints, or mark this sentence with source=None and "
        "accept the [NO_RAG_SOURCE] tag."
    )


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a handful of fake chunks that look like things the real
    # G2PRetriever emits. We deliberately use the same field names and
    # chunk_type literals as src/g2p_rag/__init__.py so a typo here would
    # also fail Pydantic validation.
    fake_chunks: list[RetrievedChunk] = [
        RetrievedChunk(
            text=(
                "PCSK9 binds to low-density lipid receptor (LDLR) on the "
                "hepatocyte surface and promotes its lysosomal degradation, "
                "raising circulating LDL cholesterol."
            ),
            gene="PCSK9",
            uniprot_id="Q8NBP7",
            chunk_type="function",
            residue_range="",
            source_url="https://g2p.broadinstitute.org/protein?gene=PCSK9",
            score=0.91,
        ),
        RetrievedChunk(
            text=(
                "The catalytic domain of PCSK9 retains a vestigial serine "
                "protease fold; autocatalytic cleavage between the prodomain "
                "and catalytic domain is required for secretion."
            ),
            gene="PCSK9",
            uniprot_id="Q8NBP7",
            chunk_type="domain",
            residue_range="153-451",
            source_url="https://g2p.broadinstitute.org/protein?gene=PCSK9",
            score=0.74,
        ),
        RetrievedChunk(
            text=(
                "Gain-of-function PCSK9 variants cluster near the LDLR-binding "
                "interface and cause autosomal dominant hypercholesterolemia."
            ),
            gene="PCSK9",
            uniprot_id="Q8NBP7",
            chunk_type="variant_cluster",
            residue_range="374-381",
            source_url="https://g2p.broadinstitute.org/protein?gene=PCSK9",
            score=0.68,
        ),
    ]

    print("=== Demo 1: supported claim, hard gate passes ===")
    evidence = assert_supported(
        "PCSK9 binds LDLR",
        fake_chunks,
        hints=[
            "binds to low-density lipid receptor",
            "low density lipoprotein receptor",  # alternate wording
        ],
    )
    print(Cited("PCSK9 binds the LDLR to promote its degradation", evidence))

    print()
    print("=== Demo 2: framing sentence, explicitly un-sourced ===")
    print(
        Cited(
            "Lowering circulating LDL-C reduces cardiovascular risk.",
            source=None,
            label="framing",
        )
    )

    print()
    print("=== Demo 3: soft lookup ===")
    hit = find_in_chunks("autocatalytic cleavage", fake_chunks)
    miss = find_in_chunks("CRISPR base editing", fake_chunks)
    assert hit is not None and hit.chunk_type == "domain", "soft lookup hit failed"
    assert miss is None, "soft lookup should have returned None"
    print(f"find_in_chunks('autocatalytic cleavage') -> {hit.chunk_type} chunk OK")
    print(f"find_in_chunks('CRISPR base editing')    -> {miss} (None) OK")

    print()
    print("=== Demo 4: unsupported claim should raise ===")
    try:
        assert_supported(
            "PCSK9 is a GPCR",
            fake_chunks,
            hints=["G protein-coupled", "seven transmembrane"],
        )
    except AssertionError as exc:
        first_line = str(exc).splitlines()[0]
        print(f"OK — AssertionError raised: {first_line}")
    else:
        raise SystemExit("FAIL: unsupported claim did not raise")

    print()
    print("=== Demo 5: empty hints list should raise ===")
    try:
        assert_supported("anything", fake_chunks, hints=[])
    except AssertionError as exc:
        print(f"OK — AssertionError raised: {str(exc).splitlines()[0]}")
    else:
        raise SystemExit("FAIL: empty hints did not raise")

    print()
    print("All _citation.py self-tests passed.")
