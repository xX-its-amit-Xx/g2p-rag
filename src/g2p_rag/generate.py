"""LangChain + Anthropic generation chain for the G2P RAG system."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from g2p_rag.retrieve import SearchResult

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise genomics assistant with expert knowledge of protein structure, "
    "function, and disease variants. You answer questions ONLY using the provided context "
    "excerpts from the Broad Institute G2P portal and ClinVar. \n\n"
    "Rules:\n"
    "1. Answer ONLY from the context. If the context does not contain enough information, "
    "say so explicitly.\n"
    "2. Cite the source of each claim using the format [Gene:UniProt:ChunkType:Residues].\n"
    "3. Use precise scientific language appropriate for a molecular biologist.\n"
    "4. Never invent protein positions, domain names, or clinical significance labels.\n"
    "5. When multiple context chunks are relevant, synthesize them coherently."
)

DEFAULT_MODEL = "claude-sonnet-4-6"

# Regex that matches citation tokens produced by the LLM
_CITATION_RE = re.compile(r"\[([A-Z0-9]+:[A-Z0-9]+:[a-z_]+:\d+-\d+)\]")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """The complete output of a single RAG generation call."""

    question: str
    answer: str
    sources: list[SearchResult]
    model: str
    citations: list[str] = field(default_factory=list)
    """Citation strings extracted from the answer, e.g. ``["BRCA1:P38398:domain:1-300"]``."""


# ---------------------------------------------------------------------------
# G2PChain
# ---------------------------------------------------------------------------


class G2PChain:
    """Retrieval-augmented generation chain backed by Claude via LangChain."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        """Initialise the LLM and prompt template.

        Args:
            model: Anthropic model identifier (e.g. ``"claude-sonnet-4-6"``).
            api_key: Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY``
                environment variable when *None*.
            max_tokens: Maximum tokens in the model response.
            temperature: Sampling temperature; 0.0 for deterministic output.
        """
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._llm = ChatAnthropic(
            model=model,
            api_key=resolved_key,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )
        self._model = model
        log.info("g2p_chain.ready", model=model, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_context(self, results: list[SearchResult]) -> str:
        """Render retrieval results as a numbered context block for the prompt.

        Each block is labelled with the chunk's provenance so the model can cite
        it using the ``[Gene:UniProt:ChunkType:Residues]`` format.

        Args:
            results: Ranked retrieval results from :class:`~g2p_rag.retrieve.HybridRetriever`.

        Returns:
            Multi-line string ready to be embedded in the human message.
        """
        sections: list[str] = []
        for i, r in enumerate(results):
            header = (
                f"--- Context {i + 1} "
                f"[Gene:{r.chunk.gene}"
                f"|UniProt:{r.chunk.uniprot_id}"
                f"|{r.chunk.chunk_type}"
                f"|Residues:{r.chunk.residue_start}-{r.chunk.residue_end}] ---"
            )
            sections.append(f"{header}\n{r.chunk.text}")
        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        results: list[SearchResult],
    ) -> GenerationResult:
        """Generate a grounded answer for *question* using *results* as context.

        Args:
            question: The user's natural-language question.
            results: Retrieved chunks that supply the factual grounding.

        Returns:
            A :class:`GenerationResult` containing the answer text, extracted
            citations, and source provenance.
        """
        context = self._format_context(results)
        human_msg = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer (cite sources using [Gene:UniProt:ChunkType:Residues] format):"
        )

        log.info(
            "g2p_chain.answer.invoke",
            question=question[:80],
            context_chunks=len(results),
            model=self._model,
        )

        messages = self._prompt.format_messages(input=human_msg)
        response = self._llm.invoke(messages)
        answer_text: str = response.content  # type: ignore[assignment]

        citations = _CITATION_RE.findall(answer_text)

        log.info(
            "g2p_chain.answer.done",
            question=question[:80],
            answer_length=len(answer_text),
            citations_found=len(citations),
        )

        return GenerationResult(
            question=question,
            answer=answer_text,
            sources=results,
            model=self._model,
            citations=citations,
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def build_chain(
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> G2PChain:
    """Build and return a :class:`G2PChain` instance.

    Args:
        model: Anthropic model identifier.
        api_key: Anthropic API key; falls back to ``ANTHROPIC_API_KEY`` env var.

    Returns:
        A ready-to-use G2PChain.
    """
    return G2PChain(model=model, api_key=api_key)
