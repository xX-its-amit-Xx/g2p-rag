"""Generation chain for the G2P RAG system.

This module exposes a stable ``G2PChain`` / ``build_chain`` API that internally
selects an LLM backend at construction time, preferring (in order):

  1. **Anthropic** via ``langchain_anthropic.ChatAnthropic`` when
     ``ANTHROPIC_API_KEY`` is set and ``langchain_anthropic`` is importable.
     The ``model`` constructor argument is honored on this branch.
  2. **Local Llama** via the ``llama_cpp`` Python bindings when a GGUF model
     file is reachable (``$LLAMA_MODEL_PATH`` or
     ``C:/llama-models/Llama-3.2-3B-Instruct-Q4_K_M.gguf``). The ``model``
     argument is ignored on this branch — the loaded GGUF defines the model.
  3. **Retrieval-only (no LLM)** — a sentinel that returns a structured
     "no LLM available" answer assembled from the retrieved chunks plus
     synthetic citation tokens, so downstream cookbooks/CLI still produce
     useful output instead of crashing.

The public API surface is preserved: callers continue to use
``G2PChain(model=...)`` and ``chain.answer(question, results) -> GenerationResult``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

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
DEFAULT_LLAMA_PATH = "C:/llama-models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

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
# Backend probes (cheap, no heavy imports)
# ---------------------------------------------------------------------------


def _anthropic_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import langchain_anthropic  # noqa: F401
    except Exception:
        return False
    return True


def _resolve_llama_path(model_llama_path: Optional[str] = None) -> Optional[str]:
    candidates: list[str] = []
    if model_llama_path:
        candidates.append(model_llama_path)
    env_path = os.environ.get("LLAMA_MODEL_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.append(DEFAULT_LLAMA_PATH)
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def _llama_available(model_llama_path: Optional[str] = None) -> bool:
    if _resolve_llama_path(model_llama_path) is None:
        return False
    try:
        import llama_cpp  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# G2PChain
# ---------------------------------------------------------------------------


class G2PChain:
    """Retrieval-augmented generation chain with a 3-tier LLM fallback.

    The public surface mirrors the previous Anthropic-only implementation:
    ``G2PChain(model=...).answer(question, results) -> GenerationResult``.
    Internally, the backend is selected at construction time.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        llama_model_path: Optional[str] = None,
    ) -> None:
        """Initialise the LLM and prompt template.

        Args:
            model: Anthropic model identifier (honored only on the Anthropic
                branch; ignored on llama / no-LLM branches).
            api_key: Anthropic API key. Falls back to ``ANTHROPIC_API_KEY``
                environment variable when *None*.
            max_tokens: Maximum tokens in the model response.
            temperature: Sampling temperature; 0.0 for deterministic output.
            llama_model_path: Optional explicit path to a GGUF model file.
        """
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._backend = "none"
        self._model = model
        self._llm: Any = None
        self._prompt: Any = None

        # 1. Anthropic
        if api_key or _anthropic_available():
            try:
                from langchain_anthropic import ChatAnthropic
                from langchain_core.prompts import ChatPromptTemplate

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
                self._backend = "anthropic"
                log.info("g2p_chain.ready", backend="anthropic", model=model, max_tokens=max_tokens)
                print(f"[g2p_rag.generate] backend=anthropic model={model}")
                return
            except Exception as exc:
                log.warning("g2p_chain.anthropic_init_failed", error=repr(exc))
                print(f"[g2p_rag.generate] anthropic init failed: {exc!r}; trying llama")

        # 2. Local Llama
        llama_path = _resolve_llama_path(llama_model_path)
        if llama_path is not None:
            try:
                from llama_cpp import Llama  # lazy

                cpu_count = os.cpu_count() or 2
                n_threads = max(1, cpu_count // 2)
                self._llm = Llama(
                    model_path=llama_path,
                    n_ctx=4096,
                    n_threads=n_threads,
                    verbose=False,
                )
                self._backend = "llama"
                self._model = f"llama_cpp::{os.path.basename(llama_path)}"
                log.info(
                    "g2p_chain.ready",
                    backend="llama",
                    model_path=llama_path,
                    max_tokens=max_tokens,
                )
                print(
                    f"[g2p_rag.generate] backend=llama model_path={llama_path} "
                    f"n_ctx=4096 n_threads={n_threads}"
                )
                return
            except Exception as exc:
                log.warning("g2p_chain.llama_init_failed", error=repr(exc))
                print(f"[g2p_rag.generate] llama init failed: {exc!r}; falling back to retrieval-only")

        # 3. No LLM
        self._backend = "none"
        self._model = "retrieval-only"
        log.info("g2p_chain.ready", backend="none")
        print(
            "[g2p_rag.generate] backend=none "
            "(no ANTHROPIC_API_KEY and no reachable Llama model) — retrieval-only mode"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_context(self, results: list[SearchResult]) -> str:
        """Render retrieval results as a numbered context block for the prompt."""
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

    def _citation_tokens(self, results: list[SearchResult]) -> list[str]:
        """Build canonical [Gene:UniProt:ChunkType:Residues] tokens for results."""
        tokens: list[str] = []
        for r in results:
            tok = (
                f"{r.chunk.gene}:{r.chunk.uniprot_id}:"
                f"{r.chunk.chunk_type}:{r.chunk.residue_start}-{r.chunk.residue_end}"
            )
            tokens.append(tok)
        return tokens

    def _retrieval_only_answer(
        self, question: str, results: list[SearchResult]
    ) -> str:
        """Compose a structured "no LLM" answer from retrieved chunks alone.

        Includes synthetic citation tokens in the canonical format so
        ``_CITATION_RE.findall`` still recovers them downstream.
        """
        if not results:
            return (
                "[no LLM available; set ANTHROPIC_API_KEY or LLAMA_MODEL_PATH] "
                "Retrieval returned no context chunks for this question."
            )
        lines: list[str] = []
        lines.append(
            "[no LLM available; set ANTHROPIC_API_KEY or LLAMA_MODEL_PATH] "
            "Returning retrieval-only summary:"
        )
        lines.append(f"Question: {question}")
        lines.append("Top retrieved context chunks:")
        for i, r in enumerate(results, 1):
            tok = (
                f"[{r.chunk.gene}:{r.chunk.uniprot_id}:"
                f"{r.chunk.chunk_type}:{r.chunk.residue_start}-{r.chunk.residue_end}]"
            )
            excerpt = r.chunk.text.replace("\n", " ").strip()
            if len(excerpt) > 220:
                excerpt = excerpt[:220] + "..."
            lines.append(f"  {i}. {tok} {excerpt}")
        return "\n".join(lines)

    def _call_llama(self, question: str, results: list[SearchResult]) -> str:
        """Call the loaded llama_cpp model with a single composed prompt."""
        context = self._format_context(results)
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer (cite sources using [Gene:UniProt:ChunkType:Residues] format):"
        )
        out = self._llm(prompt, max_tokens=self._max_tokens)
        try:
            return str(out["choices"][0]["text"])
        except (KeyError, IndexError, TypeError):
            return str(out)

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
        log.info(
            "g2p_chain.answer.invoke",
            backend=self._backend,
            question=question[:80],
            context_chunks=len(results),
            model=self._model,
        )

        if self._backend == "anthropic":
            context = self._format_context(results)
            human_msg = (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer (cite sources using [Gene:UniProt:ChunkType:Residues] format):"
            )
            messages = self._prompt.format_messages(input=human_msg)
            response = self._llm.invoke(messages)
            answer_text: str = response.content  # type: ignore[assignment]
        elif self._backend == "llama":
            answer_text = self._call_llama(question, results)
        else:
            answer_text = self._retrieval_only_answer(question, results)

        citations = _CITATION_RE.findall(answer_text)
        # When the LLM didn't produce citations, fall back to the retrieved
        # chunks' canonical tokens so callers always get traceable provenance.
        if not citations:
            citations = self._citation_tokens(results)

        log.info(
            "g2p_chain.answer.done",
            backend=self._backend,
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

    # ------------------------------------------------------------------
    # Back-compat shim — older CLI calls used .run(question=, chunks=)
    # ------------------------------------------------------------------

    def run(self, question: str, chunks: list[SearchResult]) -> str:
        """Convenience shim returning just the answer text (used by the CLI)."""
        return self.answer(question, chunks).answer


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def build_chain(
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> G2PChain:
    """Build and return a :class:`G2PChain` instance.

    Args:
        model: Anthropic model identifier (honored only when the Anthropic
            backend is selected).
        api_key: Anthropic API key; falls back to ``ANTHROPIC_API_KEY`` env var.

    Returns:
        A ready-to-use G2PChain with the best available backend selected.
    """
    return G2PChain(model=model, api_key=api_key)
