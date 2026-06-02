"""LangChain integration adapter for G2PRetriever.

Stable public API — see STABILITY.md.

Usage::

    from g2p_rag import G2PRetriever
    from g2p_rag.integrations.langchain import G2PRetrieverLangChain

    base = G2PRetriever(persist_dir="./data/chroma")
    lc_retriever = G2PRetrieverLangChain(retriever=base, k=5)

    # For LLM-driven chains, prefer the in-package G2PChain which already
    # implements the Anthropic -> local Llama -> retrieval-only fallback:
    from g2p_rag.generate import G2PChain
    chain = G2PChain()  # selects the best available backend at construction
    chunks = base.search("What domains in BRCA1 affect pathogenic variants?", k=5)
    result = chain.answer("What domains in BRCA1 affect pathogenic variants?", chunks)

    # If you must use LangChain's RetrievalQA directly, instantiate the LLM
    # through the same fallback adapter used by the cookbooks rather than
    # hard-coding ChatAnthropic, so the code still runs without an API key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import structlog

if TYPE_CHECKING:
    from g2p_rag import G2PRetriever

log = structlog.get_logger()


class G2PRetrieverLangChain(BaseRetriever):
    """LangChain BaseRetriever adapter over G2PRetriever."""

    retriever: Any  # G2PRetriever — typed as Any to avoid Pydantic forward-ref issues
    k: int = 5
    gene_filter: list[str] | None = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Retrieve documents using G2PRetriever and return as LangChain Documents."""
        chunks = self.retriever.retrieve(query, k=self.k, gene_filter=self.gene_filter)
        docs = [
            Document(
                page_content=chunk.text,
                metadata={
                    "gene": chunk.gene,
                    "uniprot_id": chunk.uniprot_id,
                    "chunk_type": chunk.chunk_type,
                    "residue_range": chunk.residue_range,
                    "source_url": chunk.source_url,
                    "score": chunk.score,
                    "source": "g2p-rag",
                },
            )
            for chunk in chunks
        ]
        log.debug("langchain_retriever.retrieved", query=query[:60], hits=len(docs))
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any,
    ) -> list[Document]:
        """Async version — delegates to sync implementation (retrieval is CPU-bound)."""
        return self._get_relevant_documents(query, run_manager=run_manager)
