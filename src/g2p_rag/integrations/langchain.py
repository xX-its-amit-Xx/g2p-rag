"""LangChain integration adapter for G2PRetriever.

Stable public API — see STABILITY.md.

Usage::

    from g2p_rag import G2PRetriever
    from g2p_rag.integrations.langchain import G2PRetrieverLangChain

    base = G2PRetriever(persist_dir="./data/chroma")
    lc_retriever = G2PRetrieverLangChain(retriever=base, k=5)

    # Use directly with LangChain chains:
    from langchain.chains import RetrievalQA
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model="claude-sonnet-4-6")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=lc_retriever)
    answer = qa.invoke({"query": "What domains in BRCA1 affect pathogenic variants?"})
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
