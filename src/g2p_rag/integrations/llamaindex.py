"""LlamaIndex integration adapter for G2PRetriever.

Stable public API — see STABILITY.md.

Requires llama-index-core to be installed::

    pip install llama-index-core

Usage::

    from g2p_rag import G2PRetriever
    from g2p_rag.integrations.llamaindex import G2PRetrieverLlamaIndex

    base = G2PRetriever(persist_dir="./data/chroma")
    li_retriever = G2PRetrieverLlamaIndex(retriever=base, k=5)

    # Use with LlamaIndex query engine:
    from llama_index.core.query_engine import RetrieverQueryEngine
    engine = RetrieverQueryEngine.from_args(li_retriever)
    response = engine.query("What BRCA1 domains overlap with pathogenic variants?")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from g2p_rag import G2PRetriever


class G2PRetrieverLlamaIndex:
    """LlamaIndex BaseRetriever adapter over G2PRetriever."""

    def __init__(
        self,
        retriever: G2PRetriever,
        k: int = 5,
        gene_filter: list[str] | None = None,
    ) -> None:
        """Initialise the LlamaIndex adapter.

        Raises:
            ImportError: If llama-index-core is not installed.
        """
        try:
            from llama_index.core.retrievers import BaseRetriever as LIBaseRetriever  # noqa: F401
            from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "llama-index-core is required for this integration. "
                "Install it with: pip install llama-index-core"
            ) from exc

        self._retriever = retriever
        self._k = k
        self._gene_filter = gene_filter
        self._log = structlog.get_logger()

    def _retrieve(self, query_bundle: Any) -> list[Any]:
        """Retrieve nodes from G2P index for use in LlamaIndex pipelines."""
        from llama_index.core.schema import NodeWithScore, TextNode

        query_str = (
            query_bundle.query_str
            if hasattr(query_bundle, "query_str")
            else str(query_bundle)
        )
        chunks = self._retriever.retrieve(
            query_str, k=self._k, gene_filter=self._gene_filter
        )

        nodes: list[Any] = []
        for chunk in chunks:
            node = NodeWithScore(
                node=TextNode(
                    text=chunk.text,
                    metadata={
                        "gene": chunk.gene,
                        "uniprot_id": chunk.uniprot_id,
                        "chunk_type": chunk.chunk_type,
                        "residue_range": chunk.residue_range,
                        "source_url": chunk.source_url,
                        "source": "g2p-rag",
                    },
                ),
                score=float(chunk.score),
            )
            nodes.append(node)

        self._log.debug(
            "llamaindex_retriever.retrieved", query=query_str[:60], hits=len(nodes)
        )
        return nodes

    def retrieve(self, query: str) -> list[Any]:
        """Convenience method matching LlamaIndex retriever interface."""
        from llama_index.core.schema import QueryBundle

        return self._retrieve(QueryBundle(query_str=query))
