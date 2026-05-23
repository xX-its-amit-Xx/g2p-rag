# API Stability Contract

## Stable (public) API

The following are part of the stable public API and will not change incompatibly
within a minor version (0.x → 0.x+1 may add fields; 1.0.0 → 1.x will not break):

| Symbol | Module | Description |
|--------|--------|-------------|
| `G2PRetriever` | `g2p_rag` | Main retriever facade |
| `RetrievedChunk` | `g2p_rag` | Pydantic result model |
| `G2PRetriever.retrieve()` | `g2p_rag` | Primary query method |
| `G2PRetrieverLangChain` | `g2p_rag.integrations.langchain` | LangChain BaseRetriever |
| `G2PRetrieverLlamaIndex` | `g2p_rag.integrations.llamaindex` | LlamaIndex BaseRetriever |

```python
# Stable — safe to depend on
from g2p_rag import G2PRetriever, RetrievedChunk
from g2p_rag.integrations.langchain import G2PRetrieverLangChain
from g2p_rag.integrations.llamaindex import G2PRetrieverLlamaIndex
```

## Internal (unstable) API

The following modules are implementation details. They may change, be renamed,
or be removed without notice between any version:

- `g2p_rag.fetch` — API clients, rate limiting, Pydantic models for raw data
- `g2p_rag.chunk` — ProteinChunker, Chunk dataclass
- `g2p_rag.embed` — EmbeddingModel protocol, embedder implementations
- `g2p_rag.retrieve` — VectorStore, BM25Index, HybridRetriever, SearchResult
- `g2p_rag.generate` — G2PChain, GenerationResult
- `g2p_rag.cli` — Typer application (for the `g2p-rag` CLI)

If you find yourself importing from these modules, open an issue requesting
that the symbol be promoted to the public API.

## Versioning

This project follows [Semantic Versioning](https://semver.org/):
- **Patch** (0.1.x): bug fixes only, no API changes
- **Minor** (0.x.0): backwards-compatible additions (new fields on RetrievedChunk,
  new optional parameters on retrieve())
- **Major** (x.0.0): breaking changes announced 2 versions in advance via deprecation warnings

## RetrievedChunk field stability

Fields guaranteed stable in 0.x:
`text`, `gene`, `uniprot_id`, `chunk_type`, `residue_range`, `source_url`, `score`

Future versions may add optional fields (e.g. `confidence`, `pmid_citations`).
Existing fields will not be removed or renamed in 0.x.
