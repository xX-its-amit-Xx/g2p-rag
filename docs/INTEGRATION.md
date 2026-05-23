# Integrating g2p-rag into Your Project

This guide shows how therapy-agent (or any downstream package) should consume g2p-rag as a library. Only the symbols documented here are stable — everything else is internal.

## Installation

```bash
pip install g2p-rag
# or
uv add g2p-rag
```

To use OpenAI embeddings instead of the default sentence-transformers model:
```bash
pip install g2p-rag openai
```

## Prerequisites

Before querying, the index must exist locally. Run once:
```bash
# Option A: build from scratch (~30 min, requires internet)
g2p-rag ingest

# Option B: download pre-built snapshot (~2 min)
make download-index
```

The default index location is `./data/chroma`. Override with `--data-dir` or `G2P_DATA_DIR` env var.

## Python Library Usage

### Minimal (3 lines)

```python
from g2p_rag import G2PRetriever

retriever = G2PRetriever(persist_dir="./data/chroma")
results = retriever.retrieve("What BRCA1 domains overlap with pathogenic missense variants?")

for chunk in results:
    print(f"[{chunk.gene}] {chunk.chunk_type} {chunk.residue_range}  score={chunk.score:.3f}")
    print(chunk.text[:200])
    print()
```

### With gene filter

```python
# Restrict to specific genes
brca_results = retriever.retrieve(
    "Which pathogenic variants cluster in DNA-binding domains?",
    k=10,
    gene_filter=["BRCA1", "BRCA2", "TP53"],
)
```

### RetrievedChunk fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Full chunk text |
| `gene` | `str` | HGNC gene symbol |
| `uniprot_id` | `str` | UniProt accession |
| `chunk_type` | `"domain" \| "variant_cluster" \| "protein_summary"` | Chunk granularity |
| `residue_range` | `str` | `"start-end"` or `""` |
| `source_url` | `str` | G2P portal link |
| `score` | `float` | RRF fusion score |

## LangChain Integration

```python
from g2p_rag import G2PRetriever
from g2p_rag.integrations.langchain import G2PRetrieverLangChain
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA

retriever = G2PRetriever(persist_dir="./data/chroma")
lc_retriever = G2PRetrieverLangChain(retriever=retriever, k=5)

llm = ChatAnthropic(model="claude-sonnet-4-6")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=lc_retriever)

answer = qa.invoke({"query": "What are the key phosphorylation sites in TP53?"})
print(answer["result"])
```

The `G2PRetrieverLangChain` yields `langchain_core.documents.Document` objects. Each document's `metadata` dict contains all `RetrievedChunk` fields plus `"source": "g2p-rag"`.

## LlamaIndex Integration

```bash
pip install llama-index-core
```

```python
from g2p_rag import G2PRetriever
from g2p_rag.integrations.llamaindex import G2PRetrieverLlamaIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.anthropic import Anthropic

retriever = G2PRetriever(persist_dir="./data/chroma")
li_retriever = G2PRetrieverLlamaIndex(retriever=retriever, k=5)

llm = Anthropic(model="claude-sonnet-4-6")
engine = RetrieverQueryEngine.from_args(li_retriever, llm=llm)

response = engine.query("Which MUC1 PTMs are near frameshift variant clusters?")
print(response)
```

The `G2PRetrieverLlamaIndex` yields `llama_index.core.schema.NodeWithScore` objects with `score` set to the RRF fusion score.

## Configuration Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | — | Required for LLM generation |
| `OPENAI_API_KEY` | — | Required only for OpenAI embeddings |
| `NCBI_API_KEY` | — | Increases ClinVar rate limit (optional) |
| `G2P_DATA_DIR` | `./data` | Index and cache directory |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |

## Troubleshooting

**`CollectionEmptyError`**: The index is empty. Run `g2p-rag ingest` or `make download-index`.

**`ImportError: llama-index-core`**: Run `pip install llama-index-core`.

**`ImportError: sentence_transformers`**: Run `pip install sentence-transformers` (requires ~2 GB for PyTorch). If disk-constrained, set `EMBEDDING_MODEL=text-embedding-3-large` and install `openai` instead.

**Slow first query**: The BM25 index is rebuilt from ChromaDB on first load (~10s for 25 genes). Subsequent queries in the same process are fast.
