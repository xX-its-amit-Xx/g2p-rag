"""Retrieval-quality regression gate for the committed g2p-rag chroma index.

This converts the old ``notebooks/01_eval.ipynb`` into a deterministic pytest run
that exercises ``g2p_rag.retrieve.HybridRetriever`` against the persisted
ChromaDB at ``data/chroma/`` using QA pairs from ``tests/eval_qa.json`` and
compares the aggregate metrics to thresholds in ``tests/eval_baseline.json``.

Marked ``@pytest.mark.eval`` so it stays out of the default ``pytest`` run — the
index is large and the embedding model has to be loaded. Run explicitly with::

    pytest tests/test_eval.py -v -m eval

If you intentionally improve retrieval, update both ``eval_baseline.json``
``measured`` values and the ``threshold = measured - buffer`` floor in the same
commit so future regressions stay detectable.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from g2p_rag.retrieve import (
    CollectionEmptyError,
    HybridRetriever,
    load_embedder,
    load_retriever,
)

# Anchor everything to this file so the test is cwd-independent.
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
CHROMA_DIR = REPO_ROOT / "data" / "chroma"
QA_PATH = TESTS_DIR / "eval_qa.json"
BASELINE_PATH = TESTS_DIR / "eval_baseline.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_qa() -> list[dict]:
    with QA_PATH.open(encoding="utf-8") as f:
        return json.load(f)["qa_pairs"]


def _load_baseline() -> dict:
    with BASELINE_PATH.open(encoding="utf-8") as f:
        return json.load(f)["metrics"]


# ---------------------------------------------------------------------------
# Metric helpers (intentionally tiny / dependency-free so the gate stays auditable)
# ---------------------------------------------------------------------------


def recall_at_k(top_genes: list[str], expected_genes: list[str]) -> float:
    """Fraction of expected genes that show up at least once in the top-k list."""
    if not expected_genes:
        return 0.0
    return len(set(expected_genes) & set(top_genes)) / len(expected_genes)


def reciprocal_rank(top_genes: list[str], expected_genes: list[str]) -> float:
    """1/rank of the first top-k gene that belongs to *expected_genes* (0 if none)."""
    expected_set = set(expected_genes)
    for rank, gene in enumerate(top_genes, start=1):
        if gene in expected_set:
            return 1.0 / rank
    return 0.0


def keyword_hit(retrieved_text: str, keywords: list[str]) -> float:
    """Fraction of *keywords* (case-insensitive substring match) found in the concatenated context."""
    if not keywords:
        return 0.0
    blob = retrieved_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in blob)
    return hits / len(keywords)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    """Load the committed chroma index once per module; skip the suite if absent."""
    if not CHROMA_DIR.exists():
        pytest.skip(f"Chroma index not found at {CHROMA_DIR}; run `g2p-rag ingest` first.")
    try:
        embedder = load_embedder(EMBEDDING_MODEL)
        return load_retriever(CHROMA_DIR, embedder)
    except CollectionEmptyError:
        pytest.skip("Chroma collection is empty; run `g2p-rag ingest` first.")


@pytest.fixture(scope="module")
def qa_pairs() -> list[dict]:
    return _load_qa()


@pytest.fixture(scope="module")
def baseline() -> dict:
    return _load_baseline()


@pytest.fixture(scope="module")
def per_question_metrics(retriever: HybridRetriever, qa_pairs: list[dict]) -> list[dict]:
    """Run every QA pair through the retriever once and return per-question scores."""
    out: list[dict] = []
    for qa in qa_pairs:
        results = retriever.search(qa["question"], k=TOP_K)
        top_genes = [r.chunk.gene for r in results[:TOP_K]]
        context_blob = " ".join(r.chunk.text for r in results[:TOP_K])
        out.append(
            {
                "id": qa["id"],
                "question": qa["question"],
                "expected_genes": qa["expected_genes"],
                "top_genes": top_genes,
                "recall_at_5": recall_at_k(top_genes, qa["expected_genes"]),
                "mrr": reciprocal_rank(top_genes, qa["expected_genes"]),
                "keyword_hit": keyword_hit(context_blob, qa.get("answer_keywords", [])),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.eval
def test_qa_corpus_loads():
    """QA file is well-formed and every entry has the required schema."""
    qa = _load_qa()
    assert len(qa) >= 8, f"Expected >= 8 QA pairs, got {len(qa)}"
    required = {"id", "question", "expected_genes", "answer_keywords"}
    for entry in qa:
        missing = required - entry.keys()
        assert not missing, f"QA entry {entry.get('id')!r} missing keys: {missing}"
        assert entry["expected_genes"], f"QA entry {entry['id']!r} has no expected_genes"


@pytest.mark.eval
def test_mean_recall_at_5_meets_baseline(
    per_question_metrics: list[dict], baseline: dict
) -> None:
    """Mean Recall@5 stays at or above the committed baseline threshold."""
    threshold = baseline["mean_recall_at_5"]["threshold"]
    mean = statistics.mean(m["recall_at_5"] for m in per_question_metrics)
    assert mean >= threshold, (
        f"Mean Recall@5 {mean:.3f} < baseline threshold {threshold:.3f}.\n"
        f"Per-question scores: "
        + ", ".join(f"{m['id']}={m['recall_at_5']:.2f}" for m in per_question_metrics)
    )


@pytest.mark.eval
def test_mean_mrr_meets_baseline(
    per_question_metrics: list[dict], baseline: dict
) -> None:
    """Mean Reciprocal Rank stays at or above the committed baseline threshold."""
    threshold = baseline["mean_mrr"]["threshold"]
    mean = statistics.mean(m["mrr"] for m in per_question_metrics)
    assert mean >= threshold, (
        f"Mean MRR {mean:.3f} < baseline threshold {threshold:.3f}.\n"
        f"Per-question scores: "
        + ", ".join(f"{m['id']}={m['mrr']:.2f}" for m in per_question_metrics)
    )


@pytest.mark.eval
def test_mean_keyword_hit_meets_baseline(
    per_question_metrics: list[dict], baseline: dict
) -> None:
    """Top-5 chunks contain a sufficient fraction of expected answer keywords.

    This is a cheap stand-in for citation accuracy from the notebook — it does
    NOT call the LLM (no API key required, deterministic) but still catches
    chunking / retrieval regressions that would starve generation of the
    relevant facts.
    """
    threshold = baseline["mean_keyword_hit"]["threshold"]
    mean = statistics.mean(m["keyword_hit"] for m in per_question_metrics)
    assert mean >= threshold, (
        f"Mean keyword-hit {mean:.3f} < baseline threshold {threshold:.3f}.\n"
        f"Per-question scores: "
        + ", ".join(f"{m['id']}={m['keyword_hit']:.2f}" for m in per_question_metrics)
    )


@pytest.mark.eval
def test_every_question_finds_at_least_one_expected_gene(
    per_question_metrics: list[dict],
) -> None:
    """No QA pair should return zero expected genes in its top-5 — a strict floor."""
    failures = [m for m in per_question_metrics if m["recall_at_5"] == 0.0]
    assert not failures, (
        "QA pairs with Recall@5 == 0 (retriever returned none of the expected genes):\n"
        + "\n".join(
            f"  {m['id']}: expected={m['expected_genes']} got_top5={m['top_genes']}"
            for m in failures
        )
    )
