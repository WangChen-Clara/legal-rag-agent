from __future__ import annotations

import pytest

from rag_law.hybrid_retriever import HybridRetriever
from rag_law.models import SearchHit


class FakeSearcher:
    def __init__(self, hits: list[SearchHit]):
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        self.calls.append((query, top_k))
        return self.hits[:top_k]


def hit(section: str, source: str, rank: int, chunk_id: str | None = None) -> SearchHit:
    return SearchHit(
        rank=rank,
        distance=1.0 / rank,
        text=f"Evidence for {section}",
        metadata={
            "section": section,
            "chunk_id": chunk_id or f"{section}:0",
            "retrieval_source": source,
        },
    )


def test_hybrid_retriever_fuses_dense_and_lexical_rankings() -> None:
    dense = FakeSearcher([hit("217.135", "semantic", 1), hit("211.31", "semantic", 2)])
    lexical = FakeSearcher([hit("211.31", "lexical", 1), hit("217.134", "lexical", 2)])
    retriever = HybridRetriever(dense, lexical, rrf_k=10)

    hits = retriever.search("double default", top_k=3, candidate_k=2)

    assert [item.metadata["section"] for item in hits] == ["211.31", "217.135", "217.134"]
    assert hits[0].metadata["retrieval_source"] == "hybrid"
    assert hits[0].metadata["retrieval_sources"] == ["semantic", "lexical"]
    assert dense.calls == [("double default", 2)]
    assert lexical.calls == [("double default", 2)]


def test_hybrid_fusion_prioritizes_explicit_citation_hits() -> None:
    hits = HybridRetriever.fuse(
        [
            [hit("217.135", "semantic", 1)],
            [hit("211.31", "explicit_citation", 10)],
        ],
        top_k=2,
        rrf_k=60,
    )

    assert [item.metadata["section"] for item in hits] == ["211.31", "217.135"]


def test_hybrid_retriever_rejects_invalid_inputs() -> None:
    retriever = HybridRetriever(FakeSearcher([]), FakeSearcher([]))

    with pytest.raises(ValueError, match="query"):
        retriever.search("   ")
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("question", top_k=0)
    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(FakeSearcher([]), FakeSearcher([]), rrf_k=0)
