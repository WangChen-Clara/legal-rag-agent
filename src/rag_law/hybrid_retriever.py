from __future__ import annotations

from typing import Protocol

from .models import SearchHit


class Searcher(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        ...


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: Searcher,
        lexical_retriever: Searcher,
        *,
        rrf_k: int = 60,
    ):
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        self.dense_retriever = dense_retriever
        self.lexical_retriever = lexical_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_k: int | None = None,
    ) -> list[SearchHit]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        candidate_limit = candidate_k or max(top_k * 3, top_k)
        dense_hits = self.dense_retriever.search(query, top_k=candidate_limit)
        lexical_hits = self.lexical_retriever.search(query, top_k=candidate_limit)
        return self.fuse(
            [dense_hits, lexical_hits],
            top_k=top_k,
            rrf_k=self.rrf_k,
        )

    @staticmethod
    def fuse(
        rankings: list[list[SearchHit]],
        *,
        top_k: int,
        rrf_k: int = 60,
    ) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")

        scores: dict[str, float] = {}
        first_hits: dict[str, SearchHit] = {}
        sources: dict[str, list[str]] = {}

        for ranking in rankings:
            for fallback_rank, hit in enumerate(ranking, start=1):
                key = _hit_key(hit)
                rank = hit.rank if hit.rank > 0 else fallback_rank
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
                first_hits.setdefault(key, hit)
                source = str(hit.metadata.get("retrieval_source") or "unknown")
                if source not in sources.setdefault(key, []):
                    sources[key].append(source)

        ordered_keys = sorted(
            scores,
            key=lambda key: (
                _source_priority(first_hits[key]),
                -scores[key],
                first_hits[key].rank,
                key,
            ),
        )[:top_k]

        fused_hits: list[SearchHit] = []
        for rank, key in enumerate(ordered_keys, start=1):
            hit = first_hits[key]
            metadata = dict(hit.metadata)
            metadata["retrieval_source"] = "hybrid"
            metadata["retrieval_sources"] = sources[key]
            metadata["rrf_score"] = scores[key]
            fused_hits.append(
                SearchHit(
                    rank=rank,
                    distance=scores[key],
                    text=hit.text,
                    metadata=metadata,
                )
            )
        return fused_hits


def _hit_key(hit: SearchHit) -> str:
    return str(
        hit.metadata.get("chunk_id")
        or hit.metadata.get("document_id")
        or hit.metadata.get("parent_document_id")
        or f"{hit.metadata.get('section')}:{hit.text[:120]}"
    )


def _source_priority(hit: SearchHit) -> int:
    source = hit.metadata.get("retrieval_source")
    if source == "explicit_citation":
        return 0
    if source == "cross_reference":
        return 1
    return 2
