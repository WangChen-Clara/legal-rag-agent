from __future__ import annotations

import json
from pathlib import Path

from rag_law.models import SearchHit
from scripts.evaluate_title12_hybrid_retrieval import (
    hit_to_row,
    load_records,
    metric_delta,
    run_context_variant,
    run_hybrid_variant,
)


class FakeContextRetriever:
    items = []

    def search_with_context(self, query: str, **kwargs: object) -> list[SearchHit]:
        return [
            SearchHit(
                rank=1,
                distance=0.9,
                text="semantic section",
                metadata={
                    "section": "9.9",
                    "chunk_id": "9.9:0",
                    "retrieval_source": "semantic",
                },
            )
        ]


class FakeLexicalRetriever:
    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        return [
            SearchHit(
                rank=1,
                distance=1.2,
                text="lexical section",
                metadata={
                    "section": "1.1",
                    "chunk_id": "1.1:0",
                    "retrieval_source": "lexical",
                },
            )
        ]


def record() -> dict[str, object]:
    return {
        "question_id": "q1",
        "candidate_id": "c1",
        "question_type": "definition",
        "question": "What is the authority section?",
        "acceptable_sections": ["1.1"],
    }


def test_hit_to_row_includes_hybrid_metadata() -> None:
    row = hit_to_row(
        SearchHit(
            rank=1,
            distance=0.5,
            text="abc",
            metadata={
                "section": "1.1",
                "retrieval_source": "hybrid",
                "retrieval_sources": ["semantic", "lexical"],
                "rrf_score": 0.2,
            },
        )
    )

    assert row["retrieval_source"] == "hybrid"
    assert row["retrieval_sources"] == ["semantic", "lexical"]
    assert row["rrf_score"] == 0.2


def test_run_hybrid_variant_can_improve_over_context_baseline() -> None:
    records = [record()]

    baseline = run_context_variant(
        FakeContextRetriever(),
        records,
        top_k=10,
        semantic_top_k=10,
        max_expanded_sections=3,
        max_chunks_per_section=1,
    )
    hybrid = run_hybrid_variant(
        FakeContextRetriever(),
        FakeLexicalRetriever(),
        records,
        top_k=10,
        semantic_top_k=10,
        lexical_top_k=10,
        max_expanded_sections=3,
        max_chunks_per_section=1,
        rrf_k=60,
    )

    assert baseline["metrics"]["hit_rate"]["hit_at_10"] == 0.0
    assert hybrid["metrics"]["hit_rate"]["hit_at_10"] == 1.0
    assert hybrid["per_question"][0]["top_hits"][0]["retrieval_source"] == "hybrid"
    assert metric_delta(baseline, hybrid)["hit_at_10"] == 1.0


def test_load_records_prefers_development_split(tmp_path: Path) -> None:
    path = tmp_path / "qa.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {"question_id": "holdout", "split": "holdout"},
                    {"question_id": "dev", "split": "development"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert [item["question_id"] for item in load_records(path)] == ["dev"]
