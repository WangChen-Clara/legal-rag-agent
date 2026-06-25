from __future__ import annotations

from rag_law.models import SearchHit
from scripts.evaluate_title12_context_retrieval import (
    focus_summary,
    hit_to_row,
    run_variant,
)


class FakeRetriever:
    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        return [
            SearchHit(
                rank=1,
                distance=0.9,
                text="semantic text",
                metadata={"section": "9.9", "chunk_id": "9.9:0"},
            )
        ]

    def search_with_context(self, query: str, **kwargs: object) -> list[SearchHit]:
        return [
            SearchHit(
                rank=1,
                distance=0.0,
                text="explicit text",
                metadata={
                    "section": "1.1",
                    "chunk_id": "1.1:0",
                    "retrieval_source": "explicit_citation",
                },
            ),
            SearchHit(
                rank=2,
                distance=0.8,
                text="semantic text",
                metadata={
                    "section": "9.9",
                    "chunk_id": "9.9:0",
                    "retrieval_source": "semantic",
                },
            ),
        ]


def test_hit_to_row_defaults_retrieval_source_to_semantic() -> None:
    row = hit_to_row(
        SearchHit(rank=1, distance=0.5, text="abc", metadata={"section": "1.1"})
    )

    assert row["retrieval_source"] == "semantic"
    assert row["section"] == "1.1"


def test_run_variant_uses_context_retrieval_when_configured() -> None:
    records = [
        {
            "question_id": "q1",
            "candidate_id": "c1",
            "question_type": "definition",
            "question": "What is 12 CFR 1.1?",
            "acceptable_sections": ["1.1"],
        }
    ]

    result = run_variant(
        FakeRetriever(),
        records,
        variant_config={
            "include_explicit_citations": True,
            "expand_cross_references": False,
            "expand_from_semantic_without_explicit": False,
            "use_context": True,
        },
        top_k=10,
        semantic_top_k=10,
        max_expanded_sections=3,
        max_chunks_per_section=1,
    )

    assert result["metrics"]["hit_rate"]["hit_at_1"] == 1.0
    assert result["per_question"][0]["top_hits"][0]["retrieval_source"] == "explicit_citation"


def test_run_variant_keeps_baseline_semantic_only() -> None:
    records = [
        {
            "question_id": "q1",
            "candidate_id": "c1",
            "question_type": "definition",
            "question": "What is 12 CFR 1.1?",
            "acceptable_sections": ["1.1"],
        }
    ]

    result = run_variant(
        FakeRetriever(),
        records,
        variant_config={
            "include_explicit_citations": False,
            "expand_cross_references": False,
            "expand_from_semantic_without_explicit": False,
            "use_context": False,
        },
        top_k=10,
        semantic_top_k=10,
        max_expanded_sections=3,
        max_chunks_per_section=1,
    )

    assert result["metrics"]["hit_rate"]["hit_at_10"] == 0.0
    assert result["per_question"][0]["top_hits"][0]["retrieval_source"] == "semantic"


def test_focus_summary_reports_focus_question_sources() -> None:
    variant = {
        "per_question": [
            {
                "question_id": "title12-dev-q001",
                "first_complete_rank": 1,
                "recall_at_10": 1.0,
                "top_hits": [{"section": "211.31", "retrieval_source": "explicit_citation"}],
            },
            {
                "question_id": "title12-dev-q018",
                "first_complete_rank": 2,
                "recall_at_10": 1.0,
                "top_hits": [
                    {"section": "217.135", "retrieval_source": "explicit_citation"},
                    {"section": "217.134", "retrieval_source": "cross_reference"},
                ],
            },
        ]
    }

    focus = focus_summary(variant)

    assert focus["title12-dev-q018"]["retrieval_sources"] == [
        "217.135:explicit_citation",
        "217.134:cross_reference",
    ]
