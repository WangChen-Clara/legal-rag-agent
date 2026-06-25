from __future__ import annotations

from scripts.compare_title12_retrieval import (
    benchmark_search,
    build_legacy_lineage,
    evaluate_rankings,
)


class FakeIndex:
    def search(self, vectors: list[object], k: int) -> tuple[list[int], list[int]]:
        return [k], [len(vectors)]


def test_legacy_lineage_uses_only_safe_alignment_statuses() -> None:
    lineage = build_legacy_lineage(
        [
            {
                "legacy_row_id": 1,
                "official_section": "3.10",
                "status": "exact",
            },
            {
                "legacy_row_id": 2,
                "official_section": "3.11",
                "status": "review_required",
                "reason_code": "similar_candidate_ambiguous",
                "legacy_document_id": "legacy:2",
            },
            {
                "legacy_row_id": 3,
                "official_section": "3.20",
                "status": "review_required",
                "reason_code": "legacy_source_truncated",
                "legacy_document_id": "legacy:3",
            },
        ]
    )
    assert [entry["section"] for entry in lineage[1]] == ["3.10"]
    assert 2 not in lineage
    assert [entry["section"] for entry in lineage[3]] == ["3.20"]


def test_metrics_accept_any_equivalent_section() -> None:
    records = [
        {
            "question_id": "q1",
            "acceptable_section_ids": ["section:3.10", "section:217.10"],
        }
    ]
    rankings = {
        "q1": [
            {"rank": 1, "section_ids": ["section:999"], "is_acceptable": False},
            {"rank": 2, "section_ids": ["section:217.10"], "is_acceptable": True},
        ]
    }
    metrics = evaluate_rankings(records, rankings)
    assert metrics["hit_rate"]["hit_at_1"] == 0.0
    assert metrics["hit_rate"]["hit_at_3"] == 1.0
    assert metrics["mrr_at_10"] == 0.5


def test_unmapped_results_are_reported() -> None:
    records = [{"question_id": "q1", "acceptable_section_ids": ["section:3.10"]}]
    rankings = {
        "q1": [
            {"rank": 1, "section_ids": [], "is_acceptable": False},
            {"rank": 2, "section_ids": ["section:3.10"], "is_acceptable": True},
        ]
    }
    metrics = evaluate_rankings(records, rankings)
    assert metrics["unmapped_results_at_10"] == 1
    assert metrics["unmapped_result_rate_at_10"] == 0.5


def test_failure_is_recorded_when_no_acceptable_result_exists() -> None:
    records = [{"question_id": "q1", "acceptable_section_ids": ["section:3.10"]}]
    rankings = {
        "q1": [
            {"rank": 1, "section_ids": ["section:999"], "is_acceptable": False}
        ]
    }
    metrics = evaluate_rankings(records, rankings)
    assert metrics["failures_at_10"] == ["q1"]
    assert metrics["mrr_at_10"] == 0.0


def test_search_benchmark_reports_repeated_batch_latency() -> None:
    scores, ids, latency = benchmark_search(FakeIndex(), [1, 2], k=10, repeats=3)
    assert scores == [10]
    assert ids == [2]
    assert latency["repeats"] == 3
    assert latency["batch_questions"] == 2
    assert latency["median_ms"] >= 0
