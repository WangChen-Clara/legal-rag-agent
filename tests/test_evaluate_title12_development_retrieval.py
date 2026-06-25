from scripts.evaluate_title12_development_retrieval import (
    evaluate_rankings,
    first_complete_rank,
    recall_at_k,
)


def test_single_section_first_complete_rank() -> None:
    record = {"question_type": "definition", "acceptable_sections": ["702.702"]}
    hits = [
        {"rank": 1, "section": "1.1"},
        {"rank": 2, "section": "702.702"},
    ]

    assert first_complete_rank(record, hits, 10) == 2
    assert recall_at_k(record, hits, 1) == 0.0
    assert recall_at_k(record, hits, 2) == 1.0


def test_cross_section_requires_all_evidence_groups() -> None:
    record = {
        "question_type": "cross_section",
        "required_evidence_groups": [["303.65"], ["303.7", "303.9"]],
    }
    hits = [
        {"rank": 1, "section": "303.9"},
        {"rank": 2, "section": "999.1"},
        {"rank": 3, "section": "303.65"},
    ]

    assert first_complete_rank(record, hits, 10) == 3
    assert recall_at_k(record, hits, 1) == 0.5
    assert recall_at_k(record, hits, 3) == 1.0


def test_cross_section_missing_group_has_no_complete_rank() -> None:
    record = {
        "question_type": "cross_section",
        "required_evidence_groups": [["303.65"], ["303.7"]],
    }
    hits = [
        {"rank": 1, "section": "303.65"},
        {"rank": 2, "section": "303.9"},
    ]

    assert first_complete_rank(record, hits, 10) is None
    assert recall_at_k(record, hits, 10) == 0.5


def test_evaluate_rankings_computes_hit_recall_and_mrr() -> None:
    records = [
        {
            "question_id": "q1",
            "question_type": "definition",
            "acceptable_sections": ["1.1"],
        },
        {
            "question_id": "q2",
            "question_type": "cross_section",
            "required_evidence_groups": [["2.1"], ["2.2"]],
        },
    ]
    rankings = {
        "q1": [{"rank": 1, "section": "1.1"}],
        "q2": [
            {"rank": 1, "section": "2.1"},
            {"rank": 2, "section": "9.9"},
            {"rank": 3, "section": "2.2"},
        ],
    }

    metrics = evaluate_rankings(records, rankings, ks=(1, 5, 10))

    assert metrics["hit_rate"] == {
        "hit_at_1": 0.5,
        "hit_at_5": 1.0,
        "hit_at_10": 1.0,
    }
    assert metrics["recall"] == {
        "recall_at_1": 0.75,
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
    }
    assert metrics["mrr_at_10"] == round((1.0 + 1 / 3) / 2, 6)
