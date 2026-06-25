from scripts.analyze_title12_development_retrieval_failures import (
    analyze_failures,
    classify_failure,
    expected_sections,
    first_section_rank,
)


def single_failure() -> dict:
    return {
        "question_id": "q1",
        "candidate_id": "c1",
        "question_type": "applicability",
        "question": "What applies?",
        "acceptable_sections": ["211.31"],
        "first_complete_rank": None,
        "top_hits": [
            {"rank": 1, "section": "211.10"},
            {"rank": 2, "section": "211.8"},
        ],
    }


def cross_failure() -> dict:
    return {
        "question_id": "q2",
        "candidate_id": "c2",
        "question_type": "cross_section",
        "question": "What cross evidence?",
        "required_evidence_groups": [["217.135"], ["217.134"]],
        "first_complete_rank": None,
        "top_hits": [
            {"rank": 1, "section": "217.135"},
            {"rank": 2, "section": "324.135"},
        ],
    }


def test_expected_sections_flattens_cross_groups() -> None:
    assert expected_sections(cross_failure()) == ["217.135", "217.134"]


def test_first_section_rank_returns_none_when_absent() -> None:
    assert first_section_rank(single_failure(), "211.31") is None
    assert first_section_rank(single_failure(), "211.10") == 1


def test_classify_same_part_ranking_issue() -> None:
    reason_code, _ = classify_failure(single_failure())

    assert reason_code == "same_part_ranking_issue"


def test_classify_cross_section_partial_retrieval_issue() -> None:
    reason_code, _ = classify_failure(cross_failure())

    assert reason_code == "cross_section_retrieval_design_issue"


def test_analyze_failures_summarizes_reason_counts() -> None:
    payload = {
        "schema": "test-schema",
        "questions": 2,
        "per_question": [single_failure(), cross_failure()],
    }

    analysis = analyze_failures(payload)

    assert analysis["failure_count"] == 2
    assert analysis["reason_counts"] == {
        "same_part_ranking_issue": 1,
        "cross_section_retrieval_design_issue": 1,
    }
