from __future__ import annotations

import pytest

from scripts.finalize_title12_retrieval_eval import finalize_records


def candidate(section: str) -> dict[str, str]:
    return {
        "section": section,
        "document_id": f"section:{section}",
        "source_url": f"https://example.test/{section}",
    }


def record(status: str, sections: list[str]) -> dict[str, object]:
    return {
        "question_id": "q1",
        "question": "Question",
        "answer": "Answer",
        "gold_text": "Evidence",
        "label_status": status,
        "candidate_sections": [candidate(section) for section in sections],
    }


def test_unique_candidate_is_finalized_automatically() -> None:
    finalized = finalize_records([record("auto_labeled", ["3.2"])], {})
    assert finalized[0]["acceptable_sections"] == ["3.2"]
    assert finalized[0]["label_method"] == "auto_unique_exact"


def test_human_can_accept_equivalent_candidate_sections() -> None:
    finalized = finalize_records(
        [record("review_required", ["3.2", "217.2"])],
        {"q1": ["3.2", "217.2"]},
    )
    assert finalized[0]["acceptable_section_ids"] == [
        "section:3.2",
        "section:217.2",
    ]
    assert finalized[0]["label_method"] == "human_confirmed_equivalence"


def test_manual_label_must_be_a_candidate() -> None:
    with pytest.raises(ValueError, match="not candidates"):
        finalize_records(
            [record("review_required", ["3.2"])],
            {"q1": ["999.9"]},
        )


def test_all_review_records_require_manual_labels() -> None:
    with pytest.raises(ValueError, match="coverage mismatch"):
        finalize_records([record("review_required", ["3.2"])], {})


def test_unmatched_record_cannot_be_finalized() -> None:
    with pytest.raises(ValueError, match="Cannot finalize"):
        finalize_records([record("unmatched", [])], {})
