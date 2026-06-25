from __future__ import annotations

import pytest

from scripts.apply_title12_eval_review_decisions import (
    apply_decisions,
    apply_replacement_selections,
)


def candidate(candidate_id: str, question_type: str, section: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "question_type": question_type,
        "section": section,
        "split": "development",
    }


def auto_review(candidate_id: str, recommendation: str = "approved") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "recommendation": recommendation,
        "evidence_excerpt": "Evidence",
    }


def test_unreviewed_auto_approval_remains_pending() -> None:
    single, cross, excluded = apply_decisions(
        [candidate("c1", "definition", "1.1")],
        [auto_review("c1")],
        {},
        {"1.1"},
    )
    assert single[0]["review_state"] == "auto_prescreened_pending"
    assert not cross and not excluded


def test_human_retype_can_keep_multiple_single_section_types() -> None:
    single, _, _ = apply_decisions(
        [candidate("c1", "authority", "1.1")],
        [auto_review("c1")],
        {
            "c1": {
                "action": "retype",
                "approved_types": ["numeric_or_date", "obligation"],
                "question_constraints": ["Keep the question narrow."],
                "reason": "Human review",
            }
        },
        {"1.1"},
    )
    assert single[0]["approved_question_types"] == [
        "numeric_or_date",
        "obligation",
    ]
    assert single[0]["review_state"] == "human_confirmed_retype"
    assert single[0]["question_constraints"] == ["Keep the question narrow."]


def test_cross_section_partner_must_exist() -> None:
    with pytest.raises(ValueError, match="Invalid partner sections"):
        apply_decisions(
            [candidate("c1", "cross_section", "1.1")],
            [auto_review("c1", "needs_pair")],
            {
                "c1": {
                    "action": "needs_pair",
                    "partner_options": [
                        {"default_partner_sections": ["9.9"], "purpose": "test"}
                    ],
                    "reason": "Human review",
                }
            },
            {"1.1"},
        )


def test_replace_excludes_candidate() -> None:
    single, cross, excluded = apply_decisions(
        [candidate("c1", "authority", "1.1")],
        [auto_review("c1")],
        {"c1": {"action": "replace", "reason": "Weak candidate"}},
        {"1.1"},
    )
    assert not single and not cross
    assert excluded[0]["candidate_id"] == "c1"


def test_human_selected_replacement_is_merged_into_single_pool() -> None:
    replacements = [
        {
            "replacement_id": "r1",
            "replacement_for": "c1",
            "intended_split": "development",
            "question_type": "applicability",
            "section": "2.1",
            "evidence_preview": "Evidence",
        }
    ]
    merged = apply_replacement_selections(
        [],
        replacements,
        {
            "c1": {
                "selected_replacement_id": "r1",
                "approved_types": ["applicability"],
                "reason": "Human selection",
            }
        },
        {"2.1"},
    )
    assert merged[0]["candidate_id"] == "r1"
    assert merged[0]["replaces_candidate_id"] == "c1"
    assert merged[0]["review_state"] == "human_confirmed_replacement"
