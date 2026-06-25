from __future__ import annotations

from scripts.audit_title12_eval_candidates import (
    audit_candidates,
    definition_term,
    evidence_window,
    TYPE_PATTERNS,
)


def section(text: str) -> dict[str, object]:
    return {
        "document_id": "section:1.1",
        "section": "1.1",
        "heading": "§ 1.1 Definitions.",
        "text": text,
    }


def candidate(question_type: str, references: list[str] | None = None) -> dict[str, object]:
    return {
        "candidate_id": "candidate-1",
        "question_type": question_type,
        "document_id": "section:1.1",
        "split": "development",
        "section": "1.1",
        "heading": "§ 1.1 Definitions.",
        "text_length": 500,
        "source_url": "https://example.test/1.1",
        "cross_section_references": references or [],
    }


def test_definition_evidence_and_term_are_extracted() -> None:
    text = "Definitions. Capital means the total qualifying amount under this rule."
    assert "Capital means" in evidence_window(text, TYPE_PATTERNS["definition"])
    assert definition_term(text) == "Capital"


def test_matching_single_section_candidate_is_approved() -> None:
    text = "Definitions. Capital means the total amount. Other term means another amount."
    reviews = audit_candidates([candidate("definition")], [section(text)], set())
    assert reviews[0]["recommendation"] == "approved"
    assert reviews[0]["suggested_question_focus"] == "definition of 'Capital'"


def test_cross_section_candidate_requires_a_valid_partner() -> None:
    source = section("See § 2.1 and § 3.1 for the required calculations.")
    partner = {**section("Partner text"), "document_id": "section:2.1", "section": "2.1"}
    reviews = audit_candidates(
        [candidate("cross_section", ["2.1", "9.9"])],
        [source, partner],
        set(),
    )
    assert reviews[0]["recommendation"] == "needs_pair"
    assert reviews[0]["suggested_partner_sections"] == ["2.1"]


def test_authority_heading_can_anchor_evidence() -> None:
    source = {
        **section("The judge may issue orders and administer oaths."),
        "heading": "§ 1.1 Authority of the judge.",
    }
    item = {**candidate("authority"), "heading": source["heading"]}
    reviews = audit_candidates([item], [source], set())
    assert reviews[0]["recommendation"] == "approved"
    assert reviews[0]["evidence_excerpt"].startswith("The judge")


def test_candidate_without_matching_evidence_is_retyped_or_replaced() -> None:
    text = "The institution must submit a report and shall retain a copy."
    reviews = audit_candidates([candidate("definition")], [section(text)], set())
    assert reviews[0]["recommendation"] == "retype"
    assert reviews[0]["proposed_type"] == "obligation"
