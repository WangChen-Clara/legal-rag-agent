from __future__ import annotations

from scripts.prepare_title12_retrieval_eval import (
    candidate_sections,
    evidence_windows,
    label_items,
    normalize_text,
)


def section(section_id: str, text: str) -> dict[str, object]:
    return {
        "document_id": f"title12:{section_id}",
        "title": "12",
        "part": section_id.split(".")[0],
        "section": section_id,
        "heading": "Heading",
        "text": text,
        "version_date": "2025-09-01",
        "source_url": f"https://example.test/{section_id}",
    }


def test_normalize_text_repairs_only_formatting_artifacts() -> None:
    assert normalize_text('  "alpha\n beta\\,"  ') == "alpha beta"


def test_evidence_windows_are_deterministic() -> None:
    text = "".join(str(index % 10) for index in range(500))
    assert evidence_windows(text, 80) == evidence_windows(text, 80)
    assert all(len(window) == 80 for window in evidence_windows(text, 80))


def test_candidate_matching_prefers_longest_exact_window() -> None:
    evidence = "prefix " + ("specific evidence " * 20) + "suffix"
    sections = [
        section("1.1", "before " + evidence + " after"),
        section("2.1", "specific evidence " * 6),
    ]
    matches, size = candidate_sections(evidence, sections)
    assert [match["section"] for match in matches] == ["1.1"]
    assert size == 160


def test_unique_candidate_is_auto_labeled() -> None:
    evidence = "unique legal evidence " * 20
    records = label_items(
        [{"Q": "Question", "A": "Answer", "Text": evidence}],
        [section("1.1", evidence)],
    )
    assert records[0]["label_status"] == "auto_labeled"
    assert records[0]["gold_section_ids"] == ["title12:1.1"]


def test_multiple_candidates_require_review() -> None:
    evidence = "duplicated legal evidence " * 20
    records = label_items(
        [{"Q": "Question", "A": "Answer", "Text": evidence}],
        [section("1.1", evidence), section("2.1", evidence)],
    )
    assert records[0]["label_status"] == "review_required"
    assert records[0]["gold_section_ids"] == []


def test_unmatched_evidence_is_not_labeled() -> None:
    records = label_items(
        [{"Q": "Question", "A": "Answer", "Text": "missing evidence " * 20}],
        [section("1.1", "different official text " * 20)],
    )
    assert records[0]["label_status"] == "unmatched"
    assert records[0]["candidate_sections"] == []
