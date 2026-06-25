from __future__ import annotations

from collections import Counter

from scripts.sample_title12_eval_replacements import sample_replacements


def section(index: int, question_type: str) -> dict[str, object]:
    if question_type == "applicability":
        heading = "Scope."
        text = "This rule applies to covered banks. " * 20
    elif question_type == "obligation":
        heading = "Requirements."
        text = "The bank must file a report and shall retain records. " * 20
    else:
        heading = "Definitions."
        text = "Covered account means an eligible account. " * 20
    return {
        "document_id": f"section:{index}",
        "part": str(index),
        "section": f"{index}.1",
        "heading": f"§ {index}.1 {heading}",
        "text": text,
        "text_sha256": str(index),
        "normalized_text_sha256": f"normalized-{index}",
        "version_date": "2025-09-01",
        "source_url": f"https://example.test/{index}.1",
        "safe_for_citation": True,
    }


def test_replacement_shortlist_is_deterministic_and_respects_quotas() -> None:
    sections = []
    for question_type, start in (("applicability", 1), ("obligation", 11), ("definition", 21)):
        sections.extend(section(index, question_type) for index in range(start, start + 5))
    first = sample_replacements(sections, set(), set(), set(), seed=7)
    second = sample_replacements(sections, set(), set(), set(), seed=7)
    assert [item["document_id"] for item in first] == [item["document_id"] for item in second]
    assert Counter(item["question_type"] for item in first) == {
        "applicability": 2,
        "obligation": 2,
        "definition": 1,
    }
    assert all(item["intended_split"] == "development" for item in first)


def test_existing_and_legacy_sections_are_excluded() -> None:
    sections = [section(index, "definition") for index in range(1, 5)]
    replacements = sample_replacements(
        sections,
        {"section:1"},
        {"2.1"},
        {"normalized-3"},
        seed=7,
        quotas={"definition": 1},
    )
    assert replacements[0]["section"] == "4.1"
