from __future__ import annotations

from collections import Counter

from scripts.sample_title12_eval_candidates import (
    assign_splits,
    classify_section,
    group_equivalent_candidates,
    select_candidates,
)


def section(section_id: str, text: str, heading: str = "Rule") -> dict[str, object]:
    return {
        "document_id": f"section:{section_id}",
        "title": "12",
        "part": section_id.split(".")[0],
        "section": section_id,
        "heading": heading,
        "text": text,
        "text_sha256": section_id,
        "normalized_text_sha256": section_id,
        "version_date": "2025-09-01",
        "source_url": f"https://example.test/{section_id}",
        "safe_for_citation": True,
    }


def test_section_classification_supports_multiple_types() -> None:
    item = section(
        "1.1",
        "This rule applies to banks. A bank must file reports within 30 days. " * 8,
        "Authority and applicability",
    )
    types = classify_section(item)
    assert {"authority", "applicability", "numeric_or_date", "obligation"} <= types


def test_candidate_selection_is_deterministic_and_respects_quotas() -> None:
    sections = [
        section(
            f"{index}.1",
            ("Term means value. Other term means another value. " * 20),
            "Definitions",
        )
        for index in range(1, 10)
    ]
    first = select_candidates(sections, {"definition": 3}, seed=7)
    second = select_candidates(sections, {"definition": 3}, seed=7)
    assert [item["document_id"] for item in first] == [
        item["document_id"] for item in second
    ]
    assert Counter(item["question_type"] for item in first) == {"definition": 3}


def test_equivalent_candidates_stay_in_one_split() -> None:
    common = "common regulatory language " * 50
    sections = [section("1.1", common), section("2.1", common), section("3.1", "different " * 80)]
    candidates = [
        {
            "document_id": item["document_id"],
            "normalized_text_sha256": item["normalized_text_sha256"],
        }
        for item in sections
    ]
    families = group_equivalent_candidates(
        candidates, {item["document_id"]: item for item in sections}
    )
    assign_splits(candidates, families, seed=7)
    assert families["section:1.1"] == families["section:2.1"]
    assert candidates[0]["split"] == candidates[1]["split"]
