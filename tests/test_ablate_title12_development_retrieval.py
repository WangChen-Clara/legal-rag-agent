from scripts.ablate_title12_development_retrieval import (
    expand_cross_references,
    query_text,
    summarize_variant,
)


def test_query_text_appends_unique_headings() -> None:
    record = {
        "question": "What applies?",
        "source_citations": [
            {"heading": "§ 1.1 Scope."},
            {"heading": "§ 1.1 Scope."},
            {"heading": "§ 1.2 Definitions."},
        ],
    }

    text = query_text(record, with_heading=True)

    assert text == "What applies?\nRelevant section heading: § 1.1 Scope.; § 1.2 Definitions."


def test_query_text_can_leave_question_unchanged() -> None:
    record = {"question": "What applies?", "source_citations": [{"heading": "§ 1.1 Scope."}]}

    assert query_text(record, with_heading=False) == "What applies?"


def test_expand_cross_references_adds_missing_groups_after_partial_hit() -> None:
    record = {
        "question_type": "cross_section",
        "required_evidence_groups": [["217.135"], ["217.134"]],
    }
    hits = [{"rank": 1, "section": "217.135"}]

    expanded = expand_cross_references(record, hits)

    assert [hit["section"] for hit in expanded] == ["217.135", "217.134"]
    assert expanded[-1]["expanded"] is True


def test_expand_cross_references_does_not_add_without_partial_hit() -> None:
    record = {
        "question_type": "cross_section",
        "required_evidence_groups": [["217.135"], ["217.134"]],
    }
    hits = [{"rank": 1, "section": "999.1"}]

    assert expand_cross_references(record, hits) == hits


def test_summarize_variant_reports_focus_questions() -> None:
    records = [
        {
            "question_id": "title12-dev-q001",
            "question_type": "definition",
            "acceptable_sections": ["211.31"],
        },
        {
            "question_id": "title12-dev-q018",
            "question_type": "cross_section",
            "required_evidence_groups": [["217.135"], ["217.134"]],
        },
    ]
    rankings = {
        "title12-dev-q001": [{"rank": 1, "section": "211.31"}],
        "title12-dev-q018": [
            {"rank": 1, "section": "217.135"},
            {"rank": 2, "section": "217.134", "expanded": True},
        ],
    }

    summary = summarize_variant(records, rankings)

    assert summary["metrics"]["hit_rate"]["hit_at_1"] == 0.5
    assert summary["metrics"]["hit_rate"]["hit_at_5"] == 1.0
    assert summary["focus_questions"]["title12-dev-q018"]["expanded_sections"] == ["217.134"]
