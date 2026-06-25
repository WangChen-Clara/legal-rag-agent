from rag_law.evaluation.answer_metrics import (
    missing_evidence_groups,
    normalize_answer,
    score_answer,
)


def test_normalize_answer_handles_case_punctuation_and_section_symbol() -> None:
    assert normalize_answer("  § 303.7, PUBLIC notice! ") == "section 3037 public notice"


def test_em_or_alias_accepts_exact_alias() -> None:
    record = {
        "metric_policy": "em_or_alias",
        "acceptable_answers": ["15 calendar days"],
        "must_contain": ["15 calendar days"],
    }

    result = score_answer(record, "15 calendar days.")

    assert result.passed is True
    assert result.matched_alias == "15 calendar days"
    assert result.missing_terms == ()


def test_em_or_alias_can_pass_with_required_terms_when_alias_is_not_exact() -> None:
    record = {
        "metric_policy": "em_or_alias",
        "acceptable_answers": ["yes"],
        "must_contain": ["yes", "E-Sign"],
    }

    result = score_answer(record, "Yes, if the E-Sign Act requirements are met.")

    assert result.passed is True
    assert result.matched_alias is None


def test_contains_all_reports_missing_terms() -> None:
    record = {
        "metric_policy": "contains_all",
        "acceptable_answers": [],
        "must_contain": ["50 percent", "$3 million", "greater"],
    }

    result = score_answer(record, "The limit is 50 percent.")

    assert result.passed is False
    assert result.missing_terms == ("$3 million", "greater")


def test_evidence_groups_require_one_section_from_each_group() -> None:
    record = {
        "metric_policy": "evidence_groups_and_contains",
        "acceptable_answers": [],
        "must_contain": ["three occasions", "303.7"],
        "required_evidence_groups": [["303.65"], ["303.7", "303.9"]],
    }

    result = score_answer(
        record,
        "Publish on three occasions and follow 303.7.",
        cited_sections=["303.65", "303.7"],
    )

    assert result.passed is True
    assert result.missing_evidence_groups == ()


def test_evidence_groups_report_missing_group() -> None:
    record = {
        "metric_policy": "evidence_groups_and_contains",
        "acceptable_answers": [],
        "must_contain": ["three occasions"],
        "required_evidence_groups": [["303.65"], ["303.7"]],
    }

    result = score_answer(
        record,
        "Publish on three occasions.",
        cited_sections=["303.65"],
    )

    assert result.passed is False
    assert result.missing_evidence_groups == (("303.7",),)


def test_missing_evidence_groups_uses_or_within_group() -> None:
    missing = missing_evidence_groups(["1012.237"], [["1010.505", "1012.237"]])

    assert missing == ()
