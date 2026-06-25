from scripts.validate_title12_development_qa import validate_payload, validate_record


def single_record() -> dict:
    return {
        "question_id": "title12-dev-q001",
        "candidate_id": "title12-exp-001",
        "split": "development",
        "question_type": "definition",
        "question": "What is the term?",
        "expected_answer": "15 calendar days",
        "metric_policy": "em_or_alias",
        "acceptable_answers": ["15 calendar days"],
        "must_contain": ["15 calendar days"],
        "acceptable_sections": ["797.19"],
        "draft_status": "needs_human_review",
    }


def cross_record() -> dict:
    record = single_record()
    record.update(
        {
            "question_id": "title12-dev-q002",
            "question_type": "cross_section",
            "metric_policy": "evidence_groups_and_contains",
            "expected_answer": "Publish on three occasions and follow 303.7.",
            "acceptable_answers": ["three occasions and 303.7"],
            "must_contain": ["three occasions", "303.7"],
            "required_evidence_groups": [["303.65"], ["303.7"]],
        }
    )
    record.pop("acceptable_sections")
    return record


def test_validate_record_accepts_single_section_record() -> None:
    assert validate_record(single_record()) == []


def test_validate_record_accepts_cross_section_record() -> None:
    assert validate_record(cross_record()) == []


def test_validate_record_reports_missing_required_fields() -> None:
    record = single_record()
    record.pop("metric_policy")

    errors = validate_record(record)

    assert "missing fields: metric_policy" in errors


def test_validate_record_checks_expected_answer_against_policy() -> None:
    record = single_record()
    record["expected_answer"] = "wrong answer"

    errors = validate_record(record)

    assert any("expected_answer does not pass metric policy" in error for error in errors)


def test_validate_payload_reports_duplicate_question_ids() -> None:
    payload = {"records": [single_record(), single_record()]}

    validation = validate_payload(payload)

    assert validation["error_count"] == 1
    assert validation["errors"][0]["errors"] == ["duplicate question_id"]
