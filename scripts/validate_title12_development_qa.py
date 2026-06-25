from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from rag_law.evaluation.answer_metrics import SUPPORTED_POLICIES, score_answer


REQUIRED_COMMON_FIELDS = {
    "question_id",
    "candidate_id",
    "split",
    "question_type",
    "question",
    "expected_answer",
    "metric_policy",
    "acceptable_answers",
    "must_contain",
    "draft_status",
}


def validate_record(record: dict[str, Any]) -> list[str]:
    errors = []
    missing = sorted(REQUIRED_COMMON_FIELDS - set(record))
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if record.get("split") != "development":
        errors.append("split must be development")
    if record.get("metric_policy") not in SUPPORTED_POLICIES:
        errors.append(f"unsupported metric_policy: {record.get('metric_policy')}")
    if not isinstance(record.get("acceptable_answers"), list) or not record.get(
        "acceptable_answers"
    ):
        errors.append("acceptable_answers must be a non-empty list")
    if not isinstance(record.get("must_contain"), list) or not record.get("must_contain"):
        errors.append("must_contain must be a non-empty list")

    if record.get("question_type") == "cross_section":
        groups = record.get("required_evidence_groups")
        if not isinstance(groups, list) or not groups:
            errors.append("cross_section records need required_evidence_groups")
        elif any(not isinstance(group, list) or not group for group in groups):
            errors.append("required_evidence_groups must be non-empty lists")
    else:
        sections = record.get("acceptable_sections")
        if not isinstance(sections, list) or not sections:
            errors.append("single-section records need acceptable_sections")

    if not errors:
        cited_sections = (
            [section for group in record.get("required_evidence_groups", []) for section in group]
            if record.get("question_type") == "cross_section"
            else record.get("acceptable_sections", [])
        )
        result = score_answer(
            record,
            record["expected_answer"],
            cited_sections=cited_sections,
        )
        if not result.passed:
            details = []
            if result.missing_terms:
                details.append(f"missing terms: {', '.join(result.missing_terms)}")
            if result.missing_evidence_groups:
                details.append(f"missing evidence groups: {result.missing_evidence_groups}")
            errors.append("expected_answer does not pass metric policy" + (
                f" ({'; '.join(details)})" if details else ""
            ))
    return errors


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records", [])
    errors = []
    seen_ids = set()
    for index, record in enumerate(records):
        question_id = record.get("question_id", f"<record-{index}>")
        if question_id in seen_ids:
            errors.append({"question_id": question_id, "errors": ["duplicate question_id"]})
            continue
        seen_ids.add(question_id)
        record_errors = validate_record(record)
        if record_errors:
            errors.append({"question_id": question_id, "errors": record_errors})
    return {
        "question_count": len(records),
        "error_count": len(errors),
        "errors": errors,
    }


def write_report(path: Path, validation: dict[str, Any]) -> None:
    if validation["errors"]:
        error_lines = []
        for item in validation["errors"]:
            error_lines.append(f"- `{item['question_id']}`: {'; '.join(item['errors'])}")
        status = "failed"
        detail = "\n".join(error_lines)
    else:
        status = "passed"
        detail = "- None"
    report = f"""# Title 12 Development QA Validation

- Status: `{status}`
- Questions: {validation['question_count']}
- Errors: {validation['error_count']}
- Holdout retrieval inspected: no

## Errors

{detail}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate Title 12 development QA draft")
    parser.add_argument(
        "--qa",
        type=Path,
        default=root / "data" / "eval" / "title12_development_qa_draft.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_development_qa_validation.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.qa.read_text(encoding="utf-8"))
    validation = validate_payload(payload)
    write_report(args.report, validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if validation["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
