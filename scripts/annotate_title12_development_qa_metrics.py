from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


METRICS: dict[str, dict[str, Any]] = {
    "title12-dev-q001": {
        "metric_policy": "em_or_alias",
        "acceptable_answers": ["eligible investors", "eligible investors as defined in the subpart"],
        "must_contain": ["eligible investors"],
    },
    "title12-dev-q002": {
        "metric_policy": "contains_all",
        "acceptable_answers": [
            "the greater of 50 percent of paid-in and unimpaired capital and surplus less public unit and nonmember shares, or $3 million",
            "50 percent or $3 million, whichever is greater",
        ],
        "must_contain": ["50 percent", "$3 million", "greater"],
    },
    "title12-dev-q003": {
        "metric_policy": "em_or_alias",
        "acceptable_answers": ["yes", "yes, subject to E-Sign Act compliance"],
        "must_contain": ["yes", "E-Sign"],
    },
    "title12-dev-q004": {
        "metric_policy": "contains_all",
        "acceptable_answers": [
            "section 19 applications filed by an IDI, depository institution holding company, or individual after denial under 12 CFR part 303, subpart L"
        ],
        "must_contain": ["section 19", "denied", "12 CFR part 303, subpart L"],
    },
    "title12-dev-q005": {
        "metric_policy": "em_or_alias",
        "acceptable_answers": [
            "Appraisal Subcommittee of the Federal Financial Institutions Examination Council"
        ],
        "must_contain": ["Appraisal Subcommittee", "Federal Financial Institutions Examination Council"],
    },
    "title12-dev-q006": {
        "metric_policy": "contains_all",
        "acceptable_answers": [
            "any customer other than specified institutional customers or an entity with total assets of at least $50 million"
        ],
        "must_contain": ["customer", "$50 million"],
    },
    "title12-dev-q007": {
        "metric_policy": "em_or_alias",
        "acceptable_answers": [
            "12-quarter reporting period",
            "the 12-quarter reporting period beginning the first day of the fiscal year in which the credit union adopts CECL",
        ],
        "must_contain": ["12-quarter", "adopts CECL"],
    },
    "title12-dev-q008": {
        "metric_policy": "contains_all",
        "acceptable_answers": [
            "assess the validity of the address change before issuing the additional or replacement card"
        ],
        "must_contain": ["assesses the validity", "address change", "additional or replacement card"],
    },
    "title12-dev-q009": {
        "metric_policy": "contains_all",
        "acceptable_answers": [
            "a business entity that a bank can direct or cause the direction of management and policies"
        ],
        "must_contain": ["business entity", "bank", "management and policies"],
    },
    "title12-dev-q010": {
        "metric_policy": "contains_all",
        "acceptable_answers": ["one year", "one year after leaving FHFA employment"],
        "must_contain": ["one year", "two or more months", "last 12 months"],
    },
    "title12-dev-q011": {
        "metric_policy": "em_or_alias",
        "acceptable_answers": ["15 calendar days", "within 15 calendar days"],
        "must_contain": ["15 calendar days"],
    },
    "title12-dev-q012": {
        "metric_policy": "contains_all",
        "acceptable_answers": [
            "the family member's interest in the same organization is imputed to the person"
        ],
        "must_contain": ["imputed", "same organization"],
    },
    "title12-dev-q013": {
        "metric_policy": "contains_all",
        "acceptable_answers": [
            "annual percentage yield earned, interest earned, fees imposed, statement period length or dates, and applicable aggregate overdraft and returned item fees"
        ],
        "must_contain": [
            "annual percentage yield earned",
            "interest",
            "fees",
            "statement period",
        ],
    },
    "title12-dev-q014": {
        "metric_policy": "contains_all",
        "acceptable_answers": [
            "informal NCUA proceedings to suspend, remove, or prohibit an institution-affiliated party of an insured credit union"
        ],
        "must_contain": ["informal proceedings", "NCUA", "institution-affiliated party"],
    },
    "title12-dev-q015": {
        "metric_policy": "em_or_alias",
        "acceptable_answers": [
            "a statistical tool or algorithm created by a third party to predict credit behaviors",
            "a statistical tool or algorithm created by a third party used to produce a value or category predicting credit behavior",
        ],
        "must_contain": ["statistical tool or algorithm", "third party", "credit behaviors"],
    },
    "title12-dev-q016": {
        "metric_policy": "contains_all",
        "acceptable_answers": [
            "when the debt is waived or found not owed, or an administrative or judicial order directs a refund"
        ],
        "must_contain": ["waived", "owed", "administrative or judicial order"],
    },
    "title12-dev-q017": {
        "metric_policy": "evidence_groups_and_contains",
        "acceptable_answers": [
            "at least three occasions at approximately equal intervals, and the notice must conform to 12 CFR 303.7"
        ],
        "must_contain": ["three occasions", "approximately equal intervals", "303.7"],
    },
    "title12-dev-q018": {
        "metric_policy": "evidence_groups_and_contains",
        "acceptable_answers": [
            "a wholesale exposure other than a sovereign exposure, with coverage described in 12 CFR 217.134(a)(1)"
        ],
        "must_contain": ["wholesale exposure", "sovereign exposure", "217.134"],
    },
    "title12-dev-q019": {
        "metric_policy": "evidence_groups_and_contains",
        "acceptable_answers": [
            "the voting procedures must comply with 12 CFR 611.340, including confidentiality and security of stockholder voting information"
        ],
        "must_contain": ["611.340", "confidential", "secure"],
    },
    "title12-dev-q020": {
        "metric_policy": "evidence_groups_and_contains",
        "acceptable_answers": [
            "it must comply with 12 CFR 1209.15(c)'s formal requirements for filed papers"
        ],
        "must_contain": ["1209.15(c)", "formal requirements", "filed papers"],
    },
}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    rows = []
    details = []
    for record in payload["records"]:
        rows.append(
            f"| {record['question_id']} | {record['candidate_id']} | "
            f"{record['question_type']} | {record['metric_policy']} | "
            f"{', '.join(record['must_contain'])} |"
        )
        details.append(
            f"## {record['question_id']} · {record['candidate_id']}\n\n"
            f"- Type: `{record['question_type']}`\n"
            f"- Metric policy: `{record['metric_policy']}`\n"
            f"- Draft status: `{record['draft_status']}`\n"
            f"- Must contain: {', '.join(record['must_contain'])}\n\n"
            f"Question: {record['question']}\n\n"
            f"Expected answer: {record['expected_answer']}\n\n"
            f"Acceptable answers:\n"
            + "\n".join(f"- {answer}" for answer in record["acceptable_answers"])
            + "\n"
        )
    report = f"""# Title 12 Development QA Draft

- Schema: `{payload['schema']}`
- Split: development
- Questions: {payload['question_count']}
- Draft status: needs human review
- Holdout QA generated: no
- Holdout retrieval inspected: no
- Metric policies: {json.dumps(payload['metric_policy_counts'], ensure_ascii=False)}

| Question ID | Candidate | Type | Metric policy | Must contain |
|---|---|---|---|---|
{chr(10).join(rows)}

{chr(10).join(details)}
"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Annotate Title 12 development QA metrics")
    parser.add_argument(
        "--qa",
        type=Path,
        default=root / "data" / "eval" / "title12_development_qa_draft.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_development_qa_draft.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.qa.read_text(encoding="utf-8"))
    missing = sorted(set(record["question_id"] for record in payload["records"]) - set(METRICS))
    if missing:
        raise ValueError(f"Missing metric annotations for: {missing}")

    for record in payload["records"]:
        metric = METRICS[record["question_id"]]
        record["metric_policy"] = metric["metric_policy"]
        record["acceptable_answers"] = metric["acceptable_answers"]
        record["must_contain"] = metric["must_contain"]

    payload["metric_policy_counts"] = dict(
        Counter(record["metric_policy"] for record in payload["records"])
    )
    atomic_write_json(args.qa, payload)
    write_report(args.report, payload)
    print(
        json.dumps(
            {
                "questions": payload["question_count"],
                "metric_policy_counts": payload["metric_policy_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
