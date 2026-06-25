from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


ANALYSIS_SCHEMA = "title12-development-retrieval-failure-analysis-v1"


def expected_sections(record: dict[str, Any]) -> list[str]:
    if record.get("acceptable_sections"):
        return [str(section) for section in record["acceptable_sections"]]
    return [
        str(section)
        for group in record.get("required_evidence_groups", [])
        for section in group
    ]


def first_section_rank(record: dict[str, Any], section: str) -> int | None:
    return next(
        (hit["rank"] for hit in record["top_hits"] if str(hit["section"]) == str(section)),
        None,
    )


def top_part_counts(record: dict[str, Any], k: int = 10) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for hit in record["top_hits"][:k]:
        counts[str(hit["section"]).split(".")[0]] += 1
    return dict(counts)


def classify_failure(record: dict[str, Any]) -> tuple[str, str]:
    expected = expected_sections(record)
    ranks = {section: first_section_rank(record, section) for section in expected}
    found = {section: rank for section, rank in ranks.items() if rank is not None}
    if record["question_type"] == "cross_section":
        if found:
            return (
                "cross_section_retrieval_design_issue",
                "The query retrieves part of the required evidence but not every required evidence group.",
            )
        return (
            "cross_section_ranking_issue",
            "None of the required evidence sections appear in the saved ranking.",
        )
    expected_parts = {section.split(".")[0] for section in expected}
    top_parts = top_part_counts(record)
    if expected_parts.intersection(top_parts):
        return (
            "same_part_ranking_issue",
            "The ranking stays in the expected CFR part but favors neighboring or competing sections.",
        )
    return (
        "ranking_issue",
        "The expected section is absent from the saved ranking and top results are not concentrated in the same part.",
    )


def analyze_failures(payload: dict[str, Any]) -> dict[str, Any]:
    failures = [
        record for record in payload["per_question"] if record["first_complete_rank"] is None
    ]
    analyses = []
    for record in failures:
        reason_code, reason = classify_failure(record)
        expected = expected_sections(record)
        analyses.append(
            {
                "question_id": record["question_id"],
                "candidate_id": record["candidate_id"],
                "question_type": record["question_type"],
                "question": record["question"],
                "expected_sections": expected,
                "expected_section_ranks": {
                    section: first_section_rank(record, section) for section in expected
                },
                "top_10_sections": [hit["section"] for hit in record["top_hits"][:10]],
                "top_part_counts": top_part_counts(record),
                "reason_code": reason_code,
                "reason": reason,
                "recommended_action": recommended_action(reason_code),
            }
        )
    return {
        "schema": ANALYSIS_SCHEMA,
        "source_eval": payload.get("schema"),
        "questions": payload["questions"],
        "failure_count": len(analyses),
        "reason_counts": dict(Counter(item["reason_code"] for item in analyses)),
        "holdout_retrieval_inspected": False,
        "failures": analyses,
    }


def recommended_action(reason_code: str) -> str:
    if reason_code == "same_part_ranking_issue":
        return (
            "Inspect the target section chunk text against higher-ranked same-part chunks; "
            "consider rewriting the development question or adding section-heading/query context."
        )
    if reason_code == "cross_section_retrieval_design_issue":
        return (
            "Do not tune the base index first; evaluate query decomposition or evidence expansion "
            "from the retrieved primary section to referenced partner sections."
        )
    if reason_code == "cross_section_ranking_issue":
        return "Inspect whether the cross-section question is too indirect or the evidence label is too strict."
    return "Inspect top-ranked chunks and target section chunks before changing index parameters."


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_report(path: Path, analysis: dict[str, Any]) -> None:
    rows = []
    details = []
    for item in analysis["failures"]:
        rows.append(
            f"| {item['question_id']} | {item['question_type']} | "
            f"{', '.join(item['expected_sections'])} | {item['reason_code']} |"
        )
        ranks = ", ".join(
            f"{section}: {rank if rank is not None else '-'}"
            for section, rank in item["expected_section_ranks"].items()
        )
        details.append(
            f"## {item['question_id']}\n\n"
            f"- Candidate: `{item['candidate_id']}`\n"
            f"- Type: `{item['question_type']}`\n"
            f"- Expected ranks: {ranks}\n"
            f"- Top-10 sections: {', '.join(item['top_10_sections'])}\n"
            f"- Reason: `{item['reason_code']}` - {item['reason']}\n"
            f"- Recommended action: {item['recommended_action']}\n\n"
            f"Question: {item['question']}\n"
        )
    report = f"""# Title 12 Development Retrieval Failure Analysis

- Schema: `{analysis['schema']}`
- Questions: {analysis['questions']}
- Top-10 failures: {analysis['failure_count']}
- Holdout retrieval inspected: no
- Reason counts: {json.dumps(analysis['reason_counts'], ensure_ascii=False)}

| Question | Type | Expected sections | Reason |
|---|---|---|---|
{chr(10).join(rows) if rows else '| - | - | - | No failures |'}

{chr(10).join(details)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Analyze Title 12 development retrieval failures")
    parser.add_argument(
        "--eval",
        type=Path,
        default=root / "reports" / "title12_development_retrieval_eval.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "title12_development_retrieval_failures.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_development_retrieval_failures.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.eval.read_text(encoding="utf-8"))
    analysis = analyze_failures(payload)
    atomic_write_json(args.output, analysis)
    write_report(args.report, analysis)
    print(
        json.dumps(
            {
                "failure_count": analysis["failure_count"],
                "reason_counts": analysis["reason_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
