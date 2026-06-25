from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.sample_title12_eval_candidates import classify_section
except ModuleNotFoundError:
    from sample_title12_eval_candidates import classify_section


REVIEW_SCHEMA = "title12-eval-candidate-review-v1"
TYPE_PATTERNS = {
    "authority": re.compile(r"(?:\(a\)\s*)?authority\.|issued pursuant to", re.I),
    "applicability": re.compile(r"\b(?:applicability|scope)\b|\bappl(?:y|ies) to\b", re.I),
    "definition": re.compile(r"\bmeans\b", re.I),
    "numeric_or_date": re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:%|percent|days?|months?|years?)\b|"
        r"\b(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+\d{1,2}(?:,\s+\d{4})?\b",
        re.I,
    ),
    "obligation": re.compile(r"\b(?:must|shall|required to|may not)\b", re.I),
    "cross_section": re.compile(r"§+\s*[0-9]+[a-z]?(?:\.[0-9a-z-]+)+", re.I),
}
TYPE_FOCUS = {
    "authority": "statutory authority, purpose, or decision-making power",
    "applicability": "entities, transactions, or situations covered by the rule",
    "definition": "the meaning of one clearly named regulatory term",
    "numeric_or_date": "a threshold, percentage, deadline, or effective date",
    "obligation": "a required, prohibited, or conditionally permitted action",
    "cross_section": "how the source rule depends on a specifically cited provision",
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def evidence_window(text: str, pattern: re.Pattern[str], width: int = 700) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    start = max(0, match.start() - width // 3)
    end = min(len(text), start + width)
    return normalize_space(text[start:end])


def definition_term(text: str) -> str | None:
    match = re.search(r"(?:^|[.;:]\s+)([A-Z][A-Za-z0-9 /'’()-]{1,80}?)\s+means\b", text)
    return normalize_space(match.group(1)) if match else None


def review_candidate(
    candidate: dict[str, Any],
    section: dict[str, Any],
    sections_by_number: dict[str, dict[str, Any]],
    legacy_sections: set[str],
) -> dict[str, Any]:
    question_type = candidate["question_type"]
    detected_types = classify_section(section)
    excerpt = evidence_window(section["text"], TYPE_PATTERNS[question_type])
    heading_label = re.sub(r"^§\s*\S+\s*", "", section["heading"].lower()).strip()
    if excerpt is None and (
        (question_type == "authority" and heading_label.startswith("authority"))
        or (
            question_type == "applicability"
            and heading_label.startswith(("applicability", "scope"))
        )
    ):
        excerpt = normalize_space(section["text"][:700])
    valid_references = [
        reference
        for reference in candidate.get("cross_section_references", [])
        if reference in sections_by_number and reference != candidate["section"]
    ]
    flags: list[str] = []
    if candidate["text_length"] > 5000:
        flags.append("long_section")
    if len(detected_types) >= 4:
        flags.append("many_possible_question_types")
    if candidate["section"] in legacy_sections:
        flags.append("overlaps_legacy_baseline")
    if question_type == "cross_section" and len(valid_references) > 5:
        flags.append("many_cross_section_choices")

    proposed_type = question_type
    if question_type == "cross_section":
        recommendation = "needs_pair" if valid_references else "replace"
    elif excerpt:
        recommendation = "approved"
    else:
        alternatives = [
            value
            for value in TYPE_PATTERNS
            if value not in {"cross_section", question_type}
            and value in detected_types
            and TYPE_PATTERNS[value].search(section["text"])
        ]
        if alternatives:
            recommendation = "retype"
            proposed_type = alternatives[0]
            excerpt = evidence_window(section["text"], TYPE_PATTERNS[proposed_type])
        else:
            recommendation = "replace"

    difficulty = "hard" if question_type == "cross_section" else (
        "medium" if candidate["text_length"] > 3000 or flags else "easy"
    )
    focus = TYPE_FOCUS[proposed_type]
    term = definition_term(section["text"]) if proposed_type == "definition" else None
    if term:
        focus = f"definition of '{term}'"

    return {
        "candidate_id": candidate["candidate_id"],
        "split": candidate["split"],
        "section": candidate["section"],
        "heading": candidate["heading"],
        "assigned_type": question_type,
        "proposed_type": proposed_type,
        "recommendation": recommendation,
        "difficulty": difficulty,
        "suggested_question_focus": focus,
        "evidence_excerpt": excerpt,
        "suggested_partner_sections": valid_references[:5],
        "quality_flags": flags,
        "source_url": candidate["source_url"],
        "human_decision": None,
        "human_notes": None,
    }


def audit_candidates(
    candidates: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    legacy_sections: set[str],
) -> list[dict[str, Any]]:
    sections_by_id = {section["document_id"]: section for section in sections}
    sections_by_number = {section["section"]: section for section in sections}
    return [
        review_candidate(
            candidate,
            sections_by_id[candidate["document_id"]],
            sections_by_number,
            legacy_sections,
        )
        for candidate in candidates
    ]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def markdown_escape(text: str | None) -> str:
    return (text or "-").replace("|", "\\|")


def write_report(path: Path, reviews: list[dict[str, Any]]) -> None:
    counts = Counter(review["recommendation"] for review in reviews)
    summary_rows = [
        f"| {review['candidate_id']} | {review['split']} | § {review['section']} | "
        f"{review['assigned_type']} | {review['recommendation']} | {review['difficulty']} | "
        f"{', '.join(review['suggested_partner_sections']) or '-'} |"
        for review in reviews
    ]
    details = []
    for review in reviews:
        details.append(
            f"### {review['candidate_id']} · § {review['section']}\n\n"
            f"- Recommendation: `{review['recommendation']}`\n"
            f"- Assigned / proposed type: `{review['assigned_type']}` / `{review['proposed_type']}`\n"
            f"- Suggested focus: {review['suggested_question_focus']}\n"
            f"- Quality flags: {', '.join(review['quality_flags']) or 'none'}\n"
            f"- Suggested partners: {', '.join('§ ' + value for value in review['suggested_partner_sections']) or 'none'}\n\n"
            f"> {markdown_escape(review['evidence_excerpt'])}\n"
        )
    report = f"""# Title 12 Evaluation Candidate Review

This is an automatic first-pass recommendation generated only from the local official
snapshot. It reduces manual reading but does not replace human approval.

## Summary

- Candidates: {len(reviews)}
- Approved recommendations: {counts['approved']}
- Retype recommendations: {counts['retype']}
- Needs-pair recommendations: {counts['needs_pair']}
- Replace recommendations: {counts['replace']}

| Candidate | Split | Section | Assigned type | Recommendation | Difficulty | Partners |
|---|---|---|---|---|---|---|
{chr(10).join(summary_rows)}

## Detailed Evidence

{chr(10).join(details)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Audit Title 12 evaluation candidates")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_expansion_candidates.json",
    )
    parser.add_argument(
        "--sections",
        type=Path,
        default=root / "data" / "canonical" / "title12_2025-09-01" / "sections.jsonl",
    )
    parser.add_argument(
        "--legacy-eval",
        type=Path,
        default=root / "data" / "eval" / "title12_retrieval_eval.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_expansion_candidate_review.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_eval_expansion_candidate_review.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    legacy_payload = json.loads(args.legacy_eval.read_text(encoding="utf-8"))
    with args.sections.open("r", encoding="utf-8") as file:
        sections = [json.loads(line) for line in file if line.strip()]
    legacy_sections = {
        section
        for record in legacy_payload["records"]
        for section in record["acceptable_sections"]
    }
    reviews = audit_candidates(candidate_payload["candidates"], sections, legacy_sections)
    payload = {
        "schema": REVIEW_SCHEMA,
        "source_candidates": str(args.candidates.resolve()),
        "source_sections": str(args.sections.resolve()),
        "human_review_status": "pending",
        "summary": dict(Counter(review["recommendation"] for review in reviews)),
        "reviews": reviews,
    }
    atomic_write_json(args.output, payload)
    write_report(args.report, reviews)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
