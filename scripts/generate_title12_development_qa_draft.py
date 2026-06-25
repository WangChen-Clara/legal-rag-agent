from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "title12-development-qa-draft-v1"


DRAFTS: dict[str, dict[str, Any]] = {
    "title12-exp-004": {
        "question": "What investors do the provisions of 12 CFR 211.31's subpart apply to?",
        "expected_answer": "They apply to eligible investors as defined in the subpart.",
    },
    "title12-exp-005": {
        "question_type": "numeric_or_date",
        "question": "What aggregate limit applies to public unit and nonmember shares received by a federal credit union under 12 CFR 701.32?",
        "expected_answer": "Except as permitted by paragraph (c), the federal credit union may not receive public unit and nonmember shares above the greater of 50 percent of the net amount of paid-in and unimpaired capital and surplus less those shares, measured when each share is accepted, or $3 million.",
    },
    "title12-exp-007": {
        "question": "May the disclosures required by Regulation X be provided electronically under 12 CFR 1024.3?",
        "expected_answer": "Yes. They may be provided in electronic form if the consumer consent and other applicable E-Sign Act requirements are satisfied.",
    },
    "title12-exp-008": {
        "question": "When do the procedures in 12 CFR 308.156 apply to a section 19 application?",
        "expected_answer": "They apply to a section 19 application filed by an insured depository institution, depository institution holding company, or individual only after the application has been denied under 12 CFR part 303, subpart L.",
    },
    "title12-exp-019": {
        "question": "What does ASC or Subcommittee mean under 12 CFR 1102.101?",
        "expected_answer": "It means the Appraisal Subcommittee of the Federal Financial Institutions Examination Council.",
    },
    "title12-exp-020": {
        "question": "Under 12 CFR 368.2, what is a non-institutional customer?",
        "expected_answer": "It is any customer other than a bank, savings association, insurance company, registered investment company, registered investment adviser, or an entity with total assets of at least $50 million.",
    },
    "title12-exp-022": {
        "question": "What is the transition period for CECL under 12 CFR 702.702?",
        "expected_answer": "It is the 12-quarter reporting period beginning on the first day of the fiscal year in which the credit union adopts CECL.",
    },
    "title12-exp-023": {
        "question": "What must a federal credit union card issuer do before issuing an additional or replacement card soon after receiving a change-of-address notice?",
        "expected_answer": "It may not issue the additional or replacement card until it assesses the validity of the address change, either by notifying the cardholder at the former address or another agreed communication method and providing a way to report incorrect changes, or by otherwise validating the change under its established policies and procedures.",
    },
    "title12-exp-025": {
        "question": "What is a controlled entity under 12 CFR 338.6?",
        "expected_answer": "It is a corporation, partnership, association, or other business entity for which a bank directly or indirectly has the power to direct or cause the direction of management and policies, whether through voting securities ownership, contract, or otherwise.",
    },
    "title12-exp-033": {
        "question": "How long after leaving FHFA is a covered senior examiner barred from knowingly accepting compensation from a regulated entity or the Office of Finance without a waiver?",
        "expected_answer": "For one year after leaving FHFA employment, if the employee served as senior examiner for two or more months during the last 12 months of FHFA employment.",
    },
    "title12-exp-034": {
        "question": "Within how many calendar days must an employee's written request to inspect or copy NCUA debt records be received?",
        "expected_answer": "The request must be received within 15 calendar days after the employee receives the Notice.",
    },
    "title12-exp-035": {
        "question": "How is a spouse's or other family member's interest treated when determining ownership or control under 12 CFR 366.6?",
        "expected_answer": "The spouse's or other family member's interest in the same organization is imputed to the person when determining ownership or control.",
    },
    "title12-exp-038": {
        "question": "What disclosures must a periodic statement include under 12 CFR 1030.6?",
        "expected_answer": "It must include the annual percentage yield earned, the dollar amount of interest earned, itemized fees imposed, the length or dates of the statement period, and any applicable aggregate overdraft and returned item fee disclosure.",
    },
    "title12-repl-001": {
        "question": "What proceedings are covered by the scope of 12 CFR 747.301?",
        "expected_answer": "It covers informal proceedings conducted by the NCUA Board or its designated Presiding Officer under section 206(i) of the Act to suspend, remove, or prohibit an institution-affiliated party of an insured credit union based on specified criminal charges, pretrial diversion or similar programs, or convictions.",
    },
    "title12-repl-021-04": {
        "question": "What is a credit score model under 12 CFR 1254.2?",
        "expected_answer": "It is a statistical tool or algorithm created by a third party and used to produce a numerical value or categorization to predict the likelihood of certain credit behaviors.",
    },
    "title12-repl-039-02": {
        "question": "When must the Corporation promptly refund deducted amounts under 12 CFR 1408.40?",
        "expected_answer": "When the Corporation is the creditor agency, it must promptly refund deducted amounts if the debt is waived or otherwise found not to be owed to the United States, unless prohibited, or if an administrative or judicial order directs the refund.",
    },
    "title12-exp-014": {
        "question": "For an FDIC merger transaction notice, how often must the applicant publish notice in the ordinary case, and what general public-notice rule supplies the notice content requirements?",
        "expected_answer": "The applicant must publish notice on at least three occasions at approximately equal intervals, and the notice must conform to the public notice requirements in 12 CFR 303.7.",
        "required_evidence_groups": [["303.65"], ["303.7"]],
    },
    "title12-exp-015": {
        "question": "For double default treatment under 12 CFR 217.135, what kind of exposure may be hedged and what related section defines the eligible guarantee or credit derivative treatment?",
        "expected_answer": "The hedged exposure must be a wholesale exposure other than a sovereign exposure, and the guarantee or credit derivative must cover an exposure described in 12 CFR 217.134(a)(1), with eligible guarantee or eligible credit derivative treatment governed by 12 CFR 217.134.",
        "required_evidence_groups": [["217.135"], ["217.134"]],
    },
    "title12-exp-017": {
        "question": "For a termination vote under 12 CFR 611.1240, what voting procedure rule must be followed, and what voting confidentiality rule does that procedure include?",
        "expected_answer": "The voting procedures must comply with 12 CFR 611.340. That rule requires policies and procedures to secure voting records and materials and keep information about how or whether an individual stockholder voted confidential, subject to specified exceptions.",
        "required_evidence_groups": [["611.1240"], ["611.340"]],
    },
    "title12-exp-018": {
        "question": "When papers are served under 12 CFR 1209.16 by electronic mail, what filing-format rule must the document satisfy?",
        "expected_answer": "A document transmitted by electronic mail for service must comply with the requirements of 12 CFR 1209.15(c), which sets formal requirements for filed papers such as identifying information, service certification, signature, caption, copies, and content format.",
        "required_evidence_groups": [["1209.16"], ["1209.15"]],
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def build_section_index(path: Path) -> dict[str, dict[str, Any]]:
    sections = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            sections[item["section"]] = item
    return sections


def question_type(candidate: dict[str, Any]) -> str:
    draft_type = DRAFTS[candidate["candidate_id"]].get("question_type")
    if draft_type:
        approved = candidate.get("approved_question_types", [])
        if draft_type not in approved:
            raise ValueError(
                f"{candidate['candidate_id']} draft type {draft_type} is not approved: {approved}"
            )
        return draft_type
    if candidate["question_type"] == "cross_section":
        return "cross_section"
    approved = candidate.get("approved_question_types", [])
    if len(approved) != 1:
        raise ValueError(
            f"{candidate['candidate_id']} needs one draft question type, got {approved}"
        )
    return approved[0]


def single_record(index: int, candidate: dict[str, Any]) -> dict[str, Any]:
    draft = DRAFTS[candidate["candidate_id"]]
    return {
        "question_id": f"title12-dev-q{index:03d}",
        "candidate_id": candidate["candidate_id"],
        "split": "development",
        "question_type": question_type(candidate),
        "question": draft["question"],
        "expected_answer": draft["expected_answer"],
        "acceptable_sections": [candidate["section"]],
        "source_urls": [candidate["source_url"]],
        "source_citations": [
            {
                "section": candidate["section"],
                "heading": candidate["heading"],
                "source_url": candidate["source_url"],
            }
        ],
        "question_constraints": candidate.get("question_constraints", []),
        "draft_status": "needs_human_review",
    }


def cross_record(
    index: int,
    candidate: dict[str, Any],
    sections: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    draft = DRAFTS[candidate["candidate_id"]]
    groups = draft["required_evidence_groups"]
    all_sections = [section for group in groups for section in group]
    return {
        "question_id": f"title12-dev-q{index:03d}",
        "candidate_id": candidate["candidate_id"],
        "split": "development",
        "question_type": "cross_section",
        "question": draft["question"],
        "expected_answer": draft["expected_answer"],
        "required_evidence_groups": groups,
        "source_urls": [sections[section]["source_url"] for section in all_sections],
        "source_citations": [
            {
                "section": section,
                "heading": sections[section]["heading"],
                "source_url": sections[section]["source_url"],
            }
            for section in all_sections
        ],
        "scoring_contract": candidate["scoring_contract"],
        "draft_status": "needs_human_review",
    }


def write_report(path: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    details = []
    for record in records:
        label = (
            ", ".join(record["acceptable_sections"])
            if "acceptable_sections" in record
            else " AND ".join(
                "(" + " OR ".join(group) + ")" for group in record["required_evidence_groups"]
            )
        )
        rows.append(
            f"| {record['question_id']} | {record['candidate_id']} | "
            f"{record['question_type']} | {label} |"
        )
        details.append(
            f"## {record['question_id']} · {record['candidate_id']}\n\n"
            f"- Type: `{record['question_type']}`\n"
            f"- Label: {label}\n"
            f"- Draft status: `{record['draft_status']}`\n\n"
            f"Question: {record['question']}\n\n"
            f"Expected answer: {record['expected_answer']}\n"
        )
    report = f"""# Title 12 Development QA Draft

- Schema: `{SCHEMA}`
- Split: development
- Questions: {len(records)}
- Draft status: needs human review
- Holdout QA generated: no
- Holdout retrieval inspected: no

| Question ID | Candidate | Type | Label |
|---|---|---|---|
{chr(10).join(rows)}

{chr(10).join(details)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate Title 12 development QA draft")
    parser.add_argument(
        "--single-candidates",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_single_section_candidates.json",
    )
    parser.add_argument(
        "--cross-candidates",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_cross_section_candidates.json",
    )
    parser.add_argument(
        "--sections",
        type=Path,
        default=root / "data" / "canonical" / "title12_2025-09-01" / "sections.jsonl",
    )
    parser.add_argument(
        "--output",
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
    single = load_json(args.single_candidates)["candidates"]
    cross = load_json(args.cross_candidates)["candidates"]
    sections = build_section_index(args.sections)
    development_single = [item for item in single if item["split"] == "development"]
    development_cross = [item for item in cross if item["split"] == "development"]
    candidates = development_single + development_cross
    missing = sorted(set(item["candidate_id"] for item in candidates) - set(DRAFTS))
    if missing:
        raise ValueError(f"Missing draft definitions for: {missing}")
    if len(candidates) != 20:
        raise ValueError(f"Expected 20 development candidates, found {len(candidates)}")

    records = []
    for index, candidate in enumerate(candidates, 1):
        if candidate["question_type"] == "cross_section":
            records.append(cross_record(index, candidate, sections))
        else:
            records.append(single_record(index, candidate))

    payload = {
        "schema": SCHEMA,
        "source_single_candidates": str(args.single_candidates.resolve()),
        "source_cross_candidates": str(args.cross_candidates.resolve()),
        "snapshot_date": "2025-09-01",
        "split": "development",
        "final_qa": False,
        "holdout_qa_generated": False,
        "holdout_retrieval_inspected": False,
        "question_count": len(records),
        "type_counts": dict(Counter(record["question_type"] for record in records)),
        "records": records,
    }
    atomic_write_json(args.output, payload)
    write_report(args.report, records)
    print(
        json.dumps(
            {
                "questions": len(records),
                "type_counts": payload["type_counts"],
                "output": str(args.output),
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
