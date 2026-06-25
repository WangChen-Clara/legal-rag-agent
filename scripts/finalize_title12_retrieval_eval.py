from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


FINAL_SCHEMA = "title12-retrieval-eval-v1"


def finalize_records(
    records: list[dict[str, Any]], manual_labels: dict[str, list[str]]
) -> list[dict[str, Any]]:
    review_ids = {
        record["question_id"]
        for record in records
        if record["label_status"] == "review_required"
    }
    label_ids = set(manual_labels)
    if review_ids != label_ids:
        missing = sorted(review_ids - label_ids)
        unexpected = sorted(label_ids - review_ids)
        raise ValueError(
            f"Manual label coverage mismatch: missing={missing}, unexpected={unexpected}"
        )

    finalized: list[dict[str, Any]] = []
    for record in records:
        candidates_by_section = {
            candidate["section"]: candidate
            for candidate in record["candidate_sections"]
        }
        if record["label_status"] == "auto_labeled":
            accepted_sections = list(candidates_by_section)
            label_method = "auto_unique_exact"
        elif record["label_status"] == "review_required":
            accepted_sections = manual_labels[record["question_id"]]
            invalid = sorted(set(accepted_sections) - set(candidates_by_section))
            if invalid:
                raise ValueError(
                    f"{record['question_id']} labels are not candidates: {invalid}"
                )
            if not accepted_sections:
                raise ValueError(f"{record['question_id']} has no accepted sections")
            label_method = "human_confirmed_equivalence"
        else:
            raise ValueError(
                f"Cannot finalize {record['question_id']} with status "
                f"{record['label_status']}"
            )

        finalized.append(
            {
                "question_id": record["question_id"],
                "question": record["question"],
                "answer": record["answer"],
                "gold_text": record["gold_text"],
                "label_method": label_method,
                "acceptable_sections": accepted_sections,
                "acceptable_section_ids": [
                    candidates_by_section[section]["document_id"]
                    for section in accepted_sections
                ],
                "acceptable_sources": [
                    candidates_by_section[section]
                    for section in accepted_sections
                ],
            }
        )
    return finalized


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_report(path: Path, records: list[dict[str, Any]], decision_rule: str) -> None:
    counts = Counter(record["label_method"] for record in records)
    rows = [
        f"| {record['question_id']} | {record['label_method']} | "
        f"{', '.join('§ ' + section for section in record['acceptable_sections'])} |"
        for record in records
    ]
    report = f"""# Title 12 Retrieval Evaluation Dataset

- Questions: {len(records)}
- Auto unique labels: {counts['auto_unique_exact']}
- Human-confirmed equivalence labels: {counts['human_confirmed_equivalence']}
- Unlabeled questions: 0

Decision rule: {decision_rule}

For Hit@K and MRR, retrieving any `acceptable_section_ids` member counts as a
correct result. Equivalent provisions are alternatives; the retriever is not required
to return every duplicate provision.

| Question | Label method | Acceptable sections |
|---|---|---|
{chr(10).join(rows)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Finalize the Title 12 retrieval set")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root / "data" / "eval" / "title12_retrieval_eval_candidates.json",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=root / "data" / "eval" / "title12_retrieval_eval_labels.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "eval" / "title12_retrieval_eval.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_retrieval_eval.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    records = finalize_records(candidates["records"], labels["labels"])
    payload = {
        "schema": FINAL_SCHEMA,
        "source_candidates": str(args.candidates.resolve()),
        "source_human_labels": str(args.labels.resolve()),
        "decision_rule": labels["decision_rule"],
        "questions": len(records),
        "records": records,
    }
    atomic_write_json(args.output, payload)
    write_report(args.report, records, labels["decision_rule"])
    print(
        json.dumps(
            {
                "questions": len(records),
                "auto_unique": sum(
                    record["label_method"] == "auto_unique_exact"
                    for record in records
                ),
                "human_confirmed": sum(
                    record["label_method"] == "human_confirmed_equivalence"
                    for record in records
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
