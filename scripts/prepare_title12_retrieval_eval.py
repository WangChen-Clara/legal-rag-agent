from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


EVAL_SCHEMA = "title12-retrieval-eval-candidates-v1"
WINDOW_SIZES = (160, 120, 80)
WINDOW_POSITIONS = (0.2, 0.4, 0.6)


def normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = normalized.strip('"')
    return normalized.rstrip("\\,").strip()


def evidence_windows(text: str, size: int) -> list[str]:
    normalized = normalize_text(text)
    if len(normalized) < size:
        return [normalized] if normalized else []
    windows: list[str] = []
    for position in WINDOW_POSITIONS:
        start = min(int(len(normalized) * position), len(normalized) - size)
        window = normalized[start : start + size]
        if window not in windows:
            windows.append(window)
    return windows


def candidate_sections(
    evidence: str,
    sections: list[dict[str, Any]],
    normalized_sections: list[str] | None = None,
) -> tuple[list[dict[str, Any]], int | None]:
    searchable_texts = (
        normalized_sections
        if normalized_sections is not None
        else [normalize_text(section["text"]) for section in sections]
    )
    for size in WINDOW_SIZES:
        windows = evidence_windows(evidence, size)
        matches = [
            section
            for section, text in zip(sections, searchable_texts)
            if any(window in text for window in windows)
        ]
        if matches:
            return matches, size
    return [], None


def label_items(
    items: list[dict[str, Any]], sections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    normalized_sections = [normalize_text(section["text"]) for section in sections]
    for index, item in enumerate(items, start=1):
        matches, window_size = candidate_sections(
            item["Text"], sections, normalized_sections
        )
        candidates = [
            {
                "document_id": match["document_id"],
                "title": match["title"],
                "part": match["part"],
                "section": match["section"],
                "heading": match["heading"],
                "version_date": match["version_date"],
                "source_url": match["source_url"],
            }
            for match in matches
        ]
        if len(candidates) == 1:
            status = "auto_labeled"
            gold_section_ids = [candidates[0]["document_id"]]
        elif candidates:
            status = "review_required"
            gold_section_ids = []
        else:
            status = "unmatched"
            gold_section_ids = []
        labeled.append(
            {
                "question_id": f"title12-q{index:03d}",
                "question": item["Q"],
                "answer": item["A"],
                "gold_text": item["Text"],
                "label_status": status,
                "gold_section_ids": gold_section_ids,
                "candidate_match_method": "exact_normalized_character_window",
                "matched_window_chars": window_size,
                "candidate_sections": candidates,
            }
        )
    return labeled


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_report(path: Path, records: list[dict[str, Any]]) -> None:
    counts = Counter(record["label_status"] for record in records)
    rows = []
    for record in records:
        candidate_labels = ", ".join(
            f"§ {candidate['section']}" for candidate in record["candidate_sections"]
        )
        rows.append(
            f"| {record['question_id']} | {record['label_status']} | "
            f"{len(record['candidate_sections'])} | {candidate_labels or '-'} |"
        )
    report = f"""# Title 12 Retrieval Evaluation Label Audit

- Questions: {len(records)}
- Auto labeled: {counts['auto_labeled']}
- Review required: {counts['review_required']}
- Unmatched: {counts['unmatched']}
- Automatic rule: exact normalized character windows only
- Fuzzy matches promoted automatically: 0

`review_required` records deliberately have empty `gold_section_ids`. A human must
select the intended section before those questions are included in Recall@K or MRR.

| Question | Status | Candidates | Candidate sections |
|---|---|---:|---|
{chr(10).join(rows)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare Title 12 retrieval labels")
    parser.add_argument("--questions", type=Path, default=root / "data" / "eval" / "biaozhu.json")
    parser.add_argument(
        "--sections",
        type=Path,
        default=root / "data" / "canonical" / "title12_2025-09-01" / "sections.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "eval" / "title12_retrieval_eval_candidates.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_retrieval_eval_label_audit.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = json.loads(args.questions.read_text(encoding="utf-8"))
    with args.sections.open("r", encoding="utf-8") as file:
        sections = [json.loads(line) for line in file if line.strip()]
    records = label_items(items, sections)
    counts = Counter(record["label_status"] for record in records)
    payload = {
        "schema": EVAL_SCHEMA,
        "source_questions": str(args.questions.resolve()),
        "source_sections": str(args.sections.resolve()),
        "summary": {
            "questions": len(records),
            "auto_labeled": counts["auto_labeled"],
            "review_required": counts["review_required"],
            "unmatched": counts["unmatched"],
        },
        "records": records,
    }
    atomic_write_json(args.output, payload)
    write_report(args.report, records)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
