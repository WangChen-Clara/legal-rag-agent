from __future__ import annotations

import argparse
import hashlib
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


REPLACEMENT_SCHEMA = "title12-eval-replacement-candidates-v1"
REPLACEMENT_QUOTAS = {
    "applicability": 2,
    "obligation": 2,
    "definition": 1,
}


def stable_key(seed: int, question_type: str, document_id: str) -> str:
    value = f"{seed}|replacement-exp-006|{question_type}|{document_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def strong_type_match(section: dict[str, Any], question_type: str) -> bool:
    heading = re.sub(r"^§\s*\S+\s*", "", section["heading"].lower()).strip()
    text = section["text"].lower()
    if question_type == "applicability":
        return heading.startswith(("applicability", "scope")) or "applies to" in text[:700]
    if question_type == "obligation":
        return len(re.findall(r"\b(?:must|shall|required to|may not)\b", text)) >= 2
    if question_type == "definition":
        return "definition" in heading and " means " in f" {text} "
    return False


def sample_replacements(
    sections: list[dict[str, Any]],
    excluded_document_ids: set[str],
    excluded_sections: set[str],
    excluded_hashes: set[str],
    *,
    seed: int,
    quotas: dict[str, int] = REPLACEMENT_QUOTAS,
) -> list[dict[str, Any]]:
    eligible = [
        section
        for section in sections
        if section.get("safe_for_citation") is True
        and 300 <= len(section["text"]) <= 5000
        and section["document_id"] not in excluded_document_ids
        and section["section"] not in excluded_sections
        and section.get("normalized_text_sha256") not in excluded_hashes
        and "[reserved]" not in section["text"].lower()
    ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used_parts: set[str] = set()
    for question_type, quota in quotas.items():
        pool = [
            section
            for section in eligible
            if section["document_id"] not in selected_ids
            and question_type in classify_section(section)
            and strong_type_match(section, question_type)
        ]
        pool.sort(key=lambda item: stable_key(seed, question_type, item["document_id"]))
        for _ in range(quota):
            unused_part = next((item for item in pool if item["part"] not in used_parts), None)
            item = unused_part or (pool[0] if pool else None)
            if item is None:
                raise ValueError(f"Not enough replacement candidates for {question_type}")
            pool.remove(item)
            selected_ids.add(item["document_id"])
            used_parts.add(item["part"])
            selected.append(
                {
                    "replacement_id": f"title12-repl-{len(selected) + 1:03d}",
                    "replacement_for": "title12-exp-006",
                    "intended_split": "development",
                    "question_type": question_type,
                    "document_id": item["document_id"],
                    "part": item["part"],
                    "section": item["section"],
                    "heading": item["heading"],
                    "version_date": item["version_date"],
                    "source_url": item["source_url"],
                    "text_length": len(item["text"]),
                    "text_sha256": item["text_sha256"],
                    "normalized_text_sha256": item["normalized_text_sha256"],
                    "evidence_preview": re.sub(r"\s+", " ", item["text"][:900]).strip(),
                    "review_status": "pending",
                }
            )
    return selected


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|")


def write_report(path: Path, candidates: list[dict[str, Any]]) -> None:
    details = []
    for item in candidates:
        details.append(
            f"### {item['replacement_id']} · § {item['section']}\n\n"
            f"- Proposed type: `{item['question_type']}`\n"
            f"- Heading: {item['heading']}\n"
            f"- Length: {item['text_length']} characters\n"
            f"- Review status: `pending`\n\n"
            f"> {markdown_escape(item['evidence_preview'])}\n"
        )
    report = f"""# Title 12 Development Replacement Shortlist

- Replacement target: `title12-exp-006`
- Shortlist size: {len(candidates)}
- Intended split: development
- Final selections made automatically: 0
- Final QA generated: no

Choose exactly one candidate after human review. The remaining four candidates stay
outside the evaluation set.

{chr(10).join(details)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Sample a shortlist to replace exp-006")
    parser.add_argument(
        "--sections",
        type=Path,
        default=root / "data" / "canonical" / "title12_2025-09-01" / "sections.jsonl",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_expansion_candidates.json",
    )
    parser.add_argument(
        "--legacy-eval",
        type=Path,
        default=root / "data" / "eval" / "title12_retrieval_eval.json",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_expansion_spec.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_exp006_replacement_shortlist.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_eval_exp006_replacement_shortlist.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.sections.open("r", encoding="utf-8") as file:
        sections = [json.loads(line) for line in file if line.strip()]
    candidate_payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    legacy_payload = json.loads(args.legacy_eval.read_text(encoding="utf-8"))
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    existing = candidate_payload["candidates"]
    legacy_sections = {
        section
        for record in legacy_payload["records"]
        for section in record["acceptable_sections"]
    }
    replacements = sample_replacements(
        sections,
        {item["document_id"] for item in existing},
        legacy_sections,
        {
            item["normalized_text_sha256"]
            for item in existing
            if item.get("normalized_text_sha256")
        },
        seed=spec["seed"],
    )
    payload = {
        "schema": REPLACEMENT_SCHEMA,
        "replacement_for": "title12-exp-006",
        "intended_split": "development",
        "selection_status": "pending_human_choice",
        "final_qa_generated": False,
        "candidate_count": len(replacements),
        "type_counts": dict(Counter(item["question_type"] for item in replacements)),
        "candidates": replacements,
    }
    atomic_write_json(args.output, payload)
    write_report(args.report, replacements)
    print(json.dumps({"candidates": len(replacements), "types": payload["type_counts"]}, indent=2))


if __name__ == "__main__":
    main()
