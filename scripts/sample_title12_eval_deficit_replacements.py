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


REPLACEMENT_SCHEMA = "title12-eval-deficit-replacement-candidates-v1"
SHORTLIST_SIZE = 5


def stable_key(seed: int, replacement_for: str, question_type: str, document_id: str) -> str:
    value = f"{seed}|deficit-replacement|{replacement_for}|{question_type}|{document_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def heading_title(heading: str) -> str:
    parts = heading.split(" ", 2)
    if len(parts) == 3 and parts[0].startswith("§"):
        return parts[2].lower().strip()
    return heading.lower().strip()


def strong_type_match(section: dict[str, Any], question_type: str) -> bool:
    heading = heading_title(section["heading"])
    text = section["text"].lower()
    first_page = text[:900]
    if question_type == "authority":
        return heading.startswith(("authority", "purpose", "scope")) or "authority" in first_page
    if question_type == "applicability":
        return (
            heading.startswith(("applicability", "scope"))
            or "applies to" in first_page
            or "applicable to" in first_page
        )
    if question_type == "definition":
        return "definition" in heading and " means " in f" {text} "
    if question_type == "numeric_or_date":
        return bool(
            re.search(
                r"\b\d+(?:\.\d+)?\s*"
                r"(?:percent|days?|months?|years?|calendar days?|business days?)\b",
                text,
                re.IGNORECASE,
            )
        )
    if question_type == "obligation":
        return len(re.findall(r"\b(?:must|shall|required to|may not)\b", text)) >= 2
    return False


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def replaced_candidates(
    candidates: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    selected_replacement_targets: set[str],
) -> list[dict[str, Any]]:
    by_id = {item["candidate_id"]: item for item in candidates}
    replaced = []
    for candidate_id, decision in decisions.items():
        if decision.get("action") != "replace":
            continue
        if candidate_id in selected_replacement_targets:
            continue
        candidate = by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Replacement decision references unknown candidate: {candidate_id}")
        replaced.append(candidate)
    return sorted(replaced, key=lambda item: item["candidate_id"])


def sample_shortlist(
    sections: list[dict[str, Any]],
    *,
    replacement_for: dict[str, Any],
    seed: int,
    excluded_document_ids: set[str],
    excluded_sections: set[str],
    excluded_hashes: set[str],
    already_selected_doc_ids: set[str],
) -> list[dict[str, Any]]:
    question_type = replacement_for["question_type"]
    eligible = [
        section
        for section in sections
        if section.get("safe_for_citation") is True
        and 300 <= len(section["text"]) <= 5000
        and section["document_id"] not in excluded_document_ids
        and section["document_id"] not in already_selected_doc_ids
        and section["section"] not in excluded_sections
        and section.get("normalized_text_sha256") not in excluded_hashes
        and "[reserved]" not in section["text"].lower()
        and question_type in classify_section(section)
        and strong_type_match(section, question_type)
    ]
    eligible.sort(
        key=lambda item: stable_key(
            seed,
            replacement_for["candidate_id"],
            question_type,
            item["document_id"],
        )
    )
    if len(eligible) < SHORTLIST_SIZE:
        raise ValueError(
            f"Not enough candidates for {replacement_for['candidate_id']} "
            f"({question_type}): {len(eligible)}"
        )
    shortlist = []
    for index, item in enumerate(eligible[:SHORTLIST_SIZE], 1):
        shortlist.append(
            {
                "replacement_id": f"{replacement_for['candidate_id'].replace('exp', 'repl')}-{index:02d}",
                "replacement_for": replacement_for["candidate_id"],
                "intended_split": replacement_for["split"],
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
    return shortlist


def write_report(path: Path, groups: list[dict[str, Any]]) -> None:
    sections = []
    for group in groups:
        sections.append(
            f"## {group['replacement_for']} · {group['intended_split']} · "
            f"`{group['question_type']}`\n\n"
            f"- Original section: § {group['original_section']}\n"
            f"- Shortlist size: {len(group['candidates'])}\n"
        )
        for item in group["candidates"]:
            sections.append(
                f"### {item['replacement_id']} · § {item['section']}\n\n"
                f"- Heading: {item['heading']}\n"
                f"- Length: {item['text_length']} characters\n"
                f"- Review status: `pending`\n\n"
                f"> {markdown_escape(item['evidence_preview'])}\n"
            )
    report = f"""# Title 12 Deficit Replacement Shortlist

- Schema: `{REPLACEMENT_SCHEMA}`
- Replacement groups: {len(groups)}
- Shortlist size per group: {SHORTLIST_SIZE}
- Final selections made automatically: 0
- Final QA generated: no

Choose one candidate per group after human review. Do not run Holdout retrieval while
reviewing Holdout replacement candidates.

{chr(10).join(sections)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Sample Title 12 deficit replacement shortlists")
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
        "--decisions",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_human_review_decisions.json",
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
        default=root / "data" / "eval" / "title12_eval_deficit_replacement_shortlist.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_eval_deficit_replacement_shortlist.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.sections.open("r", encoding="utf-8") as file:
        sections = [json.loads(line) for line in file if line.strip()]
    candidates = load_json(args.candidates)["candidates"]
    decision_payload = load_json(args.decisions)
    decisions = decision_payload["decisions"]
    replacement_selections = decision_payload.get("replacement_selections", {})
    legacy_records = load_json(args.legacy_eval)["records"]
    spec = load_json(args.spec)

    targets = replaced_candidates(candidates, decisions, set(replacement_selections))
    existing_document_ids = {item["document_id"] for item in candidates}
    existing_sections = {item["section"] for item in candidates}
    legacy_sections = {
        section for record in legacy_records for section in record["acceptable_sections"]
    }
    existing_hashes = {
        item["normalized_text_sha256"]
        for item in candidates
        if item.get("normalized_text_sha256")
    }

    groups = []
    selected_doc_ids: set[str] = set()
    for target in targets:
        shortlist = sample_shortlist(
            sections,
            replacement_for=target,
            seed=spec["seed"],
            excluded_document_ids=existing_document_ids,
            excluded_sections=existing_sections | legacy_sections,
            excluded_hashes=existing_hashes,
            already_selected_doc_ids=selected_doc_ids,
        )
        selected_doc_ids.update(item["document_id"] for item in shortlist)
        groups.append(
            {
                "replacement_for": target["candidate_id"],
                "intended_split": target["split"],
                "question_type": target["question_type"],
                "original_section": target["section"],
                "original_heading": target["heading"],
                "candidate_count": len(shortlist),
                "candidates": shortlist,
            }
        )

    payload = {
        "schema": REPLACEMENT_SCHEMA,
        "source_candidates": str(args.candidates.resolve()),
        "source_human_decisions": str(args.decisions.resolve()),
        "selection_status": "pending_human_choice",
        "final_qa_generated": False,
        "group_count": len(groups),
        "shortlist_size_per_group": SHORTLIST_SIZE,
        "type_counts": dict(Counter(group["question_type"] for group in groups)),
        "split_counts": dict(Counter(group["intended_split"] for group in groups)),
        "groups": groups,
    }
    atomic_write_json(args.output, payload)
    write_report(args.report, groups)
    print(
        json.dumps(
            {
                "groups": len(groups),
                "type_counts": payload["type_counts"],
                "split_counts": payload["split_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
