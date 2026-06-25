from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CANDIDATE_SCHEMA = "title12-eval-expansion-candidates-v1"
TYPE_ORDER = (
    "authority",
    "applicability",
    "cross_section",
    "definition",
    "numeric_or_date",
    "obligation",
)
MONTHS = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december"
)


def stable_key(seed: int, *values: str) -> str:
    value = "|".join((str(seed), *values))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def classify_section(section: dict[str, Any]) -> set[str]:
    heading = section["heading"].lower()
    text = section["text"].lower()
    heading_label = re.sub(r"^§\s*\S+\s*", "", heading).strip()
    first = text[:1000]
    types: set[str] = set()
    if "definition" in heading or (
        len(text) <= 3000 and len(re.findall(r"\bmeans\b", text)) >= 2
    ):
        types.add("definition")
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:%|percent|days?|months?|years?)\b", text) or re.search(
        rf"\b(?:{MONTHS})\s+\d{{1,2}}(?:,\s+\d{{4}})?\b", text
    ):
        types.add("numeric_or_date")
    if len(re.findall(r"\b(?:must|shall|required to|may not)\b", text)) >= 2:
        types.add("obligation")
    if "applicability" in heading or "scope" in heading or "applies to" in first:
        types.add("applicability")
    if (
        heading_label.startswith("authority")
        or first.startswith("authority.")
        or "(a) authority." in text[:500]
        or "this part is issued pursuant to" in text[:500]
    ):
        types.add("authority")
    references = set(re.findall(r"§+\s*([0-9]+[a-z]?(?:\.[0-9a-z-]+)+)", text))
    if len(references) >= 2:
        types.add("cross_section")
    return types


def length_bucket(length: int) -> str:
    if length <= 900:
        return "short"
    if length <= 3000:
        return "medium"
    return "long"


def select_candidates(
    sections: list[dict[str, Any]], quotas: dict[str, int], seed: int
) -> list[dict[str, Any]]:
    eligible = [
        section
        for section in sections
        if section.get("safe_for_citation") is True
        and 250 <= len(section["text"]) <= 20000
    ]
    section_types = {
        section["document_id"]: classify_section(section) for section in eligible
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    bucket_cycle = ("short", "medium", "long")
    for question_type in TYPE_ORDER:
        if question_type not in quotas:
            continue
        quota = quotas[question_type]
        pool = [
            section
            for section in eligible
            if question_type in section_types[section["document_id"]]
            and section["document_id"] not in selected_ids
        ]
        pool.sort(key=lambda item: stable_key(seed, question_type, item["document_id"]))
        by_bucket = {
            bucket: [item for item in pool if length_bucket(len(item["text"])) == bucket]
            for bucket in bucket_cycle
        }
        used_parts: set[str] = set()
        chosen: list[dict[str, Any]] = []
        while len(chosen) < quota:
            bucket = bucket_cycle[len(chosen) % len(bucket_cycle)]
            candidates = by_bucket[bucket] or [
                item for values in by_bucket.values() for item in values
            ]
            if not candidates:
                raise ValueError(f"Not enough candidates for {question_type}")
            unused_part = next(
                (item for item in candidates if item["part"] not in used_parts), None
            )
            item = unused_part or candidates[0]
            for values in by_bucket.values():
                if item in values:
                    values.remove(item)
            chosen.append(item)
            used_parts.add(item["part"])
        for item in chosen:
            selected_ids.add(item["document_id"])
            selected.append(
                {
                    "candidate_id": f"title12-exp-{len(selected) + 1:03d}",
                    "question_type": question_type,
                    "document_id": item["document_id"],
                    "title": item["title"],
                    "part": item["part"],
                    "section": item["section"],
                    "heading": item["heading"],
                    "version_date": item["version_date"],
                    "source_url": item["source_url"],
                    "text_sha256": item["text_sha256"],
                    "normalized_text_sha256": item["normalized_text_sha256"],
                    "text_length": len(item["text"]),
                    "length_bucket": length_bucket(len(item["text"])),
                    "detected_types": sorted(section_types[item["document_id"]]),
                    "cross_section_references": sorted(
                        set(
                            re.findall(
                                r"§+\s*([0-9]+[a-z]?(?:\.[0-9a-z-]+)+)",
                                item["text"],
                            )
                        )
                    ),
                    "evidence_preview": item["text"][:800],
                    "review_status": "pending",
                }
            )
    return selected


def group_equivalent_candidates(
    candidates: list[dict[str, Any]], sections_by_id: dict[str, dict[str, Any]], threshold: float = 0.78
) -> dict[str, str]:
    parents = {candidate["document_id"]: candidate["document_id"] for candidate in candidates}

    def find(item: str) -> str:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    token_sets = {
        candidate["document_id"]: tokenize(sections_by_id[candidate["document_id"]]["text"])
        for candidate in candidates
    }
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            left_id, right_id = left["document_id"], right["document_id"]
            same_hash = (
                left.get("normalized_text_sha256")
                and left["normalized_text_sha256"] == right.get("normalized_text_sha256")
            )
            intersection = len(token_sets[left_id] & token_sets[right_id])
            union_size = len(token_sets[left_id] | token_sets[right_id])
            similarity = intersection / union_size if union_size else 0.0
            if same_hash or similarity >= threshold:
                union(left_id, right_id)
    return {
        document_id: f"family-{stable_key(0, find(document_id))[:12]}"
        for document_id in parents
    }


def assign_splits(
    candidates: list[dict[str, Any]], families: dict[str, str], seed: int
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        family = families[candidate["document_id"]]
        candidate["equivalence_family"] = family
        grouped[family].append(candidate)
    split_counts = Counter()
    groups = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), stable_key(seed, "split", item[0])),
    )
    for _, members in groups:
        split = "development" if split_counts["development"] <= split_counts["holdout"] else "holdout"
        for member in members:
            member["split"] = split
        split_counts[split] += len(members)


def validate_candidates(candidates: list[dict[str, Any]], spec: dict[str, Any]) -> None:
    expected = spec["target_candidates"]
    if len(candidates) != expected:
        raise ValueError(f"Candidate count mismatch: {len(candidates)} != {expected}")
    if len({item["document_id"] for item in candidates}) != expected:
        raise ValueError("Candidate document IDs are not unique")
    actual_quotas = Counter(item["question_type"] for item in candidates)
    if dict(actual_quotas) != spec["question_type_quotas"]:
        raise ValueError(f"Question type quota mismatch: {dict(actual_quotas)}")
    actual_splits = Counter(item["split"] for item in candidates)
    if dict(actual_splits) != spec["split_targets"]:
        raise ValueError(f"Split target mismatch: {dict(actual_splits)}")
    family_splits: dict[str, set[str]] = defaultdict(set)
    for item in candidates:
        family_splits[item["equivalence_family"]].add(item["split"])
    leaking = [family for family, splits in family_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"Equivalence families leak across splits: {leaking}")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_report(path: Path, candidates: list[dict[str, Any]]) -> None:
    type_counts = Counter(item["question_type"] for item in candidates)
    split_counts = Counter(item["split"] for item in candidates)
    bucket_counts = Counter(item["length_bucket"] for item in candidates)
    rows = [
        f"| {item['candidate_id']} | {item['question_type']} | {item['split']} | "
        f"§ {item['section']} | {item['length_bucket']} | {item['equivalence_family']} |"
        for item in candidates
    ]
    report = f"""# Title 12 Evaluation Expansion Candidate Audit

## Summary

- Candidates: {len(candidates)}
- Development / holdout: {split_counts['development']} / {split_counts['holdout']}
- Short / medium / long: {bucket_counts['short']} / {bucket_counts['medium']} / {bucket_counts['long']}
- Equivalence grouping: exact normalized hash or token Jaccard >= 0.78
- Human-reviewed candidates: 0

Question type counts: {json.dumps(dict(type_counts), ensure_ascii=False)}

## Required Manual Review

1. Confirm that each candidate can support its assigned question type.
2. Merge cross-agency near-equivalent provisions missed by the automatic family rule.
3. Reject reserved, purely procedural, or context-dependent candidates.
4. Do not write questions until the replacement candidate list is approved.
5. Keep holdout retrieval results hidden after questions are finalized.

| Candidate | Type | Split | Section | Length | Equivalence family |
|---|---|---|---|---|---|
{chr(10).join(rows)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Sample Title 12 evaluation candidates")
    parser.add_argument(
        "--sections",
        type=Path,
        default=root / "data" / "canonical" / "title12_2025-09-01" / "sections.jsonl",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_expansion_spec.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_expansion_candidates.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_eval_expansion_candidates.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    with args.sections.open("r", encoding="utf-8") as file:
        sections = [json.loads(line) for line in file if line.strip()]
    candidates = select_candidates(sections, spec["question_type_quotas"], spec["seed"])
    sections_by_id = {section["document_id"]: section for section in sections}
    families = group_equivalent_candidates(candidates, sections_by_id)
    assign_splits(candidates, families, spec["seed"])
    validate_candidates(candidates, spec)
    payload = {
        "schema": CANDIDATE_SCHEMA,
        "source_sections": str(args.sections.resolve()),
        "spec": str(args.spec.resolve()),
        "seed": spec["seed"],
        "candidate_count": len(candidates),
        "manual_review_status": "pending",
        "candidates": candidates,
    }
    atomic_write_json(args.output, payload)
    write_report(args.report, candidates)
    print(
        json.dumps(
            {
                "candidates": len(candidates),
                "types": dict(Counter(item["question_type"] for item in candidates)),
                "splits": dict(Counter(item["split"] for item in candidates)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
