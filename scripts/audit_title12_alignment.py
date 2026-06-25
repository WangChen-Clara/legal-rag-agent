from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_law.ingestion.alignment import content_tokens, text_fingerprint
from rag_law.ingestion.legacy_recovery import write_jsonl


REVIEW_SAMPLE_PER_REASON = 15
EXACT_SAMPLE_SIZE = 30
UNMATCHED_SAMPLE_SIZE = 30
UNREFERENCED_SAMPLE_SIZE = 30
RANDOM_SEED = 20250901


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sample_rows(
    rows: Iterable[dict[str, Any]], size: int, rng: random.Random
) -> list[dict[str, Any]]:
    values = list(rows)
    if len(values) <= size:
        return values
    return rng.sample(values, size)


def text_preview(text: str | None, limit: int = 500) -> str | None:
    if text is None:
        return None
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[:limit] + "..."


def build_joined_rows(
    alignment: list[dict[str, Any]],
    official: list[dict[str, Any]],
    legacy_by_id: dict[str, dict[str, Any]],
    exact_index: dict[str, list[str]],
    token_index: dict[tuple[str, ...], list[str]],
) -> list[dict[str, Any]]:
    if len(alignment) != len(official):
        raise ValueError("Alignment and official section row counts differ")

    joined: list[dict[str, Any]] = []
    for row_number, (result, section) in enumerate(zip(alignment, official), start=1):
        if str(result["official_section"]) != str(section["section"]):
            raise ValueError(f"Section mismatch at row {row_number}")
        legacy_id = result.get("legacy_document_id")
        legacy = legacy_by_id.get(legacy_id) if legacy_id else None
        exact_candidates = exact_index.get(text_fingerprint(section["text"]), [])
        token_candidates = token_index.get(content_tokens(section["text"]), [])
        joined.append(
            {
                **result,
                "alignment_row": row_number,
                "official_heading": section["heading"],
                "official_text": section["text"],
                "official_text_preview": text_preview(section["text"]),
                "legacy_text": legacy["text"] if legacy else None,
                "legacy_text_preview": text_preview(legacy["text"]) if legacy else None,
                "exact_candidate_ids": exact_candidates,
                "token_equal_candidate_ids": token_candidates,
            }
        )
    return joined


def write_report(
    path: Path,
    *,
    alignment_count: int,
    legacy_count: int,
    statuses: Counter[str],
    review_reasons: Counter[str],
    directly_referenced: set[str],
    exact_evidenced: set[str],
    token_evidenced: set[str],
    unreferenced: list[dict[str, Any]],
    unreferenced_classes: Counter[str],
    sample_counts: Counter[str],
) -> None:
    evidence_union = directly_referenced | exact_evidenced | token_evidenced
    no_official_evidence = legacy_count - len(evidence_union)
    reason_rows = "\n".join(
        f"| {reason} | {count} |" for reason, count in sorted(review_reasons.items())
    )
    class_rows = "\n".join(
        f"| {name} | {count} |" for name, count in sorted(unreferenced_classes.items())
    )
    sample_rows_markdown = "\n".join(
        f"| {name} | {count} |" for name, count in sorted(sample_counts.items())
    )
    report = f"""# Title 12 Alignment Quality Audit

- Snapshot date: `2025-09-01`
- Alignment rows audited: {alignment_count}
- Historical documents audited: {legacy_count}
- Random seed: `{RANDOM_SEED}`

## Alignment status

| Status | Count |
|---|---:|
| exact | {statuses['exact']} |
| high_confidence | {statuses['high_confidence']} |
| review_required | {statuses['review_required']} |
| unmatched | {statuses['unmatched']} |

## Review-required reasons

| Reason | Count |
|---|---:|
{reason_rows}

## Historical-document evidence

- Directly referenced by an alignment row: {len(directly_referenced)} / {legacy_count}
- Evidenced by normalized exact equality: {len(exact_evidenced)} / {legacy_count}
- Evidenced by exact or token equality: {len(exact_evidenced | token_evidenced)} / {legacy_count}
- Evidenced by any direct/exact/token route: {len(evidence_union)} / {legacy_count}
- No official exact/token/direct evidence: {no_official_evidence}
- Not directly referenced (audit list size): {len(unreferenced)}

The unreferenced count is not a missing-data count. Ambiguous duplicate candidates are
intentionally not assigned a single `legacy_document_id`; exact and token evidence are
reconstructed in this audit to distinguish those rows from genuine evidence gaps.

### Unreferenced classifications

| Classification | Count |
|---|---:|
{class_rows}

## Manual-review sample

| Sample group | Count |
|---|---:|
{sample_rows_markdown}

Recommended review order: all `high_confidence`, all truncation cases in the sampled
review set, other `review_required` strata, exact samples, then unreferenced documents
with `no_official_evidence`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8", newline="\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Audit full Title 12 alignment quality")
    parser.add_argument(
        "--alignment-dir", type=Path, default=root / "data" / "alignment" / "full"
    )
    parser.add_argument(
        "--legacy",
        type=Path,
        default=root / "data" / "recovered" / "title12_documents.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "data" / "alignment" / "audit"
    )
    parser.add_argument(
        "--report", type=Path, default=root / "reports" / "title12_alignment_audit.md"
    )
    args = parser.parse_args()

    alignment = load_jsonl(args.alignment_dir / "alignment_results.jsonl")
    official = load_jsonl(args.alignment_dir / "official_sections.jsonl")
    legacy = load_jsonl(args.legacy)
    legacy_by_id = {row["legacy_document_id"]: row for row in legacy}

    exact_index: dict[str, list[str]] = defaultdict(list)
    token_index: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for document in legacy:
        identifier = document["legacy_document_id"]
        exact_index[text_fingerprint(document["text"])].append(identifier)
        token_index[content_tokens(document["text"])].append(identifier)

    joined = build_joined_rows(
        alignment, official, legacy_by_id, exact_index, token_index
    )
    review_rows = [row for row in joined if row["status"] == "review_required"]
    unmatched_rows = [row for row in joined if row["status"] == "unmatched"]
    exact_rows = [row for row in joined if row["status"] == "exact"]
    high_confidence_rows = [
        row for row in joined if row["status"] == "high_confidence"
    ]

    directly_referenced = {
        row["legacy_document_id"]
        for row in alignment
        if row.get("legacy_document_id") is not None
    }
    official_fingerprints = {text_fingerprint(row["text"]) for row in official}
    official_tokens = {content_tokens(row["text"]) for row in official}
    exact_evidenced = {
        document["legacy_document_id"]
        for document in legacy
        if text_fingerprint(document["text"]) in official_fingerprints
    }
    token_evidenced = {
        document["legacy_document_id"]
        for document in legacy
        if content_tokens(document["text"]) in official_tokens
    }

    normalized_counts = Counter(document["normalized_sha256"] for document in legacy)
    unreferenced: list[dict[str, Any]] = []
    for document in legacy:
        identifier = document["legacy_document_id"]
        if identifier in directly_referenced:
            continue
        if identifier in exact_evidenced:
            classification = "exact_evidence_but_ambiguous_or_unassigned"
        elif identifier in token_evidenced:
            classification = "token_equal_evidence_but_unassigned"
        elif document["quality_flags"].get("known_source_truncation"):
            classification = "truncated_without_official_evidence"
        elif normalized_counts[document["normalized_sha256"]] > 1:
            classification = "duplicate_without_official_evidence"
        else:
            classification = "no_official_evidence"
        unreferenced.append(
            {
                **document,
                "audit_classification": classification,
                "text_preview": text_preview(document["text"]),
            }
        )

    rng = random.Random(RANDOM_SEED)
    samples: list[dict[str, Any]] = []

    def add_samples(group: str, rows: Iterable[dict[str, Any]], size: int) -> None:
        for row in sample_rows(rows, size, rng):
            samples.append({"sample_group": group, **row})

    add_samples("exact", exact_rows, EXACT_SAMPLE_SIZE)
    add_samples("high_confidence", high_confidence_rows, len(high_confidence_rows))
    review_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in review_rows:
        review_by_reason[row["reason_code"]].append(row)
    for reason, rows in sorted(review_by_reason.items()):
        add_samples(f"review_required:{reason}", rows, REVIEW_SAMPLE_PER_REASON)
    add_samples("unmatched", unmatched_rows, UNMATCHED_SAMPLE_SIZE)
    add_samples("unreferenced_legacy", unreferenced, UNREFERENCED_SAMPLE_SIZE)

    failure_candidates = review_rows + [
        {"failure_group": "unmatched_sample", **row}
        for row in sample_rows(unmatched_rows, UNMATCHED_SAMPLE_SIZE, rng)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "review_required.jsonl", review_rows)
    write_jsonl(args.output_dir / "unreferenced_legacy_documents.jsonl", unreferenced)
    write_jsonl(args.output_dir / "manual_review_sample.jsonl", samples)
    write_jsonl(args.output_dir / "failure_candidates.jsonl", failure_candidates)

    statuses = Counter(row["status"] for row in alignment)
    review_reasons = Counter(row["reason_code"] for row in review_rows)
    unreferenced_classes = Counter(row["audit_classification"] for row in unreferenced)
    sample_counts = Counter(row["sample_group"] for row in samples)
    write_report(
        args.report,
        alignment_count=len(alignment),
        legacy_count=len(legacy),
        statuses=statuses,
        review_reasons=review_reasons,
        directly_referenced=directly_referenced,
        exact_evidenced=exact_evidenced,
        token_evidenced=token_evidenced,
        unreferenced=unreferenced,
        unreferenced_classes=unreferenced_classes,
        sample_counts=sample_counts,
    )
    print(
        json.dumps(
            {
                "review_required": len(review_rows),
                "unreferenced_legacy": len(unreferenced),
                "manual_review_sample": len(samples),
                "failure_candidates": len(failure_candidates),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
