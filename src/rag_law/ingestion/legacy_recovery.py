from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CJK_OR_REPLACEMENT = re.compile(r"[\u3400-\u9fff\ufffd]")
KNOWN_TRUNCATION = re.compile(r"\s*…\(已截断\)$")
SENTENCE_END = re.compile(r"[.!?;:)\]\"'”’]$")


@dataclass(frozen=True)
class RecoveryResult:
    documents: list[dict[str, Any]]
    metrics: dict[str, Any]
    issues: list[dict[str, Any]]


def _stable_row_key(row_id: Any) -> tuple[int, str]:
    if isinstance(row_id, int):
        return row_id, str(row_id)
    text = str(row_id)
    return (int(text), text) if text.isdigit() else (2**63 - 1, text)


def _normalize_for_duplicate_check(text: str) -> str:
    return " ".join(text.lower().split())


def recover_records(
    records: Iterable[Any],
    source_file: str,
    expected_chunk_size: int = 500,
) -> RecoveryResult:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    issues: list[dict[str, Any]] = []
    total_records = 0
    ignored_records = 0

    for position, item in enumerate(records):
        if not isinstance(item, dict) or item.get("source_file") != source_file:
            ignored_records += 1
            continue
        total_records += 1
        missing = [key for key in ("row_id", "chunk_id", "chunk") if key not in item]
        if missing:
            issues.append(
                {
                    "type": "invalid_record",
                    "position": position,
                    "missing_fields": missing,
                }
            )
            continue
        groups[item["row_id"]].append(item)

    documents: list[dict[str, Any]] = []
    duplicate_chunk_ids = 0
    missing_chunk_ids = 0
    nonstandard_chunk_lengths = 0
    roundtrip_failures = 0

    for row_id in sorted(groups, key=_stable_row_key):
        chunks = groups[row_id]
        counts = Counter(int(item["chunk_id"]) for item in chunks)
        duplicates = sorted(chunk_id for chunk_id, count in counts.items() if count > 1)
        actual_ids = sorted(counts)
        expected_ids = list(range(actual_ids[-1] + 1)) if actual_ids else []
        missing = sorted(set(expected_ids) - set(actual_ids))

        duplicate_chunk_ids += len(duplicates)
        missing_chunk_ids += len(missing)
        if duplicates or missing:
            issues.append(
                {
                    "type": "chunk_sequence_error",
                    "source_file": source_file,
                    "row_id": row_id,
                    "duplicate_chunk_ids": duplicates,
                    "missing_chunk_ids": missing,
                }
            )

        ordered = sorted(chunks, key=lambda item: int(item["chunk_id"]))
        text_chunks = [str(item["chunk"]) for item in ordered]
        text = "".join(text_chunks)

        invalid_lengths = [
            index
            for index, chunk in enumerate(text_chunks[:-1])
            if len(chunk) != expected_chunk_size
        ]
        nonstandard_chunk_lengths += len(invalid_lengths)
        if invalid_lengths:
            issues.append(
                {
                    "type": "nonstandard_chunk_length",
                    "source_file": source_file,
                    "row_id": row_id,
                    "chunk_ids": invalid_lengths,
                }
            )

        roundtrip = [
            text[index : index + expected_chunk_size]
            for index in range(0, len(text), expected_chunk_size)
        ]
        if roundtrip != text_chunks:
            roundtrip_failures += 1
            issues.append(
                {
                    "type": "roundtrip_failure",
                    "source_file": source_file,
                    "row_id": row_id,
                }
            )

        normalized = _normalize_for_duplicate_check(text)
        text_without_marker = KNOWN_TRUNCATION.sub("", text)
        documents.append(
            {
                "legacy_document_id": f"{source_file}:row:{row_id}",
                "source_file": source_file,
                "row_id": row_id,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "normalized_sha256": hashlib.sha256(
                    normalized.encode("utf-8")
                ).hexdigest(),
                "chunk_count": len(text_chunks),
                "chunk_ids": actual_ids,
                "quality_flags": {
                    "empty": not bool(text.strip()),
                    "very_short": 0 < len(text.strip()) < 20,
                    "reserved": text.strip().lower() == "[reserved]",
                    "over_20000_chars": len(text) > 20000,
                    "known_source_truncation": bool(KNOWN_TRUNCATION.search(text)),
                    "contains_unexplained_cjk_or_replacement": bool(
                        CJK_OR_REPLACEMENT.search(text_without_marker)
                    ),
                    "suspicious_ending": bool(text.strip())
                    and not bool(SENTENCE_END.search(text.rstrip())),
                    "chunk_sequence_valid": not duplicates and not missing,
                    "roundtrip_valid": roundtrip == text_chunks,
                },
            }
        )

    normalized_hash_counts = Counter(doc["normalized_sha256"] for doc in documents)
    duplicate_document_groups = sum(1 for count in normalized_hash_counts.values() if count > 1)
    lengths = sorted(len(doc["text"]) for doc in documents)

    def percentile(fraction: float) -> int:
        if not lengths:
            return 0
        return lengths[round((len(lengths) - 1) * fraction)]

    metrics = {
        "source_file": source_file,
        "input_record_count": total_records,
        "ignored_record_count": ignored_records,
        "recovered_document_count": len(documents),
        "duplicate_chunk_id_count": duplicate_chunk_ids,
        "missing_chunk_id_count": missing_chunk_ids,
        "nonstandard_chunk_length_count": nonstandard_chunk_lengths,
        "roundtrip_failure_count": roundtrip_failures,
        "empty_document_count": sum(doc["quality_flags"]["empty"] for doc in documents),
        "very_short_document_count": sum(
            doc["quality_flags"]["very_short"] for doc in documents
        ),
        "reserved_document_count": sum(
            doc["quality_flags"]["reserved"] for doc in documents
        ),
        "over_20000_chars_count": sum(
            doc["quality_flags"]["over_20000_chars"] for doc in documents
        ),
        "known_source_truncation_count": sum(
            doc["quality_flags"]["known_source_truncation"] for doc in documents
        ),
        "unexplained_cjk_or_replacement_document_count": sum(
            doc["quality_flags"]["contains_unexplained_cjk_or_replacement"]
            for doc in documents
        ),
        "suspicious_ending_document_count": sum(
            doc["quality_flags"]["suspicious_ending"] for doc in documents
        ),
        "duplicate_normalized_text_group_count": duplicate_document_groups,
        "text_length": {
            "min": lengths[0] if lengths else 0,
            "p25": percentile(0.25),
            "median": percentile(0.5),
            "p75": percentile(0.75),
            "p95": percentile(0.95),
            "max": lengths[-1] if lengths else 0,
        },
    }
    return RecoveryResult(documents=documents, metrics=metrics, issues=issues)


def recover_legacy_documents(
    metadata_path: str | Path,
    source_file: str,
    expected_chunk_size: int = 500,
) -> RecoveryResult:
    import numpy as np

    records = np.load(Path(metadata_path), allow_pickle=True).tolist()
    return recover_records(records, source_file, expected_chunk_size)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_review_sample(
    documents: list[dict[str, Any]], size: int = 30
) -> list[dict[str, Any]]:
    """Build a deterministic, category-aware sample without copying full documents."""
    selected: dict[str, tuple[dict[str, Any], set[str]]] = {}

    def add(document: dict[str, Any], reason: str) -> None:
        key = document["legacy_document_id"]
        if key not in selected:
            selected[key] = (document, set())
        selected[key][1].add(reason)

    categories = {
        "known_source_truncation": [
            doc for doc in documents if doc["quality_flags"]["known_source_truncation"]
        ],
        "reserved": [doc for doc in documents if doc["quality_flags"]["reserved"]],
        "suspicious_ending": [
            doc for doc in documents if doc["quality_flags"]["suspicious_ending"]
        ],
    }
    quotas = {"known_source_truncation": 6, "reserved": 3, "suspicious_ending": 5}
    for reason, candidates in categories.items():
        quota = min(quotas[reason], len(candidates))
        for index in range(quota):
            position = round(index * (len(candidates) - 1) / max(quota - 1, 1))
            add(candidates[position], reason)

    normalized_hash_counts = Counter(doc["normalized_sha256"] for doc in documents)
    duplicate_candidates = [
        doc for doc in documents if normalized_hash_counts[doc["normalized_sha256"]] > 1
    ]
    for doc in duplicate_candidates[:4]:
        add(doc, "duplicate_normalized_text")

    ordered_by_length = sorted(documents, key=lambda doc: len(doc["text"]))
    cursor = 0
    while len(selected) < min(size, len(documents)):
        position = round(cursor * (len(ordered_by_length) - 1) / max(size - 1, 1))
        add(ordered_by_length[min(position, len(ordered_by_length) - 1)], "length_distribution")
        cursor += 1
        if cursor > size * 3:
            break

    rows = []
    for document, reasons in list(selected.values())[:size]:
        text = document["text"]
        rows.append(
            {
                "legacy_document_id": document["legacy_document_id"],
                "row_id": document["row_id"],
                "text_length": len(text),
                "review_reasons": sorted(reasons),
                "quality_flags": document["quality_flags"],
                "text_start": text[:1000],
                "text_end": text[-500:] if len(text) > 1000 else "",
            }
        )
    return rows


def write_recovery_report(path: str | Path, result: RecoveryResult) -> None:
    metrics = result.metrics
    lengths = metrics["text_length"]
    go = (
        metrics["duplicate_chunk_id_count"] == 0
        and metrics["missing_chunk_id_count"] == 0
        and metrics["roundtrip_failure_count"] == 0
        and metrics["empty_document_count"] == 0
    )
    decision = "CONDITIONAL GO" if go else "NO-GO"
    report = f"""# Title 12 Legacy Recovery Report

## Scope

- Source metadata: `{metrics['source_file']}`
- Recovery method: group by `source_file + row_id`, order by `chunk_id`, concatenate without separators
- Expected historical chunk size: 500 characters
- Decision at automated integrity gate: **{decision}**

This gate only verifies mechanical reconstruction. It does not approve eCFR alignment or citation assignment.

## Integrity metrics

| Metric | Value |
|---|---:|
| Input chunks | {metrics['input_record_count']} |
| Recovered documents | {metrics['recovered_document_count']} |
| Duplicate chunk IDs | {metrics['duplicate_chunk_id_count']} |
| Missing chunk IDs | {metrics['missing_chunk_id_count']} |
| Nonstandard intermediate chunk lengths | {metrics['nonstandard_chunk_length_count']} |
| Round-trip failures | {metrics['roundtrip_failure_count']} |
| Empty documents | {metrics['empty_document_count']} |
| Very short documents | {metrics['very_short_document_count']} |
| Reserved documents | {metrics['reserved_document_count']} |
| Documents over 20,000 characters | {metrics['over_20000_chars_count']} |
| Documents carrying the historical truncation marker | {metrics['known_source_truncation_count']} |
| Documents containing unexplained CJK/replacement characters | {metrics['unexplained_cjk_or_replacement_document_count']} |
| Documents with suspicious endings | {metrics['suspicious_ending_document_count']} |
| Duplicate normalized-text groups | {metrics['duplicate_normalized_text_group_count']} |

## Text length distribution

| Min | P25 | Median | P75 | P95 | Max |
|---:|---:|---:|---:|---:|---:|
| {lengths['min']} | {lengths['p25']} | {lengths['median']} | {lengths['p75']} | {lengths['p95']} | {lengths['max']} |

## Required manual review before alignment

1. Inspect a deterministic sample of at least 30 documents across the length distribution.
2. Treat every historical truncation marker as data loss requiring official-source recovery.
3. Review every unexplained CJK/replacement character.
4. Review abnormal endings and duplicate normalized-text groups.
5. Confirm whether `2025-09-01` is a plausible source snapshot using a small official eCFR sample.
6. Do not assign section citations until the separate alignment stage passes precision checks.

## Issue records

The machine-readable issue list is stored next to this report. `review_required` and `unmatched` alignment records will be retained in the later alignment stage rather than discarded.
"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8", newline="\n")
