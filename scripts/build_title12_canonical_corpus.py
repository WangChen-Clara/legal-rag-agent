from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_law.ingestion.alignment import text_fingerprint


SCHEMA_VERSION = "title12-canonical-v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_document_id(section: str, version_date: str) -> str:
    return f"ecfr:title-12:section-{section}:version-{version_date}"


def build_records(
    official: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
    legacy_by_id: dict[str, dict[str, Any]],
    *,
    snapshot_sha256: str,
) -> list[dict[str, Any]]:
    if len(official) != len(alignment):
        raise ValueError("Official and alignment row counts differ")

    records: list[dict[str, Any]] = []
    for row_number, (section, result) in enumerate(zip(official, alignment), start=1):
        section_number = str(section["section"])
        if section_number != str(result["official_section"]):
            raise ValueError(f"Section mismatch at row {row_number}")
        text = section["text"]
        if not text.strip():
            raise ValueError(f"Empty official text at row {row_number}")

        legacy_id = result.get("legacy_document_id")
        legacy = legacy_by_id.get(legacy_id) if legacy_id else None
        legacy_flags = legacy.get("quality_flags", {}) if legacy else {}
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "document_id": canonical_document_id(
                    section_number, section["version_date"]
                ),
                "title": int(section["title"]),
                "part": str(section["part"]),
                "section": section_number,
                "heading": section["heading"],
                "text": text,
                "version_date": section["version_date"],
                "source_url": section["source_url"],
                "text_source": "official_ecfr_snapshot",
                "source_snapshot_sha256": snapshot_sha256,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "normalized_text_sha256": text_fingerprint(text),
                "safe_for_citation": True,
                "alignment_status": result["status"],
                "alignment_reason_code": result["reason_code"],
                "legacy_document_id": legacy_id,
                "legacy_row_id": result.get("legacy_row_id"),
                "legacy_source_truncated": bool(
                    legacy_flags.get("known_source_truncation")
                    or result["reason_code"] == "legacy_source_truncated"
                ),
                "legacy_text_length": len(legacy["text"]) if legacy else None,
                "official_text_length": len(text),
            }
        )

    identifiers = [record["document_id"] for record in records]
    if len(set(identifiers)) != len(identifiers):
        duplicates = [
            identifier
            for identifier, count in Counter(identifiers).items()
            if count > 1
        ]
        raise ValueError(f"Duplicate canonical document IDs: {duplicates[:10]}")
    return records


def write_report(
    path: Path,
    *,
    records: list[dict[str, Any]],
    output_path: Path,
    root: Path,
    corpus_sha256: str,
) -> None:
    statuses = Counter(record["alignment_status"] for record in records)
    truncated = sum(record["legacy_source_truncated"] for record in records)
    unique_truncated_legacy = {
        record["legacy_document_id"]
        for record in records
        if record["legacy_source_truncated"]
        and record["legacy_document_id"] is not None
    }
    report = f"""# Title 12 Canonical Corpus

- Schema: `{SCHEMA_VERSION}`
- Version date: `2025-09-01`
- Canonical documents: {len(records)}
- Unique document IDs: {len({record['document_id'] for record in records})}
- Empty official texts: {sum(not record['text'].strip() for record in records)}
- Safe for citation: {sum(record['safe_for_citation'] for record in records)}
- Canonical records linked to historical truncation: {truncated}
- Unique truncated historical documents referenced: {len(unique_truncated_legacy)}
- Corpus path: `{output_path.relative_to(root)}`
- Corpus SHA-256: `{corpus_sha256}`

## Historical alignment lineage

| Status | Count |
|---|---:|
| exact | {statuses['exact']} |
| high_confidence | {statuses['high_confidence']} |
| review_required | {statuses['review_required']} |
| unmatched | {statuses['unmatched']} |

All canonical text comes from the fixed official eCFR snapshot. Historical alignment
status is retained as migration lineage only and does not filter valid official
sections. Historical source rows remain unchanged; truncated historical rows are not
spliced or overwritten.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8", newline="\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build canonical Title 12 corpus")
    parser.add_argument(
        "--alignment-dir", type=Path, default=root / "data" / "alignment" / "full"
    )
    parser.add_argument(
        "--legacy",
        type=Path,
        default=root / "data" / "recovered" / "title12_documents.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "canonical" / "title12_2025-09-01",
    )
    parser.add_argument(
        "--report", type=Path, default=root / "reports" / "title12_canonical_corpus.md"
    )
    args = parser.parse_args()

    official = load_jsonl(args.alignment_dir / "official_sections.jsonl")
    alignment = load_jsonl(args.alignment_dir / "alignment_results.jsonl")
    legacy = load_jsonl(args.legacy)
    checkpoint = json.loads(
        (args.alignment_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    legacy_by_id = {record["legacy_document_id"]: record for record in legacy}

    records = build_records(
        official,
        alignment,
        legacy_by_id,
        snapshot_sha256=checkpoint["snapshot_sha256"],
    )
    output_path = args.output_dir / "sections.jsonl"
    write_jsonl_atomic(output_path, records)
    corpus_sha256 = file_sha256(output_path)
    truncated_links = sum(record["legacy_source_truncated"] for record in records)
    unique_truncated_legacy = {
        record["legacy_document_id"]
        for record in records
        if record["legacy_source_truncated"]
        and record["legacy_document_id"] is not None
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "title": 12,
        "version_date": "2025-09-01",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "document_count": len(records),
        "unique_document_ids": len({record["document_id"] for record in records}),
        "safe_for_citation_count": sum(
            record["safe_for_citation"] for record in records
        ),
        "historical_truncation_link_count": truncated_links,
        "unique_truncated_legacy_document_count": len(unique_truncated_legacy),
        "official_snapshot_sha256": checkpoint["snapshot_sha256"],
        "alignment_algorithm": checkpoint["alignment_algorithm"],
        "parser_schema": checkpoint["parser_schema"],
        "corpus_file": "sections.jsonl",
        "corpus_bytes": output_path.stat().st_size,
        "corpus_sha256": corpus_sha256,
    }
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    write_report(
        args.report,
        records=records,
        output_path=output_path,
        root=root,
        corpus_sha256=corpus_sha256,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
