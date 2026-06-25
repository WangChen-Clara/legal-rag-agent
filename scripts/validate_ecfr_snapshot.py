from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_law.ingestion.alignment import align_official_sections
from rag_law.ingestion.ecfr_parser import parse_ecfr_file
from rag_law.ingestion.legacy_recovery import write_jsonl


ALIGNMENT_ALGORITHM = "conservative-v2-prefix-index-autojunk"
PARSER_SCHEMA = "ecfr-sections-v2-letter-suffix"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    response = session.get(
        url,
        headers={
            "Accept": "application/xml",
            "User-Agent": "Legal-RAG-Research/0.1 (educational data validation)",
        },
        timeout=120,
    )
    response.raise_for_status()
    if not response.content.lstrip().startswith(b"<"):
        raise ValueError(f"Response from {url} is not XML")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(response.content)
    temporary.replace(destination)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def append_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_report(
    report_path: Path,
    *,
    date: str,
    source_label: str,
    source_count: int,
    legacy_count: int,
    alignment: list[dict],
) -> None:
    counts = Counter(item["status"] for item in alignment)
    reasons = Counter(item["reason_code"] for item in alignment)
    automatic = counts["exact"] + counts["high_confidence"]
    automatic_rate = automatic / len(alignment) if alignment else 0.0
    captured_legacy_ids = {
        item["legacy_document_id"]
        for item in alignment
        if item.get("legacy_document_id") is not None
    }
    matched_rate = len(captured_legacy_ids) / legacy_count if legacy_count else 0.0
    reason_rows = "\n".join(f"| {key} | {value} |" for key, value in reasons.items()) or "| - | 0 |"
    report = f"""# Title 12 Alignment Validation

- Snapshot date: `{date}`
- Source mode: `{source_label}`
- Official sections parsed: {source_count}
- Historical documents searched: {legacy_count}

| Status | Count |
|---|---:|
| exact | {counts['exact']} |
| high_confidence | {counts['high_confidence']} |
| review_required | {counts['review_required']} |
| unmatched | {counts['unmatched']} |

- Automatic exact/high-confidence coverage: {automatic_rate:.2%}
- Unique historical documents matched or review-captured: {len(captured_legacy_ids)} / {legacy_count} ({matched_rate:.2%})

## Reason codes

| Reason | Count |
|---|---:|
{reason_rows}

`exact` uses conservative normalized full-text equality. `high_confidence` requires
identical content-token sequences and a unique candidate. All substantive text
differences remain `review_required`; no citation is assigned automatically from a
fuzzy score.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate a fixed eCFR snapshot sample")
    parser.add_argument("--date", default="2025-09-01")
    parser.add_argument("--parts", nargs="+", default=["1", "3"])
    parser.add_argument(
        "--full-title",
        action="store_true",
        help="Download and align the full title XML instead of the small sample parts",
    )
    parser.add_argument(
        "--legacy",
        type=Path,
        default=root / "data" / "recovered" / "title12_documents.jsonl",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Full-title alignment batch size (default: 100)",
    )
    args = parser.parse_args()

    legacy = load_jsonl(args.legacy)

    if args.full_title:
        raw_dir = root / "data" / "raw" / "ecfr" / "title12" / args.date
        result_dir = root / "data" / "alignment" / "full"
        url = f"https://www.ecfr.gov/api/versioner/v1/full/{args.date}/title-12.xml"
        destination = raw_dir / "title-12.xml"
        if not destination.exists():
            download(url, destination)
        official_sections = [
            section.to_dict()
            for section in parse_ecfr_file(
                destination, title=12, version_date=args.date, requested_part=""
            )
        ]
        manifest = {
            "title": 12,
            "version_date": args.date,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "full-title",
            "files": [
                {
                    "url": url,
                    "path": str(destination.relative_to(root)),
                    "bytes": destination.stat().st_size,
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "parsed_sections": len(official_sections),
                }
            ],
        }
        report_path = root / "reports" / "title12_full_alignment.md"
        source_label = "full-title"
    else:
        raw_dir = root / "data" / "raw" / "ecfr" / "title12" / args.date
        result_dir = root / "data" / "alignment" / "sample"
        official_sections = []
        manifest_files = []
        for part in args.parts:
            url = (
                f"https://www.ecfr.gov/api/versioner/v1/full/{args.date}/"
                f"title-12.xml?part={part}"
            )
            destination = raw_dir / f"part-{part}.xml"
            if not destination.exists():
                download(url, destination)
            parsed = [
                section.to_dict()
                for section in parse_ecfr_file(
                    destination, title=12, version_date=args.date, requested_part=part
                )
            ]
            official_sections.extend(parsed)
            payload = destination.read_bytes()
            manifest_files.append(
                {
                    "part": part,
                    "url": url,
                    "path": str(destination.relative_to(root)),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "parsed_sections": len(parsed),
                }
            )
        manifest = {
            "title": 12,
            "version_date": args.date,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "sample-parts",
            "files": manifest_files,
        }
        report_path = root / "reports" / "title12_snapshot_validation.md"
        source_label = "sample-parts"

    (raw_dir / "snapshot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_jsonl(result_dir / "official_sections.jsonl", official_sections)

    alignment_path = result_dir / "alignment_results.jsonl"
    if args.full_title:
        if args.batch_size < 1:
            parser.error("--batch-size must be at least 1")
        checkpoint_path = result_dir / "checkpoint.json"
        snapshot_sha256 = manifest["files"][0]["sha256"]
        checkpoint = (
            json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint_path.exists()
            else None
        )
        if checkpoint is not None and (
            checkpoint.get("version_date") != args.date
            or checkpoint.get("snapshot_sha256") != snapshot_sha256
            or checkpoint.get("alignment_algorithm") != ALIGNMENT_ALGORITHM
            or checkpoint.get("parser_schema") != PARSER_SCHEMA
        ):
            raise RuntimeError(
                "Checkpoint belongs to a different snapshot; move the full alignment "
                "directory aside before starting a fresh run"
            )
        if alignment_path.exists() and checkpoint is None:
            raise RuntimeError(
                "Alignment results exist without checkpoint metadata; move them aside "
                "before starting a fresh run"
            )
        alignment = load_jsonl(alignment_path) if alignment_path.exists() else []
        if len(alignment) > len(official_sections):
            raise RuntimeError("Checkpoint has more rows than the current official snapshot")
        for index, existing in enumerate(alignment):
            expected = str(official_sections[index]["section"])
            if str(existing.get("official_section")) != expected:
                raise RuntimeError(
                    "Checkpoint does not match the current official snapshot at "
                    f"row {index + 1}; move it aside before starting a fresh run"
                )

        completed = len(alignment)
        if checkpoint is None:
            write_json_atomic(
                checkpoint_path,
                {
                    "title": 12,
                    "version_date": args.date,
                    "snapshot_sha256": snapshot_sha256,
                    "alignment_algorithm": ALIGNMENT_ALGORITHM,
                    "parser_schema": PARSER_SCHEMA,
                    "official_sections": len(official_sections),
                    "completed_sections": completed,
                },
            )
        print(
            f"Parsed {len(official_sections)} official sections; "
            f"resuming at {completed}/{len(official_sections)}",
            flush=True,
        )
        while completed < len(official_sections):
            batch_end = min(completed + args.batch_size, len(official_sections))
            batch = [
                record.to_dict()
                for record in align_official_sections(
                    official_sections[completed:batch_end], legacy
                )
            ]
            append_jsonl(alignment_path, batch)
            alignment.extend(batch)
            completed = batch_end
            write_json_atomic(
                checkpoint_path,
                {
                    "title": 12,
                    "version_date": args.date,
                    "snapshot_sha256": snapshot_sha256,
                    "alignment_algorithm": ALIGNMENT_ALGORITHM,
                    "parser_schema": PARSER_SCHEMA,
                    "official_sections": len(official_sections),
                    "completed_sections": completed,
                },
            )
            print(
                f"Aligned {completed}/{len(official_sections)} "
                f"({completed / len(official_sections):.1%}); "
                f"last batch {len(batch)} rows",
                flush=True,
            )
    else:
        alignment = [
            record.to_dict()
            for record in align_official_sections(official_sections, legacy)
        ]
        write_jsonl(alignment_path, alignment)

    _write_report(
        report_path,
        date=args.date,
        source_label=source_label,
        source_count=len(official_sections),
        legacy_count=len(legacy),
        alignment=alignment,
    )
    counts = Counter(item["status"] for item in alignment)
    print(json.dumps({"sections": len(official_sections), "statuses": dict(counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
