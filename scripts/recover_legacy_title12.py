from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_law.ingestion.legacy_recovery import (
    build_review_sample,
    recover_legacy_documents,
    write_jsonl,
    write_recovery_report,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Recover Title 12 rows from legacy chunks")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "data" / "legacy_indexes" / "bge-large-embeddings" / "metadata.npy",
    )
    parser.add_argument("--source-file", default="ecfr_sections_t12_clean.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "recovered" / "title12_documents.jsonl",
    )
    parser.add_argument(
        "--issues",
        type=Path,
        default=project_root / "data" / "recovered" / "title12_recovery_issues.json",
    )
    parser.add_argument(
        "--review-sample",
        type=Path,
        default=project_root / "data" / "recovered" / "title12_review_sample.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_root / "reports" / "title12_recovery_report.md",
    )
    args = parser.parse_args()

    result = recover_legacy_documents(args.metadata, args.source_file)
    write_jsonl(args.output, result.documents)
    args.issues.parent.mkdir(parents=True, exist_ok=True)
    args.issues.write_text(
        json.dumps(result.issues, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(args.review_sample, build_review_sample(result.documents, size=30))
    write_recovery_report(args.report, result)
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
