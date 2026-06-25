from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CHUNK_SCHEMA_VERSION = "title12-chunks-v1"
DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 150
MIN_BOUNDARY_RATIO = 0.60


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
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    index = min(math.ceil(len(values) * fraction) - 1, len(values) - 1)
    return sorted(values)[max(index, 0)]


def chunk_spans(
    text: str, *, max_chars: int = DEFAULT_MAX_CHARS, overlap_chars: int = DEFAULT_OVERLAP_CHARS
) -> list[dict[str, Any]]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must satisfy 0 <= overlap_chars < max_chars")
    if not text:
        return []

    paragraph_starts = [index + 1 for index, character in enumerate(text) if character == "\n"]
    paragraph_ends = set(paragraph_starts)
    spans: list[dict[str, Any]] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        if hard_end == len(text):
            end = hard_end
            boundary_type = "document_end"
        else:
            minimum_end = start + int(max_chars * MIN_BOUNDARY_RATIO)
            candidates = [
                boundary
                for boundary in paragraph_starts
                if minimum_end <= boundary <= hard_end
            ]
            if candidates:
                end = candidates[-1]
                boundary_type = "paragraph"
            else:
                end = hard_end
                boundary_type = "hard"

        actual_start = start
        actual_end = end
        while actual_start < actual_end and text[actual_start].isspace():
            actual_start += 1
        while actual_end > actual_start and text[actual_end - 1].isspace():
            actual_end -= 1
        if actual_start < actual_end:
            spans.append(
                {
                    "char_start": actual_start,
                    "char_end": actual_end,
                    "text": text[actual_start:actual_end],
                    "boundary_type": boundary_type,
                }
            )
        if end >= len(text):
            break

        overlap_target = max(start + 1, end - overlap_chars)
        next_boundaries = [
            boundary
            for boundary in paragraph_starts
            if overlap_target <= boundary < end
        ]
        next_start = next_boundaries[0] if next_boundaries else overlap_target
        if next_start <= start:
            next_start = min(start + max_chars - overlap_chars, end)
        start = next_start

    return spans


def build_chunks(
    documents: list[dict[str, Any]], *, max_chars: int, overlap_chars: int
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for document in documents:
        spans = chunk_spans(
            document["text"], max_chars=max_chars, overlap_chars=overlap_chars
        )
        chunk_count = len(spans)
        for index, span in enumerate(spans):
            chunk_id = f"{document['document_id']}:chunk-{index + 1:04d}"
            context_prefix = (
                f"12 CFR § {document['section']} | {document['heading']}"
            )
            chunks.append(
                {
                    "schema_version": CHUNK_SCHEMA_VERSION,
                    "chunk_id": chunk_id,
                    "parent_document_id": document["document_id"],
                    "chunk_index": index,
                    "chunk_count": chunk_count,
                    "title": document["title"],
                    "part": document["part"],
                    "section": document["section"],
                    "heading": document["heading"],
                    "text": span["text"],
                    "embedding_text": context_prefix + "\n" + span["text"],
                    "char_start": span["char_start"],
                    "char_end": span["char_end"],
                    "boundary_type": span["boundary_type"],
                    "has_overlap_before": index > 0
                    and span["char_start"] < spans[index - 1]["char_end"],
                    "version_date": document["version_date"],
                    "source_url": document["source_url"],
                    "text_source": document["text_source"],
                    "safe_for_citation": document["safe_for_citation"],
                    "alignment_status": document["alignment_status"],
                    "legacy_source_truncated": document["legacy_source_truncated"],
                }
            )
    return chunks


def validate_chunks(
    documents: list[dict[str, Any]], chunks: list[dict[str, Any]], *, max_chars: int
) -> dict[str, int]:
    parents = {document["document_id"]: document for document in documents}
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Chunk IDs are not unique")
    if any(not chunk["text"] for chunk in chunks):
        raise ValueError("Empty chunk detected")
    if any(len(chunk["text"]) > max_chars for chunk in chunks):
        raise ValueError("Chunk exceeds max_chars")

    coverage_failures = 0
    invalid_parent_references = 0
    invalid_slices = 0
    chunks_by_parent: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        chunks_by_parent.setdefault(chunk["parent_document_id"], []).append(chunk)
        parent = parents.get(chunk["parent_document_id"])
        if parent is None:
            invalid_parent_references += 1
            continue
        if parent["text"][chunk["char_start"] : chunk["char_end"]] != chunk["text"]:
            invalid_slices += 1

    for parent_id, document in parents.items():
        parent_chunks = chunks_by_parent.get(parent_id, [])
        cursor = 0
        for chunk in sorted(parent_chunks, key=lambda item: item["char_start"]):
            if document["text"][cursor : chunk["char_start"]].strip():
                coverage_failures += 1
                break
            cursor = max(cursor, chunk["char_end"])
        else:
            if document["text"][cursor:].strip():
                coverage_failures += 1

    if invalid_parent_references or invalid_slices or coverage_failures:
        raise ValueError(
            "Chunk validation failed: "
            f"invalid parents={invalid_parent_references}, "
            f"invalid slices={invalid_slices}, coverage failures={coverage_failures}"
        )
    return {
        "invalid_parent_references": invalid_parent_references,
        "invalid_slices": invalid_slices,
        "coverage_failures": coverage_failures,
    }


def write_report(
    path: Path,
    *,
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    max_chars: int,
    overlap_chars: int,
    chunk_sha256: str,
) -> None:
    lengths = [len(chunk["text"]) for chunk in chunks]
    boundaries = Counter(chunk["boundary_type"] for chunk in chunks)
    report = f"""# Title 12 Chunking Report

- Chunk schema: `{CHUNK_SCHEMA_VERSION}`
- Parent documents: {len(documents)}
- Child chunks: {len(chunks)}
- Maximum characters: {max_chars}
- Requested overlap characters: {overlap_chars}
- Empty chunks: {sum(not chunk['text'] for chunk in chunks)}
- Unique chunk IDs: {len({chunk['chunk_id'] for chunk in chunks})}
- Parent coverage failures: 0
- Chunk SHA-256: `{chunk_sha256}`

## Chunk length distribution

| Metric | Characters |
|---|---:|
| minimum | {min(lengths)} |
| median | {int(statistics.median(lengths))} |
| p95 | {percentile(lengths, 0.95)} |
| maximum | {max(lengths)} |

## Boundary types

| Boundary | Count |
|---|---:|
| paragraph | {boundaries['paragraph']} |
| hard | {boundaries['hard']} |
| document_end | {boundaries['document_end']} |

Chunks retain exact character slices into their parent section. Paragraph boundaries
are preferred after 60% of the size budget; only paragraphs without a suitable boundary
are hard-split. Every embedding input receives a Title/section/heading context prefix,
while the stored `text` remains an exact official-source slice.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8", newline="\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build structured Title 12 chunks")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=root / "data" / "canonical" / "title12_2025-09-01",
    )
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    parser.add_argument(
        "--report", type=Path, default=root / "reports" / "title12_chunking_report.md"
    )
    args = parser.parse_args()

    sections_path = args.corpus_dir / "sections.jsonl"
    manifest_path = args.corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if file_sha256(sections_path) != manifest["corpus_sha256"]:
        raise ValueError("Canonical corpus hash does not match manifest")

    documents = load_jsonl(sections_path)
    chunks = build_chunks(
        documents, max_chars=args.max_chars, overlap_chars=args.overlap_chars
    )
    validation = validate_chunks(documents, chunks, max_chars=args.max_chars)
    chunks_path = args.corpus_dir / "chunks.jsonl"
    write_jsonl_atomic(chunks_path, chunks)
    chunks_sha256 = file_sha256(chunks_path)

    lengths = [len(chunk["text"]) for chunk in chunks]
    boundaries = Counter(chunk["boundary_type"] for chunk in chunks)
    manifest["chunking"] = {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "max_chars": args.max_chars,
        "overlap_chars": args.overlap_chars,
        "chunk_count": len(chunks),
        "unique_chunk_ids": len({chunk["chunk_id"] for chunk in chunks}),
        "minimum_chars": min(lengths),
        "median_chars": int(statistics.median(lengths)),
        "p95_chars": percentile(lengths, 0.95),
        "maximum_chars_observed": max(lengths),
        "boundary_counts": dict(boundaries),
        "validation": validation,
        "chunk_file": "chunks.jsonl",
        "chunk_bytes": chunks_path.stat().st_size,
        "chunk_sha256": chunks_sha256,
    }
    write_json_atomic(manifest_path, manifest)
    write_report(
        args.report,
        documents=documents,
        chunks=chunks,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        chunk_sha256=chunks_sha256,
    )
    print(json.dumps(manifest["chunking"], ensure_ascii=False))


if __name__ == "__main__":
    main()
