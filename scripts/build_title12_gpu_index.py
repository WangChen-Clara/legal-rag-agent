from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHECKPOINT_SCHEMA = "title12-bge-index-checkpoint-v1"
MANIFEST_SCHEMA = "title12-bge-index-manifest-v1"
INDEX_DIMENSION = 1024
METADATA_FIELDS = (
    "schema_version",
    "chunk_id",
    "parent_document_id",
    "chunk_index",
    "chunk_count",
    "title",
    "part",
    "section",
    "heading",
    "text",
    "char_start",
    "char_end",
    "boundary_type",
    "version_date",
    "source_url",
    "text_source",
    "safe_for_citation",
    "alignment_status",
    "legacy_source_truncated",
)
SIGNATURE_FIELDS = (
    "schema",
    "chunks_path",
    "chunks_sha256",
    "total_chunks",
    "model_path",
    "batch_size",
    "dimension",
    "normalized_embeddings",
    "index_type",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_write_index(path: Path, index: Any) -> None:
    import faiss

    temporary = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(temporary))
    os.replace(temporary, path)


def atomic_write_metadata(path: Path, records: list[dict[str, Any]]) -> None:
    import numpy as np

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, np.asarray(records, dtype=object), allow_pickle=True)
    os.replace(temporary, path)


def checkpoint_signature(
    *, chunks_path: Path, chunks_sha256: str, total_chunks: int, model_path: Path, batch_size: int
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "chunks_path": str(chunks_path.resolve()),
        "chunks_sha256": chunks_sha256,
        "total_chunks": total_chunks,
        "model_path": str(model_path.resolve()),
        "batch_size": batch_size,
        "dimension": INDEX_DIMENSION,
        "normalized_embeddings": True,
        "index_type": "IndexFlatIP",
    }


def validate_resume_checkpoint(
    checkpoint: dict[str, Any], expected: dict[str, Any]
) -> None:
    mismatches = [
        field
        for field in SIGNATURE_FIELDS
        if checkpoint.get(field) != expected.get(field)
    ]
    if mismatches:
        raise ValueError(
            "Checkpoint is incompatible with this build: " + ", ".join(mismatches)
        )
    completed = checkpoint.get("completed_chunks")
    if not isinstance(completed, int) or not 0 <= completed <= expected["total_chunks"]:
        raise ValueError("Checkpoint has an invalid completed_chunks value")


def validate_artifact_counts(
    *, index_count: int, metadata_count: int, expected_count: int
) -> None:
    if index_count != expected_count or metadata_count != expected_count:
        raise ValueError(
            "Artifact count mismatch: "
            f"index={index_count}, metadata={metadata_count}, expected={expected_count}"
        )


def metadata_for(record: dict[str, Any]) -> dict[str, Any]:
    return {field: record[field] for field in METADATA_FIELDS if field in record}


def build_manifest(
    *,
    signature: dict[str, Any],
    index_path: Path,
    metadata_path: Path,
    index_sha256: str,
    metadata_sha256: str,
    encoded_seconds: float,
    peak_allocated_mib: float,
    peak_reserved_mib: float,
    versions: dict[str, str],
) -> dict[str, Any]:
    required = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **{field: signature[field] for field in SIGNATURE_FIELDS if field != "schema"},
        "checkpoint_schema": signature["schema"],
        "index_path": str(index_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "index_sha256": index_sha256,
        "metadata_sha256": metadata_sha256,
        "encoded_seconds": round(encoded_seconds, 6),
        "chunks_per_second": round(signature["total_chunks"] / encoded_seconds, 6),
        "peak_allocated_memory_mib": round(peak_allocated_mib, 2),
        "peak_reserved_memory_mib": round(peak_reserved_mib, 2),
        "versions": versions,
    }
    return required


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build the canonical Title 12 GPU index")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=root / "data" / "canonical" / "title12_2025-09-01" / "chunks.jsonl",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "models" / "bge-large-en-v1.5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "indexes" / "title12_bge_large_2025-09-01",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_gpu_index.md",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    import faiss
    import numpy as np
    import sentence_transformers
    import torch
    from sentence_transformers import SentenceTransformer

    args = parse_args()
    if args.batch_size < 1 or args.checkpoint_every < 1:
        raise ValueError("batch-size and checkpoint-every must be positive")
    if not args.chunks.is_file():
        raise FileNotFoundError(args.chunks)
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")

    records = load_jsonl(args.chunks)
    texts = [record["embedding_text"] for record in records]
    chunks_sha256 = file_sha256(args.chunks)
    signature = checkpoint_signature(
        chunks_path=args.chunks,
        chunks_sha256=chunks_sha256,
        total_chunks=len(records),
        model_path=args.model,
        batch_size=args.batch_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "vector_db.index"
    metadata_path = args.output_dir / "metadata.npy"
    manifest_path = args.output_dir / "manifest.json"
    checkpoint_path = args.output_dir / "checkpoint.json"

    completed = 0
    encoded_seconds = 0.0
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        validate_resume_checkpoint(checkpoint, signature)
        if not index_path.is_file():
            raise FileNotFoundError("Checkpoint exists but its partial index is missing")
        index = faiss.read_index(str(index_path))
        if index.ntotal < checkpoint["completed_chunks"] or index.ntotal > len(records):
            raise ValueError("Partial index count is inconsistent with checkpoint")
        completed = index.ntotal
        encoded_seconds = float(checkpoint.get("encoded_seconds", 0.0))
        print(f"Resuming from {completed}/{len(records)} chunks")
    else:
        if index_path.exists():
            raise FileExistsError(
                f"Index exists without a checkpoint; refusing to overwrite: {index_path}"
            )
        index = faiss.IndexFlatIP(INDEX_DIMENSION)

    model = SentenceTransformer(str(args.model), device="cuda", local_files_only=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for start in range(completed, len(records), args.checkpoint_every):
        end = min(start + args.checkpoint_every, len(records))
        torch.cuda.synchronize()
        started = time.perf_counter()
        vectors = model.encode(
            texts[start:end],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32", copy=False)
        torch.cuda.synchronize()
        encoded_seconds += time.perf_counter() - started
        if vectors.shape != (end - start, INDEX_DIMENSION):
            raise ValueError(f"Unexpected embedding shape: {vectors.shape}")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
            raise ValueError("Embedding normalization check failed")
        index.add(vectors)
        atomic_write_index(index_path, index)
        checkpoint = {
            **signature,
            "status": "in_progress" if end < len(records) else "encoded",
            "completed_chunks": end,
            "encoded_seconds": encoded_seconds,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(checkpoint_path, checkpoint)
        print(f"Encoded {end}/{len(records)} chunks ({end / len(records):.1%})")

    metadata = [metadata_for(record) for record in records]
    atomic_write_metadata(metadata_path, metadata)
    validate_artifact_counts(
        index_count=index.ntotal,
        metadata_count=len(metadata),
        expected_count=len(records),
    )
    index_sha256 = file_sha256(index_path)
    metadata_sha256 = file_sha256(metadata_path)
    manifest = build_manifest(
        signature=signature,
        index_path=index_path,
        metadata_path=metadata_path,
        index_sha256=index_sha256,
        metadata_sha256=metadata_sha256,
        encoded_seconds=encoded_seconds,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
        versions={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": str(torch.version.cuda),
            "sentence_transformers": sentence_transformers.__version__,
            "faiss": faiss.__version__,
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    )
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        checkpoint_path,
        {
            **signature,
            "status": "complete",
            "completed_chunks": len(records),
            "encoded_seconds": encoded_seconds,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    report = f"""# Title 12 BGE Large GPU Index

- Status: complete
- Source chunks: {len(records)}
- Source SHA-256: `{chunks_sha256}`
- Model: `{args.model.resolve()}`
- Batch size: {args.batch_size}
- Embedding dimension: {INDEX_DIMENSION}
- Normalized document vectors: true
- FAISS index: `IndexFlatIP`
- Encoding time: {encoded_seconds:.2f} seconds
- Encoding throughput: {len(records) / encoded_seconds:.3f} chunks/s
- Peak allocated GPU memory: {manifest['peak_allocated_memory_mib']:.2f} MiB
- Peak reserved GPU memory: {manifest['peak_reserved_memory_mib']:.2f} MiB
- Index SHA-256: `{index_sha256}`
- Metadata SHA-256: `{metadata_sha256}`
"""
    atomic_write_text(args.report, report)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
