from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_title12_gpu_index import (
    MANIFEST_SCHEMA,
    build_manifest,
    checkpoint_signature,
    metadata_for,
    validate_artifact_counts,
    validate_resume_checkpoint,
)


def signature() -> dict[str, object]:
    return checkpoint_signature(
        chunks_path=Path("chunks.jsonl"),
        chunks_sha256="abc123",
        total_chunks=10,
        model_path=Path("model"),
        batch_size=8,
    )


def test_checkpoint_accepts_matching_resume() -> None:
    checkpoint = {**signature(), "completed_chunks": 6}
    validate_resume_checkpoint(checkpoint, signature())


@pytest.mark.parametrize("field", ["chunks_sha256", "model_path", "batch_size"])
def test_checkpoint_rejects_changed_build_inputs(field: str) -> None:
    checkpoint = {**signature(), "completed_chunks": 6}
    checkpoint[field] = "changed"
    with pytest.raises(ValueError, match=field):
        validate_resume_checkpoint(checkpoint, signature())


def test_artifact_counts_must_match() -> None:
    validate_artifact_counts(index_count=10, metadata_count=10, expected_count=10)
    with pytest.raises(ValueError, match="Artifact count mismatch"):
        validate_artifact_counts(index_count=10, metadata_count=9, expected_count=10)


def test_metadata_keeps_citation_fields_without_embedding_text() -> None:
    record = {
        "chunk_id": "chunk-1",
        "text": "Official text",
        "section": "1.1",
        "version_date": "2025-09-01",
        "source_url": "https://example.test/section/1.1",
        "embedding_text": "context\nOfficial text",
    }
    metadata = metadata_for(record)
    assert metadata["text"] == "Official text"
    assert metadata["section"] == "1.1"
    assert metadata["version_date"] == "2025-09-01"
    assert metadata["source_url"].startswith("https://")
    assert "embedding_text" not in metadata


def test_manifest_contains_reproducibility_fields() -> None:
    manifest = build_manifest(
        signature=signature(),
        index_path=Path("vector_db.index"),
        metadata_path=Path("metadata.npy"),
        index_sha256="index-sha",
        metadata_sha256="metadata-sha",
        encoded_seconds=2.0,
        peak_allocated_mib=100.0,
        peak_reserved_mib=120.0,
        versions={"python": "3.10"},
    )
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["chunks_sha256"] == "abc123"
    assert manifest["index_type"] == "IndexFlatIP"
    assert manifest["normalized_embeddings"] is True
    assert manifest["chunks_per_second"] == 5.0
