from rag_law.ingestion.legacy_recovery import recover_records


def test_recovery_orders_and_concatenates_chunks() -> None:
    records = [
        {"source_file": "t12.csv", "row_id": 4, "chunk_id": 1, "chunk": "world."},
        {"source_file": "t12.csv", "row_id": 4, "chunk_id": 0, "chunk": "hello "},
        {"source_file": "other.csv", "row_id": 0, "chunk_id": 0, "chunk": "ignore"},
    ]

    result = recover_records(records, "t12.csv", expected_chunk_size=6)

    assert len(result.documents) == 1
    assert result.documents[0]["text"] == "hello world."
    assert result.documents[0]["quality_flags"]["chunk_sequence_valid"] is True
    assert result.metrics["input_record_count"] == 2


def test_recovery_reports_missing_chunk_ids() -> None:
    records = [
        {"source_file": "t12.csv", "row_id": 1, "chunk_id": 0, "chunk": "abc"},
        {"source_file": "t12.csv", "row_id": 1, "chunk_id": 2, "chunk": "xyz"},
    ]

    result = recover_records(records, "t12.csv", expected_chunk_size=3)

    assert result.metrics["missing_chunk_id_count"] == 1
    assert result.documents[0]["quality_flags"]["chunk_sequence_valid"] is False
    assert result.issues[0]["type"] == "chunk_sequence_error"


def test_known_truncation_marker_is_not_reported_as_unexplained_cjk() -> None:
    records = [
        {
            "source_file": "t12.csv",
            "row_id": 1,
            "chunk_id": 0,
            "chunk": "legal text …(已截断)",
        }
    ]

    result = recover_records(records, "t12.csv", expected_chunk_size=500)
    flags = result.documents[0]["quality_flags"]

    assert flags["known_source_truncation"] is True
    assert flags["contains_unexplained_cjk_or_replacement"] is False
