from rag_law.ingestion.alignment import align_official_sections


def official(text: str) -> dict:
    return {
        "section": "1.1",
        "part": "1",
        "text": text,
        "source_url": "https://example.test/1.1",
    }


def legacy(text: str, row_id: int = 1) -> dict:
    return {
        "legacy_document_id": f"legacy:{row_id}",
        "row_id": row_id,
        "text": text,
    }


def test_normalized_equal_is_exact() -> None:
    result = align_official_sections(
        [official("The Bank shall comply.")],
        [legacy("  the bank shall comply.\n")],
    )
    assert result[0].status == "exact"


def test_punctuation_only_difference_is_high_confidence() -> None:
    result = align_official_sections(
        [official("The bank shall comply." )],
        [legacy("The bank shall comply")],
    )
    assert result[0].status == "high_confidence"


def test_substantive_difference_requires_review() -> None:
    result = align_official_sections(
        [official("The bank shall comply with the rule.")],
        [legacy("The bank shall not comply with the rule.")],
        review_threshold=0.70,
    )
    assert result[0].status == "review_required"


def test_truncated_legacy_prefix_requires_review() -> None:
    prefix = "The bank shall comply with all requirements. " * 40
    result = align_official_sections(
        [official(prefix + "Additional official text.")],
        [legacy(prefix + "…(已截断)")],
    )
    assert result[0].status == "review_required"
    assert result[0].reason_code == "legacy_source_truncated"
