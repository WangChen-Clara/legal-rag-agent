from scripts.build_title12_chunks import build_chunks, chunk_spans, validate_chunks


def canonical_document(text: str) -> dict:
    return {
        "document_id": "ecfr:title-12:section-1.1:version-2025-09-01",
        "title": 12,
        "part": "1",
        "section": "1.1",
        "heading": "§ 1.1 Test section.",
        "text": text,
        "version_date": "2025-09-01",
        "source_url": "https://example.test/section-1.1",
        "text_source": "official_ecfr_snapshot",
        "safe_for_citation": True,
        "alignment_status": "exact",
        "legacy_source_truncated": False,
    }


def test_chunk_spans_prefer_paragraph_boundaries() -> None:
    text = ("a" * 70) + "\n" + ("b" * 70) + "\n" + ("c" * 70)

    spans = chunk_spans(text, max_chars=150, overlap_chars=20)

    assert spans[0]["boundary_type"] == "paragraph"
    assert spans[0]["text"] == ("a" * 70) + "\n" + ("b" * 70)
    assert spans[-1]["char_end"] == len(text)


def test_long_paragraph_uses_bounded_overlapping_chunks() -> None:
    text = "x" * 350

    spans = chunk_spans(text, max_chars=120, overlap_chars=20)

    assert all(len(span["text"]) <= 120 for span in spans)
    assert all(span["boundary_type"] == "hard" for span in spans[:-1])
    assert spans[1]["char_start"] < spans[0]["char_end"]
    assert spans[-1]["char_end"] == len(text)


def test_build_and_validate_chunks_preserve_parent_coverage() -> None:
    document = canonical_document(("first paragraph. " * 20) + "\n" + ("second. " * 30))

    chunks = build_chunks([document], max_chars=180, overlap_chars=30)
    validation = validate_chunks([document], chunks, max_chars=180)

    assert validation == {
        "invalid_parent_references": 0,
        "invalid_slices": 0,
        "coverage_failures": 0,
    }
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert all(chunk["parent_document_id"] == document["document_id"] for chunk in chunks)
    assert all(chunk["embedding_text"].startswith("12 CFR § 1.1") for chunk in chunks)
