from __future__ import annotations

import pytest

from rag_law.lexical_retriever import LexicalRetriever


def test_lexical_retriever_ranks_exact_terms_and_headings() -> None:
    retriever = LexicalRetriever(
        [
            {
                "chunk_id": "217.135:0",
                "section": "217.135",
                "heading": "Guarantees and credit derivatives: double default treatment.",
                "text": "A Board-regulated institution may use double default treatment.",
            },
            {
                "chunk_id": "211.31:0",
                "section": "211.31",
                "heading": "Authority, purpose, and scope.",
                "text": "This subpart applies to eligible investors.",
            },
        ]
    )

    hits = retriever.search("double default credit derivatives", top_k=2)

    assert [hit.metadata["section"] for hit in hits] == ["217.135"]
    assert hits[0].metadata["retrieval_source"] == "lexical"
    assert hits[0].distance > 0


def test_lexical_retriever_matches_section_number_tokens() -> None:
    retriever = LexicalRetriever(
        [
            {
                "chunk_id": "217.134:0",
                "section": "217.134",
                "heading": "Guarantees and credit derivatives.",
                "text": "Wholesale exposures may be covered.",
            }
        ]
    )

    hits = retriever.search("12 CFR 217.134", top_k=1)

    assert hits[0].metadata["section"] == "217.134"


def test_lexical_retriever_rejects_invalid_inputs() -> None:
    retriever = LexicalRetriever([])

    with pytest.raises(ValueError, match="query"):
        retriever.search("   ")
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("question", top_k=0)
