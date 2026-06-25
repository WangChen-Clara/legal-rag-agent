from rag_law.retriever import FaissRetriever


class DummyConfig:
    top_k = 5


def make_retriever(items: list[dict[str, object]]) -> FaissRetriever:
    retriever = object.__new__(FaissRetriever)
    retriever.items = items
    retriever.section_items = FaissRetriever._build_section_items(items)
    retriever.config = DummyConfig()
    return retriever


def test_split_item_preserves_metadata() -> None:
    item = {"doc_id": 7, "chunk": "legal text", "source_file": "part.csv"}
    text, metadata = FaissRetriever._split_item(item)

    assert text == "legal text"
    assert metadata["doc_id"] == 7
    assert metadata["source_file"] == "part.csv"


def test_split_item_accepts_plain_text() -> None:
    text, metadata = FaissRetriever._split_item("legal text")

    assert text == "legal text"
    assert metadata["raw_item"] == "legal text"


def test_extract_section_references_preserves_order_and_deduplicates() -> None:
    text = "See 12 CFR 211.31, § 217.134(a)(1), and § 217.134(b)(2)."

    assert FaissRetriever.extract_section_references(text) == ["211.31", "217.134"]


def test_build_section_items_indexes_metadata_sections() -> None:
    items = [
        {"section": "211.31", "text": "scope"},
        {"section": "217.134", "text": "guarantees"},
        {"section": "", "text": "ignored"},
    ]

    section_items = FaissRetriever._build_section_items(items)

    assert section_items == {"211.31": [0], "217.134": [1]}


def test_search_with_context_prioritizes_explicit_citation(monkeypatch) -> None:
    items = [
        {
            "chunk_id": "211.31:0",
            "section": "211.31",
            "text": "The provisions of this subpart apply to eligible investors.",
        },
        {
            "chunk_id": "211.10:0",
            "section": "211.10",
            "text": "Investor investment limits and permissible activities abroad.",
        },
    ]
    retriever = make_retriever(items)

    def fake_search(query: str, top_k: int | None = None):
        return [
            retriever._item_to_hit(1, rank=1, distance=0.9),
            retriever._item_to_hit(0, rank=2, distance=0.8),
        ]

    monkeypatch.setattr(retriever, "search", fake_search)

    hits = retriever.search_with_context("What does 12 CFR 211.31 apply to?", top_k=3)

    assert [hit.metadata["section"] for hit in hits] == ["211.31", "211.10"]
    assert hits[0].metadata["retrieval_source"] == "explicit_citation"
    assert hits[1].metadata["retrieval_source"] == "semantic"


def test_search_with_context_adds_one_hop_cross_reference(monkeypatch) -> None:
    items = [
        {
            "chunk_id": "217.135:0",
            "section": "217.135",
            "text": "Double default treatment covers an exposure described in § 217.134(a)(1).",
        },
        {
            "chunk_id": "217.134:0",
            "section": "217.134",
            "text": "This section defines eligible guarantee and credit derivative treatment.",
        },
    ]
    retriever = make_retriever(items)

    def fake_search(query: str, top_k: int | None = None):
        return [retriever._item_to_hit(0, rank=1, distance=0.95)]

    monkeypatch.setattr(retriever, "search", fake_search)

    hits = retriever.search_with_context("For double default treatment under 12 CFR 217.135", top_k=3)

    assert [hit.metadata["section"] for hit in hits] == ["217.135", "217.134"]
    assert hits[0].metadata["retrieval_source"] == "explicit_citation"
    assert hits[1].metadata["retrieval_source"] == "cross_reference"


def test_search_with_context_does_not_expand_unrelated_semantic_hits(monkeypatch) -> None:
    items = [
        {
            "chunk_id": "211.31:0",
            "section": "211.31",
            "text": "This subpart applies to eligible investors.",
        },
        {
            "chunk_id": "211.32:0",
            "section": "211.32",
            "text": "Definitions refer to § 211.1 and § 211.21.",
        },
        {"chunk_id": "211.1:0", "section": "211.1", "text": "General definitions."},
    ]
    retriever = make_retriever(items)

    def fake_search(query: str, top_k: int | None = None):
        return [
            retriever._item_to_hit(1, rank=1, distance=0.9),
            retriever._item_to_hit(0, rank=2, distance=0.8),
        ]

    monkeypatch.setattr(retriever, "search", fake_search)

    hits = retriever.search_with_context("What does 12 CFR 211.31 apply to?", top_k=5)

    assert [hit.metadata["section"] for hit in hits] == ["211.31", "211.32"]


def test_search_with_context_does_not_expand_semantic_hits_without_explicit_by_default(
    monkeypatch,
) -> None:
    items = [
        {
            "chunk_id": "217.135:0",
            "section": "217.135",
            "text": "Double default treatment covers an exposure described in § 217.134(a)(1).",
        },
        {
            "chunk_id": "217.134:0",
            "section": "217.134",
            "text": "This section defines eligible guarantee treatment.",
        },
    ]
    retriever = make_retriever(items)

    def fake_search(query: str, top_k: int | None = None):
        return [retriever._item_to_hit(0, rank=1, distance=0.95)]

    monkeypatch.setattr(retriever, "search", fake_search)

    default_hits = retriever.search_with_context("For double default treatment", top_k=3)
    expanded_hits = retriever.search_with_context(
        "For double default treatment",
        top_k=3,
        expand_from_semantic_without_explicit=True,
    )

    assert [hit.metadata["section"] for hit in default_hits] == ["217.135"]
    assert [hit.metadata["section"] for hit in expanded_hits] == ["217.134", "217.135"]


def test_search_with_context_ignores_unknown_explicit_citation(monkeypatch) -> None:
    items = [
        {
            "chunk_id": "211.10:0",
            "section": "211.10",
            "text": "Permissible activities abroad.",
        }
    ]
    retriever = make_retriever(items)

    def fake_search(query: str, top_k: int | None = None):
        return [retriever._item_to_hit(0, rank=1, distance=0.9)]

    monkeypatch.setattr(retriever, "search", fake_search)

    hits = retriever.search_with_context("What does 12 CFR 9999.99 say?", top_k=3)

    assert [hit.metadata["section"] for hit in hits] == ["211.10"]
    assert hits[0].metadata["retrieval_source"] == "semantic"


def test_search_with_context_includes_multiple_explicit_citations(monkeypatch) -> None:
    items = [
        {"chunk_id": "303.65:0", "section": "303.65", "text": "Merger notice."},
        {"chunk_id": "303.7:0", "section": "303.7", "text": "General public notice."},
        {"chunk_id": "999.1:0", "section": "999.1", "text": "Semantic fallback."},
    ]
    retriever = make_retriever(items)

    def fake_search(query: str, top_k: int | None = None):
        return [retriever._item_to_hit(2, rank=1, distance=0.9)]

    monkeypatch.setattr(retriever, "search", fake_search)

    hits = retriever.search_with_context("Compare 12 CFR 303.65 and 12 CFR 303.7.", top_k=5)

    assert [hit.metadata["section"] for hit in hits] == ["303.65", "303.7", "999.1"]
    assert [hit.metadata["retrieval_source"] for hit in hits[:2]] == [
        "explicit_citation",
        "explicit_citation",
    ]


def test_search_with_context_can_include_multiple_chunks_for_long_section(monkeypatch) -> None:
    items = [
        {"chunk_id": "217.134:0", "section": "217.134", "text": "Scope."},
        {"chunk_id": "217.134:1", "section": "217.134", "text": "Definitions."},
        {"chunk_id": "217.135:0", "section": "217.135", "text": "Semantic fallback."},
    ]
    retriever = make_retriever(items)

    def fake_search(query: str, top_k: int | None = None):
        return [retriever._item_to_hit(2, rank=1, distance=0.9)]

    monkeypatch.setattr(retriever, "search", fake_search)

    hits = retriever.search_with_context(
        "What does 12 CFR 217.134 cover?",
        top_k=5,
        max_chunks_per_section=2,
    )

    assert [hit.metadata["chunk_id"] for hit in hits] == [
        "217.134:0",
        "217.134:1",
        "217.135:0",
    ]
