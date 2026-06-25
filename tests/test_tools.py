from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_law.models import SearchHit
from rag_law.tools import RegulationToolset

TEST_TMP = Path(".tmp") / "test_tools"


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int | None]] = []

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        self.calls.append(("semantic", query, top_k))
        return [
            SearchHit(
                rank=1,
                distance=0.75,
                text="Semantic evidence.",
                metadata={
                    "title": 12,
                    "part": "211",
                    "section": "211.10",
                    "version_date": "2025-09-01",
                    "source_url": "https://example.test/211.10",
                    "chunk_id": "211.10:0",
                },
            )
        ]

    def search_with_context(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        self.calls.append(("citation_aware", query, top_k))
        return [
            SearchHit(
                rank=1,
                distance=0.0,
                text="Citation-aware evidence.",
                metadata={
                    "title": 12,
                    "part": "211",
                    "section": "211.31",
                    "version_date": "2025-09-01",
                    "source_url": "https://example.test/211.31",
                    "retrieval_source": "explicit_citation",
                    "chunk_id": "211.31:0",
                    "parent_document_id": "ecfr:title-12:section-211.31:version-2025-09-01",
                },
            )
        ]


def write_sections(path: Path) -> None:
    rows = [
        {
            "document_id": "ecfr:title-12:section-217.134:version-2025-09-01",
            "title": 12,
            "part": "217",
            "section": "217.134",
            "heading": "§ 217.134 Guarantees and credit derivatives.",
            "version_date": "2025-09-01",
            "source_url": "https://example.test/217.134",
            "text": "Full section text.",
            "safe_for_citation": True,
        }
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def sections_path(name: str) -> Path:
    TEST_TMP.mkdir(parents=True, exist_ok=True)
    return TEST_TMP / name


def test_search_regulations_uses_citation_aware_mode_by_default() -> None:
    retriever = FakeRetriever()
    toolset = RegulationToolset(retriever, sections_path("unused_sections.jsonl"))

    result = toolset.search_regulations("What does 12 CFR 211.31 apply to?", top_k=3)

    assert retriever.calls == [
        ("citation_aware", "What does 12 CFR 211.31 apply to?", 3)
    ]
    assert result.mode == "citation_aware"
    assert result.evidence[0].section == "211.31"
    assert result.evidence[0].retrieval_source == "explicit_citation"
    assert result.to_dict()["evidence"][0]["source_url"] == "https://example.test/211.31"


def test_search_regulations_can_use_semantic_mode() -> None:
    retriever = FakeRetriever()
    toolset = RegulationToolset(retriever, sections_path("unused_sections.jsonl"))

    result = toolset.search_regulations("general question", top_k=2, mode="semantic")

    assert retriever.calls == [("semantic", "general question", 2)]
    assert result.mode == "semantic"
    assert result.evidence[0].retrieval_source == "semantic"


def test_search_regulations_rejects_invalid_inputs() -> None:
    toolset = RegulationToolset(FakeRetriever(), sections_path("unused_sections.jsonl"))

    with pytest.raises(ValueError, match="query"):
        toolset.search_regulations("   ")
    with pytest.raises(ValueError, match="top_k"):
        toolset.search_regulations("question", top_k=0)
    with pytest.raises(ValueError, match="unsupported"):
        toolset.search_regulations("question", mode="bad")  # type: ignore[arg-type]


def test_fetch_section_returns_full_parent_section() -> None:
    path = sections_path("sections.jsonl")
    write_sections(path)
    toolset = RegulationToolset(FakeRetriever(), path)

    section = toolset.fetch_section("12 CFR 217.134(a)(1)")

    assert section.section == "217.134"
    assert section.heading == "§ 217.134 Guarantees and credit derivatives."
    assert section.text == "Full section text."
    assert section.safe_for_citation is True
    assert section.to_dict()["version_date"] == "2025-09-01"


def test_fetch_section_reports_missing_section() -> None:
    path = sections_path("sections.jsonl")
    write_sections(path)
    toolset = RegulationToolset(FakeRetriever(), path)

    with pytest.raises(KeyError, match="999.1"):
        toolset.fetch_section("999.1")
