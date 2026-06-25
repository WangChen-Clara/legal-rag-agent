from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .models import SearchHit
from .retriever import FaissRetriever


RetrievalMode = Literal["semantic", "citation_aware"]


@dataclass(frozen=True)
class RegulationEvidence:
    rank: int
    section: str | None
    title: int | str | None
    part: str | None
    version_date: str | None
    source_url: str | None
    retrieval_source: str
    score: float
    text: str
    chunk_id: str | None = None
    parent_document_id: str | None = None

    @classmethod
    def from_hit(cls, hit: SearchHit) -> "RegulationEvidence":
        metadata = hit.metadata
        return cls(
            rank=hit.rank,
            section=_string_or_none(metadata.get("section")),
            title=metadata.get("title"),
            part=_string_or_none(metadata.get("part")),
            version_date=_string_or_none(
                metadata.get("version_date") or metadata.get("date")
            ),
            source_url=_string_or_none(metadata.get("source_url")),
            retrieval_source=_string_or_none(metadata.get("retrieval_source"))
            or "semantic",
            score=hit.distance,
            text=hit.text,
            chunk_id=_string_or_none(metadata.get("chunk_id")),
            parent_document_id=_string_or_none(metadata.get("parent_document_id")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "section": self.section,
            "title": self.title,
            "part": self.part,
            "version_date": self.version_date,
            "source_url": self.source_url,
            "retrieval_source": self.retrieval_source,
            "score": self.score,
            "text": self.text,
            "chunk_id": self.chunk_id,
            "parent_document_id": self.parent_document_id,
        }


@dataclass(frozen=True)
class SectionRecord:
    document_id: str
    title: int | str
    part: str
    section: str
    heading: str
    version_date: str
    source_url: str
    text: str
    safe_for_citation: bool

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SectionRecord":
        return cls(
            document_id=str(row["document_id"]),
            title=row["title"],
            part=str(row["part"]),
            section=str(row["section"]),
            heading=str(row["heading"]),
            version_date=str(row["version_date"]),
            source_url=str(row["source_url"]),
            text=str(row["text"]),
            safe_for_citation=bool(row.get("safe_for_citation", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "part": self.part,
            "section": self.section,
            "heading": self.heading,
            "version_date": self.version_date,
            "source_url": self.source_url,
            "text": self.text,
            "safe_for_citation": self.safe_for_citation,
        }


@dataclass(frozen=True)
class SearchRegulationsResult:
    query: str
    mode: RetrievalMode
    evidence: list[RegulationEvidence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "evidence": [item.to_dict() for item in self.evidence],
        }


class RegulationToolset:
    def __init__(self, retriever: FaissRetriever, sections_path: str | Path):
        self.retriever = retriever
        self.sections_path = Path(sections_path)
        self._sections_by_id: dict[str, SectionRecord] | None = None

    def search_regulations(
        self,
        query: str,
        *,
        top_k: int = 10,
        mode: RetrievalMode = "citation_aware",
    ) -> SearchRegulationsResult:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if mode == "semantic":
            hits = self.retriever.search(query, top_k=top_k)
        elif mode == "citation_aware":
            hits = self.retriever.search_with_context(query, top_k=top_k)
        else:
            raise ValueError(f"unsupported retrieval mode: {mode}")

        return SearchRegulationsResult(
            query=query,
            mode=mode,
            evidence=[RegulationEvidence.from_hit(hit) for hit in hits],
        )

    def fetch_section(self, section: str) -> SectionRecord:
        references = FaissRetriever.extract_section_references(section)
        normalized = (
            references[0]
            if references
            else FaissRetriever._normalize_section_reference(section)
        )
        if not normalized:
            raise ValueError("section must not be empty")
        sections = self._load_sections()
        try:
            return sections[normalized]
        except KeyError as exc:
            raise KeyError(f"section not found: {section}") from exc

    def _load_sections(self) -> dict[str, SectionRecord]:
        if self._sections_by_id is not None:
            return self._sections_by_id
        if not self.sections_path.is_file():
            raise FileNotFoundError(f"sections file not found: {self.sections_path}")

        sections: dict[str, SectionRecord] = {}
        with self.sections_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                record = SectionRecord.from_row(row)
                normalized = FaissRetriever._normalize_section_reference(record.section)
                if normalized in sections:
                    raise ValueError(
                        f"duplicate section {record.section} at line {line_number}"
                    )
                sections[normalized] = record
        self._sections_by_id = sections
        return sections


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
