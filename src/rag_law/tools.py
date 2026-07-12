from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .models import SearchHit
from .retriever import FaissRetriever


RetrievalMode = Literal["semantic", "citation_aware"]
ToolErrorCode = Literal["INVALID_ARGUMENT", "NOT_FOUND", "RUNTIME_ERROR"]
DEFAULT_VERSION_DATE = "2025-09-01"


@dataclass(frozen=True)
class ToolError:
    code: ToolErrorCode
    message: str
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "detail": self.detail or {},
        }


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    ok: bool
    data: dict[str, Any] | None = None
    error: ToolError | None = None
    elapsed_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "ok": self.ok,
            "data": self.data,
            "error": self.error.to_dict() if self.error else None,
            "elapsed_ms": self.elapsed_ms,
        }


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
class CitationVerificationResult:
    section: str
    verified: bool
    version_date: str | None
    source_url: str | None
    safe_for_citation: bool
    checks: dict[str, bool]
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "verified": self.verified,
            "version_date": self.version_date,
            "source_url": self.source_url,
            "safe_for_citation": self.safe_for_citation,
            "checks": self.checks,
            "issues": self.issues,
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

    def call_search_regulations(
        self,
        query: str,
        *,
        top_k: int = 10,
        mode: RetrievalMode = "citation_aware",
    ) -> ToolResult:
        started = time.perf_counter()
        try:
            result = self.search_regulations(query, top_k=top_k, mode=mode)
        except ValueError as exc:
            return _tool_error_result(
                "search_regulations",
                "INVALID_ARGUMENT",
                str(exc),
                started,
                {"query": query, "top_k": top_k, "mode": mode},
            )
        except Exception as exc:
            return _tool_error_result(
                "search_regulations",
                "RUNTIME_ERROR",
                str(exc),
                started,
                {"query": query, "top_k": top_k, "mode": mode},
            )
        return ToolResult(
            tool_name="search_regulations",
            ok=True,
            data=result.to_dict(),
            elapsed_ms=_elapsed_ms(started),
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

    def verify_citation(
        self,
        citation: str,
        *,
        expected_version_date: str = DEFAULT_VERSION_DATE,
    ) -> CitationVerificationResult:
        references = FaissRetriever.extract_section_references(citation)
        normalized = (
            references[0]
            if references
            else FaissRetriever._normalize_section_reference(citation)
        )
        if not normalized:
            raise ValueError("citation must include a section")

        record = self.fetch_section(normalized)
        expected_url = (
            f"https://www.ecfr.gov/on/{expected_version_date}/"
            f"title-{record.title}/section-{record.section}"
        )
        checks = {
            "section_exists": True,
            "version_matches": record.version_date == expected_version_date,
            "source_url_matches": record.source_url == expected_url,
            "safe_for_citation": record.safe_for_citation,
        }
        issues = [name for name, passed in checks.items() if not passed]
        return CitationVerificationResult(
            section=record.section,
            verified=not issues,
            version_date=record.version_date,
            source_url=record.source_url,
            safe_for_citation=record.safe_for_citation,
            checks=checks,
            issues=issues,
        )

    def call_fetch_section(self, section: str) -> ToolResult:
        started = time.perf_counter()
        try:
            record = self.fetch_section(section)
        except ValueError as exc:
            return _tool_error_result(
                "fetch_section",
                "INVALID_ARGUMENT",
                str(exc),
                started,
                {"section": section},
            )
        except KeyError as exc:
            return _tool_error_result(
                "fetch_section",
                "NOT_FOUND",
                str(exc).strip("'"),
                started,
                {"section": section},
            )
        except Exception as exc:
            return _tool_error_result(
                "fetch_section",
                "RUNTIME_ERROR",
                str(exc),
                started,
                {"section": section},
            )
        return ToolResult(
            tool_name="fetch_section",
            ok=True,
            data=record.to_dict(),
            elapsed_ms=_elapsed_ms(started),
        )

    def call_verify_citation(
        self,
        citation: str,
        *,
        expected_version_date: str = DEFAULT_VERSION_DATE,
    ) -> ToolResult:
        started = time.perf_counter()
        try:
            result = self.verify_citation(
                citation,
                expected_version_date=expected_version_date,
            )
        except ValueError as exc:
            return _tool_error_result(
                "verify_citation",
                "INVALID_ARGUMENT",
                str(exc),
                started,
                {
                    "citation": citation,
                    "expected_version_date": expected_version_date,
                },
            )
        except KeyError as exc:
            return _tool_error_result(
                "verify_citation",
                "NOT_FOUND",
                str(exc).strip("'"),
                started,
                {
                    "citation": citation,
                    "expected_version_date": expected_version_date,
                },
            )
        except Exception as exc:
            return _tool_error_result(
                "verify_citation",
                "RUNTIME_ERROR",
                str(exc),
                started,
                {
                    "citation": citation,
                    "expected_version_date": expected_version_date,
                },
            )
        return ToolResult(
            tool_name="verify_citation",
            ok=True,
            data=result.to_dict(),
            elapsed_ms=_elapsed_ms(started),
        )

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


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _tool_error_result(
    tool_name: str,
    code: ToolErrorCode,
    message: str,
    started: float,
    detail: dict[str, Any],
) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        ok=False,
        error=ToolError(code=code, message=message, detail=detail),
        elapsed_ms=_elapsed_ms(started),
    )
