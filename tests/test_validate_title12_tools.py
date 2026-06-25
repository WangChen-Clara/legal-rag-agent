from __future__ import annotations

from pathlib import Path

from rag_law.tools import SectionRecord
from scripts.validate_title12_tools import (
    validate_fetch_section,
    validate_search_result,
    write_report,
)

TEST_TMP = Path(".tmp") / "test_validate_title12_tools"


class FakeSearchResult:
    def to_dict(self) -> dict:
        return {
            "evidence": [
                {
                    "section": "211.31",
                    "retrieval_source": "explicit_citation",
                    "text": "scope",
                },
                {"section": "211.10", "retrieval_source": "semantic", "text": "other"},
            ]
        }


class FakeToolset:
    def search_regulations(self, query: str, *, top_k: int, mode: str):
        assert mode == "citation_aware"
        return FakeSearchResult()

    def fetch_section(self, section: str) -> SectionRecord:
        return SectionRecord(
            document_id="ecfr:title-12:section-217.134:version-2025-09-01",
            title=12,
            part="217",
            section="217.134",
            heading="§ 217.134 Guarantees and credit derivatives.",
            version_date="2025-09-01",
            source_url="https://example.test/217.134",
            text="Full section text.",
            safe_for_citation=True,
        )


def test_validate_search_result_passes_expected_section_and_source() -> None:
    result = validate_search_result(
        FakeToolset(),
        {
            "question_id": "q1",
            "question": "What does 12 CFR 211.31 apply to?",
            "expected_sections": ["211.31"],
            "expected_sources": {"211.31": "explicit_citation"},
        },
        top_k=10,
    )

    assert result["passed"] is True
    assert result["missing_sections"] == []


def test_validate_search_result_reports_source_mismatch() -> None:
    result = validate_search_result(
        FakeToolset(),
        {
            "question_id": "q1",
            "question": "What does 12 CFR 211.31 apply to?",
            "expected_sections": ["211.31"],
            "expected_sources": {"211.31": "cross_reference"},
        },
        top_k=10,
    )

    assert result["passed"] is False
    assert result["source_mismatches"][0]["actual"] == "explicit_citation"


def test_validate_fetch_section_checks_required_fields() -> None:
    result = validate_fetch_section(FakeToolset(), "12 CFR 217.134(a)(1)")

    assert result["passed"] is True
    assert result["record"]["section"] == "217.134"


def test_write_report_records_status() -> None:
    payload = {
        "schema": "schema",
        "status": "passed",
        "index_path": "index",
        "sections_path": "sections",
        "device": "cpu",
        "search_validations": [
            {
                "question_id": "q1",
                "passed": True,
                "top_evidence": [
                    {"section": "211.31", "retrieval_source": "explicit_citation"}
                ],
            }
        ],
        "fetch_section_validation": {
            "section": "217.134",
            "passed": True,
            "record": {
                "section": "217.134",
                "version_date": "2025-09-01",
                "safe_for_citation": True,
                "source_url": "url",
            },
        },
    }
    TEST_TMP.mkdir(parents=True, exist_ok=True)
    report_path = TEST_TMP / "report.md"

    write_report(report_path, payload)

    assert "Status: `passed`" in report_path.read_text(encoding="utf-8")
