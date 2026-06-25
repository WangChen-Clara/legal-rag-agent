from __future__ import annotations

from pathlib import Path

from rag_law.agent import AgentState, AgentStep, FinalAnswer
from rag_law.tools import RegulationEvidence, SectionRecord
from scripts.validate_title12_agent import validate_agent_question, write_report

TEST_TMP = Path(".tmp") / "test_validate_title12_agent"


class FakeAgent:
    def run(self, question: str) -> AgentState:
        state = AgentState(question=question)
        state.steps = [
            AgentStep(1, "search_regulations", "completed", {}),
            AgentStep(2, "fetch_section", "completed", {}),
            AgentStep(3, "final_answer", "completed", {}),
        ]
        state.evidence = [
            RegulationEvidence(
                rank=1,
                section="211.31",
                title=12,
                part="211",
                version_date="2025-09-01",
                source_url="https://example.test/211.31",
                retrieval_source="explicit_citation",
                score=0.0,
                text="Evidence.",
            )
        ]
        state.fetched_sections = [
            SectionRecord(
                document_id="doc",
                title=12,
                part="211",
                section="211.31",
                heading="§ 211.31 Scope.",
                version_date="2025-09-01",
                source_url="https://example.test/211.31",
                text="Full text.",
                safe_for_citation=True,
            )
        ]
        state.final_answer = FinalAnswer(
            answer="Found evidence.",
            citations=["12 CFR 211.31 (2025-09-01)"],
        )
        state.terminated_reason = "completed"
        return state


def test_validate_agent_question_accepts_completed_state() -> None:
    result = validate_agent_question(
        FakeAgent(),
        {
            "question_id": "q1",
            "question": "What does 12 CFR 211.31 apply to?",
            "expected_sections": ["211.31"],
        },
    )

    assert result["passed"] is True
    assert result["citations"] == ["12 CFR 211.31 (2025-09-01)"]
    assert [step["action"] for step in result["steps"]] == [
        "search_regulations",
        "fetch_section",
        "final_answer",
    ]


def test_validate_agent_question_reports_missing_section() -> None:
    result = validate_agent_question(
        FakeAgent(),
        {
            "question_id": "q1",
            "question": "What does 12 CFR 999.1 say?",
            "expected_sections": ["999.1"],
        },
    )

    assert result["passed"] is False
    assert result["missing_sections"] == ["999.1"]


def test_write_report_records_agent_steps() -> None:
    TEST_TMP.mkdir(parents=True, exist_ok=True)
    report_path = TEST_TMP / "report.md"
    payload = {
        "schema": "schema",
        "status": "passed",
        "max_steps": 4,
        "max_fetch_sections": 2,
        "index_path": "index",
        "sections_path": "sections",
        "device": "cpu",
        "validations": [
            {
                "question_id": "q1",
                "passed": True,
                "terminated_reason": "completed",
                "citations": ["12 CFR 211.31 (2025-09-01)"],
                "steps": [
                    {"action": "search_regulations"},
                    {"action": "fetch_section"},
                    {"action": "final_answer"},
                ],
            }
        ],
    }

    write_report(report_path, payload)

    text = report_path.read_text(encoding="utf-8")
    assert "Status: `passed`" in text
    assert "search_regulations, fetch_section, final_answer" in text
