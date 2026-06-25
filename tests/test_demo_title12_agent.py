from __future__ import annotations

from pathlib import Path

from rag_law.agent import AgentState, AgentStep, FinalAnswer
from rag_law.tools import RegulationEvidence, SectionRecord
from scripts.demo_title12_agent import render_demo_report, write_report

TEST_TMP = Path(".tmp") / "test_demo_title12_agent"


def demo_state() -> AgentState:
    state = AgentState(question="What does 12 CFR 211.31 apply to?")
    state.steps = [
        AgentStep(
            1,
            "search_regulations",
            "completed",
            {"mode": "citation_aware", "sections": ["211.31"]},
        ),
        AgentStep(2, "fetch_section", "completed", {"section": "211.31"}),
        AgentStep(3, "final_answer", "completed", {"citations": ["12 CFR 211.31"]}),
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
            text="The provisions of this subpart apply to eligible investors.",
        )
    ]
    state.fetched_sections = [
        SectionRecord(
            document_id="doc",
            title=12,
            part="211",
            section="211.31",
            heading="§ 211.31 Authority, purpose, and scope.",
            version_date="2025-09-01",
            source_url="https://example.test/211.31",
            text="Full text.",
            safe_for_citation=True,
        )
    ]
    state.final_answer = FinalAnswer(
        answer="Relevant evidence was found.",
        citations=["12 CFR 211.31 (2025-09-01)"],
    )
    state.terminated_reason = "completed"
    return state


def test_render_demo_report_shows_agent_flow() -> None:
    report = render_demo_report(
        {
            "max_steps": 4,
            "max_fetch_sections": 2,
            "runs": [{"question_id": "q1", "state": demo_state()}],
        }
    )

    assert "# Title 12 Legal RAG Agent Demo" in report
    assert "search_regulations" in report
    assert "fetch_section" in report
    assert "211.31 | explicit_citation" in report
    assert "12 CFR 211.31 (2025-09-01)" in report
    assert "LLM called: no" in report


def test_write_report_writes_markdown() -> None:
    TEST_TMP.mkdir(parents=True, exist_ok=True)
    report_path = TEST_TMP / "demo.md"

    write_report(report_path, "# Demo\n")

    assert report_path.read_text(encoding="utf-8") == "# Demo\n"
