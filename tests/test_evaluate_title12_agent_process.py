from __future__ import annotations

from pathlib import Path

from rag_law.agent import AgentState, AgentStep, FinalAnswer
from rag_law.tools import CitationVerificationResult, RegulationEvidence, SectionRecord
from scripts.evaluate_title12_agent_process import (
    evaluate_agent_question,
    summarize_process,
    write_report,
)


class FakeAgent:
    def __init__(self, *, verification_passes: bool = True):
        self.verification_passes = verification_passes

    def run(self, question: str) -> AgentState:
        state = AgentState(question=question)
        state.steps = [
            AgentStep(1, "search_regulations", "completed", {}),
            AgentStep(2, "fetch_section", "completed", {}),
            AgentStep(
                3,
                "verify_citation",
                "completed" if self.verification_passes else "failed",
                {},
            ),
            AgentStep(4, "final_answer", "completed", {}),
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
                heading="Section heading",
                version_date="2025-09-01",
                source_url="https://example.test/211.31",
                text="Full text.",
                safe_for_citation=True,
            )
        ]
        state.citation_verifications = [
            CitationVerificationResult(
                section="211.31",
                verified=self.verification_passes,
                version_date="2025-09-01",
                source_url="https://example.test/211.31",
                safe_for_citation=True,
                checks={
                    "section_exists": True,
                    "version_matches": True,
                    "source_url_matches": self.verification_passes,
                    "safe_for_citation": True,
                },
                issues=[] if self.verification_passes else ["source_url_matches"],
            )
        ]
        state.final_answer = FinalAnswer(
            answer="Found evidence." if self.verification_passes else "Insufficient.",
            citations=["12 CFR 211.31 (2025-09-01)"] if self.verification_passes else [],
            insufficient=not self.verification_passes,
        )
        state.terminated_reason = "completed" if self.verification_passes else "insufficient_evidence"
        return state


def item() -> dict[str, object]:
    return {
        "question_id": "q1",
        "question": "What does 12 CFR 211.31 apply to?",
        "expected_sections": ["211.31"],
    }


def test_evaluate_agent_question_reports_process_success() -> None:
    result = evaluate_agent_question(FakeAgent(), item())

    assert result["tool_success"] is True
    assert result["expected_section_found"] is True
    assert result["fetch_section_success"] is True
    assert result["citation_verified"] is True
    assert result["final_answer_citation_supported"] is True
    assert result["step_actions"] == [
        "search_regulations",
        "fetch_section",
        "verify_citation",
        "final_answer",
    ]


def test_summarize_process_counts_rates() -> None:
    results = [
        evaluate_agent_question(FakeAgent(), item()),
        evaluate_agent_question(FakeAgent(verification_passes=False), item()),
    ]

    metrics = summarize_process(results)

    assert metrics["tool_success_rate"] == 0.5
    assert metrics["expected_section_found_rate"] == 1.0
    assert metrics["citation_verified_rate"] == 0.5
    assert metrics["final_answer_citation_support_rate"] == 0.5
    assert metrics["average_steps"] == 4.0
    assert metrics["termination_reason_distribution"] == {
        "completed": 1,
        "insufficient_evidence": 1,
    }


def test_write_report_records_process_metrics(tmp_path: Path) -> None:
    results = [evaluate_agent_question(FakeAgent(), item())]
    payload = {
        "schema": "schema",
        "questions": 1,
        "max_steps": 6,
        "max_fetch_sections": 2,
        "index_display_path": "index",
        "sections_display_path": "sections",
        "model_display_path": "model",
        "device": "cpu",
        "metrics": summarize_process(results),
        "results": results,
    }
    report_path = tmp_path / "process.md"

    write_report(report_path, payload)

    text = report_path.read_text(encoding="utf-8")
    assert "# Title 12 Agent Process Evaluation" in text
    assert "Citation verified rate" in text
    assert "search_regulations, fetch_section, verify_citation, final_answer" in text
