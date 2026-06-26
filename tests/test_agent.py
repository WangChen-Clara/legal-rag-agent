from __future__ import annotations

import pytest

from rag_law.agent import TRACE_SCHEMA, LegalRAGAgent
from rag_law.tools import RegulationEvidence, SearchRegulationsResult, SectionRecord


class FakeToolset:
    def __init__(self, evidence: list[RegulationEvidence]):
        self.evidence = evidence
        self.search_calls: list[tuple[str, int, str]] = []
        self.fetch_calls: list[str] = []

    def search_regulations(
        self,
        query: str,
        *,
        top_k: int = 10,
        mode: str = "citation_aware",
    ) -> SearchRegulationsResult:
        self.search_calls.append((query, top_k, mode))
        return SearchRegulationsResult(
            query=query,
            mode="citation_aware",
            evidence=self.evidence,
        )

    def fetch_section(self, section: str) -> SectionRecord:
        self.fetch_calls.append(section)
        return SectionRecord(
            document_id=f"ecfr:title-12:section-{section}:version-2025-09-01",
            title=12,
            part=section.split(".")[0],
            section=section,
            heading=f"§ {section} Heading.",
            version_date="2025-09-01",
            source_url=f"https://example.test/{section}",
            text=f"Full text for {section}.",
            safe_for_citation=True,
        )


def evidence(
    section: str,
    *,
    retrieval_source: str = "semantic",
    rank: int = 1,
) -> RegulationEvidence:
    return RegulationEvidence(
        rank=rank,
        section=section,
        title=12,
        part=section.split(".")[0],
        version_date="2025-09-01",
        source_url=f"https://example.test/{section}",
        retrieval_source=retrieval_source,
        score=0.9,
        text=f"Evidence for {section}.",
        chunk_id=f"{section}:0",
        parent_document_id=f"ecfr:title-12:section-{section}:version-2025-09-01",
    )


def test_agent_runs_search_fetch_and_final_answer() -> None:
    toolset = FakeToolset(
        [
            evidence("217.135", retrieval_source="explicit_citation", rank=1),
            evidence("217.134", retrieval_source="cross_reference", rank=2),
        ]
    )
    agent = LegalRAGAgent(toolset, max_steps=3, top_k=10, max_fetch_sections=1)

    state = agent.run("For double default treatment under 12 CFR 217.135...")

    assert state.terminated_reason == "completed"
    assert [step.action for step in state.steps] == [
        "search_regulations",
        "fetch_section",
        "final_answer",
    ]
    assert toolset.search_calls == [
        ("For double default treatment under 12 CFR 217.135...", 10, "citation_aware")
    ]
    assert toolset.fetch_calls == ["217.135"]
    assert state.final_answer is not None
    assert state.final_answer.citations == ["12 CFR 217.135 (2025-09-01)"]


def test_agent_default_steps_allow_two_fetches_and_final_answer() -> None:
    toolset = FakeToolset(
        [
            evidence("217.135", retrieval_source="explicit_citation", rank=1),
            evidence("217.134", retrieval_source="cross_reference", rank=2),
        ]
    )
    agent = LegalRAGAgent(toolset)

    state = agent.run("For double default treatment under 12 CFR 217.135...")

    assert state.terminated_reason == "completed"
    assert [step.action for step in state.steps] == [
        "search_regulations",
        "fetch_section",
        "fetch_section",
        "final_answer",
    ]
    assert toolset.fetch_calls == ["217.135", "217.134"]


def test_agent_state_exports_structured_trace() -> None:
    toolset = FakeToolset(
        [evidence("217.135", retrieval_source="explicit_citation", rank=1)]
    )
    agent = LegalRAGAgent(toolset, max_steps=3, max_fetch_sections=1)

    state = agent.run("For double default treatment under 12 CFR 217.135...")
    trace = state.to_trace_dict()

    assert trace["schema"] == TRACE_SCHEMA
    assert trace["run_id"] == state.run_id
    assert trace["question"] == "For double default treatment under 12 CFR 217.135..."
    assert [step["action"] for step in trace["steps"]] == [
        "search_regulations",
        "fetch_section",
        "final_answer",
    ]
    assert trace["evidence_summary"][0]["section"] == "217.135"
    assert trace["fetched_sections"][0]["section"] == "217.135"
    assert trace["final_answer"]["citations"] == ["12 CFR 217.135 (2025-09-01)"]
    assert trace["termination_reason"] == "completed"


def test_agent_returns_insufficient_when_no_evidence() -> None:
    toolset = FakeToolset([])
    agent = LegalRAGAgent(toolset)

    state = agent.run("unknown question")

    assert state.terminated_reason == "insufficient_evidence"
    assert state.final_answer is not None
    assert state.final_answer.insufficient is True
    assert [step.action for step in state.steps] == ["search_regulations"]


def test_agent_limits_fetch_sections() -> None:
    toolset = FakeToolset(
        [
            evidence("217.135", retrieval_source="explicit_citation", rank=1),
            evidence("217.134", retrieval_source="cross_reference", rank=2),
            evidence("217.142", retrieval_source="cross_reference", rank=3),
        ]
    )
    agent = LegalRAGAgent(toolset, max_steps=5, max_fetch_sections=2)

    state = agent.run("For double default treatment under 12 CFR 217.135...")

    assert state.terminated_reason == "completed"
    assert toolset.fetch_calls == ["217.135", "217.134"]


def test_agent_rejects_invalid_configuration() -> None:
    toolset = FakeToolset([])

    with pytest.raises(ValueError, match="max_steps"):
        LegalRAGAgent(toolset, max_steps=0)
    with pytest.raises(ValueError, match="top_k"):
        LegalRAGAgent(toolset, top_k=0)
    with pytest.raises(ValueError, match="max_fetch_sections"):
        LegalRAGAgent(toolset, max_fetch_sections=-1)
    with pytest.raises(ValueError, match="max_steps must be at least max_fetch_sections"):
        LegalRAGAgent(toolset, max_steps=3, max_fetch_sections=2)


def test_agent_rejects_empty_question() -> None:
    agent = LegalRAGAgent(FakeToolset([]))

    with pytest.raises(ValueError, match="question"):
        agent.run("  ")
