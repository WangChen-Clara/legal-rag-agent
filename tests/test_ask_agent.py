from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from rag_law.agent import AgentState, AgentStep, FinalAnswer
from rag_law.tools import CitationVerificationResult, RegulationEvidence, SectionRecord
from scripts import ask_agent
from scripts.ask_agent import render_console_output, run_question


class FakeAgent:
    def run(self, question: str) -> AgentState:
        state = AgentState(question=question, run_id="run123")
        state.steps = [
            AgentStep(1, "search_regulations", "completed", {"sections": ["211.31"]}),
            AgentStep(2, "fetch_section", "completed", {"section": "211.31"}),
            AgentStep(3, "verify_citation", "completed", {"section": "211.31"}),
            AgentStep(4, "final_answer", "completed", {"citations": ["12 CFR 211.31"]}),
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
                text="Evidence text.",
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
                verified=True,
                version_date="2025-09-01",
                source_url="https://example.test/211.31",
                safe_for_citation=True,
                checks={
                    "section_exists": True,
                    "version_matches": True,
                    "source_url_matches": True,
                    "safe_for_citation": True,
                },
                issues=[],
            )
        ]
        state.final_answer = FinalAnswer(
            answer="Relevant evidence was found.",
            citations=["12 CFR 211.31 (2025-09-01)"],
        )
        state.terminated_reason = "completed"
        return state


def test_run_question_writes_trace(tmp_path: Path) -> None:
    state, trace_path = run_question(
        FakeAgent(),
        "What does 12 CFR 211.31 apply to?",
        trace_dir=tmp_path,
    )

    assert state.run_id == "run123"
    assert trace_path == tmp_path / "run123.json"
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "legal-rag-agent-trace-v1"
    assert payload["citation_verifications"][0]["verified"] is True
    assert payload["final_answer"]["citations"] == ["12 CFR 211.31 (2025-09-01)"]


def test_render_console_output_shows_agent_result(tmp_path: Path) -> None:
    state = FakeAgent().run("question")
    output = render_console_output(state, tmp_path / "run123.json")

    assert "Answer:" in output
    assert "Relevant evidence was found." in output
    assert "12 CFR 211.31 (2025-09-01)" in output
    assert "211.31: verified" in output
    assert "Trace:" in output


def test_build_agent_injects_llm_when_enabled(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeRetriever:
        def __init__(self, retrieval_config, embedding_config):
            captured["retrieval_config"] = retrieval_config
            captured["embedding_config"] = embedding_config

    class FakeToolset:
        def __init__(self, retriever, sections):
            captured["retriever"] = retriever
            captured["sections"] = sections

    class FakeAgent:
        def __init__(self, toolset, *, max_steps, top_k, max_fetch_sections, llm_client):
            captured["toolset"] = toolset
            captured["max_steps"] = max_steps
            captured["top_k"] = top_k
            captured["max_fetch_sections"] = max_fetch_sections
            captured["llm_client"] = llm_client

    monkeypatch.setattr(ask_agent, "FaissRetriever", FakeRetriever)
    monkeypatch.setattr(ask_agent, "RegulationToolset", FakeToolset)
    monkeypatch.setattr(ask_agent, "LegalRAGAgent", FakeAgent)
    args = Namespace(
        index=tmp_path / "vector_db.index",
        metadata=tmp_path / "metadata.npy",
        sections=tmp_path / "sections.jsonl",
        model=tmp_path / "model",
        device="cpu",
        top_k=3,
        max_steps=4,
        max_fetch_sections=1,
        use_llm=True,
        llm_base_url="http://localhost:11434/v1",
        llm_model="qwen2.5:7b-instruct",
        llm_api_key="ollama",
        llm_api_key_env="LLM_API_KEY",
        llm_temperature=0.1,
        llm_top_p=0.9,
        llm_timeout=120,
    )

    ask_agent.build_agent(args)

    assert captured["max_steps"] == 4
    assert captured["top_k"] == 3
    assert captured["max_fetch_sections"] == 1
    assert captured["llm_client"] is not None
    assert captured["llm_client"].config.model_name == "qwen2.5:7b-instruct"
