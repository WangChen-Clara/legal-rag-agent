from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from rag_law.agent import AgentState, FinalAnswer
from rag_law.api import APISettings, create_app, default_settings
from rag_law.tools import CitationVerificationResult


class FakeAgent:
    def run(self, question: str) -> AgentState:
        state = AgentState(question=question, run_id="trace123")
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
            answer="API answer.",
            citations=["12 CFR 211.31 (2025-09-01)"],
        )
        state.terminated_reason = "completed"
        return state


def settings_for_test(tmp_path: Path) -> APISettings:
    base = default_settings(project_root=tmp_path)
    return APISettings(
        project_root=tmp_path,
        index_path=base.index_path,
        metadata_path=base.metadata_path,
        sections_path=base.sections_path,
        embedding_model_path=base.embedding_model_path,
        trace_dir=tmp_path / "traces",
    )


def test_health_endpoint(tmp_path: Path) -> None:
    app = create_app(agent=FakeAgent(), settings=settings_for_test(tmp_path))
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["llm_enabled"] is False


def test_ask_writes_trace_and_returns_answer(tmp_path: Path) -> None:
    app = create_app(agent=FakeAgent(), settings=settings_for_test(tmp_path))
    client = TestClient(app)

    response = client.post("/ask", json={"question": "What applies?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "API answer."
    assert payload["citations"] == ["12 CFR 211.31 (2025-09-01)"]
    assert payload["trace_id"] == "trace123"
    assert payload["termination_reason"] == "completed"
    trace_path = tmp_path / "traces" / "trace123.json"
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["run_id"] == "trace123"


def test_trace_endpoint_returns_existing_trace(tmp_path: Path) -> None:
    app = create_app(agent=FakeAgent(), settings=settings_for_test(tmp_path))
    client = TestClient(app)
    client.post("/ask", json={"question": "What applies?"})

    response = client.get("/trace/trace123")

    assert response.status_code == 200
    assert response.json()["run_id"] == "trace123"


def test_trace_endpoint_rejects_path_traversal(tmp_path: Path) -> None:
    app = create_app(agent=FakeAgent(), settings=settings_for_test(tmp_path))
    client = TestClient(app)

    response = client.get("/trace/..%5Csecret")

    assert response.status_code == 400


def test_trace_endpoint_404_for_missing_trace(tmp_path: Path) -> None:
    app = create_app(agent=FakeAgent(), settings=settings_for_test(tmp_path))
    client = TestClient(app)

    response = client.get("/trace/missing")

    assert response.status_code == 404
