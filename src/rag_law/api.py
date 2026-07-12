from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import AgentState, LegalRAGAgent
from .config import EmbeddingConfig, LLMConfig, RetrievalConfig
from .llm_client import LLMClient
from .retriever import FaissRetriever
from .tools import RegulationToolset


class AgentRunner(Protocol):
    def run(self, question: str) -> AgentState:
        ...


class LazyAgent:
    def __init__(self, settings: APISettings):
        self.settings = settings
        self._agent: LegalRAGAgent | None = None

    def run(self, question: str) -> AgentState:
        if self._agent is None:
            self._agent = build_agent(self.settings)
        return self._agent.run(question)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    trace_id: str
    trace_path: str
    termination_reason: str | None
    citation_verifications: list[dict[str, Any]]


@dataclass(frozen=True)
class APISettings:
    project_root: Path
    index_path: Path
    metadata_path: Path
    sections_path: Path
    embedding_model_path: Path
    device: str = "cpu"
    top_k: int = 10
    max_steps: int = 6
    max_fetch_sections: int = 2
    trace_dir: Path | None = None
    use_llm: bool = False
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b-instruct"
    llm_api_key: str = "ollama"
    llm_api_key_env: str = "LLM_API_KEY"
    llm_temperature: float = 0.1
    llm_top_p: float = 0.9
    llm_timeout: int = 120

    @property
    def resolved_trace_dir(self) -> Path:
        return self.trace_dir or self.project_root / "reports" / "agent_runs"


def default_settings(project_root: Path | None = None) -> APISettings:
    root = project_root or Path(__file__).resolve().parents[2]
    return APISettings(
        project_root=root,
        index_path=Path(
            os.getenv(
                "RAG_LAW_INDEX",
                root / "data" / "indexes" / "title12_bge_large_2025-09-01" / "vector_db.index",
            )
        ),
        metadata_path=Path(
            os.getenv(
                "RAG_LAW_METADATA",
                root / "data" / "indexes" / "title12_bge_large_2025-09-01" / "metadata.npy",
            )
        ),
        sections_path=Path(
            os.getenv(
                "RAG_LAW_SECTIONS",
                root / "data" / "canonical" / "title12_2025-09-01" / "sections.jsonl",
            )
        ),
        embedding_model_path=Path(
            os.getenv("RAG_LAW_EMBEDDING_MODEL", root / "models" / "bge-large-en-v1.5")
        ),
        device=os.getenv("RAG_LAW_DEVICE", "cpu"),
        top_k=int(os.getenv("RAG_LAW_TOP_K", "10")),
        max_steps=int(os.getenv("RAG_LAW_MAX_STEPS", "6")),
        max_fetch_sections=int(os.getenv("RAG_LAW_MAX_FETCH_SECTIONS", "2")),
        trace_dir=Path(os.getenv("RAG_LAW_TRACE_DIR"))
        if os.getenv("RAG_LAW_TRACE_DIR")
        else None,
        use_llm=os.getenv("RAG_LAW_USE_LLM", "").lower() in {"1", "true", "yes"},
        llm_base_url=os.getenv("RAG_LAW_LLM_BASE_URL", "http://localhost:11434/v1"),
        llm_model=os.getenv("RAG_LAW_LLM_MODEL", "qwen2.5:7b-instruct"),
        llm_api_key=os.getenv("RAG_LAW_LLM_API_KEY", "ollama"),
        llm_api_key_env=os.getenv("RAG_LAW_LLM_API_KEY_ENV", "LLM_API_KEY"),
    )


def build_agent(settings: APISettings) -> LegalRAGAgent:
    retriever = FaissRetriever(
        RetrievalConfig(
            index_path=settings.index_path,
            metadata_path=settings.metadata_path,
            top_k=settings.top_k,
            normalize_query=True,
        ),
        EmbeddingConfig(
            model_path=settings.embedding_model_path,
            device=settings.device,
        ),
    )
    toolset = RegulationToolset(retriever, settings.sections_path)
    llm_client = None
    if settings.use_llm:
        llm_client = LLMClient(
            LLMConfig(
                base_url=settings.llm_base_url,
                model_name=settings.llm_model,
                api_key_env=settings.llm_api_key_env,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                timeout_seconds=settings.llm_timeout,
            ),
            api_key=settings.llm_api_key,
        )
    return LegalRAGAgent(
        toolset,
        max_steps=settings.max_steps,
        top_k=settings.top_k,
        max_fetch_sections=settings.max_fetch_sections,
        llm_client=llm_client,
    )


def write_trace(trace_dir: Path, state: AgentState) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{state.run_id}.json"
    temporary = trace_path.with_suffix(trace_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state.to_trace_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, trace_path)
    return trace_path


def create_app(
    *,
    agent: AgentRunner | None = None,
    settings: APISettings | None = None,
) -> FastAPI:
    app_settings = settings or default_settings()
    app_agent = agent or LazyAgent(app_settings)
    trace_dir = app_settings.resolved_trace_dir
    app = FastAPI(title="Legal RAG Agent API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "llm_enabled": app_settings.use_llm,
            "trace_dir": str(trace_dir),
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        state = app_agent.run(request.question)
        trace_path = write_trace(trace_dir, state)
        answer = state.final_answer.answer if state.final_answer else ""
        citations = state.final_answer.citations if state.final_answer else []
        return AskResponse(
            answer=answer,
            citations=citations,
            trace_id=state.run_id,
            trace_path=str(trace_path),
            termination_reason=state.terminated_reason,
            citation_verifications=[
                item.to_dict() for item in state.citation_verifications
            ],
        )

    @app.get("/trace/{trace_id}")
    def trace(trace_id: str) -> dict[str, Any]:
        if not trace_id or any(char in trace_id for char in "\\/."):
            raise HTTPException(status_code=400, detail="invalid trace_id")
        trace_path = trace_dir / f"{trace_id}.json"
        if not trace_path.exists():
            raise HTTPException(status_code=404, detail="trace not found")
        return json.loads(trace_path.read_text(encoding="utf-8"))

    return app


app = create_app()
