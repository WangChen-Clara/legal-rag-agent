from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Protocol

from rag_law.agent import AgentState, LegalRAGAgent
from rag_law.config import EmbeddingConfig, LLMConfig, RetrievalConfig
from rag_law.llm_client import LLMClient
from rag_law.retriever import FaissRetriever
from rag_law.tools import RegulationToolset


class AgentRunner(Protocol):
    def run(self, question: str) -> AgentState:
        ...


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def build_agent(args: argparse.Namespace) -> LegalRAGAgent:
    retriever = FaissRetriever(
        RetrievalConfig(
            index_path=args.index,
            metadata_path=args.metadata,
            top_k=args.top_k,
            normalize_query=True,
        ),
        EmbeddingConfig(model_path=args.model, device=args.device),
    )
    toolset = RegulationToolset(retriever, args.sections)
    llm_client = None
    if args.use_llm:
        llm_client = LLMClient(
            LLMConfig(
                base_url=args.llm_base_url,
                model_name=args.llm_model,
                api_key_env=args.llm_api_key_env,
                temperature=args.llm_temperature,
                top_p=args.llm_top_p,
                timeout_seconds=args.llm_timeout,
            ),
            api_key=args.llm_api_key,
        )
    return LegalRAGAgent(
        toolset,
        max_steps=args.max_steps,
        top_k=args.top_k,
        max_fetch_sections=args.max_fetch_sections,
        llm_client=llm_client,
    )


def run_question(
    agent: AgentRunner,
    question: str,
    *,
    trace_dir: Path,
) -> tuple[AgentState, Path]:
    state = agent.run(question)
    trace_path = trace_dir / f"{state.run_id}.json"
    atomic_write_json(trace_path, state.to_trace_dict())
    return state, trace_path


def render_console_output(state: AgentState, trace_path: Path) -> str:
    final_answer = state.final_answer
    citations = final_answer.citations if final_answer else []
    fetched_sections = [section.section for section in state.fetched_sections]
    verifications = [
        f"{item.section}: {'verified' if item.verified else 'failed'}"
        for item in state.citation_verifications
    ]
    evidence = [
        f"{item.rank}. {item.section or '-'} [{item.retrieval_source}]"
        for item in state.evidence[:5]
    ]
    return "\n".join(
        [
            "Answer:",
            final_answer.answer if final_answer else "No final answer.",
            "",
            "Citations:",
            ", ".join(citations) if citations else "-",
            "",
            "Fetched Sections:",
            ", ".join(fetched_sections) if fetched_sections else "-",
            "",
            "Citation Verification:",
            ", ".join(verifications) if verifications else "-",
            "",
            "Top Evidence:",
            "\n".join(evidence) if evidence else "-",
            "",
            f"Termination: {state.terminated_reason}",
            f"Trace: {trace_path}",
        ]
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Ask the Title 12 Legal RAG Agent")
    parser.add_argument("question", help="Legal question to ask")
    parser.add_argument(
        "--index",
        type=Path,
        default=root / "data" / "indexes" / "title12_bge_large_2025-09-01" / "vector_db.index",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "data" / "indexes" / "title12_bge_large_2025-09-01" / "metadata.npy",
    )
    parser.add_argument(
        "--sections",
        type=Path,
        default=root / "data" / "canonical" / "title12_2025-09-01" / "sections.jsonl",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "models" / "bge-large-en-v1.5",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-fetch-sections", type=int, default=2)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=root / "reports" / "agent_runs",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Generate the final answer with an OpenAI-compatible chat model.",
    )
    parser.add_argument("--llm-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--llm-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--llm-api-key", default="ollama")
    parser.add_argument("--llm-api-key-env", default="LLM_API_KEY")
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--llm-top-p", type=float, default=0.9)
    parser.add_argument("--llm-timeout", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = build_agent(args)
    state, trace_path = run_question(agent, args.question, trace_dir=args.trace_dir)
    print(render_console_output(state, trace_path))
    if state.terminated_reason != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
