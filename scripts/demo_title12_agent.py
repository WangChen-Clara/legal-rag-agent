from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from rag_law.agent import AgentState, LegalRAGAgent
from rag_law.config import EmbeddingConfig, RetrievalConfig
from rag_law.retriever import FaissRetriever
from rag_law.tools import RegulationToolset


DEMO_QUESTIONS = [
    {
        "question_id": "title12-dev-q001",
        "question": "What investors do the provisions of 12 CFR 211.31's subpart apply to?",
    },
    {
        "question_id": "title12-dev-q018",
        "question": (
            "For double default treatment under 12 CFR 217.135, what kind of exposure "
            "may be hedged and what related section defines the eligible guarantee or "
            "credit derivative treatment?"
        ),
    },
]


def evidence_rows(state: AgentState, limit: int = 5) -> str:
    rows = []
    for item in state.evidence[:limit]:
        rows.append(
            f"| {item.rank} | {item.section or '-'} | {item.retrieval_source} | "
            f"{item.version_date or '-'} | {item.source_url or '-'} | "
            f"{_preview(item.text)} |"
        )
    return "\n".join(rows) if rows else "| - | - | - | - | - | No evidence |"


def fetched_rows(state: AgentState) -> str:
    rows = []
    for item in state.fetched_sections:
        rows.append(
            f"| {item.section} | {item.heading} | {item.version_date} | "
            f"{item.source_url} | {item.safe_for_citation} |"
        )
    return "\n".join(rows) if rows else "| - | - | - | - | - |"


def step_rows(state: AgentState) -> str:
    return "\n".join(
        f"| {step.step_number} | {step.action} | {step.status} | {_format_detail(step.detail)} |"
        for step in state.steps
    )


def render_demo_report(payload: dict[str, Any]) -> str:
    sections = []
    for item in payload["runs"]:
        state = item["state"]
        final_answer = state.final_answer
        citations = ", ".join(final_answer.citations) if final_answer else "-"
        sections.append(
            f"""## {item['question_id']}

**Question:** {state.question}

### Agent Steps

| Step | Action | Status | Detail |
|---:|---|---|---|
{step_rows(state)}

### Retrieved Evidence

| Rank | Section | Source | Version | URL | Preview |
|---:|---|---|---|---|---|
{evidence_rows(state)}

### Fetched Sections

| Section | Heading | Version | URL | Safe for citation |
|---|---|---|---|---|
{fetched_rows(state)}

### Final Answer

{final_answer.answer if final_answer else 'No final answer.'}

**Citations:** {citations}
"""
        )
    return f"""# Title 12 Legal RAG Agent Demo

- Demo type: deterministic Agent Harness
- Snapshot date: `2025-09-01`
- Questions: {len(payload['runs'])}
- Max steps: {payload['max_steps']}
- Max fetch sections: {payload['max_fetch_sections']}
- Holdout retrieval inspected: no
- LLM called: no

This demo shows the application-level flow: question -> agent steps -> tool calls
-> evidence -> final cited answer. The answer text is template-based; this report
is intended to demonstrate control flow, tool use, and citation plumbing.

{chr(10).join(sections)}
"""


def write_report(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _preview(text: str, limit: int = 160) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


def _format_detail(detail: dict[str, Any]) -> str:
    if not detail:
        return "-"
    parts = []
    for key, value in detail.items():
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value[:5])
        parts.append(f"{key}={value}")
    return "<br>".join(parts)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render a Title 12 Agent demo report")
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
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-fetch-sections", type=int, default=2)
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_agent_demo.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    agent = LegalRAGAgent(
        toolset,
        max_steps=args.max_steps,
        top_k=args.top_k,
        max_fetch_sections=args.max_fetch_sections,
    )
    payload = {
        "max_steps": args.max_steps,
        "max_fetch_sections": args.max_fetch_sections,
        "runs": [
            {"question_id": item["question_id"], "state": agent.run(item["question"])}
            for item in DEMO_QUESTIONS
        ],
    }
    report = render_demo_report(payload)
    write_report(args.report, report)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
