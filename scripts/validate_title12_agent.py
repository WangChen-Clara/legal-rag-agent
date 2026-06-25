from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from rag_law.agent import LegalRAGAgent
from rag_law.config import EmbeddingConfig, RetrievalConfig
from rag_law.retriever import FaissRetriever
from rag_law.tools import RegulationToolset


RESULT_SCHEMA = "title12-agent-validation-v1"
VALIDATION_QUESTIONS = [
    {
        "question_id": "title12-dev-q001",
        "question": "What investors do the provisions of 12 CFR 211.31's subpart apply to?",
        "expected_sections": ["211.31"],
    },
    {
        "question_id": "title12-dev-q018",
        "question": (
            "For double default treatment under 12 CFR 217.135, what kind of exposure "
            "may be hedged and what related section defines the eligible guarantee or "
            "credit derivative treatment?"
        ),
        "expected_sections": ["217.135", "217.134"],
    },
]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def validate_agent_question(agent: LegalRAGAgent, item: dict[str, Any]) -> dict[str, Any]:
    state = agent.run(item["question"])
    fetched_sections = [section.section for section in state.fetched_sections]
    evidence_sections = [evidence.section for evidence in state.evidence if evidence.section]
    missing_sections = [
        section
        for section in item["expected_sections"]
        if section not in evidence_sections and section not in fetched_sections
    ]
    passed = state.terminated_reason == "completed" and not missing_sections
    return {
        "question_id": item["question_id"],
        "question": item["question"],
        "passed": passed,
        "terminated_reason": state.terminated_reason,
        "missing_sections": missing_sections,
        "steps": [step.to_dict() for step in state.steps],
        "citations": state.final_answer.citations if state.final_answer else [],
        "evidence_sections": evidence_sections[:10],
        "fetched_sections": fetched_sections,
    }


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def write_report(path: Path, payload: dict[str, Any]) -> None:
    rows = []
    for item in payload["validations"]:
        rows.append(
            f"| {item['question_id']} | {'passed' if item['passed'] else 'failed'} | "
            f"{item['terminated_reason']} | "
            f"{', '.join(item['citations']) or '-'} | "
            f"{', '.join(step['action'] for step in item['steps'])} |"
        )
    index_path = payload.get("index_display_path", payload["index_path"])
    sections_path = payload.get("sections_display_path", payload["sections_path"])
    report = f"""# Title 12 Agent Validation

- Schema: `{payload['schema']}`
- Status: `{payload['status']}`
- Questions: {len(payload['validations'])}
- Max steps: {payload['max_steps']}
- Max fetch sections: {payload['max_fetch_sections']}
- Index: `{index_path}`
- Sections: `{sections_path}`
- Device: `{payload['device']}`
- Holdout retrieval inspected: no
- LLM called: no

## Validations

| Question | Status | Termination | Citations | Steps |
|---|---|---|---|---|
{chr(10).join(rows)}

## Interpretation

This validates the Phase 5 minimal deterministic Agent Harness. The loop is:
`search_regulations` -> optional `fetch_section` -> `final_answer`, bounded by
`max_steps`. The generated answer is template-based and is intended to validate
control flow and evidence/citation plumbing, not final legal answer quality.

Remaining work: add a structured LLM decision step, improve answer generation,
add durable JSON trace output, and later fold in `ToolRegistry` / `ToolResult`
once the harness shape is stable.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate Title 12 minimal agent loop")
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
        "--output",
        type=Path,
        default=root / "reports" / "title12_agent_validation.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_agent_validation.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
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
    validations = [validate_agent_question(agent, item) for item in VALIDATION_QUESTIONS]
    status = "passed" if all(item["passed"] for item in validations) else "failed"
    payload = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "index_path": str(args.index.resolve()),
        "metadata_path": str(args.metadata.resolve()),
        "sections_path": str(args.sections.resolve()),
        "model_path": str(args.model.resolve()),
        "index_display_path": display_path(args.index, root),
        "sections_display_path": display_path(args.sections, root),
        "model_display_path": display_path(args.model, root),
        "device": args.device,
        "top_k": args.top_k,
        "max_steps": args.max_steps,
        "max_fetch_sections": args.max_fetch_sections,
        "holdout_retrieval_inspected": False,
        "llm_called": False,
        "validations": validations,
    }
    atomic_write_json(args.output, payload)
    write_report(args.report, payload)
    print(json.dumps({"status": status}, ensure_ascii=False, indent=2))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
