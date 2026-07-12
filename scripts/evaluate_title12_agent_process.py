from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from rag_law.agent import AgentState, LegalRAGAgent
from rag_law.config import EmbeddingConfig, RetrievalConfig
from rag_law.retriever import FaissRetriever
from rag_law.tools import RegulationToolset

try:
    from scripts.validate_title12_agent import VALIDATION_QUESTIONS, display_path
except ModuleNotFoundError:
    from validate_title12_agent import VALIDATION_QUESTIONS, display_path


RESULT_SCHEMA = "title12-agent-process-eval-v1"


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


def evaluate_agent_question(agent: AgentRunner, item: dict[str, Any]) -> dict[str, Any]:
    state = agent.run(item["question"])
    expected_sections = item["expected_sections"]
    evidence_sections = [evidence.section for evidence in state.evidence if evidence.section]
    fetched_sections = [section.section for section in state.fetched_sections]
    verified_sections = [
        verification.section
        for verification in state.citation_verifications
        if verification.verified
    ]
    final_citations = state.final_answer.citations if state.final_answer else []
    step_actions = [step.action for step in state.steps]
    failed_steps = [step.to_dict() for step in state.steps if step.status == "failed"]
    expected_found = all(
        section in evidence_sections or section in fetched_sections
        for section in expected_sections
    )
    fetched_expected = all(section in fetched_sections for section in expected_sections)
    verified_expected = all(section in verified_sections for section in fetched_sections)
    final_supported = all(
        any(f"12 CFR {section}" in citation for section in verified_sections)
        for citation in final_citations
    )
    final_supported = bool(final_citations) and final_supported
    return {
        "question_id": item["question_id"],
        "question": item["question"],
        "expected_sections": expected_sections,
        "terminated_reason": state.terminated_reason,
        "steps": [step.to_dict() for step in state.steps],
        "step_actions": step_actions,
        "failed_steps": failed_steps,
        "tool_success": not failed_steps,
        "expected_section_found": expected_found,
        "fetch_section_success": fetched_expected,
        "citation_verified": bool(state.citation_verifications) and verified_expected,
        "final_answer_citation_supported": final_supported,
        "average_step_count_item": len(state.steps),
        "evidence_sections": evidence_sections[:10],
        "fetched_sections": fetched_sections,
        "verified_sections": verified_sections,
        "final_citations": final_citations,
        "trace": state.to_trace_dict(),
    }


def summarize_process(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    if count == 0:
        raise ValueError("results must not be empty")
    termination_counts = Counter(item["terminated_reason"] for item in results)
    return {
        "questions": count,
        "tool_success_rate": _rate(results, "tool_success"),
        "expected_section_found_rate": _rate(results, "expected_section_found"),
        "fetch_section_success_rate": _rate(results, "fetch_section_success"),
        "citation_verified_rate": _rate(results, "citation_verified"),
        "final_answer_citation_support_rate": _rate(
            results,
            "final_answer_citation_supported",
        ),
        "average_steps": round(
            sum(len(item["steps"]) for item in results) / count,
            6,
        ),
        "termination_reason_distribution": dict(sorted(termination_counts.items())),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    rows = []
    for item in payload["results"]:
        rows.append(
            f"| {item['question_id']} | {item['terminated_reason']} | "
            f"{item['tool_success']} | {item['expected_section_found']} | "
            f"{item['citation_verified']} | "
            f"{item['final_answer_citation_supported']} | "
            f"{', '.join(item['step_actions'])} |"
        )
    report = f"""# Title 12 Agent Process Evaluation

## Setup

- Schema: `{payload['schema']}`
- Questions: {payload['questions']}
- Max steps: {payload['max_steps']}
- Max fetch sections: {payload['max_fetch_sections']}
- Index: `{payload['index_display_path']}`
- Sections: `{payload['sections_display_path']}`
- Model: `{payload['model_display_path']}`
- Device: `{payload['device']}`
- Holdout retrieval inspected: no
- LLM called: no

## Metrics

| Metric | Value |
|---|---:|
| Tool success rate | {metrics['tool_success_rate']:.3f} |
| Expected section found rate | {metrics['expected_section_found_rate']:.3f} |
| Fetch section success rate | {metrics['fetch_section_success_rate']:.3f} |
| Citation verified rate | {metrics['citation_verified_rate']:.3f} |
| Final answer citation support rate | {metrics['final_answer_citation_support_rate']:.3f} |
| Average steps | {metrics['average_steps']:.3f} |

## Termination Reasons

```json
{json.dumps(metrics['termination_reason_distribution'], ensure_ascii=False, indent=2)}
```

## Per Question

| Question | Termination | Tool success | Expected section found | Citation verified | Final citation supported | Steps |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
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
    return LegalRAGAgent(
        toolset,
        max_steps=args.max_steps,
        top_k=args.top_k,
        max_fetch_sections=args.max_fetch_sections,
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate Title 12 Agent process reliability")
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
        "--output",
        type=Path,
        default=root / "reports" / "title12_agent_process_eval.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_agent_process_eval.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    agent = build_agent(args)
    results = [evaluate_agent_question(agent, item) for item in VALIDATION_QUESTIONS]
    metrics = summarize_process(results)
    payload = {
        "schema": RESULT_SCHEMA,
        "questions": len(results),
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
        "metrics": metrics,
        "results": results,
    }
    atomic_write_json(args.output, payload)
    write_report(args.report, payload)
    print(json.dumps({"metrics": metrics}, ensure_ascii=False, indent=2))


def _rate(results: list[dict[str, Any]], key: str) -> float:
    return round(sum(bool(item[key]) for item in results) / len(results), 6)


if __name__ == "__main__":
    main()
