from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from rag_law.agent import AgentState, LegalRAGAgent
from rag_law.config import EmbeddingConfig, LLMConfig, RetrievalConfig
from rag_law.llm_client import LLMClient
from rag_law.llm_judge import JudgeScore, evaluate_answer_with_judge
from rag_law.retriever import FaissRetriever
from rag_law.tools import RegulationToolset

try:
    from scripts.validate_title12_agent import VALIDATION_QUESTIONS, display_path
except ModuleNotFoundError:
    from validate_title12_agent import VALIDATION_QUESTIONS, display_path


RESULT_SCHEMA = "title12-llm-judge-eval-v1"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def build_llm_client(args: argparse.Namespace, *, prefix: str) -> LLMClient:
    return LLMClient(
        LLMConfig(
            base_url=getattr(args, f"{prefix}_base_url"),
            model_name=getattr(args, f"{prefix}_model"),
            api_key_env=getattr(args, f"{prefix}_api_key_env"),
            temperature=getattr(args, f"{prefix}_temperature"),
            top_p=getattr(args, f"{prefix}_top_p"),
            timeout_seconds=getattr(args, f"{prefix}_timeout"),
        ),
        api_key=getattr(args, f"{prefix}_api_key"),
    )


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
    answer_llm = None if args.no_answer_llm else build_llm_client(args, prefix="answer")
    return LegalRAGAgent(
        toolset,
        max_steps=args.max_steps,
        top_k=args.top_k,
        max_fetch_sections=args.max_fetch_sections,
        llm_client=answer_llm,
    )


def evaluate_question(
    agent: LegalRAGAgent,
    judge: LLMClient,
    item: dict[str, Any],
) -> dict[str, Any]:
    state = agent.run(item["question"])
    score = evaluate_answer_with_judge(state, judge)
    return {
        "question_id": item["question_id"],
        "question": item["question"],
        "expected_sections": item["expected_sections"],
        "terminated_reason": state.terminated_reason,
        "answer": state.final_answer.answer if state.final_answer else "",
        "citations": state.final_answer.citations if state.final_answer else [],
        "judge_score": score.to_dict(),
        "trace": state.to_trace_dict(),
    }


def summarize_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("results must not be empty")
    keys = [
        "answer_relevance",
        "faithfulness",
        "citation_support",
        "legal_caution",
        "overall",
    ]
    count = len(results)
    return {
        "questions": count,
        "pass_rate": round(
            sum(bool(item["judge_score"]["pass"]) for item in results) / count,
            6,
        ),
        **{
            f"average_{key}": round(
                sum(int(item["judge_score"][key]) for item in results) / count,
                6,
            )
            for key in keys
        },
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    rows = []
    for item in payload["results"]:
        score = item["judge_score"]
        rows.append(
            f"| {item['question_id']} | {score['pass']} | "
            f"{score['answer_relevance']} | {score['faithfulness']} | "
            f"{score['citation_support']} | {score['legal_caution']} | "
            f"{score['overall']} | {', '.join(score['issues']) or '-'} |"
        )
    report = f"""# Title 12 LLM-as-Judge Evaluation

## Setup

- Schema: `{payload['schema']}`
- Questions: {payload['questions']}
- Answer model: `{payload['answer_model']}`
- Judge model: `{payload['judge_model']}`
- Same model for answer and judge: {str(payload['same_answer_and_judge_model']).lower()}
- Index: `{payload['index_display_path']}`
- Sections: `{payload['sections_display_path']}`
- Embedding model: `{payload['model_display_path']}`
- Device: `{payload['device']}`

## Metrics

| Metric | Value |
|---|---:|
| Pass rate | {metrics['pass_rate']:.3f} |
| Average answer relevance | {metrics['average_answer_relevance']:.3f} |
| Average faithfulness | {metrics['average_faithfulness']:.3f} |
| Average citation support | {metrics['average_citation_support']:.3f} |
| Average legal caution | {metrics['average_legal_caution']:.3f} |
| Average overall | {metrics['average_overall']:.3f} |

## Interpretation

If the same local model is used for both answer generation and judging, this report
only proves that the LLM-as-Judge evaluation loop is runnable. More serious answer
quality evaluation should use an independent, stronger judge model.

## Per Question

| Question | Pass | Relevance | Faithfulness | Citation support | Legal caution | Overall | Issues |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate Title 12 answers with LLM-as-Judge")
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
    parser.add_argument("--no-answer-llm", action="store_true")
    parser.add_argument("--answer-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--answer-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--answer-api-key", default="ollama")
    parser.add_argument("--answer-api-key-env", default="LLM_API_KEY")
    parser.add_argument("--answer-temperature", type=float, default=0.1)
    parser.add_argument("--answer-top-p", type=float, default=0.9)
    parser.add_argument("--answer-timeout", type=int, default=120)
    parser.add_argument("--judge-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--judge-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--judge-api-key", default="ollama")
    parser.add_argument("--judge-api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-top-p", type=float, default=0.9)
    parser.add_argument("--judge-timeout", type=int, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "title12_llm_judge_eval.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_llm_judge_eval.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    agent = build_agent(args)
    judge = build_llm_client(args, prefix="judge")
    results = [evaluate_question(agent, judge, item) for item in VALIDATION_QUESTIONS]
    metrics = summarize_scores(results)
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
        "answer_model": "deterministic" if args.no_answer_llm else args.answer_model,
        "judge_model": args.judge_model,
        "same_answer_and_judge_model": (
            not args.no_answer_llm and args.answer_model == args.judge_model
        ),
        "metrics": metrics,
        "results": results,
    }
    atomic_write_json(args.output, payload)
    write_report(args.report, payload)
    print(json.dumps({"metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
