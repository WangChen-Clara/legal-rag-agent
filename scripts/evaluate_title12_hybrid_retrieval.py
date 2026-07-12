from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Protocol

from rag_law.config import EmbeddingConfig, RetrievalConfig
from rag_law.hybrid_retriever import HybridRetriever
from rag_law.lexical_retriever import LexicalRetriever
from rag_law.models import SearchHit
from rag_law.retriever import FaissRetriever

try:
    from scripts.evaluate_title12_development_retrieval import (
        DEFAULT_KS,
        evaluate_rankings,
        first_complete_rank,
        recall_at_k,
    )
except ModuleNotFoundError:
    from evaluate_title12_development_retrieval import (
        DEFAULT_KS,
        evaluate_rankings,
        first_complete_rank,
        recall_at_k,
    )


RESULT_SCHEMA = "title12-hybrid-retrieval-eval-v1"


class ContextRetriever(Protocol):
    items: list[Any]

    def search_with_context(self, query: str, **kwargs: Any) -> list[SearchHit]:
        ...


class Searcher(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
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


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"]
    development = [record for record in records if record.get("split") == "development"]
    return development or records


def hit_to_row(hit: SearchHit) -> dict[str, Any]:
    return {
        "rank": hit.rank,
        "score": hit.distance,
        "chunk_id": hit.metadata.get("chunk_id"),
        "section": hit.metadata.get("section"),
        "parent_document_id": hit.metadata.get("parent_document_id"),
        "source_url": hit.metadata.get("source_url"),
        "retrieval_source": hit.metadata.get("retrieval_source", "semantic"),
        "retrieval_sources": hit.metadata.get("retrieval_sources"),
        "rrf_score": hit.metadata.get("rrf_score"),
        "text_preview": hit.text[:260],
    }


def run_context_variant(
    retriever: ContextRetriever,
    records: list[dict[str, Any]],
    *,
    top_k: int,
    semantic_top_k: int,
    max_expanded_sections: int,
    max_chunks_per_section: int,
) -> dict[str, Any]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    started = time.perf_counter()
    for record in records:
        hits = retriever.search_with_context(
            record["question"],
            top_k=top_k,
            include_explicit_citations=True,
            semantic_top_k=semantic_top_k,
            expand_cross_references=True,
            expand_from_semantic_without_explicit=False,
            max_expanded_sections=max_expanded_sections,
            max_chunks_per_section=max_chunks_per_section,
        )
        rankings[record["question_id"]] = [hit_to_row(hit) for hit in hits]
    return _variant_payload(records, rankings, started)


def run_hybrid_variant(
    retriever: ContextRetriever,
    lexical_retriever: Searcher,
    records: list[dict[str, Any]],
    *,
    top_k: int,
    semantic_top_k: int,
    lexical_top_k: int,
    max_expanded_sections: int,
    max_chunks_per_section: int,
    rrf_k: int,
) -> dict[str, Any]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    started = time.perf_counter()
    for record in records:
        context_hits = retriever.search_with_context(
            record["question"],
            top_k=semantic_top_k,
            include_explicit_citations=True,
            semantic_top_k=semantic_top_k,
            expand_cross_references=True,
            expand_from_semantic_without_explicit=False,
            max_expanded_sections=max_expanded_sections,
            max_chunks_per_section=max_chunks_per_section,
        )
        lexical_hits = lexical_retriever.search(record["question"], top_k=lexical_top_k)
        fused = HybridRetriever.fuse(
            [context_hits, lexical_hits],
            top_k=top_k,
            rrf_k=rrf_k,
        )
        rankings[record["question_id"]] = [hit_to_row(hit) for hit in fused]
    return _variant_payload(records, rankings, started)


def _variant_payload(
    records: list[dict[str, Any]],
    rankings: dict[str, list[dict[str, Any]]],
    started: float,
) -> dict[str, Any]:
    elapsed_ms = (time.perf_counter() - started) * 1000
    metrics = evaluate_rankings(records, rankings)
    per_question = []
    for record in records:
        hits = rankings[record["question_id"]]
        per_question.append(
            {
                "question_id": record["question_id"],
                "candidate_id": record.get("candidate_id"),
                "question_type": record.get("question_type", "single_section"),
                "question": record["question"],
                "acceptable_sections": record.get("acceptable_sections"),
                "required_evidence_groups": record.get("required_evidence_groups"),
                "first_complete_rank": first_complete_rank(record, hits, max(DEFAULT_KS)),
                "recall_at_10": recall_at_k(record, hits, 10),
                "top_hits": hits,
            }
        )
    return {
        "elapsed_ms": round(elapsed_ms, 6),
        "metrics": metrics,
        "per_question": per_question,
    }


def metric_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float]:
    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    return {
        "hit_at_1": round(
            candidate_metrics["hit_rate"]["hit_at_1"]
            - baseline_metrics["hit_rate"]["hit_at_1"],
            6,
        ),
        "hit_at_5": round(
            candidate_metrics["hit_rate"]["hit_at_5"]
            - baseline_metrics["hit_rate"]["hit_at_5"],
            6,
        ),
        "hit_at_10": round(
            candidate_metrics["hit_rate"]["hit_at_10"]
            - baseline_metrics["hit_rate"]["hit_at_10"],
            6,
        ),
        "recall_at_10": round(
            candidate_metrics["recall"]["recall_at_10"]
            - baseline_metrics["recall"]["recall_at_10"],
            6,
        ),
        "mrr_at_10": round(
            candidate_metrics["mrr_at_10"] - baseline_metrics["mrr_at_10"],
            6,
        ),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    rows = []
    for name, variant in payload["variants"].items():
        metrics = variant["metrics"]
        rows.append(
            f"| {name} | {metrics['hit_rate']['hit_at_1']:.3f} | "
            f"{metrics['hit_rate']['hit_at_5']:.3f} | "
            f"{metrics['hit_rate']['hit_at_10']:.3f} | "
            f"{metrics['recall']['recall_at_10']:.3f} | "
            f"{metrics['mrr_at_10']:.3f} | {variant['elapsed_ms']:.1f} |"
        )

    delta = payload["hybrid_delta"]
    failure_rows = []
    for name, variant in payload["variants"].items():
        for question_id in variant["metrics"]["failures_at_10"]:
            failure_rows.append(f"| {name} | {question_id} |")

    report = f"""# Title 12 Hybrid Retrieval Evaluation

## Setup

- Schema: `{payload['schema']}`
- Questions: {payload['questions']}
- Index: `{payload['index_path']}`
- Metadata: `{payload['metadata_path']}`
- Model: `{payload['model_path']}`
- Device: `{payload['device']}`
- Top K: {payload['top_k']}
- Semantic candidate K: {payload['semantic_top_k']}
- Lexical candidate K: {payload['lexical_top_k']}
- RRF K: {payload['rrf_k']}
- Holdout retrieval inspected: no

## Metrics

| Variant | Hit@1 | Hit@5 | Hit@10 | Recall@10 | MRR@10 | Elapsed ms |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Hybrid Delta

Delta is `hybrid_rrf - citation_aware_context`.

| Metric | Delta |
|---|---:|
| Hit@1 | {delta['hit_at_1']:.3f} |
| Hit@5 | {delta['hit_at_5']:.3f} |
| Hit@10 | {delta['hit_at_10']:.3f} |
| Recall@10 | {delta['recall_at_10']:.3f} |
| MRR@10 | {delta['mrr_at_10']:.3f} |

## Failures At 10

| Variant | Question |
|---|---|
{chr(10).join(failure_rows) if failure_rows else '| - | No top-10 failures |'}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate Title 12 hybrid retrieval")
    parser.add_argument(
        "--qa",
        type=Path,
        default=root / "data" / "eval" / "title12_development_qa_draft.json",
    )
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
        "--model",
        type=Path,
        default=root / "models" / "bge-large-en-v1.5",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--semantic-top-k", type=int, default=20)
    parser.add_argument("--lexical-top-k", type=int, default=20)
    parser.add_argument("--max-expanded-sections", type=int, default=3)
    parser.add_argument("--max-chunks-per-section", type=int, default=1)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "title12_hybrid_retrieval_eval.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_hybrid_retrieval_eval.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < max(DEFAULT_KS):
        raise ValueError(f"top-k must be at least {max(DEFAULT_KS)}")

    records = load_records(args.qa)
    retrieval = RetrievalConfig(
        index_path=args.index,
        metadata_path=args.metadata,
        top_k=args.top_k,
        normalize_query=True,
    )
    embedding = EmbeddingConfig(model_path=args.model, device=args.device)
    retriever = FaissRetriever(retrieval, embedding)
    lexical_retriever = LexicalRetriever(retriever.items)

    baseline = run_context_variant(
        retriever,
        records,
        top_k=args.top_k,
        semantic_top_k=args.semantic_top_k,
        max_expanded_sections=args.max_expanded_sections,
        max_chunks_per_section=args.max_chunks_per_section,
    )
    hybrid = run_hybrid_variant(
        retriever,
        lexical_retriever,
        records,
        top_k=args.top_k,
        semantic_top_k=args.semantic_top_k,
        lexical_top_k=args.lexical_top_k,
        max_expanded_sections=args.max_expanded_sections,
        max_chunks_per_section=args.max_chunks_per_section,
        rrf_k=args.rrf_k,
    )

    payload = {
        "schema": RESULT_SCHEMA,
        "qa_path": str(args.qa.resolve()),
        "index_path": str(args.index.resolve()),
        "metadata_path": str(args.metadata.resolve()),
        "model_path": str(args.model.resolve()),
        "device": args.device,
        "questions": len(records),
        "top_k": args.top_k,
        "semantic_top_k": args.semantic_top_k,
        "lexical_top_k": args.lexical_top_k,
        "max_expanded_sections": args.max_expanded_sections,
        "max_chunks_per_section": args.max_chunks_per_section,
        "rrf_k": args.rrf_k,
        "holdout_retrieval_inspected": False,
        "variants": {
            "citation_aware_context": baseline,
            "hybrid_rrf": hybrid,
        },
        "hybrid_delta": metric_delta(baseline, hybrid),
    }
    atomic_write_json(args.output, payload)
    write_report(args.report, payload)
    print(
        json.dumps(
            {
                "citation_aware_context": baseline["metrics"],
                "hybrid_rrf": hybrid["metrics"],
                "hybrid_delta": payload["hybrid_delta"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
