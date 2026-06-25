from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from rag_law.config import EmbeddingConfig, RetrievalConfig
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


RESULT_SCHEMA = "title12-context-retrieval-eval-v1"
VARIANTS = {
    "baseline": {
        "include_explicit_citations": False,
        "expand_cross_references": False,
        "expand_from_semantic_without_explicit": False,
        "use_context": False,
    },
    "explicit_only": {
        "include_explicit_citations": True,
        "expand_cross_references": False,
        "expand_from_semantic_without_explicit": False,
        "use_context": True,
    },
    "semantic_cross_reference": {
        "include_explicit_citations": False,
        "expand_cross_references": True,
        "expand_from_semantic_without_explicit": True,
        "use_context": True,
    },
    "full_context": {
        "include_explicit_citations": True,
        "expand_cross_references": True,
        "expand_from_semantic_without_explicit": False,
        "use_context": True,
    },
}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def hit_to_row(hit: SearchHit) -> dict[str, Any]:
    return {
        "rank": hit.rank,
        "score": hit.distance,
        "chunk_id": hit.metadata.get("chunk_id"),
        "section": hit.metadata.get("section"),
        "parent_document_id": hit.metadata.get("parent_document_id"),
        "source_url": hit.metadata.get("source_url"),
        "retrieval_source": hit.metadata.get("retrieval_source", "semantic"),
        "text_preview": hit.text[:260],
    }


def run_variant(
    retriever: FaissRetriever,
    records: list[dict[str, Any]],
    *,
    variant_config: dict[str, Any],
    top_k: int,
    semantic_top_k: int,
    max_expanded_sections: int,
    max_chunks_per_section: int,
) -> dict[str, Any]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    started = time.perf_counter()
    for record in records:
        if variant_config["use_context"]:
            hits = retriever.search_with_context(
                record["question"],
                top_k=top_k,
                include_explicit_citations=variant_config["include_explicit_citations"],
                semantic_top_k=semantic_top_k,
                expand_cross_references=variant_config["expand_cross_references"],
                expand_from_semantic_without_explicit=variant_config[
                    "expand_from_semantic_without_explicit"
                ],
                max_expanded_sections=max_expanded_sections,
                max_chunks_per_section=max_chunks_per_section,
            )
        else:
            hits = retriever.search(record["question"], top_k=top_k)
            hits = [
                SearchHit(
                    rank=hit.rank,
                    distance=hit.distance,
                    text=hit.text,
                    metadata={**hit.metadata, "retrieval_source": "semantic"},
                )
                for hit in hits
            ]
        rankings[record["question_id"]] = [hit_to_row(hit) for hit in hits]
    elapsed_ms = (time.perf_counter() - started) * 1000
    metrics = evaluate_rankings(records, rankings)
    per_question = []
    for record in records:
        hits = rankings[record["question_id"]]
        per_question.append(
            {
                "question_id": record["question_id"],
                "candidate_id": record["candidate_id"],
                "question_type": record["question_type"],
                "question": record["question"],
                "acceptable_sections": record.get("acceptable_sections"),
                "required_evidence_groups": record.get("required_evidence_groups"),
                "first_complete_rank": first_complete_rank(record, hits, max(DEFAULT_KS)),
                "recall_at_10": recall_at_k(record, hits, 10),
                "top_hits": hits,
            }
        )
    return {
        "config": variant_config,
        "elapsed_ms": round(elapsed_ms, 6),
        "metrics": metrics,
        "per_question": per_question,
    }


def focus_summary(variant: dict[str, Any]) -> dict[str, Any]:
    focus = {}
    for question_id in ("title12-dev-q001", "title12-dev-q018"):
        question = next(
            row for row in variant["per_question"] if row["question_id"] == question_id
        )
        focus[question_id] = {
            "first_complete_rank": question["first_complete_rank"],
            "recall_at_10": question["recall_at_10"],
            "top_10_sections": [hit["section"] for hit in question["top_hits"][:10]],
            "retrieval_sources": [
                f"{hit['section']}:{hit['retrieval_source']}"
                for hit in question["top_hits"][:10]
            ],
        }
    return focus


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def write_report(path: Path, payload: dict[str, Any]) -> None:
    metric_rows = []
    focus_rows = []
    failure_rows = []
    for name, variant in payload["variants"].items():
        metrics = variant["metrics"]
        metric_rows.append(
            f"| {name} | {metrics['hit_rate']['hit_at_1']:.3f} | "
            f"{metrics['hit_rate']['hit_at_5']:.3f} | "
            f"{metrics['hit_rate']['hit_at_10']:.3f} | "
            f"{metrics['recall']['recall_at_10']:.3f} | "
            f"{metrics['mrr_at_10']:.3f} | {variant['elapsed_ms']:.1f} |"
        )
        for question_id, focus in variant["focus_questions"].items():
            focus_rows.append(
                f"| {name} | {question_id} | {focus['first_complete_rank'] or '-'} | "
                f"{focus['recall_at_10']:.2f} | "
                f"{', '.join(focus['retrieval_sources'])} |"
            )
        for question_id in metrics["failures_at_10"]:
            failure_rows.append(f"| {name} | {question_id} |")

    report = f"""# Title 12 Context Retrieval Evaluation

## Setup

- Schema: `{payload['schema']}`
- Split: development
- Questions: {payload['questions']}
- Index: `{payload['index_display_path']}`
- Model: `{payload['model_display_path']}`
- Device: `{payload['device']}`
- Holdout retrieval inspected: no

## Metrics

| Variant | Hit@1 | Hit@5 | Hit@10 | Recall@10 | MRR@10 | Elapsed ms |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

## Focus Questions

| Variant | Question | First complete rank | Recall@10 | Top sources |
|---|---|---:|---:|---|
{chr(10).join(focus_rows)}

## Failures At 10

| Variant | Question |
|---|---|
{chr(10).join(failure_rows) if failure_rows else '| - | No top-10 failures |'}

## Interpretation

`full_context` is a candidate strategy named `citation_aware_context_retrieval`,
not a final default. It should be enabled when the user query contains an explicit
CFR section reference. On this Development split it combines explicit citation
priority with one-hop cross-reference expansion from the explicit section.

Compared with baseline, `full_context` removes the q001 and q018 Top-10 failures
and improves Hit@10, Recall@10, and MRR@10. `semantic_cross_reference` performs
worse than baseline, so cross-reference expansion from ordinary semantic hits
should not be enabled by default.

Remaining risks before Holdout: an existing but incorrect user citation can still
pollute evidence, cross-reference expansion is only one hop and reference-order
based, and long sections may require more than the first chunk or a parent-section
fetch path.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate Title 12 context retrieval variants")
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
    parser.add_argument("--semantic-top-k", type=int, default=10)
    parser.add_argument("--max-expanded-sections", type=int, default=3)
    parser.add_argument("--max-chunks-per-section", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "title12_context_retrieval_eval.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_context_retrieval_eval.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.top_k < max(DEFAULT_KS):
        raise ValueError(f"top-k must be at least {max(DEFAULT_KS)}")
    qa = json.loads(args.qa.read_text(encoding="utf-8"))
    records = [record for record in qa["records"] if record["split"] == "development"]
    if len(records) != 20:
        raise ValueError(f"Expected 20 development records, found {len(records)}")

    retrieval = RetrievalConfig(
        index_path=args.index,
        metadata_path=args.metadata,
        top_k=args.top_k,
        normalize_query=True,
    )
    embedding = EmbeddingConfig(model_path=args.model, device=args.device)
    retriever = FaissRetriever(retrieval, embedding)

    variants = {}
    for name, config in VARIANTS.items():
        variant = run_variant(
            retriever,
            records,
            variant_config=config,
            top_k=args.top_k,
            semantic_top_k=args.semantic_top_k,
            max_expanded_sections=args.max_expanded_sections,
            max_chunks_per_section=args.max_chunks_per_section,
        )
        variant["focus_questions"] = focus_summary(variant)
        variants[name] = variant

    payload = {
        "schema": RESULT_SCHEMA,
        "qa_path": str(args.qa.resolve()),
        "index_path": str(args.index.resolve()),
        "metadata_path": str(args.metadata.resolve()),
        "model_path": str(args.model.resolve()),
        "index_display_path": display_path(args.index, root),
        "model_display_path": display_path(args.model, root),
        "device": args.device,
        "questions": len(records),
        "top_k": args.top_k,
        "semantic_top_k": args.semantic_top_k,
        "max_expanded_sections": args.max_expanded_sections,
        "max_chunks_per_section": args.max_chunks_per_section,
        "holdout_retrieval_inspected": False,
        "variants": variants,
    }
    atomic_write_json(args.output, payload)
    write_report(args.report, payload)
    print(
        json.dumps(
            {name: result["metrics"] for name, result in variants.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
