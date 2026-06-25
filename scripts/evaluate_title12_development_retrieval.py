from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "title12-development-retrieval-eval-v1"
DEFAULT_KS = (1, 5, 10)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def group_hit_rank(hits: list[dict[str, Any]], group: list[str]) -> int | None:
    accepted = set(group)
    return next((hit["rank"] for hit in hits if hit["section"] in accepted), None)


def first_complete_rank(record: dict[str, Any], hits: list[dict[str, Any]], max_k: int) -> int | None:
    if record["question_type"] == "cross_section":
        ranks = [
            group_hit_rank(hits[:max_k], group)
            for group in record.get("required_evidence_groups", [])
        ]
        if not ranks or any(rank is None for rank in ranks):
            return None
        return max(rank for rank in ranks if rank is not None)
    accepted = set(record.get("acceptable_sections", []))
    return next((hit["rank"] for hit in hits[:max_k] if hit["section"] in accepted), None)


def recall_at_k(record: dict[str, Any], hits: list[dict[str, Any]], k: int) -> float:
    if record["question_type"] == "cross_section":
        groups = record.get("required_evidence_groups", [])
        if not groups:
            return 0.0
        found = sum(group_hit_rank(hits[:k], group) is not None for group in groups)
        return found / len(groups)
    return 1.0 if first_complete_rank(record, hits, k) is not None else 0.0


def evaluate_rankings(
    records: list[dict[str, Any]],
    rankings: dict[str, list[dict[str, Any]]],
    *,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, Any]:
    hit_counts = {k: 0 for k in ks}
    recall_totals = {k: 0.0 for k in ks}
    reciprocal_ranks = []
    failures = []
    max_k = max(ks)
    for record in records:
        hits = rankings[record["question_id"]]
        first_rank = first_complete_rank(record, hits, max_k)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        if first_rank is None:
            failures.append(record["question_id"])
        for k in ks:
            hit_counts[k] += int(first_complete_rank(record, hits, k) is not None)
            recall_totals[k] += recall_at_k(record, hits, k)
    count = len(records)
    return {
        "questions": count,
        "hit_rate": {f"hit_at_{k}": round(hit_counts[k] / count, 6) for k in ks},
        "recall": {f"recall_at_{k}": round(recall_totals[k] / count, 6) for k in ks},
        "mrr_at_10": round(sum(reciprocal_ranks) / count, 6),
        "failures_at_10": failures,
    }


def build_ranking(
    record: dict[str, Any],
    scores: Any,
    item_ids: Any,
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    hits = []
    for rank, (score, item_id) in enumerate(zip(scores, item_ids), start=1):
        item = metadata[int(item_id)]
        hits.append(
            {
                "rank": rank,
                "score": float(score),
                "item_id": int(item_id),
                "chunk_id": item["chunk_id"],
                "section": item["section"],
                "parent_document_id": item["parent_document_id"],
                "source_url": item.get("source_url"),
                "text_preview": item.get("text", "")[:260],
                "is_complete_hit_at_rank": first_complete_rank(record, hits + [
                    {
                        "rank": rank,
                        "score": float(score),
                        "item_id": int(item_id),
                        "chunk_id": item["chunk_id"],
                        "section": item["section"],
                        "parent_document_id": item["parent_document_id"],
                        "source_url": item.get("source_url"),
                        "text_preview": item.get("text", "")[:260],
                    }
                ], rank) is not None,
            }
        )
    return hits


def write_report(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    rows = []
    for question in payload["per_question"]:
        rows.append(
            f"| {question['question_id']} | {question['question_type']} | "
            f"{question['first_complete_rank'] or '-'} | "
            f"{question['recall_at_10']:.2f} | {question['question']} |"
        )
    failure_rows = []
    for question in payload["per_question"]:
        if question["first_complete_rank"] is not None:
            continue
        expected = question.get("acceptable_sections") or question.get("required_evidence_groups")
        top_sections = ", ".join(hit["section"] for hit in question["top_hits"][:5])
        failure_rows.append(
            f"| {question['question_id']} | {expected} | {top_sections} | {question['question']} |"
        )
    report = f"""# Title 12 Development Retrieval Evaluation

## Setup

- Schema: `{payload['schema']}`
- Split: development
- Questions: {payload['questions']}
- Index: `{payload['index_path']}`
- Model: `{payload['model_path']}`
- Device: `{payload['device']}`
- Query normalized: true
- Holdout retrieval inspected: no

## Metrics

| Metric | Value |
|---|---:|
| Hit@1 | {metrics['hit_rate']['hit_at_1']:.3f} |
| Hit@5 | {metrics['hit_rate']['hit_at_5']:.3f} |
| Hit@10 | {metrics['hit_rate']['hit_at_10']:.3f} |
| Recall@1 | {metrics['recall']['recall_at_1']:.3f} |
| Recall@5 | {metrics['recall']['recall_at_5']:.3f} |
| Recall@10 | {metrics['recall']['recall_at_10']:.3f} |
| MRR@10 | {metrics['mrr_at_10']:.3f} |
| Query encoding ms | {payload['query_encoding_ms']:.3f} |
| Search ms | {payload['search_ms']:.3f} |

## Per Question

| Question | Type | First complete rank | Recall@10 | Text |
|---|---|---:|---:|---|
{chr(10).join(rows)}

## Top-10 Failures

| Question | Expected evidence | Top-5 sections | Text |
|---|---|---|---|
{chr(10).join(failure_rows) if failure_rows else '| - | - | - | No top-10 failures |'}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate Title 12 development retrieval")
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
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "title12_development_retrieval_eval.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_development_retrieval_eval.md",
    )
    return parser.parse_args()


def main() -> None:
    import faiss
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    args = parse_args()
    qa = json.loads(args.qa.read_text(encoding="utf-8"))
    records = [record for record in qa["records"] if record["split"] == "development"]
    if len(records) != 20:
        raise ValueError(f"Expected 20 development records, found {len(records)}")
    if args.top_k < max(DEFAULT_KS):
        raise ValueError(f"top-k must be at least {max(DEFAULT_KS)}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(args.model), device=device, local_files_only=True)
    questions = [record["question"] for record in records]
    if device == "cuda":
        torch.cuda.synchronize()
    encode_started = time.perf_counter()
    vectors = model.encode(
        questions,
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32", copy=False)
    if device == "cuda":
        torch.cuda.synchronize()
    encoding_ms = (time.perf_counter() - encode_started) * 1000

    index = faiss.read_index(str(args.index))
    metadata = np.load(args.metadata, allow_pickle=True).tolist()
    if index.ntotal != len(metadata):
        raise ValueError("Index and metadata counts do not match")
    if index.d != vectors.shape[1]:
        raise ValueError(f"Index dimension {index.d} != query dimension {vectors.shape[1]}")
    search_started = time.perf_counter()
    scores, item_ids = index.search(vectors, args.top_k)
    search_ms = (time.perf_counter() - search_started) * 1000

    rankings = {
        record["question_id"]: build_ranking(record, scores[row], item_ids[row], metadata)
        for row, record in enumerate(records)
    }
    metrics = evaluate_rankings(records, rankings)
    per_question = []
    for record in records:
        hits = rankings[record["question_id"]]
        first_rank = first_complete_rank(record, hits, max(DEFAULT_KS))
        per_question.append(
            {
                "question_id": record["question_id"],
                "candidate_id": record["candidate_id"],
                "question_type": record["question_type"],
                "question": record["question"],
                "acceptable_sections": record.get("acceptable_sections"),
                "required_evidence_groups": record.get("required_evidence_groups"),
                "first_complete_rank": first_rank,
                "recall_at_10": recall_at_k(record, hits, 10),
                "top_hits": hits,
            }
        )

    payload = {
        "schema": RESULT_SCHEMA,
        "qa_path": str(args.qa.resolve()),
        "index_path": str(args.index.resolve()),
        "metadata_path": str(args.metadata.resolve()),
        "model_path": str(args.model.resolve()),
        "device": device,
        "questions": len(records),
        "top_k": args.top_k,
        "query_normalized": True,
        "holdout_retrieval_inspected": False,
        "query_encoding_ms": round(encoding_ms, 6),
        "search_ms": round(search_ms, 6),
        "metrics": metrics,
        "per_question": per_question,
    }
    atomic_write_json(args.output, payload)
    write_report(args.report, payload)
    print(json.dumps({"questions": len(records), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
