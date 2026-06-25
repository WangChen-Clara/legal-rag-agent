from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "title12-retrieval-comparison-v1"
DEFAULT_KS = (1, 3, 5, 10)
SAFE_ALIGNMENT_STATUSES = {"exact", "high_confidence"}
RETRIEVAL_ONLY_REASON_CODES = {"legacy_source_truncated"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_legacy_lineage(
    alignment_records: list[dict[str, Any]],
) -> dict[int, list[dict[str, str]]]:
    lineage: dict[int, list[dict[str, str]]] = defaultdict(list)
    for record in alignment_records:
        row_id = record.get("legacy_row_id")
        is_safe_status = record.get("status") in SAFE_ALIGNMENT_STATUSES
        is_direct_truncated_lineage = (
            record.get("reason_code") in RETRIEVAL_ONLY_REASON_CODES
            and record.get("legacy_document_id") is not None
        )
        if row_id is None or not (is_safe_status or is_direct_truncated_lineage):
            continue
        lineage[int(row_id)].append(
            {
                "document_id": (
                    f"ecfr:title-12:section-{record['official_section']}:"
                    "version-2025-09-01"
                ),
                "section": str(record["official_section"]),
            }
        )
    return dict(lineage)


def evaluate_rankings(
    eval_records: list[dict[str, Any]],
    rankings: dict[str, list[dict[str, Any]]],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, Any]:
    reciprocal_ranks: list[float] = []
    hit_counts = {k: 0 for k in ks}
    failures: list[str] = []
    unmapped = 0
    result_count = 0
    for record in eval_records:
        hits = rankings[record["question_id"]]
        first_rank = next((hit["rank"] for hit in hits if hit["is_acceptable"]), None)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        if first_rank is None:
            failures.append(record["question_id"])
        for k in ks:
            hit_counts[k] += int(first_rank is not None and first_rank <= k)
        unmapped += sum(not hit["section_ids"] for hit in hits)
        result_count += len(hits)
    count = len(eval_records)
    return {
        "questions": count,
        "hit_rate": {
            f"hit_at_{k}": round(hit_counts[k] / count, 6) for k in ks
        },
        "mrr_at_10": round(sum(reciprocal_ranks) / count, 6),
        "failures_at_10": failures,
        "unmapped_results_at_10": unmapped,
        "unmapped_result_rate_at_10": round(unmapped / result_count, 6),
    }


def benchmark_search(index: Any, vectors: Any, *, k: int, repeats: int = 20) -> tuple[Any, Any, dict[str, Any]]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    index.search(vectors, k)
    timings: list[float] = []
    scores = ids = None
    for _ in range(repeats):
        started = time.perf_counter()
        scores, ids = index.search(vectors, k)
        timings.append((time.perf_counter() - started) * 1000)
    ordered = sorted(timings)
    p95_index = min(math.ceil(len(ordered) * 0.95) - 1, len(ordered) - 1)
    return scores, ids, {
        "repeats": repeats,
        "batch_questions": len(vectors),
        "mean_ms": round(statistics.fmean(timings), 6),
        "median_ms": round(statistics.median(timings), 6),
        "p95_ms": round(ordered[p95_index], 6),
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


def write_report(path: Path, payload: dict[str, Any]) -> None:
    old = payload["indexes"]["old_fixed_chunks"]
    new = payload["indexes"]["new_structured_chunks"]
    metric_rows = []
    for metric in ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10"):
        metric_rows.append(
            f"| {metric} | {old['metrics']['hit_rate'][metric]:.3f} | "
            f"{new['metrics']['hit_rate'][metric]:.3f} |"
        )
    failure_rows = []
    for record in payload["per_question"]:
        if record["old_first_acceptable_rank"] is None or record["new_first_acceptable_rank"] is None:
            failure_rows.append(
                f"| {record['question_id']} | "
                f"{record['old_first_acceptable_rank'] or '-'} | "
                f"{record['new_first_acceptable_rank'] or '-'} | "
                f"{record['new_first_acceptable_rank_at_100'] or '-'} | "
                f"{record['question']} |"
            )
    report = f"""# Title 12 Old vs New Retrieval Comparison

## Setup

- Questions: {payload['questions']}
- Query model: `{payload['query_model']}`
- Query normalization: true
- Query prefix: none
- Top K: 10
- Acceptable-equivalent rule: any accepted section counts as correct
- Legacy truncated direct lineage: allowed for retrieval scoring only, not citation safety
- Old index: normalized vectors with `IndexFlatL2`, fixed 500-character chunks
- New index: normalized vectors with `IndexFlatIP`, structured section-aware chunks

Because both query and document vectors are normalized, L2 and inner-product rankings
are mathematically equivalent. The indexes differ in corpus scope and chunk construction:
the old index contains 99,238 broad historical vectors, while the new index contains the
27,750 canonical Title 12 chunks. Results should be interpreted as an end-to-end index
comparison, not as an embedding-model-only experiment.

## Metrics

| Metric | Old | New |
|---|---:|---:|
{chr(10).join(metric_rows)}
| MRR@10 | {old['metrics']['mrr_at_10']:.3f} | {new['metrics']['mrr_at_10']:.3f} |
| Search latency median, 20-query batch ms | {old['search_latency_ms']['median_ms']:.3f} | {new['search_latency_ms']['median_ms']:.3f} |
| Search latency p95, 20-query batch ms | {old['search_latency_ms']['p95_ms']:.3f} | {new['search_latency_ms']['p95_ms']:.3f} |
| Unmapped top-10 rate | {old['metrics']['unmapped_result_rate_at_10']:.3f} | {new['metrics']['unmapped_result_rate_at_10']:.3f} |

Shared query encoding took {payload['query_encoding_total_ms']:.3f} ms for
{payload['questions']} questions.

## Failures

| Question | Old rank@10 | New rank@10 | New rank@100 | Text |
|---|---:|---:|---:|---|
{chr(10).join(failure_rows) if failure_rows else '| - | - | - | - | No top-10 failures |'}

The new-index failures are rank displacement rather than missing-corpus failures:
the acceptable sections remain present below rank 10. Their queries target a short
definition or a short rule embedded within a longer structured chunk, while competing
definition-like chunks rank higher. This is an end-to-end chunking and ranking issue;
the current 20-question sample does not isolate the embedding model as the sole cause.

## Conclusion

On this 20-question baseline, the old fixed-chunk index has higher Hit@K and MRR.
The new canonical index searches fewer vectors substantially faster and provides complete
section/version/source metadata, but it does not yet improve retrieval quality. The next
optimization should focus on the two recorded failure cases before changing models.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Compare old and new Title 12 indexes")
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=root / "data" / "eval" / "title12_retrieval_eval.json",
    )
    parser.add_argument(
        "--alignment",
        type=Path,
        default=root / "data" / "alignment" / "full" / "alignment_results.jsonl",
    )
    parser.add_argument(
        "--old-index",
        type=Path,
        default=root / "data" / "legacy_indexes" / "bge-large-embeddings" / "vector_db.index",
    )
    parser.add_argument(
        "--old-metadata",
        type=Path,
        default=root / "data" / "legacy_indexes" / "bge-large-embeddings" / "metadata.npy",
    )
    parser.add_argument(
        "--new-index",
        type=Path,
        default=root / "data" / "indexes" / "title12_bge_large_2025-09-01" / "vector_db.index",
    )
    parser.add_argument(
        "--new-metadata",
        type=Path,
        default=root / "data" / "indexes" / "title12_bge_large_2025-09-01" / "metadata.npy",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "models" / "bge-large-en-v1.5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "title12_retrieval_comparison.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_retrieval_comparison.md",
    )
    return parser.parse_args()


def main() -> None:
    import faiss
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    args = parse_args()
    evaluation = json.loads(args.eval_set.read_text(encoding="utf-8"))
    eval_records = evaluation["records"]
    lineage = build_legacy_lineage(load_jsonl(args.alignment))
    old_index = faiss.read_index(str(args.old_index))
    new_index = faiss.read_index(str(args.new_index))
    old_metadata = np.load(args.old_metadata, allow_pickle=True).tolist()
    new_metadata = np.load(args.new_metadata, allow_pickle=True).tolist()
    if old_index.ntotal != len(old_metadata) or new_index.ntotal != len(new_metadata):
        raise ValueError("Index and metadata counts do not match")
    if old_index.d != new_index.d:
        raise ValueError("Old and new index dimensions do not match")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured comparison run")

    model = SentenceTransformer(str(args.model), device="cuda", local_files_only=True)
    questions = [record["question"] for record in eval_records]
    torch.cuda.synchronize()
    encode_started = time.perf_counter()
    query_vectors = model.encode(
        questions,
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32", copy=False)
    torch.cuda.synchronize()
    encoding_ms = (time.perf_counter() - encode_started) * 1000

    old_scores, old_ids, old_search_latency = benchmark_search(
        old_index, query_vectors, k=10
    )
    new_scores, new_ids, new_search_latency = benchmark_search(
        new_index, query_vectors, k=10
    )
    _, new_ids_at_100 = new_index.search(query_vectors, 100)

    rankings: dict[str, dict[str, list[dict[str, Any]]]] = {
        "old_fixed_chunks": {},
        "new_structured_chunks": {},
    }
    per_question: list[dict[str, Any]] = []
    for row, record in enumerate(eval_records):
        acceptable = set(record["acceptable_section_ids"])
        old_hits = []
        for rank, (score, item_id) in enumerate(zip(old_scores[row], old_ids[row]), start=1):
            item = old_metadata[int(item_id)]
            mapped = lineage.get(int(item["row_id"]), [])
            section_ids = [entry["document_id"] for entry in mapped]
            old_hits.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "item_id": int(item_id),
                    "row_id": int(item["row_id"]),
                    "chunk_id": int(item["chunk_id"]),
                    "sections": [entry["section"] for entry in mapped],
                    "section_ids": section_ids,
                    "is_acceptable": bool(acceptable.intersection(section_ids)),
                    "text_preview": item.get("chunk", "")[:240],
                }
            )
        new_hits = []
        for rank, (score, item_id) in enumerate(zip(new_scores[row], new_ids[row]), start=1):
            item = new_metadata[int(item_id)]
            section_ids = [item["parent_document_id"]]
            new_hits.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "item_id": int(item_id),
                    "chunk_id": item["chunk_id"],
                    "sections": [item["section"]],
                    "section_ids": section_ids,
                    "is_acceptable": bool(acceptable.intersection(section_ids)),
                    "text_preview": item.get("text", "")[:240],
                    "source_url": item.get("source_url"),
                }
            )
        rankings["old_fixed_chunks"][record["question_id"]] = old_hits
        rankings["new_structured_chunks"][record["question_id"]] = new_hits
        per_question.append(
            {
                "question_id": record["question_id"],
                "question": record["question"],
                "acceptable_sections": record["acceptable_sections"],
                "old_first_acceptable_rank": next(
                    (hit["rank"] for hit in old_hits if hit["is_acceptable"]), None
                ),
                "new_first_acceptable_rank": next(
                    (hit["rank"] for hit in new_hits if hit["is_acceptable"]), None
                ),
                "new_first_acceptable_rank_at_100": next(
                    (
                        rank
                        for rank, item_id in enumerate(new_ids_at_100[row], start=1)
                        if new_metadata[int(item_id)]["parent_document_id"] in acceptable
                    ),
                    None,
                ),
            }
        )

    payload = {
        "schema": RESULT_SCHEMA,
        "questions": len(eval_records),
        "eval_set": str(args.eval_set.resolve()),
        "query_model": str(args.model.resolve()),
        "query_normalized": True,
        "query_prefix": None,
        "query_encoding_total_ms": round(encoding_ms, 6),
        "indexes": {
            "old_fixed_chunks": {
                "index_path": str(args.old_index.resolve()),
                "index_type": type(old_index).__name__,
                "vectors": old_index.ntotal,
                "search_latency_ms": old_search_latency,
                "metrics": evaluate_rankings(eval_records, rankings["old_fixed_chunks"]),
            },
            "new_structured_chunks": {
                "index_path": str(args.new_index.resolve()),
                "index_type": type(new_index).__name__,
                "vectors": new_index.ntotal,
                "search_latency_ms": new_search_latency,
                "metrics": evaluate_rankings(eval_records, rankings["new_structured_chunks"]),
            },
        },
        "per_question": per_question,
        "rankings": rankings,
    }
    atomic_write_json(args.output, payload)
    write_report(args.report, payload)
    print(json.dumps(payload["indexes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
