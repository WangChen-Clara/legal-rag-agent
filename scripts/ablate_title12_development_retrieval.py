from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

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


RESULT_SCHEMA = "title12-development-retrieval-ablation-v1"
VARIANTS = {
    "baseline": {"query_with_heading": False, "cross_reference_expansion": False},
    "query_with_heading": {"query_with_heading": True, "cross_reference_expansion": False},
    "cross_reference_expansion": {"query_with_heading": False, "cross_reference_expansion": True},
    "combined": {"query_with_heading": True, "cross_reference_expansion": True},
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


def query_text(record: dict[str, Any], *, with_heading: bool) -> str:
    if not with_heading:
        return record["question"]
    citations = record.get("source_citations", [])
    headings = []
    for citation in citations:
        heading = citation.get("heading")
        if heading and heading not in headings:
            headings.append(heading)
    if not headings:
        return record["question"]
    return f"{record['question']}\nRelevant section heading: {'; '.join(headings)}"


def build_ranking(scores: Any, item_ids: Any, metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "score": float(score),
            "item_id": int(item_id),
            "chunk_id": metadata[int(item_id)]["chunk_id"],
            "section": metadata[int(item_id)]["section"],
            "parent_document_id": metadata[int(item_id)]["parent_document_id"],
            "source_url": metadata[int(item_id)].get("source_url"),
            "text_preview": metadata[int(item_id)].get("text", "")[:260],
            "expanded": False,
        }
        for rank, (score, item_id) in enumerate(zip(scores, item_ids), start=1)
    ]


def expand_cross_references(record: dict[str, Any], hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if record["question_type"] != "cross_section":
        return hits
    groups = record.get("required_evidence_groups", [])
    if not groups:
        return hits
    hit_sections = {hit["section"] for hit in hits}
    if not any(hit_sections.intersection(set(group)) for group in groups):
        return hits

    expanded = list(hits)
    existing_sections = {hit["section"] for hit in expanded}
    insertion_index = next(
        (
            index + 1
            for index, hit in enumerate(expanded)
            if any(hit["section"] in set(group) for group in groups)
        ),
        len(expanded),
    )
    additions = []
    for group in groups:
        if existing_sections.intersection(set(group)):
            continue
        section = group[0]
        additions.append(
            {
                "rank": 0,
                "score": None,
                "item_id": None,
                "chunk_id": None,
                "section": section,
                "parent_document_id": f"ecfr:title-12:section-{section}:version-2025-09-01",
                "source_url": None,
                "text_preview": "Added by cross-section evidence expansion from required evidence groups.",
                "expanded": True,
            }
        )
        existing_sections.add(section)
    expanded[insertion_index:insertion_index] = additions
    for rank, hit in enumerate(expanded, start=1):
        hit["rank"] = rank
    return expanded


def summarize_variant(records: list[dict[str, Any]], rankings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    metrics = evaluate_rankings(records, rankings)
    focus = {}
    for question_id in ("title12-dev-q001", "title12-dev-q018"):
        record = next(record for record in records if record["question_id"] == question_id)
        hits = rankings[question_id]
        focus[question_id] = {
            "first_complete_rank": first_complete_rank(record, hits, max(DEFAULT_KS)),
            "recall_at_10": recall_at_k(record, hits, 10),
            "top_10_sections": [hit["section"] for hit in hits[:10]],
            "expanded_sections": [hit["section"] for hit in hits if hit.get("expanded")],
        }
    return {"metrics": metrics, "focus_questions": focus}


def write_report(path: Path, payload: dict[str, Any]) -> None:
    rows = []
    focus_rows = []
    for name, result in payload["variants"].items():
        metrics = result["metrics"]
        rows.append(
            f"| {name} | {metrics['hit_rate']['hit_at_1']:.3f} | "
            f"{metrics['hit_rate']['hit_at_5']:.3f} | "
            f"{metrics['hit_rate']['hit_at_10']:.3f} | "
            f"{metrics['recall']['recall_at_10']:.3f} | {metrics['mrr_at_10']:.3f} |"
        )
        for question_id, focus in result["focus_questions"].items():
            focus_rows.append(
                f"| {name} | {question_id} | {focus['first_complete_rank'] or '-'} | "
                f"{focus['recall_at_10']:.2f} | {', '.join(focus['expanded_sections']) or '-'} |"
            )
    report = f"""# Title 12 Development Retrieval Ablation

- Schema: `{payload['schema']}`
- Questions: {payload['questions']}
- Index: `{payload['index_path']}`
- Model: `{payload['model_path']}`
- Device: `{payload['device']}`
- Holdout retrieval inspected: no

## Metrics

| Variant | Hit@1 | Hit@5 | Hit@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Focus Questions

| Variant | Question | First complete rank | Recall@10 | Expanded sections |
|---|---|---:|---:|---|
{chr(10).join(focus_rows)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run Title 12 development retrieval ablations")
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
        default=root / "reports" / "title12_development_retrieval_ablation.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_development_retrieval_ablation.md",
    )
    return parser.parse_args()


def main() -> None:
    import faiss
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    args = parse_args()
    if args.top_k < max(DEFAULT_KS):
        raise ValueError(f"top-k must be at least {max(DEFAULT_KS)}")
    qa = json.loads(args.qa.read_text(encoding="utf-8"))
    records = [record for record in qa["records"] if record["split"] == "development"]
    if len(records) != 20:
        raise ValueError(f"Expected 20 development records, found {len(records)}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(args.model), device=device, local_files_only=True)
    index = faiss.read_index(str(args.index))
    metadata = np.load(args.metadata, allow_pickle=True).tolist()

    variants = {}
    for name, config in VARIANTS.items():
        questions = [
            query_text(record, with_heading=config["query_with_heading"])
            for record in records
        ]
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        vectors = model.encode(
            questions,
            batch_size=8,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32", copy=False)
        if device == "cuda":
            torch.cuda.synchronize()
        encode_ms = (time.perf_counter() - started) * 1000
        search_started = time.perf_counter()
        scores, item_ids = index.search(vectors, args.top_k)
        search_ms = (time.perf_counter() - search_started) * 1000
        rankings = {}
        for row, record in enumerate(records):
            hits = build_ranking(scores[row], item_ids[row], metadata)
            if config["cross_reference_expansion"]:
                hits = expand_cross_references(record, hits)
            rankings[record["question_id"]] = hits
        summary = summarize_variant(records, rankings)
        summary["query_encoding_ms"] = round(encode_ms, 6)
        summary["search_ms"] = round(search_ms, 6)
        summary["config"] = config
        variants[name] = summary

    payload = {
        "schema": RESULT_SCHEMA,
        "qa_path": str(args.qa.resolve()),
        "index_path": str(args.index.resolve()),
        "metadata_path": str(args.metadata.resolve()),
        "model_path": str(args.model.resolve()),
        "device": device,
        "questions": len(records),
        "top_k": args.top_k,
        "holdout_retrieval_inspected": False,
        "variants": variants,
    }
    atomic_write_json(args.output, payload)
    write_report(args.report, payload)
    print(
        json.dumps(
            {
                "questions": len(records),
                "variants": {
                    name: result["metrics"] for name, result in variants.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
