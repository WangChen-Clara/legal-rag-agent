from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from rag_law.config import EmbeddingConfig, RetrievalConfig
from rag_law.retriever import FaissRetriever
from rag_law.tools import RegulationToolset


RESULT_SCHEMA = "title12-tools-validation-v1"
VALIDATION_QUESTIONS = [
    {
        "question_id": "title12-dev-q001",
        "question": "What investors do the provisions of 12 CFR 211.31's subpart apply to?",
        "expected_sections": ["211.31"],
        "expected_sources": {"211.31": "explicit_citation"},
    },
    {
        "question_id": "title12-dev-q018",
        "question": (
            "For double default treatment under 12 CFR 217.135, what kind of exposure "
            "may be hedged and what related section defines the eligible guarantee or "
            "credit derivative treatment?"
        ),
        "expected_sections": ["217.135", "217.134"],
        "expected_sources": {
            "217.135": "explicit_citation",
            "217.134": "cross_reference",
        },
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


def validate_search_result(
    toolset: RegulationToolset,
    question: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    result = toolset.search_regulations(
        question["question"],
        top_k=top_k,
        mode="citation_aware",
    )
    evidence = result.to_dict()["evidence"]
    top_sections = [item["section"] for item in evidence]
    source_by_section = {}
    for item in evidence:
        section = item["section"]
        if section and section not in source_by_section:
            source_by_section[section] = item["retrieval_source"]

    missing_sections = [
        section for section in question["expected_sections"] if section not in top_sections
    ]
    source_mismatches = [
        {
            "section": section,
            "expected": expected_source,
            "actual": source_by_section.get(section),
        }
        for section, expected_source in question["expected_sources"].items()
        if source_by_section.get(section) != expected_source
    ]
    passed = not missing_sections and not source_mismatches
    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "passed": passed,
        "missing_sections": missing_sections,
        "source_mismatches": source_mismatches,
        "top_evidence": evidence[:top_k],
    }


def validate_fetch_section(toolset: RegulationToolset, section: str) -> dict[str, Any]:
    record = toolset.fetch_section(section)
    payload = record.to_dict()
    required_fields = [
        "document_id",
        "title",
        "part",
        "section",
        "heading",
        "version_date",
        "source_url",
        "text",
        "safe_for_citation",
    ]
    missing = [field for field in required_fields if payload.get(field) in (None, "")]
    return {
        "section": section,
        "passed": not missing and payload["safe_for_citation"] is True,
        "missing_fields": missing,
        "record": payload,
    }


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def write_report(path: Path, payload: dict[str, Any]) -> None:
    search_rows = []
    for item in payload["search_validations"]:
        top_sources = ", ".join(
            f"{evidence['section']}:{evidence['retrieval_source']}"
            for evidence in item["top_evidence"][:5]
        )
        search_rows.append(
            f"| {item['question_id']} | {'passed' if item['passed'] else 'failed'} | "
            f"{top_sources} |"
        )
    fetch = payload["fetch_section_validation"]
    index_path = payload.get("index_display_path", payload["index_path"])
    sections_path = payload.get("sections_display_path", payload["sections_path"])
    report = f"""# Title 12 Tools Validation

- Schema: `{payload['schema']}`
- Status: `{payload['status']}`
- Search questions: {len(payload['search_validations'])}
- Fetch section: `{fetch['section']}`
- Index: `{index_path}`
- Sections: `{sections_path}`
- Device: `{payload['device']}`
- Holdout retrieval inspected: no
- LLM called: no

## Search Validations

| Question | Status | Top evidence |
|---|---|---|
{chr(10).join(search_rows)}

## Fetch Section

- Status: `{'passed' if fetch['passed'] else 'failed'}`
- Section: `{fetch['record'].get('section')}`
- Version date: `{fetch['record'].get('version_date')}`
- Safe for citation: `{fetch['record'].get('safe_for_citation')}`
- Source URL: `{fetch['record'].get('source_url')}`

## Interpretation

The Phase 4 tool prototype is read-only and validates against the fixed Title 12
`2025-09-01` canonical corpus. `search_regulations` returns structured evidence
only; it does not generate an answer. `fetch_section` returns the full official
parent section for citation display or later verification.

Remaining work: define a formal `Tool` interface, add `ToolRegistry` and
`ToolResult`, add timeout/error typing, implement `verify_citation`, and implement
`compare_versions`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate Title 12 read-only tools")
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
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "title12_tools_validation.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_tools_validation.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.top_k < 1:
        raise ValueError("top-k must be at least 1")
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
    search_validations = [
        validate_search_result(toolset, question, top_k=args.top_k)
        for question in VALIDATION_QUESTIONS
    ]
    fetch_validation = validate_fetch_section(toolset, "12 CFR 217.134(a)(1)")
    status = (
        "passed"
        if all(item["passed"] for item in search_validations) and fetch_validation["passed"]
        else "failed"
    )
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
        "holdout_retrieval_inspected": False,
        "llm_called": False,
        "search_validations": search_validations,
        "fetch_section_validation": fetch_validation,
    }
    atomic_write_json(args.output, payload)
    write_report(args.report, payload)
    print(json.dumps({"status": status}, ensure_ascii=False, indent=2))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
