from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


SINGLE_SCHEMA = "title12-single-section-candidates-v1"
CROSS_SCHEMA = "title12-cross-section-candidates-v1"
VALID_ACTIONS = {"approve", "retype", "replace", "needs_pair"}


def referenced_partner_sections(decision: dict[str, Any]) -> set[str]:
    return {
        section
        for option in decision.get("partner_options", [])
        for key in ("default_partner_sections", "conditional_partner_sections")
        for section in option.get(key, [])
    }


def apply_decisions(
    candidates: list[dict[str, Any]],
    auto_reviews: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    official_sections: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates_by_id = {item["candidate_id"]: item for item in candidates}
    reviews_by_id = {item["candidate_id"]: item for item in auto_reviews}
    unknown = sorted(set(decisions) - set(candidates_by_id))
    if unknown:
        raise ValueError(f"Human decisions reference unknown candidates: {unknown}")

    single: list[dict[str, Any]] = []
    cross: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for candidate_id, candidate in candidates_by_id.items():
        auto_review = reviews_by_id[candidate_id]
        decision = decisions.get(candidate_id)
        if decision:
            action = decision.get("action")
            if action not in VALID_ACTIONS:
                raise ValueError(f"Invalid action for {candidate_id}: {action}")
        else:
            action = None

        if action == "replace":
            excluded.append(
                {
                    "candidate_id": candidate_id,
                    "section": candidate["section"],
                    "split": candidate["split"],
                    "reason": decision["reason"],
                }
            )
            continue

        if action == "needs_pair":
            partners = referenced_partner_sections(decision)
            invalid = sorted(partners - official_sections)
            if invalid:
                raise ValueError(f"Invalid partner sections for {candidate_id}: {invalid}")
            cross.append(
                {
                    **candidate,
                    "review_state": "human_pairing_confirmed",
                    "partner_options": decision["partner_options"],
                    "human_reason": decision["reason"],
                    "scoring_contract": (
                        "AND across required evidence groups; OR only within an "
                        "explicit alternative group"
                    ),
                }
            )
            continue

        if candidate["question_type"] == "cross_section":
            raise ValueError(f"Cross-section candidate lacks a human pairing decision: {candidate_id}")

        if action == "retype":
            approved_types = decision.get("approved_types", [])
            if not approved_types:
                raise ValueError(f"Retype decision lacks approved_types: {candidate_id}")
            review_state = "human_confirmed_retype"
            human_reason = decision["reason"]
        elif action == "approve":
            approved_types = [candidate["question_type"]]
            review_state = "human_confirmed"
            human_reason = decision["reason"]
        elif auto_review["recommendation"] == "approved":
            approved_types = [candidate["question_type"]]
            review_state = "auto_prescreened_pending"
            human_reason = None
        else:
            raise ValueError(
                f"Non-approved automatic recommendation needs a human decision: {candidate_id}"
            )
        single.append(
            {
                **candidate,
                "approved_question_types": approved_types,
                "review_state": review_state,
                "human_reason": human_reason,
                "question_constraints": decision.get("question_constraints", []) if decision else [],
                "automatic_recommendation": auto_review["recommendation"],
                "automatic_evidence_excerpt": auto_review["evidence_excerpt"],
            }
        )
    return single, cross, excluded


def apply_replacement_selections(
    single: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
    selections: dict[str, dict[str, Any]],
    official_sections: set[str],
) -> list[dict[str, Any]]:
    shortlist_by_id = {item["replacement_id"]: item for item in shortlist}
    selected_ids: set[str] = set()
    merged = list(single)
    for replaced_id, selection in selections.items():
        replacement_id = selection.get("selected_replacement_id")
        if replacement_id not in shortlist_by_id:
            raise ValueError(f"Unknown replacement selection for {replaced_id}: {replacement_id}")
        if replacement_id in selected_ids:
            raise ValueError(f"Replacement selected more than once: {replacement_id}")
        item = shortlist_by_id[replacement_id]
        if item["replacement_for"] != replaced_id:
            raise ValueError(
                f"Replacement target mismatch: {replacement_id} targets "
                f"{item['replacement_for']}, not {replaced_id}"
            )
        if item["section"] not in official_sections:
            raise ValueError(f"Replacement section does not exist: {item['section']}")
        approved_types = selection.get("approved_types", [])
        if not approved_types:
            raise ValueError(f"Replacement lacks approved_types: {replacement_id}")
        selected_ids.add(replacement_id)
        merged.append(
            {
                **item,
                "candidate_id": replacement_id,
                "split": item["intended_split"],
                "replaces_candidate_id": replaced_id,
                "approved_question_types": approved_types,
                "review_state": "human_confirmed_replacement",
                "human_reason": selection["reason"],
                "question_constraints": selection.get("question_constraints", []),
                "automatic_recommendation": None,
                "automatic_evidence_excerpt": item["evidence_preview"],
            }
        )
    return merged


def load_replacement_candidates(paths: list[Path]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "groups" in payload:
            candidates.extend(
                item
                for group in payload["groups"]
                for item in group.get("candidates", [])
            )
        else:
            candidates.extend(payload.get("candidates", []))
    return candidates


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_report(
    path: Path,
    single: list[dict[str, Any]],
    cross: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
) -> None:
    single_states = Counter(item["review_state"] for item in single)
    single_splits = Counter(item["split"] for item in single)
    cross_splits = Counter(item["split"] for item in cross)
    combined_development = single_splits["development"] + cross_splits["development"]
    combined_holdout = single_splits["holdout"] + cross_splits["holdout"]
    single_rows = [
        f"| {item['candidate_id']} | {item['split']} | § {item['section']} | "
        f"{', '.join(item['approved_question_types'])} | {item['review_state']} |"
        for item in single
    ]
    cross_rows = [
        f"| {item['candidate_id']} | {item['split']} | § {item['section']} | "
        f"{'; '.join(', '.join(option.get('default_partner_sections', option.get('conditional_partner_sections', []))) for option in item['partner_options'])} |"
        for item in cross
    ]
    report = f"""# Title 12 Human Review Summary

## Status Meaning

- `human_confirmed`: explicitly approved by human review.
- `human_confirmed_retype`: explicitly retained with corrected question types.
- `auto_prescreened_pending`: automatic prescreen only; not final human approval.
- Cross-section candidates use a separate pool and separate scoring contract.

## Single-Section Pool

- Candidates: {len(single)}
- Development / holdout: {single_splits['development']} / {single_splits['holdout']}
- Human confirmed: {single_states['human_confirmed']}
- Human-confirmed retype: {single_states['human_confirmed_retype']}
- Human-confirmed replacements: {single_states['human_confirmed_replacement']}
- Still pending human review: {single_states['auto_prescreened_pending']}
- The single-section pool is not independently required to contain 20/20 candidates.

| Candidate | Split | Section | Approved question types | Review state |
|---|---|---|---|---|
{chr(10).join(single_rows)}

## Cross-Section Pool

- Candidates: {len(cross)}
- Development / holdout: {cross_splits['development']} / {cross_splits['holdout']}
- These candidates must not be included in ordinary single-section Hit@K/MRR.

## Combined Sampling Target

- Single + cross Development: {combined_development} / 20
- Single + cross Holdout: {combined_holdout} / 20
- Replacement deficit: Development {max(0, 20 - combined_development)}, Holdout {max(0, 20 - combined_holdout)}

| Candidate | Split | Source section | Partner options |
|---|---|---|---|
{chr(10).join(cross_rows)}

## Excluded

{chr(10).join(f"- {item['candidate_id']} · § {item['section']}: {item['reason']}" for item in excluded) or '- None'}

No final QA has been generated. The next step is human confirmation of the remaining
single-section candidates and replacement sampling for the reported split deficits.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Apply human evaluation review decisions")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_expansion_candidates.json",
    )
    parser.add_argument(
        "--auto-review",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_expansion_candidate_review.json",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_human_review_decisions.json",
    )
    parser.add_argument(
        "--replacements",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_exp006_replacement_shortlist.json",
    )
    parser.add_argument(
        "--deficit-replacements",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_deficit_replacement_shortlist.json",
    )
    parser.add_argument(
        "--sections",
        type=Path,
        default=root / "data" / "canonical" / "title12_2025-09-01" / "sections.jsonl",
    )
    parser.add_argument(
        "--single-output",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_single_section_candidates.json",
    )
    parser.add_argument(
        "--cross-output",
        type=Path,
        default=root / "data" / "eval" / "title12_eval_cross_section_candidates.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "title12_eval_human_review_summary.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))["candidates"]
    auto_reviews = json.loads(args.auto_review.read_text(encoding="utf-8"))["reviews"]
    decision_payload = json.loads(args.decisions.read_text(encoding="utf-8"))
    replacement_candidates = load_replacement_candidates(
        [args.replacements, args.deficit_replacements]
    )
    with args.sections.open("r", encoding="utf-8") as file:
        official_sections = {
            json.loads(line)["section"] for line in file if line.strip()
        }
    single, cross, excluded = apply_decisions(
        candidates,
        auto_reviews,
        decision_payload["decisions"],
        official_sections,
    )
    single = apply_replacement_selections(
        single,
        replacement_candidates,
        decision_payload.get("replacement_selections", {}),
        official_sections,
    )
    atomic_write_json(
        args.single_output,
        {
            "schema": SINGLE_SCHEMA,
            "source_candidates": str(args.candidates.resolve()),
            "source_human_decisions": str(args.decisions.resolve()),
            "final_qa_generated": False,
            "candidate_count": len(single),
            "candidates": single,
        },
    )
    atomic_write_json(
        args.cross_output,
        {
            "schema": CROSS_SCHEMA,
            "source_candidates": str(args.candidates.resolve()),
            "source_human_decisions": str(args.decisions.resolve()),
            "final_qa_generated": False,
            "candidate_count": len(cross),
            "candidates": cross,
        },
    )
    write_report(args.report, single, cross, excluded)
    print(
        json.dumps(
            {
                "single_section": len(single),
                "cross_section": len(cross),
                "excluded": len(excluded),
                "single_review_states": dict(
                    Counter(item["review_state"] for item in single)
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
