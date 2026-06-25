from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Iterable


SUPPORTED_POLICIES = {
    "em_or_alias",
    "contains_all",
    "evidence_groups_and_contains",
}


@dataclass(frozen=True)
class AnswerMetricResult:
    passed: bool
    metric_policy: str
    matched_alias: str | None
    missing_terms: tuple[str, ...]
    missing_evidence_groups: tuple[tuple[str, ...], ...]


def normalize_answer(value: str) -> str:
    """Normalize short legal QA answers for alias-style exact matching."""
    value = value.lower()
    value = value.replace("§", " section ")
    value = re.sub(r"\s+", " ", value)
    value = value.strip()
    value = value.translate(str.maketrans("", "", string.punctuation.replace("$", "")))
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def contains_term(answer: str, term: str) -> bool:
    return normalize_answer(term) in normalize_answer(answer)


def missing_required_terms(answer: str, terms: Iterable[str]) -> tuple[str, ...]:
    return tuple(term for term in terms if not contains_term(answer, term))


def matched_acceptable_answer(answer: str, acceptable_answers: Iterable[str]) -> str | None:
    normalized = normalize_answer(answer)
    for acceptable in acceptable_answers:
        if normalized == normalize_answer(acceptable):
            return acceptable
    return None


def missing_evidence_groups(
    cited_sections: Iterable[str],
    required_evidence_groups: Iterable[Iterable[str]],
) -> tuple[tuple[str, ...], ...]:
    cited = {str(section) for section in cited_sections}
    missing = []
    for group in required_evidence_groups:
        normalized_group = tuple(str(section) for section in group)
        if not cited.intersection(normalized_group):
            missing.append(normalized_group)
    return tuple(missing)


def score_answer(
    record: dict[str, Any],
    answer: str,
    *,
    cited_sections: Iterable[str] = (),
) -> AnswerMetricResult:
    policy = record["metric_policy"]
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported metric_policy: {policy}")

    acceptable_answers = record.get("acceptable_answers", [])
    must_contain = record.get("must_contain", [])
    missing_terms = missing_required_terms(answer, must_contain)
    matched_alias = matched_acceptable_answer(answer, acceptable_answers)
    missing_groups: tuple[tuple[str, ...], ...] = ()

    if policy == "em_or_alias":
        passed = matched_alias is not None or not missing_terms
    elif policy == "contains_all":
        passed = not missing_terms
    else:
        missing_groups = missing_evidence_groups(
            cited_sections,
            record.get("required_evidence_groups", []),
        )
        passed = not missing_terms and not missing_groups

    return AnswerMetricResult(
        passed=passed,
        metric_policy=policy,
        matched_alias=matched_alias,
        missing_terms=missing_terms,
        missing_evidence_groups=missing_groups,
    )
