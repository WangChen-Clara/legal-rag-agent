from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Any, Iterable


TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "−": "-",
        "搂": "§",
        "\u00a0": " ",
    }
)
TOKEN = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*|§", re.I)
KNOWN_TRUNCATION = re.compile(r"\s*…\(已截断\)$")


def normalize_conservative(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).translate(TRANSLATION).lower()
    return re.sub(r"\s+", " ", value).strip()


def content_tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN.findall(normalize_conservative(text)))


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_conservative(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AlignmentRecord:
    official_section: str
    official_part: str
    status: str
    reason_code: str
    legacy_document_id: str | None
    legacy_row_id: Any | None
    similarity: float | None
    score_margin: float | None
    source_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def align_official_sections(
    official_sections: Iterable[dict[str, Any]],
    legacy_documents: Iterable[dict[str, Any]],
    *,
    review_threshold: float = 0.85,
    minimum_length_ratio: float = 0.80,
) -> list[AlignmentRecord]:
    legacy = list(legacy_documents)
    exact_index: dict[str, list[dict[str, Any]]] = {}
    token_index: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    normalized_cache: dict[str, str] = {}
    token_cache: dict[str, tuple[str, ...]] = {}
    prefix_index: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    truncated_legacy: list[tuple[str, dict[str, Any]]] = []
    for document in legacy:
        identifier = document["legacy_document_id"]
        normalized = normalize_conservative(document["text"])
        tokens = content_tokens(document["text"])
        normalized_cache[identifier] = normalized
        token_cache[identifier] = tokens
        prefix_index.setdefault(tokens[:3], []).append(document)
        exact_index.setdefault(hashlib.sha256(normalized.encode()).hexdigest(), []).append(document)
        token_index.setdefault(tokens, []).append(document)
        if KNOWN_TRUNCATION.search(document["text"]):
            truncated_legacy.append(
                (
                    normalize_conservative(KNOWN_TRUNCATION.sub("", document["text"])),
                    document,
                )
            )

    results: list[AlignmentRecord] = []
    official_list = list(official_sections)
    for official in official_list:
        official_text = official["text"]
        normalized = normalize_conservative(official_text)
        exact = exact_index.get(hashlib.sha256(normalized.encode()).hexdigest(), [])
        if len(exact) == 1:
            candidate = exact[0]
            results.append(
                AlignmentRecord(
                    official_section=str(official["section"]),
                    official_part=str(official["part"]),
                    status="exact",
                    reason_code="normalized_text_equal",
                    legacy_document_id=candidate["legacy_document_id"],
                    legacy_row_id=candidate["row_id"],
                    similarity=1.0,
                    score_margin=None,
                    source_url=official["source_url"],
                )
            )
            continue
        if len(exact) > 1:
            results.append(
                AlignmentRecord(
                    official_section=str(official["section"]),
                    official_part=str(official["part"]),
                    status="review_required",
                    reason_code="multiple_exact_candidates",
                    legacy_document_id=None,
                    legacy_row_id=None,
                    similarity=1.0,
                    score_margin=0.0,
                    source_url=official["source_url"],
                )
            )
            continue

        tokens = content_tokens(official_text)
        token_matches = token_index.get(tokens, [])
        if len(token_matches) == 1:
            candidate = token_matches[0]
            results.append(
                AlignmentRecord(
                    official_section=str(official["section"]),
                    official_part=str(official["part"]),
                    status="high_confidence",
                    reason_code="content_tokens_equal",
                    legacy_document_id=candidate["legacy_document_id"],
                    legacy_row_id=candidate["row_id"],
                    similarity=1.0,
                    score_margin=None,
                    source_url=official["source_url"],
                )
            )
            continue
        if len(token_matches) > 1:
            results.append(
                AlignmentRecord(
                    official_section=str(official["section"]),
                    official_part=str(official["part"]),
                    status="review_required",
                    reason_code="multiple_token_equal_candidates",
                    legacy_document_id=None,
                    legacy_row_id=None,
                    similarity=1.0,
                    score_margin=0.0,
                    source_url=official["source_url"],
                )
            )
            continue

        truncation_matches = [
            (prefix, document)
            for prefix, document in truncated_legacy
            if len(prefix) >= 1000 and normalized.startswith(prefix)
        ]
        if len(truncation_matches) == 1:
            prefix, candidate = truncation_matches[0]
            results.append(
                AlignmentRecord(
                    official_section=str(official["section"]),
                    official_part=str(official["part"]),
                    status="review_required",
                    reason_code="legacy_source_truncated",
                    legacy_document_id=candidate["legacy_document_id"],
                    legacy_row_id=candidate["row_id"],
                    similarity=round(len(prefix) / max(len(normalized), 1), 6),
                    score_margin=None,
                    source_url=official["source_url"],
                )
            )
            continue

        scored: list[tuple[float, dict[str, Any]]] = []
        official_length = max(len(normalized), 1)
        # Blocking only limits expensive candidate scoring. Three leading tokens keep
        # near-duplicates discoverable even when a legally material token such as
        # "not" appears immediately afterwards; scoring never promotes them to an
        # automatic match.
        official_prefix = tokens[:3]
        for document in prefix_index.get(official_prefix, []):
            candidate_text = normalized_cache[document["legacy_document_id"]]
            candidate_tokens = token_cache[document["legacy_document_id"]]
            if official_prefix and candidate_tokens[:3] != official_prefix:
                continue
            length_ratio = min(len(candidate_text), official_length) / max(
                len(candidate_text), official_length, 1
            )
            if length_ratio < minimum_length_ratio:
                continue
            score = SequenceMatcher(None, normalized, candidate_text, autojunk=True).ratio()
            if score >= review_threshold:
                scored.append((score, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            results.append(
                AlignmentRecord(
                    official_section=str(official["section"]),
                    official_part=str(official["part"]),
                    status="unmatched",
                    reason_code="no_candidate_above_review_threshold",
                    legacy_document_id=None,
                    legacy_row_id=None,
                    similarity=None,
                    score_margin=None,
                    source_url=official["source_url"],
                )
            )
            continue
        best_score, best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        results.append(
            AlignmentRecord(
                official_section=str(official["section"]),
                official_part=str(official["part"]),
                status="review_required",
                reason_code=(
                    "similar_candidate_ambiguous"
                    if len(scored) > 1 and best_score - second_score < 0.01
                    else "content_difference_detected"
                ),
                legacy_document_id=best["legacy_document_id"],
                legacy_row_id=best["row_id"],
                similarity=round(best_score, 6),
                score_margin=round(best_score - second_score, 6),
                source_url=official["source_url"],
            )
        )
    return results
