from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchHit:
    rank: int
    distance: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_label(self) -> str:
        fields = ("source_file", "title", "part", "section", "date")
        parts = [str(self.metadata[key]) for key in fields if self.metadata.get(key)]
        return " | ".join(parts) or f"chunk:{self.metadata.get('doc_id', self.rank)}"


@dataclass(frozen=True)
class AnswerRecord:
    question: str
    answer: str
    evidence: list[SearchHit]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

