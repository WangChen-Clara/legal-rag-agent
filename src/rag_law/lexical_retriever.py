from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable

from .models import SearchHit


TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9-]+)*", re.IGNORECASE)


class LexicalRetriever:
    def __init__(
        self,
        items: Iterable[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.items = [dict(item) for item in items]
        self.k1 = k1
        self.b = b
        self._texts = [self._item_text(item) for item in self.items]
        self._doc_tokens = [self.tokenize(text) for text in self._texts]
        self._doc_lengths = [len(tokens) for tokens in self._doc_tokens]
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )
        self._term_frequencies = [Counter(tokens) for tokens in self._doc_tokens]
        self._document_frequencies = self._build_document_frequencies(self._doc_tokens)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]

    @staticmethod
    def _item_text(item: dict[str, Any]) -> str:
        parts = [
            item.get("section"),
            item.get("heading"),
            item.get("title"),
            item.get("part"),
            item.get("text") or item.get("chunk") or item.get("content"),
        ]
        return " ".join(str(part) for part in parts if part)

    @staticmethod
    def _build_document_frequencies(doc_tokens: list[list[str]]) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for tokens in doc_tokens:
            frequencies.update(set(tokens))
        return frequencies

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self.items:
            return []

        query_terms = self.tokenize(query)
        scores = [
            (index, self._bm25_score(query_terms, index))
            for index in range(len(self.items))
        ]
        ranked = [
            (index, score)
            for index, score in sorted(scores, key=lambda item: (-item[1], item[0]))
            if score > 0
        ][:top_k]

        hits: list[SearchHit] = []
        for rank, (index, score) in enumerate(ranked, start=1):
            metadata = dict(self.items[index])
            metadata["retrieval_source"] = "lexical"
            hits.append(
                SearchHit(
                    rank=rank,
                    distance=float(score),
                    text=self._texts[index].strip(),
                    metadata=metadata,
                )
            )
        return hits

    def _bm25_score(self, query_terms: list[str], item_index: int) -> float:
        if not query_terms:
            return 0.0

        score = 0.0
        total_documents = len(self.items)
        doc_length = self._doc_lengths[item_index]
        term_frequencies = self._term_frequencies[item_index]
        for term in query_terms:
            term_frequency = term_frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = self._document_frequencies[term]
            idf = math.log(
                1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * doc_length / (self._avg_doc_length or 1.0)
            )
            score += idf * (term_frequency * (self.k1 + 1)) / denominator
        return score
