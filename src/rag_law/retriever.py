from __future__ import annotations

import re
from typing import Any

from .config import EmbeddingConfig, RetrievalConfig
from .models import SearchHit


SECTION_REFERENCE_RE = re.compile(
    r"(?:12\s+CFR\s+|§+\s*)([0-9]+[a-z]?(?:\.[0-9a-z-]+)+)",
    re.IGNORECASE,
)


class FaissRetriever:
    def __init__(self, retrieval: RetrievalConfig, embedding: EmbeddingConfig):
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if not retrieval.index_path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {retrieval.index_path}")
        if not retrieval.metadata_path.is_file():
            raise FileNotFoundError(f"Metadata not found: {retrieval.metadata_path}")

        self.config = retrieval
        self.index = faiss.read_index(str(retrieval.index_path))
        self.items = np.load(retrieval.metadata_path, allow_pickle=True).tolist()
        if self.index.ntotal != len(self.items):
            raise ValueError(
                f"Index/metadata size mismatch: {self.index.ntotal} != {len(self.items)}"
            )
        self.section_items = self._build_section_items(self.items)

        self.embedder = SentenceTransformer(str(embedding.model_path), device=embedding.device)

    @staticmethod
    def _split_item(item: Any) -> tuple[str, dict[str, Any]]:
        if isinstance(item, dict):
            text = next(
                (
                    item[key]
                    for key in ("text", "chunk", "content", "page_content")
                    if isinstance(item.get(key), str)
                ),
                "",
            )
            return text, dict(item)
        return str(item), {"raw_item": str(item)}

    @staticmethod
    def _normalize_section_reference(section: str) -> str:
        return section.strip().rstrip(").,;:").lower()

    @classmethod
    def extract_section_references(cls, text: str) -> list[str]:
        sections: list[str] = []
        seen: set[str] = set()
        for match in SECTION_REFERENCE_RE.finditer(text):
            section = cls._normalize_section_reference(match.group(1))
            if section and section not in seen:
                sections.append(section)
                seen.add(section)
        return sections

    @classmethod
    def _build_section_items(cls, items: list[Any]) -> dict[str, list[int]]:
        section_items: dict[str, list[int]] = {}
        for index, item in enumerate(items):
            _, metadata = cls._split_item(item)
            section = metadata.get("section")
            if not isinstance(section, str) or not section.strip():
                continue
            normalized = cls._normalize_section_reference(section)
            section_items.setdefault(normalized, []).append(index)
        return section_items

    def _item_to_hit(
        self,
        item_index: int,
        *,
        rank: int,
        distance: float,
        retrieval_source: str | None = None,
    ) -> SearchHit | None:
        text, metadata = self._split_item(self.items[item_index])
        if not text.strip():
            return None
        if retrieval_source:
            metadata["retrieval_source"] = retrieval_source
        return SearchHit(
            rank=rank,
            distance=distance,
            text=text.strip(),
            metadata=metadata,
        )

    def _section_hits(
        self,
        sections: list[str],
        *,
        retrieval_source: str,
        max_chunks_per_section: int,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for section in sections:
            item_indices = self.section_items.get(self._normalize_section_reference(section), [])
            for item_index in item_indices[:max_chunks_per_section]:
                hit = self._item_to_hit(
                    item_index,
                    rank=0,
                    distance=0.0,
                    retrieval_source=retrieval_source,
                )
                if hit:
                    hits.append(hit)
        return hits

    @staticmethod
    def _dedupe_hits(hits: list[SearchHit]) -> list[SearchHit]:
        deduped: list[SearchHit] = []
        seen: set[str] = set()
        for hit in hits:
            key = str(
                hit.metadata.get("chunk_id")
                or hit.metadata.get("document_id")
                or hit.metadata.get("parent_document_id")
                or f"{hit.metadata.get('section')}:{hit.text[:120]}"
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
        return [
            SearchHit(
                rank=rank,
                distance=hit.distance,
                text=hit.text,
                metadata=hit.metadata,
            )
            for rank, hit in enumerate(deduped, start=1)
        ]

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        import faiss

        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")

        k = top_k or self.config.top_k
        vector = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        if self.config.normalize_query:
            faiss.normalize_L2(vector)

        distances, indices = self.index.search(vector, k)
        hits: list[SearchHit] = []
        for rank, (distance, index) in enumerate(zip(distances[0], indices[0]), start=1):
            if not 0 <= index < len(self.items):
                continue
            hit = self._item_to_hit(index, rank=rank, distance=float(distance))
            if hit:
                hits.append(hit)
        return hits

    def search_with_context(
        self,
        query: str,
        top_k: int | None = None,
        *,
        include_explicit_citations: bool = True,
        semantic_top_k: int | None = None,
        expand_cross_references: bool = True,
        expand_from_semantic_without_explicit: bool = False,
        max_expanded_sections: int = 5,
        max_chunks_per_section: int = 1,
    ) -> list[SearchHit]:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")
        if max_expanded_sections < 0:
            raise ValueError("max_expanded_sections must not be negative")
        if max_chunks_per_section < 1:
            raise ValueError("max_chunks_per_section must be at least 1")

        k = top_k or self.config.top_k
        semantic_hits = self.search(query, top_k=semantic_top_k or k)
        semantic_hits = [
            SearchHit(
                rank=hit.rank,
                distance=hit.distance,
                text=hit.text,
                metadata={**hit.metadata, "retrieval_source": "semantic"},
            )
            for hit in semantic_hits
        ]

        explicit_sections = self.extract_section_references(query) if include_explicit_citations else []
        explicit_hits = self._section_hits(
            explicit_sections,
            retrieval_source="explicit_citation",
            max_chunks_per_section=max_chunks_per_section,
        )

        cross_reference_hits: list[SearchHit] = []
        if expand_cross_references and max_expanded_sections:
            explicit_set = {self._normalize_section_reference(section) for section in explicit_sections}
            semantic_sections = {
                self._normalize_section_reference(str(hit.metadata.get("section", "")))
                for hit in semantic_hits
                if hit.metadata.get("section")
            }
            if explicit_set:
                expansion_source_hits = [
                    hit
                    for hit in explicit_hits + semantic_hits
                    if self._normalize_section_reference(str(hit.metadata.get("section", "")))
                    in explicit_set
                ]
            elif expand_from_semantic_without_explicit:
                expansion_source_hits = semantic_hits
            else:
                expansion_source_hits = []
            referenced_sections: list[str] = []
            seen_references: set[str] = set()
            for hit in expansion_source_hits:
                for section in self.extract_section_references(hit.text):
                    if section in explicit_set or section in semantic_sections:
                        continue
                    if section in seen_references:
                        continue
                    referenced_sections.append(section)
                    seen_references.add(section)
                    if len(referenced_sections) >= max_expanded_sections:
                        break
                if len(referenced_sections) >= max_expanded_sections:
                    break
            cross_reference_hits = self._section_hits(
                referenced_sections,
                retrieval_source="cross_reference",
                max_chunks_per_section=max_chunks_per_section,
            )

        return self._dedupe_hits(explicit_hits + cross_reference_hits + semantic_hits)[:k]
