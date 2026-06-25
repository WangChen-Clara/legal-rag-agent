from __future__ import annotations

from .config import AppConfig
from .llm_client import LLMClient
from .models import AnswerRecord, SearchHit
from .retriever import FaissRetriever


class RAGPipeline:
    def __init__(
        self,
        config: AppConfig,
        retriever: FaissRetriever | None = None,
        llm: LLMClient | None = None,
    ):
        self.config = config
        self.retriever = retriever or FaissRetriever(config.retrieval, config.embedding)
        self.llm = llm or LLMClient(config.llm)

    @staticmethod
    def build_prompt(question: str, hits: list[SearchHit]) -> str:
        evidence = "\n\n".join(
            f"[{hit.rank}] Source: {hit.source_label}\n{hit.text}" for hit in hits
        )
        return f"""
Answer the legal question using only the numbered evidence below.
If the evidence is insufficient, answer exactly: Insufficient information in the provided evidence.
Do not invent authorities, dates, or holdings. Keep the answer concise.
When possible, cite supporting evidence using bracketed numbers such as [1].

Evidence:
{evidence}

Question: {question}
Answer:
""".strip()

    def answer(self, question: str, top_k: int | None = None) -> AnswerRecord:
        hits = self.retriever.search(question, top_k=top_k)
        if not hits:
            return AnswerRecord(
                question=question,
                answer="Insufficient information in the provided evidence.",
                evidence=[],
            )
        answer = self.llm.complete(self.build_prompt(question, hits))
        return AnswerRecord(question=question, answer=answer, evidence=hits)

