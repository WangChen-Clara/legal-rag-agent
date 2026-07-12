from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from uuid import uuid4

from .llm_client import LLMClientError
from .tools import (
    CitationVerificationResult,
    RegulationEvidence,
    RegulationToolset,
    SectionRecord,
)


TerminationReason = Literal[
    "completed",
    "insufficient_evidence",
    "max_steps_exceeded",
]
TRACE_SCHEMA = "legal-rag-agent-trace-v1"


class AnswerLLM(Protocol):
    def complete(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class AgentStep:
    step_number: int
    action: str
    status: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "action": self.action,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FinalAnswer:
    answer: str
    citations: list[str]
    insufficient: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "insufficient": self.insufficient,
        }


@dataclass
class AgentState:
    question: str
    run_id: str = field(default_factory=lambda: uuid4().hex)
    steps: list[AgentStep] = field(default_factory=list)
    evidence: list[RegulationEvidence] = field(default_factory=list)
    fetched_sections: list[SectionRecord] = field(default_factory=list)
    citation_verifications: list[CitationVerificationResult] = field(default_factory=list)
    final_answer: FinalAnswer | None = None
    terminated_reason: TerminationReason | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "question": self.question,
            "steps": [step.to_dict() for step in self.steps],
            "evidence": [item.to_dict() for item in self.evidence],
            "fetched_sections": [item.to_dict() for item in self.fetched_sections],
            "citation_verifications": [
                item.to_dict() for item in self.citation_verifications
            ],
            "final_answer": self.final_answer.to_dict() if self.final_answer else None,
            "terminated_reason": self.terminated_reason,
        }

    def to_trace_dict(self, *, evidence_limit: int = 10) -> dict[str, Any]:
        return {
            "schema": TRACE_SCHEMA,
            "run_id": self.run_id,
            "question": self.question,
            "steps": [step.to_dict() for step in self.steps],
            "evidence_summary": [
                {
                    "rank": item.rank,
                    "section": item.section,
                    "retrieval_source": item.retrieval_source,
                    "version_date": item.version_date,
                    "source_url": item.source_url,
                    "score": item.score,
                    "chunk_id": item.chunk_id,
                    "parent_document_id": item.parent_document_id,
                }
                for item in self.evidence[:evidence_limit]
            ],
            "fetched_sections": [
                {
                    "section": item.section,
                    "heading": item.heading,
                    "version_date": item.version_date,
                    "source_url": item.source_url,
                    "safe_for_citation": item.safe_for_citation,
                }
                for item in self.fetched_sections
            ],
            "citation_verifications": [
                {
                    "section": item.section,
                    "verified": item.verified,
                    "version_date": item.version_date,
                    "source_url": item.source_url,
                    "safe_for_citation": item.safe_for_citation,
                    "issues": item.issues,
                }
                for item in self.citation_verifications
            ],
            "final_answer": self.final_answer.to_dict() if self.final_answer else None,
            "termination_reason": self.terminated_reason,
        }


class LegalRAGAgent:
    def __init__(
        self,
        toolset: RegulationToolset,
        *,
        max_steps: int = 6,
        top_k: int = 10,
        max_fetch_sections: int = 2,
        llm_client: AnswerLLM | None = None,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if max_fetch_sections < 0:
            raise ValueError("max_fetch_sections must not be negative")
        if max_steps < (max_fetch_sections * 2) + 2:
            raise ValueError("max_steps must be at least max_fetch_sections * 2 + 2")
        self.toolset = toolset
        self.max_steps = max_steps
        self.top_k = top_k
        self.max_fetch_sections = max_fetch_sections
        self.llm_client = llm_client

    def run(self, question: str) -> AgentState:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        state = AgentState(question=question)
        if not self._can_take_step(state):
            self._terminate_max_steps(state)
            return state

        search_result = self.toolset.search_regulations(
            question,
            top_k=self.top_k,
            mode="citation_aware",
        )
        state.evidence = search_result.evidence
        self._record_step(
            state,
            action="search_regulations",
            status="completed",
            detail={
                "mode": search_result.mode,
                "evidence_count": len(search_result.evidence),
                "sections": [item.section for item in search_result.evidence[: self.top_k]],
            },
        )

        if not state.evidence:
            state.final_answer = FinalAnswer(
                answer="Insufficient information in the provided evidence.",
                citations=[],
                insufficient=True,
            )
            state.terminated_reason = "insufficient_evidence"
            return state

        sections_to_fetch = self._sections_to_fetch(state.evidence)
        for section in sections_to_fetch:
            if not self._can_take_step(state):
                self._terminate_max_steps(state)
                return state
            fetched = self.toolset.fetch_section(section)
            state.fetched_sections.append(fetched)
            self._record_step(
                state,
                action="fetch_section",
                status="completed",
                detail={
                    "section": fetched.section,
                    "source_url": fetched.source_url,
                    "safe_for_citation": fetched.safe_for_citation,
                },
            )

        for section in state.fetched_sections:
            if not self._can_take_step(state):
                self._terminate_max_steps(state)
                return state
            verification = self.toolset.verify_citation(section.section)
            state.citation_verifications.append(verification)
            self._record_step(
                state,
                action="verify_citation",
                status="completed" if verification.verified else "failed",
                detail={
                    "section": verification.section,
                    "verified": verification.verified,
                    "issues": verification.issues,
                },
            )

        if not self._can_take_step(state):
            self._terminate_max_steps(state)
            return state

        state.final_answer = self._build_final_answer(state)
        state.terminated_reason = (
            "insufficient_evidence" if state.final_answer.insufficient else "completed"
        )
        self._record_step(
            state,
            action="final_answer",
            status=state.terminated_reason,
            detail={
                "citations": state.final_answer.citations,
                "llm_used": self.llm_client is not None
                and not state.final_answer.insufficient
                and not state.final_answer.answer.startswith("Relevant evidence was found"),
            },
        )
        return state

    def _sections_to_fetch(self, evidence: list[RegulationEvidence]) -> list[str]:
        sections: list[str] = []
        for item in evidence:
            if item.retrieval_source not in {"explicit_citation", "cross_reference"}:
                continue
            if not item.section or item.section in sections:
                continue
            sections.append(item.section)
            if len(sections) >= self.max_fetch_sections:
                break
        if len(sections) >= self.max_fetch_sections:
            return sections
        for item in evidence:
            if not item.section or item.section in sections:
                continue
            sections.append(item.section)
            if len(sections) >= self.max_fetch_sections:
                break
        return sections

    def _build_final_answer(self, state: AgentState) -> FinalAnswer:
        if not state.evidence:
            return FinalAnswer(
                answer="Insufficient information in the provided evidence.",
                citations=[],
                insufficient=True,
            )

        citation_sections = []
        verified_sections = {
            item.section for item in state.citation_verifications if item.verified
        }
        for section in state.fetched_sections:
            if state.citation_verifications and section.section not in verified_sections:
                continue
            if section.section not in citation_sections:
                citation_sections.append(section.section)
        if not citation_sections and not state.citation_verifications:
            for item in state.evidence:
                if item.section and item.section not in citation_sections:
                    citation_sections.append(item.section)
                if len(citation_sections) >= 2:
                    break
        if not citation_sections:
            return FinalAnswer(
                answer="Insufficient information in the verified evidence.",
                citations=[],
                insufficient=True,
            )

        citations = [
            self._citation_for_section(section, state)
            for section in citation_sections
        ]
        if self.llm_client is not None:
            llm_answer = self._try_build_llm_answer(state, citation_sections, citations)
            if llm_answer is not None:
                return llm_answer

        return self._build_deterministic_answer(citations)

    @staticmethod
    def _build_deterministic_answer(citations: list[str]) -> FinalAnswer:
        answer = (
            "Relevant evidence was found in "
            + ", ".join(citations)
            + ". Use the cited fixed-snapshot sections to answer the question."
        )
        return FinalAnswer(answer=answer, citations=citations)

    def _try_build_llm_answer(
        self,
        state: AgentState,
        citation_sections: list[str],
        citations: list[str],
    ) -> FinalAnswer | None:
        if self.llm_client is None:
            return None
        prompt = self._build_llm_answer_prompt(state, citation_sections, citations)
        try:
            answer = self.llm_client.complete(prompt).strip()
        except LLMClientError:
            return None
        if not answer:
            return None
        used_citations = [
            citation
            for section, citation in zip(citation_sections, citations, strict=False)
            if citation in answer or f"12 CFR {section}" in answer
        ]
        return FinalAnswer(answer=answer, citations=used_citations or citations)

    @staticmethod
    def _build_llm_answer_prompt(
        state: AgentState,
        citation_sections: list[str],
        citations: list[str],
    ) -> str:
        evidence_blocks = []
        section_by_citation = dict(zip(citation_sections, citations, strict=False))
        included_sections = set()
        for section in state.fetched_sections:
            if section.section not in citation_sections:
                continue
            citation = section_by_citation[section.section]
            evidence_blocks.append(
                "\n".join(
                    [
                        f"Citation: {citation}",
                        f"Heading: {section.heading}",
                        f"Version date: {section.version_date}",
                        "Text:",
                        section.text,
                    ]
                )
            )
            included_sections.add(section.section)
        for item in state.evidence:
            if item.section not in citation_sections:
                continue
            if item.section in included_sections:
                continue
            citation = section_by_citation[item.section]
            evidence_blocks.append(
                "\n".join(
                    [
                        f"Citation: {citation}",
                        f"Section: {item.section}",
                        f"Version date: {item.version_date}",
                        "Evidence excerpt:",
                        item.text,
                    ]
                )
            )
            included_sections.add(item.section)
        evidence = "\n\n---\n\n".join(evidence_blocks)
        allowed_citations = ", ".join(citations)
        return f"""
Answer the user's legal research question using only the verified evidence below.
Use only these citations in the answer: {allowed_citations}.

Rules:
- Before writing the final answer, internally perform this Evidence-to-Claim Checklist:
  1. Identify each requested entity, action, condition, and referenced section.
  2. Extract all conditions, qualifiers, exceptions, and definitions from the verified evidence.
  3. Ensure the final answer includes every condition that limits applicability.
  4. If a condition is not directly supported by evidence, state that the verified evidence is insufficient for that part.
- Do not output the checklist. Output only the final answer.
- Identify each sub-question or requested fact in the user's question, then answer each one directly.
- Every legal claim must include one citation from the allowed citation list.
- Do not invent authorities, dates, section numbers, facts, exceptions, definitions, or citations.
- Do not treat purpose, background, or policy language as a definition unless the verified evidence explicitly defines the term.
- If multiple sections contain parallel rules for different regulated entities, use the section whose scope matches the entity in the question.
- Do not say the evidence is insufficient when an evidence block directly matches the requested entity and action.
- If only one section supports the answer, cite only that section even if other allowed citations are listed.
- If multiple verified sections are provided and each is relevant, state what each section supports.
- If the verified evidence does not directly answer a requested fact, say that the verified evidence is insufficient for that part.
- Keep the answer concise. Prefer 2-4 short sentences or a compact bullet list for multi-part questions.

Question:
{state.question}

Verified evidence:
{evidence}

Answer:
""".strip()

    @staticmethod
    def _citation_for_section(section: str, state: AgentState) -> str:
        for fetched in state.fetched_sections:
            if fetched.section == section:
                return f"12 CFR {section} ({fetched.version_date})"
        for item in state.evidence:
            if item.section == section and item.version_date:
                return f"12 CFR {section} ({item.version_date})"
        return f"12 CFR {section}"

    def _record_step(
        self,
        state: AgentState,
        *,
        action: str,
        status: str,
        detail: dict[str, Any],
    ) -> None:
        state.steps.append(
            AgentStep(
                step_number=len(state.steps) + 1,
                action=action,
                status=status,
                detail=detail,
            )
        )

    def _can_take_step(self, state: AgentState) -> bool:
        return len(state.steps) < self.max_steps

    def _terminate_max_steps(self, state: AgentState) -> None:
        state.final_answer = FinalAnswer(
            answer="Insufficient information in the provided evidence.",
            citations=[],
            insufficient=True,
        )
        state.terminated_reason = "max_steps_exceeded"
