from __future__ import annotations

import pytest

from rag_law.agent import AgentState, FinalAnswer
from rag_law.llm_judge import (
    build_judge_prompt,
    evaluate_answer_with_judge,
    parse_judge_score,
)
from rag_law.tools import CitationVerificationResult, SectionRecord


class FakeJudge:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class RepairingJudge:
    def __init__(self):
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            return "The answer looks good overall, but this is not JSON."
        return """
        {
          "answer_relevance": 4,
          "faithfulness": 4,
          "citation_support": 4,
          "legal_caution": 4,
          "overall": 4,
          "pass": true,
          "issues": []
        }
        """


def judged_state() -> AgentState:
    state = AgentState(question="What does 12 CFR 211.31 apply to?")
    state.fetched_sections = [
        SectionRecord(
            document_id="doc",
            title=12,
            part="211",
            section="211.31",
            heading="Authority, purpose, and scope.",
            version_date="2025-09-01",
            source_url="https://www.ecfr.gov/on/2025-09-01/title-12/section-211.31",
            text="This subpart applies to eligible investors.",
            safe_for_citation=True,
        )
    ]
    state.citation_verifications = [
        CitationVerificationResult(
            section="211.31",
            verified=True,
            version_date="2025-09-01",
            source_url="https://www.ecfr.gov/on/2025-09-01/title-12/section-211.31",
            safe_for_citation=True,
            checks={
                "section_exists": True,
                "version_matches": True,
                "source_url_matches": True,
                "safe_for_citation": True,
            },
            issues=[],
        )
    ]
    state.final_answer = FinalAnswer(
        answer="It applies to eligible investors. 12 CFR 211.31 (2025-09-01)",
        citations=["12 CFR 211.31 (2025-09-01)"],
    )
    state.terminated_reason = "completed"
    return state


def test_parse_judge_score_accepts_fenced_json() -> None:
    score = parse_judge_score(
        """```json
        {
          "answer_relevance": 5,
          "faithfulness": 4,
          "citation_support": 5,
          "legal_caution": 4,
          "overall": 4,
          "pass": true,
          "issues": []
        }
        ```"""
    )

    assert score.answer_relevance == 5
    assert score.faithfulness == 4
    assert score.passed is True
    assert score.issues == []


def test_parse_judge_score_rejects_out_of_range_score() -> None:
    with pytest.raises(ValueError, match="out of range"):
        parse_judge_score(
            """
            {
              "answer_relevance": 6,
              "faithfulness": 4,
              "citation_support": 5,
              "legal_caution": 4,
              "overall": 4,
              "pass": true,
              "issues": []
            }
            """
        )


def test_build_judge_prompt_contains_only_verified_evidence() -> None:
    state = judged_state()
    prompt = build_judge_prompt(state)

    assert "What does 12 CFR 211.31 apply to?" in prompt
    assert "This subpart applies to eligible investors." in prompt
    assert "Do not use outside knowledge" in prompt


def test_evaluate_answer_with_judge_returns_structured_score() -> None:
    judge = FakeJudge(
        """
        {
          "answer_relevance": 5,
          "faithfulness": 5,
          "citation_support": 5,
          "legal_caution": 4,
          "overall": 5,
          "pass": true,
          "issues": ["Could be more explicit about scope."]
        }
        """
    )

    score = evaluate_answer_with_judge(judged_state(), judge)

    assert score.overall == 5
    assert score.passed is True
    assert judge.prompts
    assert "verified evidence" in judge.prompts[0].lower()


def test_evaluate_answer_with_judge_repairs_non_json_response() -> None:
    judge = RepairingJudge()

    score = evaluate_answer_with_judge(judged_state(), judge)

    assert score.overall == 4
    assert score.passed is True
    assert len(judge.prompts) == 2
    assert "Convert the previous judge response" in judge.prompts[1]
