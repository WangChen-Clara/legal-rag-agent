from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .agent import AgentState
from .llm_client import LLMClientError


class JudgeLLM(Protocol):
    def complete(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class JudgeScore:
    answer_relevance: int
    faithfulness: int
    citation_support: int
    legal_caution: int
    overall: int
    passed: bool
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_relevance": self.answer_relevance,
            "faithfulness": self.faithfulness,
            "citation_support": self.citation_support,
            "legal_caution": self.legal_caution,
            "overall": self.overall,
            "pass": self.passed,
            "issues": self.issues,
        }


def build_judge_prompt(state: AgentState) -> str:
    answer = state.final_answer.answer if state.final_answer else ""
    citations = state.final_answer.citations if state.final_answer else []
    verified_sections = {
        verification.section
        for verification in state.citation_verifications
        if verification.verified
    }
    evidence = []
    for section in state.fetched_sections:
        if section.section not in verified_sections:
            continue
        evidence.append(
            {
                "section": section.section,
                "heading": section.heading,
                "version_date": section.version_date,
                "source_url": section.source_url,
                "text": section.text,
            }
        )
    payload = {
        "question": state.question,
        "answer": answer,
        "citations": citations,
        "verified_evidence": evidence,
    }
    return f"""
You are judging a legal RAG answer. Use only the provided verified evidence.
Do not use outside knowledge.

Score each metric from 1 to 5:
- answer_relevance: Does the answer address the question?
- faithfulness: Is the answer fully grounded in the verified evidence?
- citation_support: Are the answer's claims supported by the listed citations?
- legal_caution: Does the answer avoid unsupported legal advice or overclaiming?
- overall: Overall answer quality.

Return only valid JSON with this schema:
{{
  "answer_relevance": 1,
  "faithfulness": 1,
  "citation_support": 1,
  "legal_caution": 1,
  "overall": 1,
  "pass": false,
  "issues": ["short issue text"]
}}

Evaluation input:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def evaluate_answer_with_judge(state: AgentState, judge: JudgeLLM) -> JudgeScore:
    prompt = build_judge_prompt(state)
    try:
        raw = judge.complete(prompt)
    except LLMClientError as error:
        raise ValueError("judge LLM failed") from error
    try:
        return parse_judge_score(raw)
    except ValueError:
        try:
            repaired = judge.complete(build_json_repair_prompt(raw))
        except LLMClientError as error:
            raise ValueError("judge LLM failed during JSON repair") from error
        return parse_judge_score(repaired)


def build_json_repair_prompt(raw_response: str) -> str:
    return f"""
Convert the previous judge response into valid JSON only.
Do not add markdown or explanation.

Required schema:
{{
  "answer_relevance": 1,
  "faithfulness": 1,
  "citation_support": 1,
  "legal_caution": 1,
  "overall": 1,
  "pass": false,
  "issues": ["short issue text"]
}}

Rules:
- Scores must be integers from 1 to 5.
- "pass" must be a boolean.
- "issues" must be a list of strings.
- Return only one JSON object.

Previous response:
{raw_response}
""".strip()


def parse_judge_score(raw: str) -> JudgeScore:
    payload = _extract_json_object(raw)
    required_ints = [
        "answer_relevance",
        "faithfulness",
        "citation_support",
        "legal_caution",
        "overall",
    ]
    scores = {key: _coerce_score(payload, key) for key in required_ints}
    passed = bool(payload.get("pass", payload.get("passed", False)))
    issues_raw = payload.get("issues", [])
    if isinstance(issues_raw, str):
        issues = [issues_raw]
    elif isinstance(issues_raw, list):
        issues = [str(item) for item in issues_raw]
    else:
        issues = [f"Unexpected issues field: {issues_raw!r}"]
    return JudgeScore(
        answer_relevance=scores["answer_relevance"],
        faithfulness=scores["faithfulness"],
        citation_support=scores["citation_support"],
        legal_caution=scores["legal_caution"],
        overall=scores["overall"],
        passed=passed,
        issues=issues,
    )


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    elif not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("judge response does not contain a JSON object")
        text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("judge response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("judge response JSON must be an object")
    return payload


def _coerce_score(payload: dict[str, Any], key: str) -> int:
    if key not in payload:
        raise ValueError(f"judge response missing score: {key}")
    score = int(payload[key])
    if score < 1 or score > 5:
        raise ValueError(f"judge score out of range for {key}: {score}")
    return score
