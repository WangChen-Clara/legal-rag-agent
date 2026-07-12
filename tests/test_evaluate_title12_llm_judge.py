from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.evaluate_title12_llm_judge import (
    load_questions,
    normalize_question_record,
)


def test_normalize_question_record_accepts_expected_sections() -> None:
    record = {
        "question_id": "q1",
        "question": "What applies?",
        "expected_sections": ["211.31"],
    }

    assert normalize_question_record(record) == {
        "question_id": "q1",
        "question": "What applies?",
        "expected_sections": ["211.31"],
    }


def test_normalize_question_record_accepts_development_acceptable_sections() -> None:
    record = {
        "question_id": "q1",
        "question": "What applies?",
        "acceptable_sections": ["211.31"],
    }

    assert normalize_question_record(record)["expected_sections"] == ["211.31"]


def test_load_questions_uses_validation_dataset() -> None:
    args = Namespace(
        dataset="validation",
        questions_path=Path("unused.json"),
        question_limit=1,
    )

    questions = load_questions(args)

    assert len(questions) == 1
    assert questions[0]["question_id"] == "title12-dev-q001"
    assert questions[0]["expected_sections"] == ["211.31"]


def test_load_questions_uses_development_dataset(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        """
        {
          "records": [
            {
              "question_id": "dev-1",
              "question": "Question 1?",
              "acceptable_sections": ["1.1"]
            },
            {
              "question_id": "dev-2",
              "question": "Question 2?",
              "acceptable_sections": ["2.1"]
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    args = Namespace(
        dataset="development",
        questions_path=questions_path,
        question_limit=1,
    )

    questions = load_questions(args)

    assert questions == [
        {
            "question_id": "dev-1",
            "question": "Question 1?",
            "expected_sections": ["1.1"],
        }
    ]
