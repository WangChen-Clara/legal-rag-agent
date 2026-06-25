from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_law import RAGPipeline, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answers for a legal QA set")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--questions", default="data/eval/biaozhu.json")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    with Path(args.questions).open("r", encoding="utf-8") as file:
        questions = json.load(file)

    pipeline = RAGPipeline(config)
    records = []
    for item in questions:
        question = item.get("Q") or item.get("question")
        if question:
            records.append(pipeline.answer(question).to_dict())

    output = Path(args.output) if args.output else config.output.directory / "answers.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} records to {output}")


if __name__ == "__main__":
    main()

