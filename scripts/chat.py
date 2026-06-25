from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_law import RAGPipeline, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive legal RAG")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    pipeline = RAGPipeline(load_config(args.config))
    print("Legal RAG ready. Enter 'exit' to stop.")
    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            return
        if not question:
            continue
        record = pipeline.answer(question)
        print(f"\n{record.answer}\n")


if __name__ == "__main__":
    main()

