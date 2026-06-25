from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model_name: str
    api_key_env: str
    temperature: float = 0.2
    top_p: float = 0.95
    timeout_seconds: int = 60


@dataclass(frozen=True)
class EmbeddingConfig:
    model_path: Path
    device: str = "cpu"


@dataclass(frozen=True)
class RetrievalConfig:
    index_path: Path
    metadata_path: Path
    top_k: int = 7
    normalize_query: bool = True


@dataclass(frozen=True)
class OutputConfig:
    directory: Path


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    output: OutputConfig


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}

    project_root = config_path.parent.parent
    llm = raw["llm"]
    embedding = raw["embedding"]
    retrieval = raw["retrieval"]
    output = raw.get("output", {})

    return AppConfig(
        llm=LLMConfig(**llm),
        embedding=EmbeddingConfig(
            model_path=_resolve_path(embedding["model_path"], project_root),
            device=embedding.get("device", "cpu"),
        ),
        retrieval=RetrievalConfig(
            index_path=_resolve_path(retrieval["index_path"], project_root),
            metadata_path=_resolve_path(retrieval["metadata_path"], project_root),
            top_k=int(retrieval.get("top_k", 7)),
            normalize_query=bool(retrieval.get("normalize_query", True)),
        ),
        output=OutputConfig(
            directory=_resolve_path(output.get("directory", "outputs"), project_root)
        ),
    )

