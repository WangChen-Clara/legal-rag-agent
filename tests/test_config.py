from pathlib import Path

from rag_law.config import load_config


def test_default_config_uses_project_relative_public_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml")

    assert config.embedding.model_path == root / "models" / "bge-large-en-v1.5"
    assert (
        config.retrieval.index_path
        == root / "data" / "indexes" / "title12_bge_large_2025-09-01" / "vector_db.index"
    )
    assert (
        config.retrieval.metadata_path
        == root / "data" / "indexes" / "title12_bge_large_2025-09-01" / "metadata.npy"
    )
    assert config.llm.api_key_env == "RAG_LAW_API_KEY"
