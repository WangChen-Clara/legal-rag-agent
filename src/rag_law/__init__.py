"""Reusable legal RAG core."""

from .config import AppConfig, load_config
from .pipeline import RAGPipeline

__all__ = ["AppConfig", "RAGPipeline", "load_config"]

