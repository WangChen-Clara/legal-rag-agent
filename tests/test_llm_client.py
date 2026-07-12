from __future__ import annotations

import pytest
import requests

from rag_law.config import LLMConfig
from rag_law.llm_client import LLMClient, LLMClientError


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def test_llm_client_calls_openai_compatible_chat_completions(monkeypatch) -> None:
    calls = []

    def fake_post(url, *, headers, json, timeout):
        calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return FakeResponse(
            {"choices": [{"message": {"content": "  Generated answer.  "}}]}
        )

    monkeypatch.setattr(requests, "post", fake_post)
    client = LLMClient(
        LLMConfig(
            base_url="http://localhost:11434/v1/",
            model_name="qwen2.5:7b-instruct",
            api_key_env="LLM_API_KEY",
            timeout_seconds=7,
        ),
        api_key="ollama",
    )

    answer = client.complete("Question")

    assert answer == "Generated answer."
    assert calls[0]["url"] == "http://localhost:11434/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer ollama"
    assert calls[0]["json"]["model"] == "qwen2.5:7b-instruct"
    assert calls[0]["json"]["messages"][1]["content"] == "Question"
    assert calls[0]["timeout"] == 7


def test_llm_client_raises_when_api_key_missing(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_LLM_API_KEY", raising=False)
    client = LLMClient(
        LLMConfig(
            base_url="http://localhost:11434/v1",
            model_name="qwen2.5:7b-instruct",
            api_key_env="MISSING_LLM_API_KEY",
        )
    )

    with pytest.raises(LLMClientError, match="Missing API key"):
        client.complete("Question")


def test_llm_client_rejects_unexpected_response_schema(monkeypatch) -> None:
    def fake_post(url, *, headers, json, timeout):
        return FakeResponse({"choices": []})

    monkeypatch.setattr(requests, "post", fake_post)
    client = LLMClient(
        LLMConfig(
            base_url="http://localhost:11434/v1",
            model_name="qwen2.5:7b-instruct",
            api_key_env="LLM_API_KEY",
        ),
        api_key="ollama",
    )

    with pytest.raises(LLMClientError, match="Unexpected"):
        client.complete("Question")
