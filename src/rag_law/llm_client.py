from __future__ import annotations

import os

import requests

from .config import LLMConfig


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: LLMConfig, *, api_key: str | None = None):
        self.config = config
        self.api_key = api_key

    def complete(self, prompt: str) -> str:
        api_key = (self.api_key or os.getenv(self.config.api_key_env, "")).strip()
        if not api_key:
            raise LLMClientError(
                f"Missing API key environment variable: {self.config.api_key_env}"
            )

        try:
            response = requests.post(
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a careful legal research assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                },
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise LLMClientError("LLM chat-completions request failed") from error

        payload = response.json()
        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as error:
            raise LLMClientError("Unexpected chat-completions response schema") from error
