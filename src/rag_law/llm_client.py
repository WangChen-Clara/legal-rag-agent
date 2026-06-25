from __future__ import annotations

import os

import requests

from .config import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, prompt: str) -> str:
        api_key = os.getenv(self.config.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable: {self.config.api_key_env}"
            )

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
        payload = response.json()
        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("Unexpected chat-completions response schema") from error

