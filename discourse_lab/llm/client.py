"""LLM client for the offline realization pass (spec §2.10). Talks to
Ollama Cloud (https://ollama.com) by default — the same `/api/chat` shape as
local Ollama, just pointed at the hosted service with a bearer token, so
swapping back to a local `ollama serve` is a `base_url` change.

Get an API key at https://ollama.com/settings/keys and export it as
`OLLAMA_API_KEY`. Cloud model ids carry a `-cloud` suffix (e.g.
`gpt-oss:120b-cloud`, `qwen3-coder:480b-cloud`, `deepseek-v3.1:671b-cloud`).
"""

from __future__ import annotations

import os
from typing import Protocol

import requests

DEFAULT_BASE_URL = "https://ollama.com"


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    """Structural interface — a `FakeLLMClient` in tests need not subclass
    `OllamaCloudClient`, only match this shape.
    """

    def chat(self, messages: list[dict], *, temperature: float = 0.7, max_tokens: int | None = None) -> str: ...


class OllamaCloudClient:
    def __init__(
        self,
        model: str = "gpt-oss:120b-cloud",
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, messages: list[dict], *, temperature: float = 0.7, max_tokens: int | None = None) -> str:
        if not self.api_key:
            raise LLMError(
                "OLLAMA_API_KEY is not set. Get a key at https://ollama.com/settings/keys and "
                "`export OLLAMA_API_KEY=...`, or pass api_key= explicitly."
            )
        options = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False, "options": options},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise LLMError(f"could not reach Ollama Cloud at {self.base_url}: {e}") from e

        if resp.status_code != 200:
            raise LLMError(f"Ollama Cloud returned {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as e:
            raise LLMError(f"unexpected response shape from Ollama Cloud: {data!r}") from e
