"""
OpenAI-compatible provider -- works with any server that speaks the
OpenAI chat completions API (OpenAI, OpenRouter, vLLM, llama.cpp, etc.).
"""

import logging
from typing import Optional

import httpx

from forge_iml.providers.base import LLMProvider

log = logging.getLogger("forge_iml.providers.openai_compat")


class OpenAICompatProvider(LLMProvider):
    """LLM provider for any OpenAI-compatible API endpoint.

    Args:
        base_url: API base URL (e.g. ``https://api.openai.com/v1``,
                  ``https://openrouter.ai/api/v1``).
        api_key: Bearer token for authentication.
        model: Default model name (e.g. ``gpt-4o``, ``deepseek/deepseek-r1``).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4o",
        timeout: float = 60.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        model: Optional[str] = None,
        json_schema: Optional[dict] = None,
    ) -> Optional[dict]:
        """Send a chat-completion request to an OpenAI-compatible endpoint."""
        payload: dict = {
            "model": model or self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code != 200:
                    log.warning("OpenAI-compat returned %d: %s",
                                resp.status_code, resp.text[:200])
                    return None

                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return None

                content = choices[0].get("message", {}).get("content", "")
                return {"response": content}

        except Exception as e:
            log.warning("OpenAI-compat completion failed: %s", e)
            return None
