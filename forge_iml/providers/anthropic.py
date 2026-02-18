"""
Anthropic provider -- wraps the Anthropic Messages API.

Placeholder implementation that follows the LLMProvider contract.
Requires the ``anthropic`` package for actual use.
"""

import logging
from typing import Optional

from forge_iml.providers.base import LLMProvider

log = logging.getLogger("forge_iml.providers.anthropic")


class AnthropicProvider(LLMProvider):
    """LLM provider for the Anthropic Messages API.

    Args:
        api_key: Anthropic API key.
        model: Default model name (e.g. ``claude-sonnet-4-20250514``).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
        timeout: float = 60.0,
    ):
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
        """Send a messages request to the Anthropic API.

        Requires the ``anthropic`` Python package to be installed.
        Falls back to httpx if not available.
        """
        try:
            import anthropic
        except ImportError:
            log.warning("anthropic package not installed -- using httpx fallback")
            return self._httpx_fallback(messages, temperature=temperature,
                                        max_tokens=max_tokens, model=model)

        effective_model = model or self._model
        # Convert OpenAI-style messages to Anthropic format
        system_msg = ""
        api_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg.get("content", "")
            else:
                api_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        try:
            client = anthropic.Anthropic(api_key=self._api_key)
            resp = client.messages.create(
                model=effective_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_msg if system_msg else anthropic.NOT_GIVEN,
                messages=api_messages,
            )
            content = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    content += block.text
            return {"response": content}
        except Exception as e:
            log.warning("Anthropic completion failed: %s", e)
            return None

    def _httpx_fallback(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> Optional[dict]:
        """Raw HTTP fallback when the anthropic package is not installed."""
        import httpx

        effective_model = model or self._model
        system_msg = ""
        api_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg.get("content", "")
            else:
                api_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        payload: dict = {
            "model": effective_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system_msg:
            payload["system"] = system_msg

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload,
                    headers={
                        "x-api-key": self._api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                )
                if resp.status_code != 200:
                    log.warning("Anthropic returned %d: %s",
                                resp.status_code, resp.text[:200])
                    return None

                data = resp.json()
                content = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        content += block.get("text", "")
                return {"response": content}
        except Exception as e:
            log.warning("Anthropic httpx fallback failed: %s", e)
            return None
