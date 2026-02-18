"""
MLX provider -- wraps an mlx_lm.server HTTP endpoint.

This is the concrete provider for local MLX-served models
(e.g. Qwen3.5 via ``mlx_lm.server``).
"""

import logging
from typing import Optional

import httpx

from forge_iml.providers.base import LLMProvider

log = logging.getLogger("forge_iml.providers.mlx")


class MLXProvider(LLMProvider):
    """LLM provider backed by a local ``mlx_lm.server`` instance.

    Args:
        base_url: The HTTP base URL of the server (e.g. ``http://127.0.0.1:8081``).
        model: Default model path to include in requests.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8081",
        model: str = "",
        timeout: float = 120.0,
    ):
        self._base_url = base_url.rstrip("/")
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
        """Send a chat-completion request to the MLX server."""
        payload: dict = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        effective_model = model or self._model
        if effective_model:
            payload["model"] = effective_model

        # mlx_lm.server supports chat_template_kwargs for thinking control
        payload["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                )
                if resp.status_code != 200:
                    log.warning("MLX server returned %d: %s",
                                resp.status_code, resp.text[:200])
                    return None

                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return None

                content = choices[0].get("message", {}).get("content", "")
                return {"response": content}

        except Exception as e:
            log.warning("MLX completion failed: %s", e)
            return None
