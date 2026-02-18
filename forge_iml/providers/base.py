"""
Abstract base classes for IML providers.

LLMProvider  -- text completion (structured or freeform)
ToolProvider -- tool registry and execution
MemoryProvider -- search and storage of memories/facts
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class LLMProvider(ABC):
    """Abstract LLM backend for the IML pipeline.

    Implementations must provide a synchronous ``complete`` method.
    The pipeline will call this from async contexts via ``run_in_executor``
    when needed.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        model: Optional[str] = None,
        json_schema: Optional[dict] = None,
    ) -> Optional[dict]:
        """Run a completion and return the result.

        Args:
            messages: OpenAI-style messages list
                      ``[{"role": "system"|"user"|"assistant", "content": "..."}]``
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            model: Optional model override (provider-specific).
            json_schema: Optional JSON schema hint for structured output.

        Returns:
            A dict with at least ``{"response": "..."}`` on success,
            or ``None`` on failure.
        """
        ...


class ToolProvider(ABC):
    """Abstract tool registry and executor."""

    @abstractmethod
    def list_tools(self) -> list[dict]:
        """Return available tools.

        Each entry should have at least ``{"name": "...", "description": "..."}``.
        """
        ...

    @abstractmethod
    async def execute(self, tool_name: str, args: dict) -> dict:
        """Execute a tool by name with the given arguments.

        Returns a dict with at least ``{"success": bool}`` and
        ``{"result": ...}`` or ``{"error": "..."}`` on failure.
        """
        ...


class MemoryProvider(ABC):
    """Abstract memory/fact store."""

    @abstractmethod
    def search(self, query: str, *, limit: int = 5,
               namespace: Optional[str] = None) -> list[dict]:
        """Search for memories matching *query*.

        Returns a list of dicts, each with at least
        ``{"id": "...", "content": "...", "importance": int}``.
        """
        ...

    @abstractmethod
    def save(self, content: str, *, namespace: Optional[str] = None,
             importance: int = 3, tags: Optional[list[str]] = None,
             agent: str = "forge-iml",
             entry_type: str = "memory") -> Optional[str]:
        """Persist a memory entry.

        Returns the new entry ID on success, or ``None``.
        """
        ...

    def update(self, entry_id: str, content: str, *,
               importance: Optional[int] = None,
               tags: Optional[list[str]] = None) -> bool:
        """Update an existing entry. Returns True on success."""
        return False  # default: not supported
