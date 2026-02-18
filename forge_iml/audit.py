"""
IML Audit -- Audit log writer for pipeline executions.

Traces every execution to:
  1. Memory provider (summary, optional)
  2. Local JSONL file (detailed, append-only)
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from forge_iml.schema import IntentGraph, PipelineResult
from forge_iml.providers.base import MemoryProvider

logger = logging.getLogger("forge_iml.audit")


# -- IMLAudit -----------------------------------------------------------------

class IMLAudit:
    """Audit log writer for IML pipeline executions.

    Args:
        log_path: Path to the JSONL audit log file.
        memory_provider: Optional MemoryProvider for posting summaries.
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        memory_provider: Optional[MemoryProvider] = None,
    ):
        self._lock = threading.Lock()
        self._log_path = log_path or Path("iml_audit.jsonl")
        self._memory = memory_provider
        self._total_executions = 0
        self._total_cache_hits = 0
        self._total_duration_ms = 0.0
        self._total_tokens = 0
        # Ensure log directory exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_execution(self, graph: IntentGraph, result: PipelineResult,
                      stages: list[dict]) -> None:
        """Log a pipeline execution to disk and (optionally) memory."""
        now = datetime.now(timezone.utc).isoformat()

        self._total_executions += 1
        if result.cache_hit:
            self._total_cache_hits += 1
        self._total_duration_ms += result.total_duration_ms
        self._total_tokens += result.total_tokens

        jsonl_entry = {
            "timestamp": now,
            "graph_id": graph.id,
            "user_handle": graph.source.user_handle,
            "channel": graph.source.channel,
            "stages": stages,
            "total_duration_ms": result.total_duration_ms,
            "total_tokens": result.total_tokens,
            "policy_gates_checked": result.policy_gates_checked,
            "policy_gates_triggered": result.policy_gates_triggered,
            "result": result.status,
        }

        self._append_jsonl(jsonl_entry)

        cache_str = f"hit/{result.cache_layer}" if result.cache_hit else "miss"
        step_count = len(result.step_results)
        summary = (
            f"IML execution: {graph.id} | "
            f"user={graph.source.user_handle} | "
            f"channel={graph.source.channel} | "
            f"cache={cache_str} | "
            f"steps={step_count} | "
            f"tokens={result.total_tokens} | "
            f"{result.total_duration_ms:.0f}ms | "
            f"status={result.status}"
        )

        if self._memory is not None:
            self._post_memory(summary)

        logger.info("audit: %s", summary)

    def get_recent(self, limit: int = 20) -> list[dict]:
        """Read the most recent audit entries from the JSONL file."""
        if not self._log_path.exists():
            return []
        entries = []
        try:
            with self._lock:
                with open(self._log_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entries.append(json.loads(line))
        except Exception as e:
            logger.warning("failed to read audit log: %s", e)
            return []
        return entries[-limit:]

    def get_stats(self) -> dict:
        """Aggregate stats from the current session."""
        return {
            "total_executions": self._total_executions,
            "total_cache_hits": self._total_cache_hits,
            "cache_hit_rate": (self._total_cache_hits / self._total_executions
                               if self._total_executions > 0 else 0.0),
            "avg_duration_ms": (self._total_duration_ms / self._total_executions
                                if self._total_executions > 0 else 0.0),
            "total_tokens": self._total_tokens,
            "avg_tokens": (self._total_tokens / self._total_executions
                           if self._total_executions > 0 else 0),
        }

    # -- Internal -------------------------------------------------------------

    def _append_jsonl(self, entry: dict) -> None:
        try:
            with self._lock:
                with open(self._log_path, "a") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.warning("failed to write audit JSONL: %s", e)

    def _post_memory(self, summary: str) -> None:
        try:
            self._memory.save(
                summary,
                namespace="forge-iml",
                importance=3,
                agent="forge-iml-audit",
                entry_type="iml_audit",
            )
        except Exception as e:
            logger.debug("memory post failed: %s", e)
