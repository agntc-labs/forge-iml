"""
IML Memory -- Memory claim retrieval with budget enforcement.

Queries a MemoryProvider and optional user profiles to inject MemoryClaim
nodes into the IntentGraph. Respects budget limits on claim count and
token usage.
"""

import logging
import re
from typing import Optional

from forge_iml.schema import IntentGraph, Node, NodeType, EdgeRelation
from forge_iml.providers.base import MemoryProvider

log = logging.getLogger("forge_iml.memory")

CHARS_PER_TOKEN = 4


# -- Helpers ------------------------------------------------------------------

def _extract_spo(content: str) -> tuple[str, str, str]:
    """Best-effort (subject, predicate, object) extraction from a string."""
    patterns = [
        r"^(.+?)\s+(is|are|was|were|has|have|had|born|lives? in|located at|prefers?|uses?|runs? on)\s+(.+)$",
        r"^(.+?):\s+(.+?)\s*[-=]\s*(.+)$",
    ]
    for pat in patterns:
        m = re.match(pat, content.strip(), re.IGNORECASE)
        if m:
            return (m.group(1).strip(), m.group(2).strip(), m.group(3).strip())
    words = content.strip().split()
    if len(words) >= 3:
        return (words[0], "relates_to", " ".join(words[1:]))
    return ("?", "relates_to", content.strip())


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


# -- Main Class ---------------------------------------------------------------

class IMLMemory:
    """Retrieves relevant memories and injects them as MemoryClaim nodes.

    Args:
        memory_provider: MemoryProvider for searching memories.
        user_profile_fn: Optional callable ``(handle: str) -> list[dict]``
            that returns curated facts for a user. Each dict should have
            at least ``{"content": "...", "importance": int, "confidence": float}``.
    """

    def __init__(
        self,
        memory_provider: Optional[MemoryProvider] = None,
        user_profile_fn=None,
    ):
        self._memory = memory_provider
        self._user_profile_fn = user_profile_fn

    # -- Public API -----------------------------------------------------------

    async def inject_claims(self, graph: IntentGraph) -> IntentGraph:
        """Mutate *graph* in-place by adding MemoryClaim nodes. Returns the graph."""
        max_claims = graph.budget.retrieval_max_claims
        max_tokens = graph.budget.retrieval_max_memory_tokens

        if max_claims <= 0:
            return graph

        queries = self._queries_from_goals(graph)
        if not queries:
            return graph

        candidates: list[dict] = []

        # User profile facts (high confidence)
        user_handle = graph.source.user_handle
        if user_handle and self._user_profile_fn:
            try:
                profile_claims = self._user_profile_fn(user_handle)
                candidates.extend(profile_claims)
            except Exception as e:
                log.warning("User profile lookup failed for %s: %s", user_handle, e)

        # Memory provider search
        if self._memory is not None:
            lake_claims = self._search_memory(queries)
            candidates.extend(lake_claims)

        if not candidates:
            return graph

        candidates.sort(
            key=lambda c: (c.get("importance", 0), c.get("confidence", 0.5)),
            reverse=True,
        )

        selected: list[dict] = []
        token_budget_remaining = max_tokens
        for claim in candidates:
            if len(selected) >= max_claims:
                break
            claim_tokens = _estimate_tokens(claim.get("content", ""))
            if claim_tokens > token_budget_remaining:
                continue
            token_budget_remaining -= claim_tokens
            selected.append(claim)

        goal_ids = [n.id for n in graph.get_goals()]
        for i, claim in enumerate(selected):
            subj = claim.get("subject", "?")
            pred = claim.get("predicate", "relates_to")
            obj = claim.get("object", "")
            node = graph.add_node(
                type=NodeType.MEMORY_CLAIM,
                content=claim.get("content", ""),
                confidence=claim.get("confidence", 0.7),
                metadata={
                    "source": claim.get("source", "memory"),
                    "fact_id": claim.get("fact_id", ""),
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "timestamp": claim.get("timestamp", ""),
                    "privacy_level": claim.get("privacy_level", "general"),
                },
                node_id=f"mc{i+1}",
            )
            for gid in goal_ids:
                graph.add_edge(gid, node.id, EdgeRelation.INFORMED_BY)

        return graph

    # -- Internal -------------------------------------------------------------

    def _queries_from_goals(self, graph: IntentGraph) -> list[str]:
        queries: list[str] = []
        for goal in graph.get_goals():
            queries.append(goal.content)
            obj = goal.metadata.get("object", "")
            if obj and obj != goal.content:
                queries.append(obj)
        seen: set[str] = set()
        unique: list[str] = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower and q_lower not in seen:
                seen.add(q_lower)
                unique.append(q.strip())
        return unique

    def _search_memory(self, queries: list[str]) -> list[dict]:
        claims: list[dict] = []
        seen_ids: set[str] = set()

        for query in queries:
            try:
                memories = self._memory.search(query, limit=5)
                for mem in memories:
                    mid = str(mem.get("id", ""))
                    if mid in seen_ids:
                        continue
                    seen_ids.add(mid)
                    content = mem.get("content", "")
                    subj, pred, obj = _extract_spo(content)
                    claims.append({
                        "source": "memory",
                        "fact_id": mid,
                        "content": content,
                        "subject": subj,
                        "predicate": pred,
                        "object": obj,
                        "importance": mem.get("importance", 3),
                        "confidence": 0.7,
                        "timestamp": mem.get("created_at", ""),
                        "privacy_level": "general",
                    })
            except Exception as e:
                log.warning("Memory query error for %s: %s", query, e)

        return claims
