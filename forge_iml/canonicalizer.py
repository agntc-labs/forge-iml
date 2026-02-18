"""
IML Canonicalizer -- Stage B of the pipeline.

Normalizes an IntentGraph into canonical form and generates a
deterministic cache key. Pure Python -- no LLM calls.

INVARIANT: Stage B MUST NEVER add facts. It only reorganizes
what Stage A extracted.
"""

import hashlib
import logging
from typing import Optional

from forge_iml.schema import (
    EdgeRelation,
    IntentGraph,
    Node,
    NodeType,
)

log = logging.getLogger("forge_iml.canonicalizer")

# -- Verb normalization -------------------------------------------------------

VERB_MAP: dict[str, list[str]] = {
    "get": ["fetch", "retrieve", "check", "look up", "find", "show", "tell", "what", "whats",
            "verify", "confirm", "validate", "test", "how", "hows",
            "give", "gimme"],
    "send": ["text", "message", "notify", "forward", "share"],
    "create": ["make", "build", "generate", "write", "draft", "compose"],
    "delete": ["remove", "clear", "erase", "drop", "destroy"],
    "search": ["google", "look up", "research", "find out"],
    "list": ["show all", "enumerate", "display"],
    "set": ["update", "change", "modify", "configure"],
    "run": ["execute", "start", "launch", "trigger"],
}

_VERB_LOOKUP: dict[str, str] = {}
for _canon, _aliases in VERB_MAP.items():
    for _alias in _aliases:
        _VERB_LOOKUP[_alias.lower()] = _canon
    _VERB_LOOKUP[_canon.lower()] = _canon

# -- Target normalization -----------------------------------------------------

# NOTE: Override with your own target aliases via
# ``IMLCanonicalizer(target_aliases={"dad": ["father", ...]})``.
DEFAULT_TARGET_ALIASES: dict[str, list[str]] = {}

# -- Tool normalization -------------------------------------------------------

TOOL_ALIASES: dict[str, list[str]] = {
    "weather": ["forecast", "temperature", "rain", "weather"],
    "web_search": ["google", "search", "look up", "browse"],
    "generate_image": ["image", "picture", "photo", "draw", "art"],
    "memory_search": ["remember", "recall", "memory", "memorize"],
}


def _build_lookup(aliases: dict[str, list[str]]) -> dict[str, str]:
    """Build inverted alias -> canonical lookup from an aliases dict."""
    lookup: dict[str, str] = {}
    for canon, alts in aliases.items():
        for alt in alts:
            lookup[alt.lower()] = canon
        lookup[canon.lower()] = canon
    return lookup


_TOOL_LOOKUP = _build_lookup(TOOL_ALIASES)


# -- Normalization helpers ----------------------------------------------------

def _normalize_verb(verb: str) -> str:
    """Map a verb to its canonical form."""
    return _VERB_LOOKUP.get(verb.lower().strip(), verb.lower().strip())


def _normalize_target(target: str, target_lookup: dict[str, str]) -> Optional[str]:
    """Map a target name to its canonical handle."""
    if not target:
        return None
    t = target.lower().strip()
    if t in target_lookup:
        return target_lookup[t]
    for prefix in ("my ", "our ", "the ", "his ", "her "):
        if t.startswith(prefix):
            stripped = t[len(prefix):]
            if stripped in target_lookup:
                return target_lookup[stripped]
    return t


def _normalize_tool(tool: str) -> str:
    """Map a tool reference to its canonical name."""
    return _TOOL_LOOKUP.get(tool.lower().strip(), tool.lower().strip())


# -- Cache key generation -----------------------------------------------------

def _generate_cache_key(graph: IntentGraph) -> str:
    """Generate a deterministic SHA256 cache key from the canonical graph.

    Includes: canonical verbs, objects, sorted constraints, user handle.
    Excludes: preferences, output specs, ambiguities, timestamps.
    """
    parts = []

    goals = graph.get_nodes_by_type(NodeType.USER_GOAL)
    goal_strs = []
    for g in goals:
        verb = g.metadata.get("verb", "")
        obj = g.metadata.get("object", "")
        target = g.metadata.get("target") or ""
        goal_strs.append(f"{verb}|{obj}|{target}")
    goal_strs.sort()
    parts.append(";".join(goal_strs))

    constraints = graph.get_nodes_by_type(NodeType.CONSTRAINT)
    cst_strs = []
    for c in constraints:
        fld = c.metadata.get("field", "")
        op = c.metadata.get("operator", "")
        val = c.metadata.get("value", "")
        cst_strs.append(f"{fld}:{op}:{val}")
    cst_strs.sort()
    parts.append(";".join(cst_strs))

    parts.append(graph.source.user_handle or "anonymous")

    key_input = "||".join(parts)
    return hashlib.sha256(key_input.encode()).hexdigest()


# -- IMLCanonicalizer class ---------------------------------------------------

class IMLCanonicalizer:
    """Stage B: normalize IntentGraph into canonical form + cache key.

    Args:
        target_aliases: Optional dict mapping canonical target names to
            lists of aliases.  e.g. ``{"dad": ["father", "papa", "justin"]}``.
        tool_aliases: Optional dict of additional tool aliases to merge
            with the built-in defaults.
    """

    def __init__(
        self,
        target_aliases: Optional[dict[str, list[str]]] = None,
        tool_aliases: Optional[dict[str, list[str]]] = None,
    ):
        self._canonicalize_count = 0
        self._target_lookup = _build_lookup(target_aliases or DEFAULT_TARGET_ALIASES)
        if tool_aliases:
            extra = _build_lookup(tool_aliases)
            global _TOOL_LOOKUP
            merged = dict(_TOOL_LOOKUP)
            merged.update(extra)
            self._tool_lookup = merged
        else:
            self._tool_lookup = _TOOL_LOOKUP

    @property
    def stats(self) -> dict:
        return {"canonicalize_count": self._canonicalize_count}

    def canonicalize(self, graph: IntentGraph) -> IntentGraph:
        """Normalize the graph in-place and return it.

        Mutations:
        1. Normalize verb/object/target in UserGoal nodes
        2. Normalize tool references in ToolPermission and PlanStep nodes
        3. Sort constraint nodes alphabetically by field
        4. Generate and set cache_key
        """
        self._normalize_goals(graph)
        self._normalize_tool_nodes(graph)
        self._sort_constraints(graph)
        graph.cache_key = _generate_cache_key(graph)

        self._canonicalize_count += 1
        log.info(
            "IML canonicalize: %d nodes, cache_key=%s",
            len(graph.nodes),
            graph.cache_key[:16],
        )
        return graph

    # -- Internal normalization passes ----------------------------------------

    def _normalize_goals(self, graph: IntentGraph) -> None:
        for node in graph.get_nodes_by_type(NodeType.USER_GOAL):
            md = node.metadata
            if "verb" in md:
                md["verb"] = _normalize_verb(md["verb"])
            if "target" in md:
                md["target"] = _normalize_target(md.get("target"), self._target_lookup)
            verb = md.get("verb", "do")
            obj = md.get("object", "something")
            target = md.get("target")
            node.content = f"{verb} {obj}"
            if target:
                node.content += f" -> {target}"

    def _normalize_tool_nodes(self, graph: IntentGraph) -> None:
        for node in graph.get_nodes_by_type(NodeType.TOOL_PERMISSION):
            canonical = self._tool_lookup.get(node.content.lower().strip(),
                                               node.content.lower().strip())
            node.content = canonical
            node.metadata["tool"] = canonical

        for node in graph.get_nodes_by_type(NodeType.PLAN_STEP):
            tool = node.metadata.get("tool", "")
            if tool:
                canonical = self._tool_lookup.get(tool.lower().strip(),
                                                   tool.lower().strip())
                node.metadata["tool"] = canonical
                node.content = f"call:{canonical}"

    def _sort_constraints(self, graph: IntentGraph) -> None:
        constraints = graph.get_nodes_by_type(NodeType.CONSTRAINT)
        if len(constraints) <= 1:
            return
        constraints.sort(key=lambda n: n.metadata.get("field", ""))
        non_constraints = [n for n in graph.nodes if n.type != NodeType.CONSTRAINT]
        insert_idx = 0
        for i, n in enumerate(non_constraints):
            if n.type == NodeType.USER_GOAL:
                insert_idx = i + 1
        graph.nodes = (
            non_constraints[:insert_idx]
            + constraints
            + non_constraints[insert_idx:]
        )
