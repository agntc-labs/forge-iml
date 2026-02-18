"""
IML Parser -- Stage A of the pipeline.

Takes raw user input and produces an IntentGraph via structured
extraction through an LLM provider.
"""

import json
import logging
import re
import time
from typing import Optional

from forge_iml.schema import (
    Ambiguity,
    Budget,
    Edge,
    EdgeRelation,
    IntentGraph,
    Node,
    NodeType,
    Source,
)
from forge_iml.providers.base import LLMProvider

log = logging.getLogger("forge_iml.parser")


# -- Extraction prompt --------------------------------------------------------

_EXTRACTION_SYSTEM = """\
You are a structured intent extractor. Given a user message, output ONLY valid JSON (no markdown, no explanation) with these fields:

{
  "goals": [{"verb": "...", "object": "...", "target": "...", "urgency": "low|normal|high"}],
  "constraints": [{"field": "...", "operator": "eq|gt|lt|contains|before|after", "value": "..."}],
  "preferences": [{"field": "...", "value": "...", "relaxable": true}],
  "tools_needed": ["tool_name"],
  "ambiguities": [{"question": "...", "why_it_matters": "...", "default": "...", "can_proceed": true}],
  "sentiment": "neutral",
  "complexity": "simple|complex|memory_only"
}

Rules:
- verb: the core action (get, send, create, delete, search, list, set, check, run)
- object: what the action applies to (weather, message, image, balance, etc.)
- target: who the action is directed at (a person name, or null)
- tools_needed: list tool names if obvious (weather, imessage_send, web_search, generate_image, etc.)
- complexity: "simple" for single-tool requests, "complex" for multi-step, "memory_only" for recall/conversation
- If the request is a plain conversation (greeting, question about self, etc.), use complexity "memory_only" and empty tools_needed
- Return ONLY the JSON object, nothing else"""


# -- JSON response parsing ----------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_JSON_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_llm_response(raw: str) -> Optional[dict]:
    """Extract JSON from LLM response, handling markdown blocks and bare JSON."""
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try extracting the first { ... } block
    m = _JSON_BRACE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


# -- Graph builder ------------------------------------------------------------

def _graph_from_parsed(
    parsed: dict,
    raw_input: str,
    user_handle: str,
    channel: str,
) -> IntentGraph:
    """Build an IntentGraph from the structured extraction output."""
    graph = IntentGraph(
        source=Source(
            user_handle=user_handle,
            channel=channel,
            language="en",
            raw_input=raw_input,
        ),
    )

    # Budget
    complexity = parsed.get("complexity", "simple")
    if complexity == "complex":
        graph.budget = Budget.for_complex()
    elif complexity == "memory_only":
        graph.budget = Budget.for_memory_only()
    else:
        graph.budget = Budget.for_simple()

    # Goal nodes
    goal_ids = []
    for i, g in enumerate(parsed.get("goals", [])):
        verb = g.get("verb", "do")
        obj = g.get("object", "something")
        target = g.get("target")
        urgency = g.get("urgency", "normal")
        content = f"{verb} {obj}"
        if target:
            content += f" -> {target}"
        node = graph.add_node(
            type=NodeType.USER_GOAL,
            content=content,
            confidence=0.9,
            metadata={"verb": verb, "object": obj, "target": target, "urgency": urgency},
            node_id=f"goal_{i}",
        )
        goal_ids.append(node.id)

    # Default goal if none extracted
    if not goal_ids:
        node = graph.add_node(
            type=NodeType.USER_GOAL,
            content=raw_input[:120],
            confidence=0.5,
            metadata={"verb": "respond", "object": "query", "target": None, "urgency": "normal"},
            node_id="goal_0",
        )
        goal_ids.append(node.id)

    # Constraint nodes
    for i, c in enumerate(parsed.get("constraints", [])):
        fld = c.get("field", "unknown")
        op = c.get("operator", "eq")
        val = c.get("value", "")
        node = graph.add_node(
            type=NodeType.CONSTRAINT,
            content=f"{fld} {op} {val}",
            metadata={"field": fld, "operator": op, "value": val},
            node_id=f"cst_{i}",
        )
        if goal_ids:
            graph.add_edge(node.id, goal_ids[0], EdgeRelation.CONSTRAINS)

    # Preference nodes
    sentiment = parsed.get("sentiment", "neutral")
    for i, p in enumerate(parsed.get("preferences", [])):
        fld = p.get("field", "")
        val = p.get("value", "")
        relaxable = p.get("relaxable", True)
        node = graph.add_node(
            type=NodeType.PREFERENCE,
            content=f"{fld}: {val}",
            metadata={"field": fld, "value": val, "relaxable": relaxable, "sentiment": sentiment},
            node_id=f"pref_{i}",
        )
        if goal_ids:
            graph.add_edge(goal_ids[0], node.id, EdgeRelation.PREFERS)

    # Tool permission nodes
    tools = parsed.get("tools_needed", [])
    tool_perm_ids = []
    for i, tool_name in enumerate(tools):
        node = graph.add_node(
            type=NodeType.TOOL_PERMISSION,
            content=tool_name,
            metadata={"tool": tool_name, "granted": True},
            node_id=f"tp_{i}",
        )
        tool_perm_ids.append(node.id)

    # Plan step nodes
    step_ids = []
    for i, tool_name in enumerate(tools):
        node = graph.add_node(
            type=NodeType.PLAN_STEP,
            content=f"call:{tool_name}",
            metadata={"tool": tool_name, "order": i, "args": {}},
            node_id=f"step_{i}",
        )
        step_ids.append(node.id)

        if i < len(tool_perm_ids):
            graph.add_edge(node.id, tool_perm_ids[i], EdgeRelation.NEEDS_TOOL)
        if goal_ids:
            graph.add_edge(goal_ids[0], node.id, EdgeRelation.REQUIRES)
        if i > 0:
            graph.add_edge(node.id, step_ids[i - 1], EdgeRelation.DEPENDS_ON)

    # Output spec node
    output_node = graph.add_node(
        type=NodeType.OUTPUT_SPEC,
        content="default",
        metadata={"format": "text", "channel": channel},
        node_id="output_0",
    )
    if step_ids:
        graph.add_edge(step_ids[-1], output_node.id, EdgeRelation.PRODUCES)
    elif goal_ids:
        graph.add_edge(goal_ids[0], output_node.id, EdgeRelation.PRODUCES)

    # Ambiguity ledger
    for a in parsed.get("ambiguities", []):
        amb = Ambiguity(
            question=a.get("question", ""),
            why_it_matters=a.get("why_it_matters", ""),
            allowed_defaults=[a.get("default", "")] if a.get("default") else [],
            can_proceed_without=a.get("can_proceed", True),
            default_used=a.get("default") if a.get("can_proceed", True) else None,
            confidence=0.5,
        )
        graph.ambiguity_ledger.append(amb)

    return graph


# -- IMLParser class ----------------------------------------------------------

class IMLParser:
    """Stage A: raw text -> IntentGraph via LLM structured extraction.

    Args:
        llm: An LLMProvider instance used for structured extraction.
        user_display_fn: Optional callable ``(handle: str) -> str`` that
            resolves a user handle to a display name. If not provided
            the raw handle is used as-is.
    """

    def __init__(
        self,
        llm: LLMProvider,
        user_display_fn=None,
    ):
        self._llm = llm
        self._user_display_fn = user_display_fn or (lambda h: h)
        self._parse_count = 0
        self._total_latency_ms = 0.0
        self._fail_count = 0

    @property
    def avg_latency_ms(self) -> float:
        if self._parse_count == 0:
            return 0.0
        return self._total_latency_ms / self._parse_count

    @property
    def stats(self) -> dict:
        return {
            "parse_count": self._parse_count,
            "fail_count": self._fail_count,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }

    async def parse(
        self,
        raw_input: str,
        user_handle: str,
        channel: str = "api",
    ) -> Optional[IntentGraph]:
        """Parse raw user input into an IntentGraph.

        Returns None if extraction fails (caller should fall back
        to an alternative routing pipeline).
        """
        if not raw_input or not raw_input.strip():
            return None

        t0 = time.monotonic()

        user_name = self._user_display_fn(user_handle)
        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM},
            {"role": "user", "content": f"User ({user_name}): {raw_input}"},
        ]

        result = self._llm.complete(
            messages,
            temperature=0.1,
            max_tokens=1024,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000

        if result is None:
            self._fail_count += 1
            log.warning("IML parse: LLM returned None (%.0fms)", elapsed_ms)
            return None

        raw_response = result.get("response", "")
        if not raw_response:
            self._fail_count += 1
            log.warning("IML parse: empty response from LLM (%.0fms)", elapsed_ms)
            return None

        parsed = _parse_llm_response(raw_response)
        if parsed is None:
            self._fail_count += 1
            log.warning(
                "IML parse: JSON extraction failed (%.0fms). Raw: %s",
                elapsed_ms,
                raw_response[:200],
            )
            return None

        graph = _graph_from_parsed(parsed, raw_input, user_handle, channel)

        self._parse_count += 1
        self._total_latency_ms += elapsed_ms

        log.info(
            "IML parse OK: %d nodes, %d edges, %d ambiguities (%.0fms)",
            len(graph.nodes),
            len(graph.edges),
            len(graph.ambiguity_ledger),
            elapsed_ms,
        )

        return graph
