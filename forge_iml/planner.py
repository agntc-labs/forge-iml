"""
IML Planner -- Stage C of the pipeline.

Takes an IntentGraph (with UserGoal nodes from the parser) and produces
an optimized execution plan: PlanStep nodes wired into a dependency DAG.

Fast path: deterministic single-tool mapping (no LLM).
Slow path: LLM-assisted multi-step planning.
"""

import asyncio
import json
import logging
import re
import time
from typing import Optional

from forge_iml.schema import (
    Budget,
    Edge,
    EdgeRelation,
    IntentGraph,
    Node,
    NodeType,
)
from forge_iml.providers.base import LLMProvider, ToolProvider

log = logging.getLogger("forge_iml.planner")


# -- Known single-tool patterns (fast path) -----------------------------------

KNOWN_PATTERNS: dict[str, dict] = {
    "get_weather":     {"tool": "weather",         "input_fields": ["location"]},
    "check_weather":   {"tool": "weather",         "input_fields": ["location"]},
    "get_time":        {"tool": "system_info",     "input_fields": []},
    "get_balance":     {"tool": "mercury_balance", "input_fields": []},
    "check_balance":   {"tool": "mercury_balance", "input_fields": []},
    "send_message":    {"tool": "imessage_send",   "input_fields": ["recipient", "message"]},
    "send_imessage":   {"tool": "imessage_send",   "input_fields": ["recipient", "message"]},
    "web_search":      {"tool": "web_search",      "input_fields": ["query"]},
    "search_web":      {"tool": "web_search",      "input_fields": ["query"]},
    "get_calendar":    {"tool": "calendar",        "input_fields": []},
    "check_calendar":  {"tool": "calendar",        "input_fields": []},
    "disk_usage":      {"tool": "disk_usage",      "input_fields": []},
    "system_info":     {"tool": "system_info",     "input_fields": []},
    "process_list":    {"tool": "process_list",    "input_fields": []},
    "screen_capture":  {"tool": "screen_capture",  "input_fields": []},
    "take_screenshot": {"tool": "screen_capture",  "input_fields": []},
    "generate_image":  {"tool": "generate_image",  "input_fields": ["prompt"]},
    "create_image":    {"tool": "generate_image",  "input_fields": ["prompt"]},
    "read_file":       {"tool": "read_file",       "input_fields": ["path"]},
    "list_files":      {"tool": "list_directory",  "input_fields": ["path"]},
    "memory_search":   {"tool": "memory_search",   "input_fields": ["query"]},
    "recall":          {"tool": "memory_search",   "input_fields": ["query"]},
}


# -- Planner ------------------------------------------------------------------

class IMLPlanner:
    """Stage C: IntentGraph -> IntentGraph with PlanStep nodes + DAG edges.

    Args:
        llm: LLM provider used for slow-path multi-step planning.
        tool_provider: Optional ToolProvider for validating tool names.
        known_patterns: Override the default fast-path pattern map.
    """

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        tool_provider: Optional[ToolProvider] = None,
        known_patterns: Optional[dict[str, dict]] = None,
    ):
        self._llm = llm
        self._tool_provider = tool_provider
        self._known_patterns = known_patterns if known_patterns is not None else KNOWN_PATTERNS
        self._plan_count = 0
        self._tool_names: list[str] = []
        if tool_provider:
            try:
                self._tool_names = [t["name"] for t in tool_provider.list_tools()]
            except Exception:
                pass

    # -- Main entry -----------------------------------------------------------

    async def plan(self, graph: IntentGraph) -> IntentGraph:
        """Produce an execution plan for the intent graph.

        Adds PlanStep nodes and dependency/parallel edges.
        Returns the same graph, modified in place.
        """
        t0 = time.monotonic()

        if graph.get_plan_steps():
            log.debug("Graph %s already has plan steps, skipping", graph.id)
            return graph

        goals = graph.get_goals()
        if not goals:
            log.warning("Graph %s has no goals -- nothing to plan", graph.id)
            return graph

        if graph.budget.planner_max_plan_steps == 0:
            log.debug("Graph %s budget allows 0 plan steps -- memory only", graph.id)
            return graph

        # Fast path
        if len(goals) == 1:
            matched = self._try_fast_path(graph, goals[0])
            if matched:
                self._plan_count += 1
                dt = (time.monotonic() - t0) * 1000
                log.info("Planned %s via fast path in %.1fms", graph.id, dt)
                return graph

        # Slow path: LLM-assisted planning
        if graph.budget.planner_max_model_calls > 0 and self._llm is not None:
            await self._slow_path(graph, goals)
            self._plan_count += 1
            dt = (time.monotonic() - t0) * 1000
            log.info("Planned %s via slow path in %.1fms", graph.id, dt)
        else:
            log.debug("Graph %s: no fast-path match and no LLM, no plan", graph.id)

        return graph

    # -- Fast Path ------------------------------------------------------------

    def _try_fast_path(self, graph: IntentGraph, goal: Node) -> bool:
        verb = goal.metadata.get("verb", "")
        obj = goal.metadata.get("object", "")
        entities = goal.metadata.get("entities", {})

        key = f"{verb}_{obj}".lower().replace(" ", "_")

        pattern = self._known_patterns.get(key)
        if pattern is None:
            pattern = self._known_patterns.get(verb.lower().replace(" ", "_"))
        if pattern is None:
            return False

        tool_name = pattern["tool"]
        if self._tool_names and tool_name not in self._tool_names:
            log.warning("Fast-path tool %s not in registry, skipping", tool_name)
            return False

        inputs = {}
        for field_name in pattern["input_fields"]:
            val = entities.get(field_name) or goal.metadata.get(field_name, "")
            if val:
                inputs[field_name] = val
            elif field_name == "query":
                inputs["query"] = goal.content
            elif field_name == "prompt":
                inputs["prompt"] = goal.content

        step_id = "ps1"
        graph.add_node(
            NodeType.PLAN_STEP,
            f"Execute {tool_name}",
            confidence=goal.confidence,
            metadata={
                "tool": tool_name,
                "inputs": inputs,
                "expected_output": goal.metadata.get("expected_output", ""),
                "stop_condition": "tool_returns",
                "depends_on": [],
                "parallel_with": [],
                "fast_path": True,
            },
            node_id=step_id,
        )

        perm_id = "tp1"
        graph.add_node(
            NodeType.TOOL_PERMISSION,
            f"Permission to use {tool_name}",
            confidence=1.0,
            metadata={"tool": tool_name, "safety": "read"},
            node_id=perm_id,
        )

        graph.add_edge(goal.id, step_id, EdgeRelation.PRODUCES)
        graph.add_edge(step_id, perm_id, EdgeRelation.NEEDS_TOOL)

        return True

    # -- Slow Path (LLM) -----------------------------------------------------

    async def _slow_path(self, graph: IntentGraph, goals: list[Node]) -> None:
        goals_desc = []
        for g in goals:
            goals_desc.append({
                "id": g.id,
                "content": g.content,
                "verb": g.metadata.get("verb", ""),
                "object": g.metadata.get("object", ""),
                "entities": g.metadata.get("entities", {}),
            })

        available = self._tool_names[:60] if self._tool_names else list(self._known_patterns.keys())

        constraints = [
            n.content for n in graph.get_nodes_by_type(NodeType.CONSTRAINT)
        ]
        risk_gates = [
            {"id": n.id, "content": n.content, "metadata": n.metadata}
            for n in graph.get_risk_gates()
        ]

        prompt = self._build_planner_prompt(goals_desc, available, constraints,
                                            risk_gates, graph.budget)

        messages = [{"role": "user", "content": prompt}]

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._llm.complete(
                messages,
                temperature=0.1,
                max_tokens=2048,
            ),
        )

        if not response:
            log.error("LLM returned empty response during planning")
            return

        text = response.get("response", "")
        plan_data = self._parse_plan_json(text)
        if not plan_data:
            log.error("Failed to parse plan JSON from LLM response")
            return

        self._apply_plan(graph, goals, plan_data)

    def _build_planner_prompt(
        self,
        goals: list[dict],
        tools: list[str],
        constraints: list[str],
        risk_gates: list[dict],
        budget: Budget,
    ) -> str:
        return f"""You are an execution planner. Given user goals and available tools,
produce a JSON execution plan.

GOALS:
{json.dumps(goals, indent=2)}

AVAILABLE TOOLS (use exact names):
{json.dumps(tools)}

CONSTRAINTS:
{json.dumps(constraints)}

RISK GATES:
{json.dumps(risk_gates, indent=2)}

BUDGET:
- Max steps: {budget.planner_max_plan_steps}
- Max retries per step: {budget.planner_max_tool_loops}

Return ONLY a JSON object with this schema:
{{
  "steps": [
    {{
      "id": "ps1",
      "tool": "tool_name",
      "inputs": {{"key": "value"}},
      "description": "what this step does",
      "depends_on": [],
      "parallel_with": []
    }}
  ]
}}

Rules:
- Use the fewest steps possible
- Steps with no data dependency should list each other in parallel_with
- depends_on lists step IDs that must complete first
- inputs can reference prior step outputs as ${{step_id.output}}
- Every tool must be from the AVAILABLE TOOLS list
- Max {budget.planner_max_plan_steps} steps"""

    def _parse_plan_json(self, text: str) -> Optional[list[dict]]:
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "steps" in data:
                return data["steps"]
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                if isinstance(data, dict) and "steps" in data:
                    return data["steps"]
            except json.JSONDecodeError:
                pass

        m = re.search(r"\{[^{}]*\"steps\"\s*:\s*\[.*?\]\s*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                return data.get("steps", [])
            except json.JSONDecodeError:
                pass

        return None

    def _apply_plan(self, graph: IntentGraph, goals: list[Node],
                    steps: list[dict]) -> None:
        max_steps = graph.budget.planner_max_plan_steps
        steps = steps[:max_steps]

        step_ids = []
        for i, step in enumerate(steps):
            sid = step.get("id", f"ps{i+1}")
            tool = step.get("tool", "")
            inputs = step.get("inputs", {})
            description = step.get("description", f"Execute {tool}")
            depends = step.get("depends_on", [])
            parallel = step.get("parallel_with", [])

            if self._tool_names and tool not in self._tool_names:
                log.warning("LLM suggested unknown tool %s, skipping step %s", tool, sid)
                continue

            graph.add_node(
                NodeType.PLAN_STEP,
                description,
                confidence=0.9,
                metadata={
                    "tool": tool,
                    "inputs": inputs,
                    "expected_output": "",
                    "stop_condition": "tool_returns",
                    "depends_on": depends,
                    "parallel_with": parallel,
                    "fast_path": False,
                },
                node_id=sid,
            )
            step_ids.append(sid)

            perm_id = f"tp{i+1}"
            graph.add_node(
                NodeType.TOOL_PERMISSION,
                f"Permission to use {tool}",
                confidence=1.0,
                metadata={"tool": tool},
                node_id=perm_id,
            )
            graph.add_edge(sid, perm_id, EdgeRelation.NEEDS_TOOL)

        for step in steps:
            sid = step.get("id", "")
            if sid not in step_ids:
                continue
            for dep_id in step.get("depends_on", []):
                if dep_id in step_ids:
                    graph.add_edge(sid, dep_id, EdgeRelation.DEPENDS_ON)
            for par_id in step.get("parallel_with", []):
                if par_id in step_ids:
                    graph.add_edge(sid, par_id, EdgeRelation.PARALLEL_WITH)

        roots = [s for s in step_ids if not steps[step_ids.index(s)].get("depends_on")]
        for goal in goals:
            for root_id in roots:
                graph.add_edge(goal.id, root_id, EdgeRelation.PRODUCES)

    # -- DAG optimization -----------------------------------------------------

    def optimize_parallelism(self, graph: IntentGraph) -> None:
        """Post-pass: find steps with no dependency and mark as parallel."""
        plan_steps = graph.get_plan_steps()
        if len(plan_steps) < 2:
            return

        dep_map: dict[str, set[str]] = {}
        for step in plan_steps:
            deps = set(step.metadata.get("depends_on", []))
            dep_map[step.id] = deps

        for i, s1 in enumerate(plan_steps):
            for s2 in plan_steps[i + 1:]:
                if s2.id in dep_map.get(s1.id, set()):
                    continue
                if s1.id in dep_map.get(s2.id, set()):
                    continue
                if self._has_transitive_dep(s1.id, s2.id, dep_map):
                    continue
                if self._has_transitive_dep(s2.id, s1.id, dep_map):
                    continue

                existing = graph.parallel_with(s1.id)
                if s2.id not in existing:
                    graph.add_edge(s1.id, s2.id, EdgeRelation.PARALLEL_WITH)
                    s1.metadata.setdefault("parallel_with", [])
                    if s2.id not in s1.metadata["parallel_with"]:
                        s1.metadata["parallel_with"].append(s2.id)

    def _has_transitive_dep(self, start: str, target: str,
                            dep_map: dict[str, set[str]]) -> bool:
        visited: set[str] = set()
        queue = [start]
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            deps = dep_map.get(current, set())
            if target in deps:
                return True
            queue.extend(deps)
        return False

    @property
    def plan_count(self) -> int:
        return self._plan_count
