"""
IML Executor -- Stage D of the pipeline.

Walks the execution DAG produced by the planner, enforces policy gates,
executes tools via the ToolProvider, handles failures with bounded retry.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from forge_iml.schema import (
    EdgeRelation,
    IntentGraph,
    Node,
    NodeType,
    PipelineResult,
    StepResult,
)
from forge_iml.providers.base import ToolProvider

log = logging.getLogger("forge_iml.executor")


# -- Execution Context --------------------------------------------------------

@dataclass
class ExecutionContext:
    user_handle: str = ""
    channel: str = "api"
    user_allowed_tools: set = field(default_factory=set)
    step_results: dict = field(default_factory=dict)  # step_id -> StepResult


# -- Variable interpolation ---------------------------------------------------

_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _interpolate(value: Any, step_results: dict[str, StepResult]) -> Any:
    """Replace ${step_id.output} references with actual outputs."""
    if not isinstance(value, str):
        return value

    def _replace(m: re.Match) -> str:
        ref = m.group(1)
        parts = ref.split(".", 1)
        step_id = parts[0]
        result = step_results.get(step_id)
        if result is None or not result.success:
            return m.group(0)

        output = result.output
        if len(parts) > 1:
            path = parts[1]
            if path.startswith("output."):
                path = path[7:]
            elif path == "output":
                return str(output)
            for key in path.split("."):
                if isinstance(output, dict):
                    output = output.get(key, "")
                else:
                    break
        return str(output) if output is not None else ""

    return _VAR_PATTERN.sub(_replace, value)


def _interpolate_inputs(inputs: dict, step_results: dict[str, StepResult]) -> dict:
    result = {}
    for k, v in inputs.items():
        if isinstance(v, str):
            result[k] = _interpolate(v, step_results)
        elif isinstance(v, dict):
            result[k] = _interpolate_inputs(v, step_results)
        elif isinstance(v, list):
            result[k] = [
                _interpolate(item, step_results) if isinstance(item, str) else item
                for item in v
            ]
        else:
            result[k] = v
    return result


# -- Executor -----------------------------------------------------------------

class IMLExecutor:
    """Stage D: walk the execution DAG, run tools, enforce policy.

    Args:
        tool_provider: The ToolProvider to execute tools through.
        policy_engine: Optional PolicyEngine instance. If not provided
            no policy checks are performed.
    """

    def __init__(
        self,
        tool_provider: ToolProvider,
        policy_engine=None,
    ):
        self._tool_provider = tool_provider
        self._policy_engine = policy_engine
        self._exec_count = 0

    # -- Main entry -----------------------------------------------------------

    async def execute(
        self,
        graph: IntentGraph,
        user_handle: str = "",
        channel: str = "api",
        user_allowed_tools: Optional[set] = None,
    ) -> PipelineResult:
        """Execute all PlanStep nodes in topological order.

        Returns a PipelineResult with step results and final output.
        """
        t0 = time.monotonic()
        result = PipelineResult(graph_id=graph.id)

        plan_steps = graph.get_plan_steps()
        if not plan_steps:
            result.status = "success"
            result.total_duration_ms = (time.monotonic() - t0) * 1000
            return result

        ctx = ExecutionContext(
            user_handle=user_handle,
            channel=channel,
            user_allowed_tools=user_allowed_tools or set(),
        )

        order = self._topological_sort(graph, plan_steps)
        if order is None:
            result.status = "failed"
            result.error = "Cycle detected in execution DAG"
            result.total_duration_ms = (time.monotonic() - t0) * 1000
            return result

        waves = self._build_waves(order, plan_steps)
        max_retries = graph.budget.planner_max_tool_loops

        for wave in waves:
            tasks = []
            for step_node in wave:
                gate_ok, gate_result = self._check_policy(step_node, ctx, result)
                if not gate_ok:
                    if gate_result == "block":
                        sr = StepResult(
                            step_id=step_node.id,
                            tool=step_node.metadata.get("tool", ""),
                            success=False,
                            error="Blocked by policy gate",
                        )
                        ctx.step_results[step_node.id] = sr
                        result.step_results.append(sr)
                        continue
                    elif gate_result == "needs_confirmation":
                        result.status = "needs_confirmation"
                        sr = StepResult(
                            step_id=step_node.id,
                            tool=step_node.metadata.get("tool", ""),
                            success=False,
                            error="Requires user confirmation",
                        )
                        ctx.step_results[step_node.id] = sr
                        result.step_results.append(sr)
                        result.total_duration_ms = (time.monotonic() - t0) * 1000
                        return result

                tasks.append(self._execute_step(step_node, ctx, max_retries))

            if tasks:
                step_results = await asyncio.gather(*tasks, return_exceptions=True)
                for sr in step_results:
                    if isinstance(sr, Exception):
                        log.error("Step execution raised exception: %s", sr)
                        continue
                    ctx.step_results[sr.step_id] = sr
                    result.step_results.append(sr)
                    result.total_tokens += sr.tokens_used

        self._mark_blocked_downstream(graph, plan_steps, ctx, result)
        result.status = self._compute_status(result)

        for sr in reversed(result.step_results):
            if sr.success and sr.output is not None:
                result.final_output = sr.output
                break

        result.total_duration_ms = (time.monotonic() - t0) * 1000
        self._exec_count += 1
        log.info(
            "Executed %s: %d steps, status=%s, %.1fms",
            graph.id, len(result.step_results), result.status,
            result.total_duration_ms,
        )

        return result

    # -- Topological Sort -----------------------------------------------------

    def _topological_sort(self, graph: IntentGraph,
                          plan_steps: list[Node]) -> Optional[list[str]]:
        step_ids = {s.id for s in plan_steps}

        in_degree: dict[str, int] = {sid: 0 for sid in step_ids}
        dependents: dict[str, list[str]] = {sid: [] for sid in step_ids}

        for step in plan_steps:
            deps = step.metadata.get("depends_on", [])
            for dep_id in deps:
                if dep_id in step_ids:
                    in_degree[step.id] += 1
                    dependents[dep_id].append(step.id)

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            queue.sort()
            current = queue.pop(0)
            order.append(current)
            for dep in dependents.get(current, []):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        if len(order) != len(step_ids):
            return None
        return order

    # -- Wave Builder ---------------------------------------------------------

    def _build_waves(self, order: list[str],
                     plan_steps: list[Node]) -> list[list[Node]]:
        step_map = {s.id: s for s in plan_steps}
        completed: set[str] = set()
        waves: list[list[Node]] = []

        remaining = list(order)
        while remaining:
            wave = []
            wave_ids = set()
            next_remaining = []

            for sid in remaining:
                step = step_map[sid]
                deps = set(step.metadata.get("depends_on", []))
                if deps.issubset(completed):
                    wave.append(step)
                    wave_ids.add(sid)
                else:
                    next_remaining.append(sid)

            if not wave:
                for sid in next_remaining:
                    waves.append([step_map[sid]])
                break

            waves.append(wave)
            completed.update(wave_ids)
            remaining = next_remaining

        return waves

    # -- Step Execution -------------------------------------------------------

    async def _execute_step(self, step_node: Node, ctx: ExecutionContext,
                            max_retries: int) -> StepResult:
        tool_name = step_node.metadata.get("tool", "")
        raw_inputs = step_node.metadata.get("inputs", {})
        inputs = _interpolate_inputs(raw_inputs, ctx.step_results)

        retries = 0
        last_error = None

        while retries <= max_retries:
            t0 = time.monotonic()
            try:
                tool_result = await self._tool_provider.execute(tool_name, inputs)
                duration = (time.monotonic() - t0) * 1000
                success = tool_result.get("success", True)
                error_msg = tool_result.get("error")

                if success or error_msg is None:
                    return StepResult(
                        step_id=step_node.id,
                        tool=tool_name,
                        success=True,
                        output=tool_result.get("result", tool_result.get("output", tool_result)),
                        duration_ms=duration,
                        tokens_used=0,
                        retries=retries,
                    )
                else:
                    last_error = error_msg
            except Exception as e:
                last_error = str(e)

            retries += 1
            if retries <= max_retries:
                await asyncio.sleep(1.0)

        return StepResult(
            step_id=step_node.id,
            tool=tool_name,
            success=False,
            error=last_error or "Unknown error",
            retries=retries - 1,
        )

    # -- Policy Gate ----------------------------------------------------------

    def _check_policy(self, step_node: Node, ctx: ExecutionContext,
                      result: PipelineResult) -> tuple[bool, str]:
        tool_name = step_node.metadata.get("tool", "")
        result.policy_gates_checked.append(f"{step_node.id}:{tool_name}")

        if ctx.user_allowed_tools and tool_name not in ctx.user_allowed_tools:
            result.policy_gates_triggered.append(
                f"{step_node.id}:{tool_name}:user_tool_gate"
            )
            return False, "block"

        if self._policy_engine is not None:
            try:
                allowed, gate_name, message = self._policy_engine.check_step(step_node, {
                    "user_handle": ctx.user_handle,
                    "channel": ctx.channel,
                    "step_results": {
                        k: v.to_dict() for k, v in ctx.step_results.items()
                    },
                })
                if not allowed:
                    result.policy_gates_triggered.append(
                        f"{step_node.id}:{tool_name}:{gate_name}"
                    )
                    action = "block"
                    if "confirm" in (gate_name or "").lower() or "confirm" in (message or "").lower():
                        action = "needs_confirmation"
                    return False, action
            except Exception as e:
                log.warning("Policy check failed: %s", e)

        return True, "allow"

    # -- Downstream Blocking --------------------------------------------------

    def _mark_blocked_downstream(self, graph: IntentGraph,
                                  plan_steps: list[Node],
                                  ctx: ExecutionContext,
                                  result: PipelineResult) -> None:
        failed_ids = {sr.step_id for sr in result.step_results if not sr.success}
        if not failed_ids:
            return

        for step in plan_steps:
            if step.id in ctx.step_results:
                continue
            deps = set(step.metadata.get("depends_on", []))
            if deps & failed_ids:
                sr = StepResult(
                    step_id=step.id,
                    tool=step.metadata.get("tool", ""),
                    success=False,
                    error=f"Blocked: dependency {deps & failed_ids} failed",
                )
                ctx.step_results[step.id] = sr
                result.step_results.append(sr)
                failed_ids.add(step.id)

    # -- Status Computation ---------------------------------------------------

    def _compute_status(self, result: PipelineResult) -> str:
        if not result.step_results:
            return "success"

        all_success = all(sr.success for sr in result.step_results)
        any_success = any(sr.success for sr in result.step_results)
        any_blocked = any("Blocked" in (sr.error or "") for sr in result.step_results)

        if result.status == "needs_confirmation":
            return "needs_confirmation"
        if all_success:
            return "success"
        if any_blocked and not any_success:
            return "blocked"
        if any_success:
            return "success"
        return "failed"

    @property
    def exec_count(self) -> int:
        return self._exec_count
