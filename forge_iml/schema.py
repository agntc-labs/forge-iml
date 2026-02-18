"""
IML Schema — Internal Machine Language type definitions.

Defines the Intent Graph: the intermediate representation between
raw human input and deterministic tool execution.

All IML modules import from here.
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# -- Node Types ---------------------------------------------------------------

class NodeType(str, Enum):
    USER_GOAL = "UserGoal"
    CONSTRAINT = "Constraint"
    PREFERENCE = "Preference"
    INPUT = "Input"
    ARTIFACT = "Artifact"
    OUTPUT_SPEC = "OutputSpec"
    RISK_GATE = "RiskGate"
    TOOL_PERMISSION = "ToolPermission"
    ASSUMPTION = "Assumption"
    QUESTION = "Question"
    SUBTASK = "Subtask"
    PLAN_STEP = "PlanStep"
    MEMORY_CLAIM = "MemoryClaim"
    AMBIGUITY = "Ambiguity"
    CONTEXT = "Context"
    RESULT = "Result"


# -- Edge Relations -----------------------------------------------------------

class EdgeRelation(str, Enum):
    REQUIRES = "requires"
    PREFERS = "prefers"
    NEEDS_TOOL = "needs_tool"
    BLOCKED_BY = "blocked_by"
    DEPENDS_ON = "depends_on"
    PARALLEL_WITH = "parallel_with"
    INFORMED_BY = "informed_by"
    RESOLVES = "resolves"
    PRODUCES = "produces"
    CONSTRAINS = "constrains"


# -- Policy Actions -----------------------------------------------------------

class PolicyAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_CONFIRMATION = "require_confirmation"


# -- Core Dataclasses ---------------------------------------------------------

@dataclass
class Node:
    id: str
    type: NodeType
    content: str
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(
            id=d["id"],
            type=NodeType(d["type"]),
            content=d["content"],
            confidence=d.get("confidence", 1.0),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Edge:
    source: str  # from node id
    target: str  # to node id
    relation: EdgeRelation

    def to_dict(self) -> dict:
        return {
            "from": self.source,
            "to": self.target,
            "relation": self.relation.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(
            source=d["from"],
            target=d["to"],
            relation=EdgeRelation(d["relation"]),
        )


@dataclass
class Ambiguity:
    question: str
    why_it_matters: str
    allowed_defaults: list[str] = field(default_factory=list)
    can_proceed_without: bool = False
    default_used: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "why_it_matters": self.why_it_matters,
            "allowed_defaults": self.allowed_defaults,
            "can_proceed_without": self.can_proceed_without,
            "default_used": self.default_used,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Ambiguity":
        return cls(
            question=d["question"],
            why_it_matters=d["why_it_matters"],
            allowed_defaults=d.get("allowed_defaults", []),
            can_proceed_without=d.get("can_proceed_without", False),
            default_used=d.get("default_used"),
            confidence=d.get("confidence", 0.0),
        )


@dataclass
class Budget:
    """Token and call budgets for each pipeline stage."""
    normalizer_max_tokens: int = 500
    normalizer_max_passes: int = 1
    planner_max_model_calls: int = 1
    planner_max_plan_steps: int = 5
    planner_max_tool_loops: int = 2
    retrieval_max_memory_tokens: int = 400
    retrieval_max_claims: int = 5
    retrieval_require_provenance: bool = False

    def to_dict(self) -> dict:
        return {
            "normalizer": {
                "max_tokens": self.normalizer_max_tokens,
                "max_passes": self.normalizer_max_passes,
            },
            "planner": {
                "max_model_calls": self.planner_max_model_calls,
                "max_plan_steps": self.planner_max_plan_steps,
                "max_tool_loops": self.planner_max_tool_loops,
            },
            "retrieval": {
                "max_memory_tokens": self.retrieval_max_memory_tokens,
                "max_claims": self.retrieval_max_claims,
                "require_provenance": self.retrieval_require_provenance,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Budget":
        n = d.get("normalizer", {})
        p = d.get("planner", {})
        r = d.get("retrieval", {})
        return cls(
            normalizer_max_tokens=n.get("max_tokens", 500),
            normalizer_max_passes=n.get("max_passes", 1),
            planner_max_model_calls=p.get("max_model_calls", 1),
            planner_max_plan_steps=p.get("max_plan_steps", 5),
            planner_max_tool_loops=p.get("max_tool_loops", 2),
            retrieval_max_memory_tokens=r.get("max_memory_tokens", 400),
            retrieval_max_claims=r.get("max_claims", 5),
            retrieval_require_provenance=r.get("require_provenance", False),
        )

    @classmethod
    def for_simple(cls) -> "Budget":
        """Budget for simple single-tool requests."""
        return cls(
            normalizer_max_tokens=300,
            planner_max_model_calls=0,
            planner_max_plan_steps=2,
            planner_max_tool_loops=1,
            retrieval_max_memory_tokens=200,
            retrieval_max_claims=3,
        )

    @classmethod
    def for_complex(cls) -> "Budget":
        """Budget for multi-step tool workflows."""
        return cls(
            normalizer_max_tokens=800,
            planner_max_model_calls=1,
            planner_max_plan_steps=8,
            planner_max_tool_loops=3,
            retrieval_max_memory_tokens=600,
            retrieval_max_claims=10,
            retrieval_require_provenance=True,
        )

    @classmethod
    def for_memory_only(cls) -> "Budget":
        """Budget for recall-only requests (no tools)."""
        return cls(
            normalizer_max_tokens=400,
            planner_max_model_calls=0,
            planner_max_plan_steps=0,
            planner_max_tool_loops=0,
            retrieval_max_memory_tokens=800,
            retrieval_max_claims=15,
            retrieval_require_provenance=True,
        )


@dataclass
class Source:
    """Origin metadata for the raw input."""
    user_handle: str = ""
    channel: str = "api"
    language: str = "en"
    raw_input: str = ""

    def to_dict(self) -> dict:
        return {
            "user_handle": self.user_handle,
            "channel": self.channel,
            "language": self.language,
            "raw_input": self.raw_input,
        }


@dataclass
class IntentGraph:
    """The core IML representation -- a typed DAG of intent."""
    version: str = "1.0.0"
    id: str = field(default_factory=lambda: f"ig_{uuid.uuid4().hex[:12]}")
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: Source = field(default_factory=Source)
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    ambiguity_ledger: list[Ambiguity] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    cache_key: Optional[str] = None

    # -- Node helpers ---------------------------------------------------------

    def add_node(self, type=None, content: str = "",
                 confidence: float = 1.0, metadata: dict = None,
                 node_id: str = None) -> Node:
        """Add a node and return it. Accepts either (NodeType, content, ...) or a pre-built Node."""
        if isinstance(type, Node):
            self.nodes.append(type)
            return type
        nid = node_id or f"{type.value[0].lower()}{len(self.nodes)+1}"
        node = Node(id=nid, type=type, content=content,
                    confidence=confidence, metadata=metadata or {})
        self.nodes.append(node)
        return node

    def add_edge(self, source: str, target: str, relation: EdgeRelation) -> Edge:
        """Add an edge and return it."""
        edge = Edge(source=source, target=target, relation=relation)
        self.edges.append(edge)
        return edge

    def get_nodes_by_type(self, type: NodeType) -> list[Node]:
        return [n for n in self.nodes if n.type == type]

    def get_plan_steps(self) -> list[Node]:
        return self.get_nodes_by_type(NodeType.PLAN_STEP)

    def get_goals(self) -> list[Node]:
        return self.get_nodes_by_type(NodeType.USER_GOAL)

    def get_memory_claims(self) -> list[Node]:
        return self.get_nodes_by_type(NodeType.MEMORY_CLAIM)

    def get_risk_gates(self) -> list[Node]:
        return self.get_nodes_by_type(NodeType.RISK_GATE)

    def has_blocking_ambiguity(self) -> bool:
        """True if any ambiguity prevents proceeding."""
        return any(
            not a.can_proceed_without and a.default_used is None
            for a in self.ambiguity_ledger
        )

    def edges_from(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source == node_id]

    def edges_to(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.target == node_id]

    def dependencies_of(self, node_id: str) -> list[str]:
        """Get IDs of nodes this node depends on."""
        return [
            e.target for e in self.edges
            if e.source == node_id and e.relation == EdgeRelation.DEPENDS_ON
        ]

    def parallel_with(self, node_id: str) -> list[str]:
        """Get IDs of nodes that can run in parallel with this one."""
        return [
            e.target for e in self.edges
            if e.source == node_id and e.relation == EdgeRelation.PARALLEL_WITH
        ]

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "id": self.id,
            "timestamp": self.timestamp,
            "source": self.source.to_dict(),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "ambiguity_ledger": [a.to_dict() for a in self.ambiguity_ledger],
            "budget": self.budget.to_dict(),
            "cache_key": self.cache_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IntentGraph":
        graph = cls(
            version=d.get("version", "1.0.0"),
            id=d.get("id", ""),
            timestamp=d.get("timestamp", ""),
            source=Source(**d.get("source", {})) if d.get("source") else Source(),
            budget=Budget.from_dict(d.get("budget", {})),
            cache_key=d.get("cache_key"),
        )
        for nd in d.get("nodes", []):
            graph.nodes.append(Node.from_dict(nd))
        for ed in d.get("edges", []):
            graph.edges.append(Edge.from_dict(ed))
        for ad in d.get("ambiguity_ledger", []):
            graph.ambiguity_ledger.append(Ambiguity.from_dict(ad))
        return graph

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2)


# -- Execution Result ---------------------------------------------------------

@dataclass
class StepResult:
    """Result from executing a single PlanStep."""
    step_id: str
    tool: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    tokens_used: int = 0
    retries: int = 0

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "tool": self.tool,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "retries": self.retries,
        }


@dataclass
class PipelineResult:
    """Full result from the IML pipeline."""
    graph_id: str
    cache_hit: bool = False
    cache_layer: Optional[str] = None
    step_results: list[StepResult] = field(default_factory=list)
    final_output: Any = None
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    policy_gates_checked: list[str] = field(default_factory=list)
    policy_gates_triggered: list[str] = field(default_factory=list)
    status: str = "success"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "graph_id": self.graph_id,
            "cache_hit": self.cache_hit,
            "cache_layer": self.cache_layer,
            "step_results": [s.to_dict() for s in self.step_results],
            "final_output": self.final_output,
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "policy_gates_checked": self.policy_gates_checked,
            "policy_gates_triggered": self.policy_gates_triggered,
            "status": self.status,
            "error": self.error,
        }
