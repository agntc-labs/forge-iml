"""Tests for forge_iml.schema -- pure dataclass validation."""

import json

from forge_iml.schema import (
    Ambiguity,
    Budget,
    Edge,
    EdgeRelation,
    IntentGraph,
    Node,
    NodeType,
    PipelineResult,
    Source,
    StepResult,
)


def test_node_creation():
    node = Node(id="n1", type=NodeType.USER_GOAL, content="get weather")
    assert node.id == "n1"
    assert node.type == NodeType.USER_GOAL
    assert node.confidence == 1.0
    assert node.metadata == {}


def test_node_round_trip():
    node = Node(id="n1", type=NodeType.PLAN_STEP, content="call:weather",
                confidence=0.9, metadata={"tool": "weather"})
    d = node.to_dict()
    restored = Node.from_dict(d)
    assert restored.id == node.id
    assert restored.type == node.type
    assert restored.confidence == node.confidence
    assert restored.metadata == node.metadata


def test_edge_round_trip():
    edge = Edge(source="a", target="b", relation=EdgeRelation.DEPENDS_ON)
    d = edge.to_dict()
    assert d["from"] == "a"
    assert d["to"] == "b"
    restored = Edge.from_dict(d)
    assert restored.source == "a"
    assert restored.target == "b"
    assert restored.relation == EdgeRelation.DEPENDS_ON


def test_graph_add_nodes_and_edges():
    g = IntentGraph()
    goal = g.add_node(NodeType.USER_GOAL, "get weather", node_id="g1")
    step = g.add_node(NodeType.PLAN_STEP, "call:weather", node_id="s1")
    g.add_edge("g1", "s1", EdgeRelation.PRODUCES)

    assert len(g.nodes) == 2
    assert len(g.edges) == 1
    assert g.get_goals() == [goal]
    assert g.get_plan_steps() == [step]
    assert g.edges_from("g1")[0].target == "s1"


def test_graph_serialization():
    g = IntentGraph(source=Source(user_handle="test", channel="api"))
    g.add_node(NodeType.USER_GOAL, "hello world", node_id="g1")
    g.ambiguity_ledger.append(Ambiguity(question="which?", why_it_matters="needed"))

    d = g.to_dict()
    j = json.loads(json.dumps(d))
    restored = IntentGraph.from_dict(j)

    assert len(restored.nodes) == 1
    assert restored.nodes[0].content == "hello world"
    assert restored.source.user_handle == "test"
    assert len(restored.ambiguity_ledger) == 1


def test_budget_presets():
    simple = Budget.for_simple()
    assert simple.planner_max_model_calls == 0
    assert simple.planner_max_plan_steps == 2

    complex_ = Budget.for_complex()
    assert complex_.planner_max_model_calls == 1
    assert complex_.planner_max_plan_steps == 8

    mem = Budget.for_memory_only()
    assert mem.planner_max_plan_steps == 0
    assert mem.retrieval_max_claims == 15


def test_pipeline_result():
    pr = PipelineResult(graph_id="test_1")
    pr.step_results.append(StepResult(step_id="s1", tool="weather", success=True,
                                       output={"temp": 85}))
    d = pr.to_dict()
    assert d["graph_id"] == "test_1"
    assert len(d["step_results"]) == 1
    assert d["step_results"][0]["success"] is True
