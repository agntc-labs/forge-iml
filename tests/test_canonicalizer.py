"""Tests for forge_iml.canonicalizer -- pure Python normalization."""

from forge_iml.schema import (
    EdgeRelation,
    IntentGraph,
    Node,
    NodeType,
    Source,
)
from forge_iml.canonicalizer import IMLCanonicalizer, _normalize_verb


def test_verb_normalization():
    assert _normalize_verb("fetch") == "get"
    assert _normalize_verb("retrieve") == "get"
    assert _normalize_verb("tell") == "get"
    assert _normalize_verb("text") == "send"
    assert _normalize_verb("make") == "create"
    assert _normalize_verb("remove") == "delete"
    assert _normalize_verb("google") == "search"
    assert _normalize_verb("get") == "get"
    assert _normalize_verb("unknown_verb") == "unknown_verb"


def test_canonicalize_normalizes_verb():
    g = IntentGraph(source=Source(user_handle="test"))
    g.add_node(NodeType.USER_GOAL, "fetch weather",
               metadata={"verb": "fetch", "object": "weather", "target": None},
               node_id="g1")
    c = IMLCanonicalizer()
    c.canonicalize(g)

    goal = g.get_goals()[0]
    assert goal.metadata["verb"] == "get"
    assert goal.content == "get weather"


def test_canonicalize_target_aliases():
    g = IntentGraph(source=Source(user_handle="test"))
    g.add_node(NodeType.USER_GOAL, "send message -> father",
               metadata={"verb": "send", "object": "message", "target": "father"},
               node_id="g1")
    c = IMLCanonicalizer(target_aliases={"dad": ["father", "papa", "justin"]})
    c.canonicalize(g)

    goal = g.get_goals()[0]
    assert goal.metadata["target"] == "dad"
    assert "dad" in goal.content


def test_canonicalize_generates_cache_key():
    g = IntentGraph(source=Source(user_handle="alice"))
    g.add_node(NodeType.USER_GOAL, "get weather",
               metadata={"verb": "get", "object": "weather", "target": None},
               node_id="g1")
    c = IMLCanonicalizer()
    c.canonicalize(g)

    assert g.cache_key is not None
    assert len(g.cache_key) == 64  # SHA256 hex


def test_cache_key_deterministic():
    """Same input should always produce the same cache key."""
    def make_graph():
        g = IntentGraph(source=Source(user_handle="bob"))
        g.add_node(NodeType.USER_GOAL, "get balance",
                   metadata={"verb": "get", "object": "balance", "target": None},
                   node_id="g1")
        return g

    c = IMLCanonicalizer()
    g1 = make_graph()
    g2 = make_graph()
    c.canonicalize(g1)
    c.canonicalize(g2)

    assert g1.cache_key == g2.cache_key


def test_constraint_sorting():
    g = IntentGraph(source=Source(user_handle="test"))
    g.add_node(NodeType.USER_GOAL, "search", node_id="g1")
    g.add_node(NodeType.CONSTRAINT, "zz_field eq 1",
               metadata={"field": "zz_field", "operator": "eq", "value": "1"},
               node_id="c1")
    g.add_node(NodeType.CONSTRAINT, "aa_field eq 2",
               metadata={"field": "aa_field", "operator": "eq", "value": "2"},
               node_id="c2")

    c = IMLCanonicalizer()
    c.canonicalize(g)

    constraints = g.get_nodes_by_type(NodeType.CONSTRAINT)
    assert constraints[0].metadata["field"] == "aa_field"
    assert constraints[1].metadata["field"] == "zz_field"


def test_tool_normalization():
    g = IntentGraph(source=Source(user_handle="test"))
    g.add_node(NodeType.USER_GOAL, "search web", node_id="g1")
    g.add_node(NodeType.TOOL_PERMISSION, "google",
               metadata={"tool": "google"}, node_id="tp1")
    g.add_node(NodeType.PLAN_STEP, "call:google",
               metadata={"tool": "google"}, node_id="ps1")

    c = IMLCanonicalizer()
    c.canonicalize(g)

    tp = g.get_nodes_by_type(NodeType.TOOL_PERMISSION)[0]
    ps = g.get_plan_steps()[0]
    assert tp.metadata["tool"] == "web_search"
    assert ps.metadata["tool"] == "web_search"
    assert ps.content == "call:web_search"
