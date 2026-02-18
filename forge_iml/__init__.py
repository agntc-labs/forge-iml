"""
forge-iml -- IML Agent Compiler

Parse human intent into typed execution graphs.

Quick start::

    from forge_iml import IMLParser, IMLCanonicalizer, IMLPlanner, IMLExecutor
    from forge_iml.providers.mlx import MLXProvider

    llm = MLXProvider(base_url="http://localhost:8081")
    parser = IMLParser(llm=llm)

    graph = await parser.parse("check the weather in Austin", user_handle="user1")
    canon = IMLCanonicalizer()
    graph = canon.canonicalize(graph)
"""

__version__ = "0.1.0"

from forge_iml.schema import (
    IntentGraph,
    Node,
    Edge,
    Budget,
    Source,
    Ambiguity,
    NodeType,

    EdgeRelation,
    PolicyAction,
    StepResult,
    PipelineResult,
)
from forge_iml.parser import IMLParser
from forge_iml.canonicalizer import IMLCanonicalizer
from forge_iml.cache import IMLCache
from forge_iml.planner import IMLPlanner
from forge_iml.executor import IMLExecutor
from forge_iml.policy import PolicyEngine, PolicyGate
from forge_iml.audit import IMLAudit
from forge_iml.memory import IMLMemory
from forge_iml.fact_extractor import IMLFactExtractor
from forge_iml.skill_learner import IMLSkillLearner
from forge_iml.providers.base import LLMProvider, ToolProvider, MemoryProvider

__all__ = [
    # Schema types
    "IntentGraph",
    "Node",
    "Edge",
    "Budget",
    "Source",
    "Ambiguity",
    "NodeType",
    "EdgeRelation",
    "PolicyAction",
    "StepResult",
    "PipelineResult",
    # Pipeline stages
    "IMLParser",
    "IMLCanonicalizer",
    "IMLCache",
    "IMLPlanner",
    "IMLExecutor",
    "PolicyEngine",
    "PolicyGate",
    "IMLAudit",
    "IMLMemory",
    "IMLFactExtractor",
    "IMLSkillLearner",
    # Providers
    "LLMProvider",
    "ToolProvider",
    "MemoryProvider",
]
