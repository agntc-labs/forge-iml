# forge-iml

**IML Agent Compiler** -- Parse human intent into typed execution graphs.

IML (Internal Machine Language) is a structured intermediate representation that sits between natural language input and deterministic tool execution. It transforms fuzzy human requests into a typed directed acyclic graph (IntentGraph) of goals, constraints, plan steps, memory claims, and policy gates -- then walks that graph to execute tools with full audit trails, caching, and safety enforcement.

## Quick Start

```python
import asyncio
from forge_iml import IMLParser, IMLCanonicalizer, IMLPlanner, IMLExecutor, PolicyEngine
from forge_iml.providers.mlx import MLXProvider

# 1. Configure an LLM provider
llm = MLXProvider(base_url="http://localhost:8081")

# 2. Build the pipeline stages
parser    = IMLParser(llm=llm)
canon     = IMLCanonicalizer()
planner   = IMLPlanner(llm=llm)
policy    = PolicyEngine()

async def run(text: str):
    # Stage A: Parse intent
    graph = await parser.parse(text, user_handle="user1")
    if graph is None:
        return "Could not parse intent"

    # Stage B: Canonicalize (pure Python, no LLM)
    graph = canon.canonicalize(graph)

    # Stage C: Plan execution steps
    graph = await planner.plan(graph)

    # Stage D: Execute (requires a ToolProvider -- see below)
    # executor = IMLExecutor(tool_provider=my_tools, policy_engine=policy)
    # result = await executor.execute(graph)

    return graph.to_json()

print(asyncio.run(run("check the weather in Austin")))
```

## Architecture

```
                        forge-iml Pipeline
 +---------------------------------------------------------------------------+
 |                                                                           |
 |  Raw Text                                                                 |
 |     |                                                                     |
 |     v                                                                     |
 |  [Stage A: Parser] -----> IntentGraph (goals, constraints, prefs)         |
 |     |                        |                                            |
 |     |   LLMProvider          |                                            |
 |     |                        v                                            |
 |  [Stage B: Canonicalizer] -> Normalized graph + cache key  (pure Python)  |
 |                              |                                            |
 |                              v                                            |
 |  [IMLCache] -- hit? ------> Return cached result                          |
 |       |                                                                   |
 |       | miss                                                              |
 |       v                                                                   |
 |  [IMLMemory] ------------> Inject MemoryClaim nodes (MemoryProvider)      |
 |       |                                                                   |
 |       v                                                                   |
 |  [Stage C: Planner] -----> PlanStep nodes + dependency DAG                |
 |       |                        |                                          |
 |       |   LLMProvider          |                                          |
 |       |                        v                                          |
 |  [PolicyEngine] ----------> Gate check (block / confirm / allow)          |
 |       |                                                                   |
 |       v                                                                   |
 |  [Stage D: Executor] ----> Run tools in topological order                 |
 |       |                        |                                          |
 |       |   ToolProvider         |                                          |
 |       |                        v                                          |
 |  [IMLAudit] -------------> Log to JSONL + MemoryProvider                  |
 |  [SkillLearner] ---------> Track patterns, promote to templates           |
 |  [FactExtractor] --------> Extract (S,P,O) triples, store in memory      |
 |                                                                           |
 +---------------------------------------------------------------------------+
```

## Provider System

forge-iml is decoupled from any specific LLM, tool registry, or memory store
through three abstract provider interfaces:

### LLMProvider

Any backend that can do text completion:

```python
from forge_iml.providers.base import LLMProvider

class MyLLM(LLMProvider):
    def complete(self, messages, *, temperature=0.1, max_tokens=1024,
                 model=None, json_schema=None):
        # Call your LLM and return {"response": "..."}
        ...
```

Built-in implementations:
- `forge_iml.providers.mlx.MLXProvider` -- local MLX server
- `forge_iml.providers.openai_compat.OpenAICompatProvider` -- OpenAI, OpenRouter, vLLM, etc.
- `forge_iml.providers.anthropic.AnthropicProvider` -- Anthropic Messages API

### ToolProvider

Any registry that can list and execute tools:

```python
from forge_iml.providers.base import ToolProvider

class MyTools(ToolProvider):
    def list_tools(self):
        return [{"name": "weather", "description": "Get weather"}]

    async def execute(self, tool_name, args):
        # Run tool and return {"success": True, "result": ...}
        ...
```

### MemoryProvider

Any store that can search and persist memories:

```python
from forge_iml.providers.base import MemoryProvider

class MyMemory(MemoryProvider):
    def search(self, query, *, limit=5, namespace=None):
        return [{"id": "1", "content": "...", "importance": 5}]

    def save(self, content, *, namespace=None, importance=3,
             tags=None, agent="forge-iml", entry_type="memory"):
        return "new_id"
```

## Pipeline Stages

| Stage | Module | LLM? | Purpose |
|-------|--------|------|---------|
| A | `parser.py` | Yes | Raw text -> IntentGraph via structured extraction |
| B | `canonicalizer.py` | No | Normalize verbs, targets, tools; generate cache key |
| - | `cache.py` | No | 2-layer cache (exact + semantic) |
| - | `memory.py` | No | Inject MemoryClaim nodes from memory store |
| C | `planner.py` | Maybe | Fast path (deterministic) or slow path (LLM) |
| - | `policy.py` | No | Safety gates: block secrets, destructive ops, etc. |
| D | `executor.py` | No | Walk DAG, execute tools, handle retries |
| - | `audit.py` | No | JSONL logging + memory summaries |
| - | `fact_extractor.py` | Yes | Extract (S,P,O) triples from conversations |
| - | `skill_learner.py` | No | Track patterns, promote to reusable templates |

## Benchmarks

Tested against a live deployment (Qwen3.5-397B-A17B, 208GB MoE, Apple Silicon):

| Metric | Value |
|--------|-------|
| Success Rate | 95% (19/20 queries) |
| IML Routed | 95% (18/19 through pipeline) |
| Cache Hit Rate | 79% (15/19 served from cache) |
| Median Latency | 43ms (cache hits) |
| Avg Latency (cache hit) | 1.8s |
| Avg Latency (cache miss) | 8.0s |
| Token Savings | 78.9% vs raw LLM baseline |
| Constrained Gen | 100% first-try valid JSON |
| Fast-Path Hit Rate | 59.5% (regex patterns skip LLM) |

Cache hits are **4.4x faster** than cache misses. Fast-path patterns handle common intents in **<1ms** with zero LLM calls.

### Cost Comparison

```
Raw LLM baseline:   13,300 tokens for 19 queries (700 tokens/query)
With IML pipeline:    2,800 tokens for 19 queries (cache + fast-path)
Token savings:           79% reduction
```

## API Reference

### Core Classes

#### `IntentGraph`
The central data structure — a typed DAG representing parsed user intent.

```python
from forge_iml import IntentGraph

graph = IntentGraph(
    raw_input="check the weather in Austin",
    user_handle="user1",
    nodes=[...],     # List of typed nodes (UserGoal, Constraint, PlanStep, etc.)
    edges=[...],     # Edges with typed relations (REQUIRES, BLOCKS, INFORMS, etc.)
    cache_key="...", # Generated by canonicalizer
)
```

**Node Types** (16): `UserGoal`, `SubGoal`, `Constraint`, `Preference`, `PlanStep`, `ToolCall`, `ToolResult`, `MemoryClaim`, `Observation`, `Hypothesis`, `Decision`, `PolicyGate`, `ErrorRecovery`, `ContextSlot`, `Dependency`, `Output`

**Edge Relations** (10): `REQUIRES`, `BLOCKS`, `INFORMS`, `CONTRADICTS`, `REFINES`, `SEQUENCED_BEFORE`, `PARALLEL_WITH`, `FALLBACK_FOR`, `GATES`, `PRODUCES`

#### `IMLParser`
Stage A — converts raw text to IntentGraph.

```python
parser = IMLParser(llm=my_llm)
graph = await parser.parse("check weather", user_handle="user1")
# Returns IntentGraph with goals, constraints, tool needs
```

Features: fast-path regex patterns (19 built-in), constrained generation (Pydantic validation + repair), multilingual support.

#### `IMLCanonicalizer`
Stage B — normalizes the graph and generates cache keys.

```python
canon = IMLCanonicalizer()
graph = canon.canonicalize(graph)
# graph.cache_key is now set (deterministic SHA-256)
```

Features: verb normalization (50+ aliases), target canonicalization, language-agnostic keys.

#### `IMLCache`
Two-layer caching: Layer 1 (exact SHA-256, <1ms) and Layer 2 (semantic SequenceMatcher, ~50ms).

```python
from forge_iml import IMLCache
cache = IMLCache(max_size=1000)
cached = cache.get(cache_key, semantic_query="whats the weather")
cache.put(cache_key, result_data, result_type="tool_output", raw_query="weather")
```

#### `IMLPlanner`
Stage C — generates PlanStep nodes with dependency ordering.

```python
planner = IMLPlanner(llm=my_llm)
graph = await planner.plan(graph)
# PlanStep nodes added with topological ordering
```

#### `PolicyEngine`
Safety gates — checks before execution.

```python
from forge_iml import PolicyEngine
policy = PolicyEngine()
gates = policy.check(graph)
# Returns list of triggered gates (block/confirm/allow)
```

Built-in gates: secret detection, destructive operations, cost limits, rate limiting, PII filtering.

#### `IMLExecutor`
Stage D — walks the DAG and executes tools.

```python
executor = IMLExecutor(tool_provider=my_tools, policy_engine=policy)
result = await executor.execute(graph)
```

#### Support Modules

| Module | Purpose |
|--------|---------|
| `IMLAudit` | JSONL logging with full execution traces |
| `FactExtractor` | Extract (Subject, Predicate, Object) triples |
| `SkillLearner` | Track patterns, promote to reusable templates |
| `IMLMemory` | Privacy-tier filtering, budget enforcement |

## Installation

```bash
pip install forge-iml
```

Or install from source:

```bash
git clone https://github.com/agntc-labs/forge-iml.git
cd forge-iml
pip install -e ".[dev]"
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

Apache 2.0
