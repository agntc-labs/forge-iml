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
