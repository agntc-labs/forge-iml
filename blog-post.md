# How We Cut Agent Costs 79% with a Compiler Layer

*TL;DR: We added a structured intermediate representation (IR) between natural language input and tool execution. The result: 79% fewer tokens, 43ms median latency (down from 8 seconds), and zero hallucinated tool calls.*

---

## The Problem

We run [Leila](https://github.com/agntc-labs), a personal AI assistant with 152 tools, on a Mac Studio with a 397B parameter model (Qwen3.5-397B-A17B running via MLX on 208GB of unified memory). She handles everything from weather queries to financial data to smart home control across iMessage, Telegram, an iOS app, and a REST API.

The problem? Every message — even "what time is it" — was burning a full LLM round-trip. That's ~700 tokens of input (system prompt + tool descriptions + conversation history) plus ~200 tokens of output, taking 5-10 seconds. For a 397B model running locally, that's a lot of compute for "it's 2:30 PM."

We needed a way to route simple queries instantly while preserving the full power of the LLM for complex reasoning.

## The Insight: Treat Intent as an IR

Programming language compilers don't interpret source code directly. They parse it into an intermediate representation (IR), optimize it, then emit machine code. We applied the same idea to natural language:

```
Raw text → Parse → Canonicalize → Cache → Plan → Execute → Render
```

We call this IR the **IntentGraph** — a typed directed acyclic graph with 16 node types and 10 edge relation types. Each node represents something like a `UserGoal`, `Constraint`, `PlanStep`, `ToolCall`, `MemoryClaim`, or `PolicyGate`. Edges express relationships: `REQUIRES`, `BLOCKS`, `PARALLEL_WITH`, `GATES`, etc.

## The Pipeline

### Stage A: Parse (Fast-Path + LLM)

The parser first checks 19 regex patterns for common intents. "What's the weather?" matches instantly (<1ms) and produces a valid IntentGraph without any LLM call. Our fast-path handles 59.5% of real traffic.

For everything else, the parser calls the LLM with **constrained generation** — a Pydantic schema that guarantees valid JSON output. No more hallucinated node types or malformed tool calls. First-try success rate: 100%.

### Stage B: Canonicalize (Pure Python, No LLM)

The canonicalizer normalizes the graph:
- **Verb normalization**: "check", "look at", "show me", "tell me about" → `get`
- **Target canonicalization**: "weather", "forecast", "temperature" → `weather`
- **Language-agnostic keys**: Indonesian "cuaca" maps to the same cache key as English "weather"
- **Deterministic cache key**: SHA-256 hash of the canonical form

This is pure Python string manipulation — no LLM call, sub-millisecond.

### Cache: Two Layers

**Layer 1 (Exact)**: SHA-256 match of the canonical form. Sub-millisecond. Handles exact repeats and trivially different phrasings that canonicalize identically.

**Layer 2 (Semantic)**: SequenceMatcher fuzzy matching against stored queries. ~50ms. Catches paraphrases that the canonicalizer doesn't collapse: "is it raining?" matches "what's the weather like?"

Cache hit rate after warmup: **79%**.

### Stages C/D: Plan & Execute

On cache miss, the planner generates a DAG of execution steps (sometimes via LLM, sometimes deterministically for simple tool calls). The executor walks the DAG in topological order, running tools and collecting results.

A **PolicyEngine** checks every execution against 10 safety gates: secret detection, destructive operations, cost limits, rate limiting, and PII filtering.

### Support Modules

- **Audit**: Every execution is logged as JSONL with full traces. Queryable via API.
- **FactExtractor**: Extracts (Subject, Predicate, Object) triples from conversations and stores them in the memory system.
- **SkillLearner**: Tracks recurring patterns and promotes them to reusable templates. We've learned 52 skill templates so far.

## Results

After deploying IML on all channels (iMessage, Telegram, iOS app, REST API), here's what we measured across a 20-query benchmark:

| Metric | Before (Raw LLM) | After (IML Pipeline) |
|--------|-------------------|----------------------|
| Median Latency | ~5,000ms | **43ms** |
| Token Usage | 700/query | **147/query avg** |
| Cache Hit Rate | 0% | **79%** |
| Parse Failures | ~5% | **0%** |
| Hallucinated Tools | Occasional | **Zero** (constrained gen) |
| Fast-Path Coverage | 0% | **59.5%** |

The headline number: **79% fewer tokens** consumed. For the 59.5% of queries hitting fast-path + cache, the savings are 100% — zero LLM tokens at all.

## What We Learned

**1. Most agent traffic is repetitive.** People ask the same things in slightly different ways. A good canonicalizer + two-layer cache handles 79% of queries without touching the LLM.

**2. Constrained generation eliminates parse failures.** Pydantic schema validation with a repair layer means the parser produces valid IntentGraphs 100% of the time. No more "sorry, I didn't understand that."

**3. The compiler metaphor is powerful.** Treating natural language as source code that gets compiled into an IR opens up all the classical compiler optimizations: constant folding (fast-path), common subexpression elimination (cache), dead code elimination (policy gates), and register allocation (tool scheduling).

**4. You don't need a cloud.** This runs on a single Mac Studio (M4 Ultra, 512GB RAM) with a 397B model via MLX. No API calls, no cloud costs, full privacy. The IML layer makes local inference practical by reducing how often you actually need the LLM.

## Open Source

forge-iml is available as a pip-installable package:

```bash
pip install forge-iml
```

It's provider-agnostic — bring your own LLM (MLX, OpenAI, Anthropic, local vLLM), your own tool registry, and your own memory store. The pipeline doesn't care where the intelligence comes from.

Repository: [github.com/agntc-labs/forge-iml](https://github.com/agntc-labs/forge-iml)

---

*Built at [Agentic Labs](https://agentic-labs.web.app) by Justin. Leila runs 24/7 on a Mac Studio handling family messages in English and Indonesian.*
