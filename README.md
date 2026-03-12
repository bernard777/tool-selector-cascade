# tool-selector-cascade

> A 3-level cascading tool selector for AI agents.
> **Filter 1 000 tools down to the 1 most relevant** in ~450 ms at ~$0.0001/call.

[![CI](https://github.com/bernard777/tool-selector-cascade/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/bernard777/tool-selector-cascade/actions/workflows/ci.yml)
[![Security](https://github.com/bernard777/tool-selector-cascade/actions/workflows/security.yml/badge.svg?branch=master)](https://github.com/bernard777/tool-selector-cascade/actions/workflows/security.yml)
[![PyPI version](https://img.shields.io/pypi/v/tool-selector-cascade.svg)](https://pypi.org/project/tool-selector-cascade/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![codecov](https://codecov.io/gh/bernard777/tool-selector-cascade/branch/master/graph/badge.svg)](https://codecov.io/gh/bernard777/tool-selector-cascade)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

---

## Why?

When an AI agent has hundreds of tools, naively passing all of them to the LLM:

- Wastes tokens (cost ↑, speed ↓)
- Confuses the LLM (quality ↓)
- Hits context window limits

`tool-selector-cascade` solves this with a **3-level cascade**:

```
[1 000 tools]
      |
      v  Level 1 — Embedding   (~20 ms, local, ~$0)
      |  intfloat/multilingual-e5-base · cosine similarity
      |
[top 20]
      |
      v  Level 2 — Reranker    (~243 ms, local, ~$0)
      |  Alibaba-NLP/gte-reranker-modernbert-base · cross-encoder
      |
[top 5]
      |
      v  Level 3 — Micro-LLM  (~150 ms, API, ~$0.0001)
      |  claude-haiku-4-5 / gemini-2.5-flash-lite / gpt-4o-mini
      |
[1 tool]
```

**Total budget per selection:** ~570 MB RAM · ~400–600 ms · ~$0.0001

---

## Quick Start

### Install

```bash
pip install tool-selector-cascade[all]
```

Or with specific level dependencies:

```bash
pip install tool-selector-cascade[level1,level2]                    # local models only
pip install tool-selector-cascade[level1,level2,level3-anthropic]   # + Anthropic
pip install tool-selector-cascade[level1,level2,level3-google]      # + Google
pip install tool-selector-cascade[level1,level2,level3-openai]      # + OpenAI
```

### Basic usage

```python
import asyncio
from tool_selector_cascade import CascadeSelector

# Any object with .name and .description works (LangChain StructuredTool,
# plain dict, or custom dataclass)
class Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

tools = [
    Tool("web_search",   "Search the web for information"),
    Tool("gmail_send",   "Send an email via Gmail"),
    Tool("calendar_add", "Create a calendar event"),
    # ... up to 1 000+ tools
]

selector = CascadeSelector()
selector.warm_up(tools)   # pre-load models once at startup

async def main():
    result, metrics = await selector.aselect(
        intent="send an email to the team",
        tools=tools,
    )
    print(result[0].name)       # "gmail_send"
    print(metrics.as_dict())    # latency + cost breakdown

asyncio.run(main())
```

### Synchronous (Level 1 + 2 only, no LLM cost)

```python
candidates, metrics = selector.select(
    intent="search for Python tutorials",
    tools=tools,
    top_k=5,
)
```

### Convenience functions (no setup required)

```python
from tool_selector_cascade import select_tools_for_intent, warm_up
import threading

# Pre-load models at startup (non-blocking)
threading.Thread(target=warm_up, kwargs={"tools": all_tools}, daemon=True).start()

# One-liner — Level 1 + 2 only, no LLM cost
relevant = select_tools_for_intent(
    intent="send an email to the team",
    tools=tools,
    top_k=5,
)
```

---

## Architecture

### Level 1 — Embedding Filter

| Property     | Value |
|--------------|-------|
| Model        | [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) |
| Parameters   | 278 M |
| RAM          | ~270 MB |
| Languages    | 100+ (FR, EN, DE, ES, ZH, …) |
| Latency      | ~20 ms (warm, CPU) |
| Cost         | $0 (local) |

Computes **cosine similarity** between the intent embedding and all tool
`"name: description"` embeddings.  Tool embeddings are cached by pool hash
(MD5) so repeated calls on the same pool are near-instant.

### Level 2 — Cross-Encoder Reranker

| Property     | Value |
|--------------|-------|
| Model        | [`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) |
| Parameters   | 149 M |
| RAM          | ~300 MB |
| Latency      | ~243 ms / 100 pairs (CPU) |
| Cost         | $0 (local) |

Applies **cross-attention** between the intent and each candidate — richer
than bi-encoder cosine similarity at the cost of ~10× more compute.

### Level 3 — Micro-LLM Picker

| Provider  | Model                   | Latency  | Cost / 1M input tokens |
|-----------|-------------------------|----------|------------------------|
| Anthropic | `claude-haiku-4-5`      | ~150 ms  | $0.25                  |
| Google    | `gemini-2.5-flash-lite` | ~150 ms  | $0.075                 |
| OpenAI    | `gpt-4o-mini`           | ~200 ms  | $0.15                  |

Sends the top-5 candidates to the LLM and asks it to **reason** about which
single tool best matches the intent.

### Graceful Degradation

Each level degrades gracefully:

- Level 1 unavailable → returns `tools[:top_k]`
- Level 2 unavailable → returns Level 1 result
- Level 3 unavailable / API error → returns Level 2 top-1

---

## Configuration

```python
from tool_selector_cascade import CascadeSelector, CascadeConfig

selector = CascadeSelector(CascadeConfig(
    # Level 1
    embedder_model="intfloat/multilingual-e5-base",
    embedder_top_k=20,
    embedder_min_pool=20,

    # Level 2
    reranker_model="Alibaba-NLP/gte-reranker-modernbert-base",
    reranker_top_k=5,
    reranker_enabled=True,

    # Level 3
    llm_provider="anthropic",       # "anthropic" | "google" | "openai"
    llm_model="claude-haiku-4-5",
    llm_api_key=None,               # reads ANTHROPIC_API_KEY if None
    llm_enabled=True,
    llm_timeout=30.0,

    # General
    always_include_prefixes=["web_search"],  # always present in output
    min_pool_size=5,                # skip all filtering below this count
))
```

### Environment Variables

| Variable           | Description                          |
|--------------------|--------------------------------------|
| `ANTHROPIC_API_KEY`| Anthropic API key (Level 3)          |
| `OPENAI_API_KEY`   | OpenAI API key (Level 3)             |
| `GOOGLE_API_KEY`   | Google API key (Level 3)             |

---

## Benchmarks

Measured on AMD Ryzen 9 5900X, 32 GB RAM, CPU-only inference:

| Scenario           | Tools      | Latency   | LLM Cost      |
|--------------------|------------|-----------|---------------|
| Level 1 only       | 1000 → 20  | ~20 ms    | $0            |
| Level 1 + 2        | 1000 → 5   | ~263 ms   | $0            |
| Full cascade       | 1000 → 1   | ~420 ms   | ~$0.0001      |
| Warm cache (L1)    | 1000 → 20  | ~5 ms     | $0            |

---

## Integration

Drop `tool-selector-cascade` into any agent framework by replacing your
existing tool-filtering step with :func:`select_tools_for_intent` for an
immediate quality improvement:

```python
# Before (single-level bi-encoder)
from your_framework.tools import filter_tools
relevant = filter_tools(intent, all_tools)[:5]

# After (3-level cascade) — same interface
from tool_selector_cascade import select_tools_for_intent
relevant = select_tools_for_intent(intent, all_tools, top_k=5)
```

For the full 3-level cascade with LLM reasoning:

```python
from tool_selector_cascade import CascadeSelector

selector = CascadeSelector()

# In your async reasoning step:
result, metrics = await selector.aselect(intent, tools)
best_tool = result[0]
```

---

## Development

```bash
git clone https://github.com/tool-selector-cascade/tool-selector-cascade
cd tool-selector-cascade
pip install -e ".[dev,all]"
pytest tests/ -v
```

---

## License

MIT © 2026 [Jean Bernard NDONGO AMBASSA](mailto:ndongoambassa7@gmail.com)

