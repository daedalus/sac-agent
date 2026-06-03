**sac** — Search as Code: agentic search orchestration using LLMs.

[![PyPI](https://img.shields.io/pypi/v/sac.svg)](https://pypi.org/project/sac/)
[![Python](https://img.shields.io/pypi/pyversions/sac.svg)](https://pypi.org/project/sac/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/master/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Search%20as%20Code-blue)](https://deepwiki.com/daedalus/SAC)

## Install

```bash
pip install sac
uv pip install sac-agent
```

## Usage

```python
from sac import AgenticSearchSDK, SaCAgent

sdk = AgenticSearchSDK()
agent = SaCAgent(task="Research the latest AI papers on retrieval-augmented generation", sdk=sdk)
answer = agent.run()
print(answer)
```

## CLI

```bash
sac "What are the latest developments in LLM agents?"
sac -v "Verbose research mode"
sac --endpoint https://opencode.ai/zen/v1 --model big-pickle "task"
sac --final-report ./report.md "task"
sac --final-report-format json --final-report ./output "task"
sac  # interactive mode
```

## API

- `AgenticSearchSDK` — Composable SDK with `search`, `llm`, `fs`, `utils` attributes
- `SaCAgent` — Multi-turn research agent that generates and executes search code
- `SearchSDK` — Web search via Exa MCP (free) or Brave Search API
- `LLMSDKClient` — LLM integration via OpenAI-compatible API
- `FilesystemSDK` — Persistent key-value store
- `UtilsSDK` — Deduplication, filtering, flattening, coverage summarization
- `Sandbox` — Executes generated Python code in a restricted namespace

## Code Library

Every code snippet the agent generates can be saved as a reusable, callable function with `--with-code-library`:

```bash
sac --with-code-library "Research the latest AI papers"
```

Functions are grouped by concern into `~/.cache/sac-agent/library/` (classification priority: extraction > synthesis > search > storage > analysis > pipeline):

| File | Concern |
|------|---------|
| `extraction.py` | `sdk.llm.extract_many` operations |
| `synthesis.py` | `sdk.llm.synthesize` operations |
| `search.py` | `sdk.search.*` operations |
| `storage.py` | `sdk.fs.*` operations |
| `analysis.py` | `sdk.utils.*` operations |
| `pipeline.py` | Mixed / catch-all |

Execution flow:

```
CLI (sac "task")
  → SaCAgent.run()
    → _call_model()                   # LLM generates JSON with "code"
    → sandbox.execute(code)           # exec() in restricted namespace
    → if successful: library.collect()  # save snippet in memory
    → repeat up to 6 turns
    → on synthesis: library.flush_all()
      → _classify_code() per snippet  # extraction > synthesis > search > ...
      → append to category file       # search.py / extraction.py / pipeline.py
```

Each function accepts `sdk` as its first parameter and has a Google-style docstring:

```python
def search_fanout(sdk, queries, limit_per_query=5, concurrency=3):
    """Execute parallel searches across multiple queries and flatten results.

    Args:
        sdk: AgenticSearchSDK instance
        queries: List of search query strings
        limit_per_query: Results per query (default: 5)
        concurrency: Number of parallel workers (default: 3)

    Returns:
        Flattened list of SearchResult objects
    """
    results = sdk.search.web_many(queries, ...)
    return sdk.utils.flatten(results)
```

Load functions by name at runtime:

```python
from sac.library import load_function

fn = load_function("search_fanout_1234")
results = fn(sdk)
```

## Development

```bash
git clone https://github.com/daedalus/SAC.git
cd SAC
pip install -e ".[test]"

# run tests
pytest

# format
ruff format src/ tests/

# format markdown
mdformat .

# lint + type check (prospector runs ruff check + mypy + pylint together)
prospector --with-tool ruff --with-tool mypy --with-tool pylint src/

# find unused code (vulture reports dead code with 90%+ confidence)
vulture --min-confidence 90 src/

# analyze code complexity (lizard reports cyclomatic complexity, NLOC, etc.)
lizard src/ --CCN=15

## References

This project is inspired by and implements the architecture described in:

> Perplexity AI. "Rethinking Search as Code Generation." *Perplexity Research*, June 1, 2026.
> <https://research.perplexity.ai/articles/rethinking-search-as-code-generation>
```
