**sac** — Search as Code: agentic search orchestration using LLMs.

[![PyPI](https://img.shields.io/pypi/v/sac.svg)](https://pypi.org/project/sac/)
[![Python](https://img.shields.io/pypi/pyversions/sac.svg)](https://pypi.org/project/sac/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/master/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Search%20as%20Code-blue)](https://deepwiki.ai/repo/daedalus/SAC)

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
