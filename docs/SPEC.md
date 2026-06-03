# SPEC.md — SAC

## Purpose

Search as Code (SaC) is a CLI tool and Python library for agentic research orchestration.
It uses an LLM (OpenAI-compatible API) to generate and execute Python code that performs
web searches, extracts structured data, and synthesizes findings — all within a sandboxed
execution environment with persistent key-value storage.

## Scope

### What IS in scope

- Multi-turn research agent that generates Python code to search, extract, and synthesize
- Web search via Exa MCP (free) or Brave Search API (with key)
- LLM integration via any OpenAI-compatible API (default: opencode Zen)
- Persistent key-value file store for inter-turn state
- Code sandbox that executes generated Python with a restricted `sdk` namespace
- Auto-fix loop: when generated code errors, the LLM is asked to fix it (up to 3 attempts)
- Interactive mode (REPL-style) and one-shot CLI mode
- Utility functions: deduplication, filtering, flattening, coverage summarization
- Simulated search results when no real search backend is available

### What is NOT in scope

- Multi-agent coordination (single agent only)
- Concurrent agent runs
- Web UI or API server
- Database-backed persistence (only pickle files)
- Plugin system for custom search backends
- Caching across separate CLI invocations

## Public API / Interface

### `sac.core.SearchResult`

```python
@dataclass
class SearchResult:
    url: str = ""
    title: str = ""
    snippet: str = ""
    domain: str = ""
```

- `__post_init__` auto-populates `domain` from `url` if empty
- `__repr__` returns `SearchResult(domain=..., title=...)`

### `sac.core._extract_domain(url: str) -> str`

Extracts netloc from URL. Returns empty string if parsing fails.

### `sac.core._format_items(items: list[Any], max_chars: int = 8000) -> str`

JSON-dumps items, truncating if over `max_chars`.

### `sac.search.SearchSDK`

```python
class SearchSDK:
    def __init__(self, brave_key: str | None = None)
    def web(self, query: str, limit: int = 8) -> list[SearchResult]
    def web_many(self, queries: list[str | dict], limit_per_query: int = 8, concurrency: int = 6) -> list[list[SearchResult]]
    def neural(self, query: str, limit: int = 8) -> list[SearchResult]
```

- `web()` and `neural()` are aliases for `_search_one`
- Caches results by `query:::limit` key
- Falls back through: Brave → Exa MCP → simulation

### `sac.llm.LLMSDKClient`

```python
class LLMSDKClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str = "big-pickle", max_tokens: int = 4096)
    def synthesize(self, items: list[Any], instruction: str) -> str
    def plan(self, context: str, goal: str) -> str
    def extract_many(self, items: list[dict], instruction: str, schema: dict[str, type | str]) -> list[dict]
```

- `synthesize()` wraps items in `<context>` tags with an instruction
- `plan()` produces a search plan from context and goal
- `extract_many()` chunks items by 10 and calls `_extract_chunk()` for JSON extraction
- `_parse_json_list()` extracts JSON array from LLM response text

### `sac.storage.FilesystemSDK`

```python
class FilesystemSDK:
    def __init__(self, fs_dir: str | Path | None = None)
    def write(self, key: str, data: Any) -> None
    def read(self, key: str) -> Any
    def list() -> list[str]
    def exists(self, key: str) -> bool
```

- Default dir is a temp directory
- Keys are sanitized with `_safe_key()` (replaces non-alphanumeric with `_`)
- Pickle-based serialization

### `sac.utils.UtilsSDK`

```python
class UtilsSDK:
    @staticmethod
    def dedupe_by(items: list, key: str | Callable) -> list
    @staticmethod
    def filter_by(items: list, field: str | Callable, value: Any | None = None) -> list
    @staticmethod
    def summarize_coverage(items: list, by_fields: list[str]) -> str
    @staticmethod
    def flatten(list_of_lists: list[list]) -> list
    @staticmethod
    def join_result_fields(result: Any) -> str
```

### `sac.sdk.AgenticSearchSDK`

```python
class AgenticSearchSDK:
    def __init__(self, llm_base_url: str | None = None, llm_api_key: str | None = None, llm_model: str = "big-pickle", fs_dir: str | Path | None = None, brave_key: str | None = None)
    # Attributes: search, llm, fs, utils
```

Composable SDK container. All sub-SDKs are public attributes.

### `sac.sandbox.Sandbox`

```python
class Sandbox:
    def __init__(self, sdk: AgenticSearchSDK)
    def execute(self, code: str) -> str
```

Executes Python code with `sdk`, `SearchResult`, `json`, and custom `print()` in scope.
Returns output string. Errors are captured as `--- ERROR ---\n<traceback>`.

### `sac.agent.SaCAgent`

```python
class SaCAgent:
    def __init__(self, task: str, sdk: AgenticSearchSDK | None = None, max_turns: int = 6, max_fixes_per_turn: int = 3, base_url: str | None = None, api_key: str | None = None, model: str = "big-pickle")
    def run() -> str
```

Multi-turn research agent. Returns final synthesis string.

### CLI

```bash
sac <research task>
sac -v <research task>   # verbose mode
sac                       # interactive mode
```

## Data Formats

- Search cache: `dict[str, list[SearchResult]]` keyed by `query:::limit`
- Filesystem: pickle files named `<safe_key>.pkl` in a directory
- Agent history: list of dicts with `role` and `content` keys
- Agent response: JSON with `turn_type` ("code" or "synthesis"), `reasoning`, and `code`/`answer`
- Exa MCP: JSON-RPC 2.0 over HTTP with SSE-style `data:` lines

## Edge Cases

1. Empty query string — returns simulated results with default domain
1. No search backends available — gracefully falls back to simulated results
1. LLM API unreachable — `LLMSDKClient` raises `ImportError` or API error
1. Missing `openai` or `requests` packages — explicit `ImportError` at construction time (not at module import)
1. Corrupt pickle files — `FilesystemSDK.read()` raises `pickle.UnpicklingError`
1. Generated code has syntax errors — `Sandbox.execute()` captures and returns error
1. Agent response is not valid JSON — `_parse_response()` returns `None`
1. Empty history on first turn — agent constructs initial prompt with task + empty fs keys
1. Max turns reached without synthesis — agent force-synthesizes from persisted state
1. Fix loop yields no improvement — agent gives up and returns last error output
1. File key with special characters — sanitized to alphanumeric + underscore + hyphen
1. Very large items in `_format_items` — truncated at `max_chars` with `[truncated]` suffix

## Performance & Constraints

- No hard O(n) requirements
- Concurrency in `web_many()` defaults to 6 parallel threads
- `extract_many()` processes chunks of 10 items sequentially
- Filesystem operations are O(1) per key (hash-based lookup)
- LLM calls are the primary latency bottleneck
- No streaming support
