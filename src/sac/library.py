from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, cast

from sac.core import _httpx_client, _log

CACHE_DIR = Path.home() / ".cache" / "sac-agent"
LIBRARY_DIR = CACHE_DIR / "library"
INDEX_FILE = CACHE_DIR / "index.json"


CATEGORY_FILES = {
    "search": "search",
    "extraction": "extraction",
    "synthesis": "synthesis",
    "analysis": "analysis",
    "storage": "storage",
    "pipeline": "pipeline",
}


def _classify_code(code: str) -> str:
    if "sdk.llm.extract_many" in code or "sdk.llm.extract" in code:
        return "extraction"
    if "sdk.llm.synthesize" in code:
        return "synthesis"
    if "sdk.search." in code:
        return "search"
    if "sdk.fs." in code:
        return "storage"
    if "sdk.utils." in code:
        return "analysis"
    return "pipeline"


def _category_file(category: str) -> Path:
    filename = CATEGORY_FILES.get(category, "pipeline")
    return LIBRARY_DIR / f"{filename}.py"


def _ensure_dirs() -> None:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    _sync_init()


def _sync_init() -> None:
    category_files = sorted(
        {_category_file(c) for c in CATEGORY_FILES.values()}
        | {LIBRARY_DIR / f"{v}.py" for v in set(CATEGORY_FILES.values())}
    )
    seen = set()
    lines = [
        "# Auto-generated library of reusable research functions.\n",
        "# Each function accepts `sdk` (AgenticSearchSDK) as its first argument.\n",
        "\nfrom __future__ import annotations\n\n",
    ]
    for f in sorted(LIBRARY_DIR.glob("*.py")):
        mod = f.stem
        if mod == "__init__" or mod in seen:
            continue
        seen.add(mod)
        lines.append(f"from sac.library.{mod} import *\n")
    init_path = CACHE_DIR / "__init__.py"
    init_path.write_text("".join(lines))


def _load_index() -> dict[str, Any]:
    if INDEX_FILE.exists():
        return cast("dict[str, Any]", json.loads(INDEX_FILE.read_text()))
    return {}


def _save_index(index: dict[str, Any]) -> None:
    INDEX_FILE.write_text(json.dumps(index, indent=2))


def _safe_func_name(task: str, code: str) -> str:
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", task.lower())
    keywords = {
        "and",
        "or",
        "not",
        "the",
        "a",
        "an",
        "in",
        "of",
        "to",
        "for",
        "is",
        "are",
        "with",
        "what",
        "how",
        "why",
    }
    filtered = [w for w in words if w not in keywords and len(w) > 1]
    stem = "_".join(filtered[:4]) if filtered else "research_fn"
    # Derive a short hash from the code
    code_hash = abs(hash(code)) % 10**4
    return f"{stem}_{code_hash}"


def _extract_sdk_calls(code: str) -> str:
    lines = code.strip().split("\n")
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("print("):
            continue
        if "sdk." in stripped:
            out.append(line)
        elif stripped and not stripped.startswith("print("):
            out.append(line)
    return "\n".join(out)


def _generate_function(
    code: str,
    task: str,
    base_url: str,
    api_key: str,
    model: str,
) -> tuple[str, str]:
    func_name = _safe_func_name(task, code)
    cleaned = _extract_sdk_calls(code)
    body_indent = _indent_body(cleaned)
    docstring = _make_docstring(task, cleaned)
    func_code = (
        f"def {func_name}(sdk):\n"
        f'    """{docstring}'
        f"    Args:\n"
        f"        sdk: AgenticSearchSDK instance\n"
        f'    """\n'
        f"{body_indent}"
    )
    return func_name, func_code


def _indent_body(code: str) -> str:
    lines = code.strip().split("\n")
    return "\n".join(f"    {line}" for line in lines)


def _make_docstring(task: str, code: str) -> str:
    first_line = task.strip().split("\n")[0][:80]
    sdk_methods = re.findall(r"sdk\.(\w+\.\w+)", code)
    if sdk_methods:
        methods = ", ".join(sorted(set(sdk_methods)))
        return f"Execute {methods} for: {first_line}.\n\n"
    return f"Research operation for: {first_line}.\n\n"


def list_functions() -> list[dict[str, Any]]:
    index = _load_index()
    return [
        {
            "name": name,
            "task": meta.get("task", ""),
            "category": meta.get("category", ""),
            "file": str(meta.get("file", "")),
            "created": meta.get("created", ""),
        }
        for name, meta in index.items()
    ]


def load_function(name: str) -> Any | None:  # noqa: ANN401
    """Dynamically import a function from the library by name."""
    import importlib.util  # noqa: PLC0415

    index = _load_index()
    meta = index.get(name)
    if meta is None:
        return None
    filepath = Path(meta["file"])
    if not filepath.exists():
        return None
    spec = importlib.util.spec_from_file_location(meta["category"], str(filepath))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name, None)


class CodeLibrary:
    """Manages a persistent library of reusable research functions."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "big-pickle",
    ) -> None:
        self._base_url = (
            base_url
            or os.environ.get("OPENAI_API_BASE")
            or "https://opencode.ai/zen/v1"
        )
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or "public"
        self._model = model or os.environ.get("SAC_MODEL") or "big-pickle"
        _ensure_dirs()
        self._pending: list[tuple[str, str]] = []

    def collect(self, code: str, task: str) -> None:
        self._pending.append((code, task))

    def flush_all(self) -> list[str]:
        names: list[str] = []
        for code, task in self._pending:
            try:
                name = self._save_one(code, task)
                names.append(name)
            except Exception as e:
                _log(f"Failed to save library function: {e}")
        self._pending.clear()
        return names

    def _save_one(self, code: str, task: str) -> str:
        index = _load_index()
        dedup_key = code.strip()[:200]
        for existing_name, existing_meta in index.items():
            if existing_meta.get("code_preview") == dedup_key:
                return existing_name

        func_name, func_code = _generate_function(
            code,
            task,
            self._base_url,
            self._api_key,
            self._model,
        )
        category = _classify_code(code)
        filepath = _category_file(category)
        if filepath.exists():
            existing = filepath.read_text().rstrip()
            filepath.write_text(f"{existing}\n\n\n{func_code}\n")
        else:
            filepath.write_text(f"{func_code}\n")

        index[func_name] = {
            "task": task,
            "category": category,
            "file": str(filepath),
            "code_preview": dedup_key,
            "created": os.path.getmtime(filepath) if filepath.exists() else 0,
        }
        _save_index(index)
        _sync_init()
        return func_name
