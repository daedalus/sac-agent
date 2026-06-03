from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


class UtilsSDK:
    @staticmethod
    def _get_value(item: Any, key: str | Callable[..., Any]) -> Any:  # noqa: ANN401
        if callable(key):
            return key(item)
        if isinstance(item, dict):
            return item.get(key)
        return getattr(item, key, None)

    @staticmethod
    def dedupe_by(items: list[Any], key: str | Callable[..., Any]) -> list[Any]:
        seen: set[Any] = set()
        result: list[Any] = []
        for item in items:
            val = UtilsSDK._get_value(item, key)
            if val not in seen:
                seen.add(val)
                result.append(item)
        return result

    @staticmethod
    def filter_by(
        items: list[Any], field: str | Callable[..., Any], value: Any | None = None
    ) -> list[Any]:  # noqa: ANN401
        if callable(field):
            return [item for item in items if field(item)]
        return [item for item in items if UtilsSDK._get_value(item, field) == value]

    @staticmethod
    def summarize_coverage(items: list[Any], by_fields: list[str]) -> str:
        lines = []
        for field in by_fields:
            counts: dict[str, int] = {}
            for item in items:
                val = UtilsSDK._get_value(item, field)
                val_str = str(val) if val is not None else "None"
                counts[val_str] = counts.get(val_str, 0) + 1
            lines.append(f"Coverage by '{field}':")
            for val, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {val}: {cnt}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def flatten(list_of_lists: list[list[Any]]) -> list[Any]:
        return [item for sublist in list_of_lists for item in sublist]

    @staticmethod
    def join_result_fields(result: Any) -> str:  # noqa: ANN401
        if isinstance(result, dict):
            return f"{result.get('title', '')} | {result.get('snippet', '')}"
        return f"{getattr(result, 'title', '')} | {getattr(result, 'snippet', '')}"

    @staticmethod
    def normalize_url(url: str) -> str:
        parsed = re.split(r"(https?://)", url.strip(), maxsplit=1)
        if len(parsed) < 2:
            return url.strip().rstrip("/")
        scheme = parsed[1].lower()
        rest = parsed[2].rstrip("/")
        return f"{scheme}{rest}"

    @staticmethod
    def filter_by_regex(items: list[Any], field: str, pattern: str) -> list[Any]:  # noqa: ANN401
        compiled = re.compile(pattern)
        return [
            item
            for item in items
            if compiled.search(str(UtilsSDK._get_value(item, field)))
        ]

    @staticmethod
    def chunk(items: list[Any], size: int = 10) -> list[list[Any]]:
        return [items[i : i + size] for i in range(0, len(items), size)]
