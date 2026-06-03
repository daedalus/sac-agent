from __future__ import annotations

import json
import traceback
from typing import TYPE_CHECKING, Any

from sac.core import SearchResult

if TYPE_CHECKING:
    from sac.sdk import AgenticSearchSDK


class Sandbox:
    def __init__(self, sdk: AgenticSearchSDK) -> None:
        self._sdk = sdk

    def execute(self, code: str) -> str:
        namespace: dict[str, Any] = {
            "sdk": self._sdk,
            "SearchResult": SearchResult,
            "json": json,
        }
        output_lines: list[str] = []

        def _print(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            text = " ".join(str(a) for a in args)
            end = kwargs.get("end", "\n")
            output_lines.append(text + end.rstrip("\n"))

        namespace["print"] = _print

        try:
            exec(code, namespace)
        except Exception:
            output_lines.append("--- ERROR ---")
            output_lines.append(traceback.format_exc())

        return "\n".join(output_lines)
