from __future__ import annotations

import json
import traceback
from typing import TYPE_CHECKING, Any

from sac.core import SearchResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import ClassVar

    from sac.sdk import AgenticSearchSDK


class Sandbox:
    _docker_session: ClassVar[Callable[..., Any] | None] = None

    def __init__(self, sdk: AgenticSearchSDK, backend: str = "exec") -> None:
        self._sdk = sdk
        self._backend = backend

    def execute(self, code: str) -> str:
        if self._backend == "docker":
            return self._execute_docker(code)
        return self._execute_exec(code)

    def _execute_exec(self, code: str) -> str:
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

    def _execute_docker(self, code: str) -> str:
        if Sandbox._docker_session is None:
            try:
                from llm_sandbox import SandboxSession  # type: ignore[import-untyped]
            except ImportError as e:
                raise ImportError(
                    "llm-sandbox is required for backend='docker'. "
                    "Install: pip install sac-agent[docker]"
                ) from e

            Sandbox._docker_session = SandboxSession(
                lang="python", keep_template=True
            )
            Sandbox._docker_session.__enter__()

        result = Sandbox._docker_session.run(code)
        output = result.text if hasattr(result, "text") else str(result)
        if result.return_code != 0:
            output = f"--- ERROR ---\n{output}"
        return output

    def close(self) -> None:
        if Sandbox._docker_session is not None:
            try:
                Sandbox._docker_session.__exit__(None, None, None)
            except Exception:
                pass
            Sandbox._docker_session = None
